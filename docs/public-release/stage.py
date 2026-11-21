#!/usr/bin/env python3
"""Create a new, auditable public source candidate; never publish or delete originals."""
import argparse, hashlib, json, os, re, shutil
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--audit',type=Path,required=True); p.add_argument('--private-name',action='append',default=[]); a=p.parse_args()
root=Path(__file__).resolve().parents[2]; out=a.out.resolve(); audit=a.audit.resolve()
if out.exists() or out==root or root in out.parents: raise SystemExit('output must be NEW and outside source tree')
out.mkdir(parents=True); audit.mkdir(parents=True,exist_ok=True)
roots={'audio','cli','crates','glue','image','kernels','llm','model-cards','model3d','vendor','vision','wfe'}
blocked={'target','dist','build','node_modules','__pycache__','.git','evidence','reference','integration','validation','reference-sd15-prompt','vae_diagnostic_v1'}
exts={'.rs','.toml','.lock','.h','.hpp','.hip','.c','.cpp','.cu','.cuh','.js','.mjs','.ts','.tsx','.json','.css','.html','.svg','.param','.txt','.map','.bcmap','.pfb','.ttf','.woff','.woff2'}
special={'Cargo.toml','Cargo.lock','README.md','.gitignore'}
explicit={'reference/inkling2-mini/nvfp4-decoder-v1/LICENSE','reference/inkling2-mini/nvfp4-decoder-v1/quant_upstream_oracle.hpp','reference/inkling2-mini/nvfp4-decoder-v1/upstream_oracle.hpp','reference/qwenimage21-turbo/softmax_source_scalar.hpp','reference/qwenimage21-turbo/silu_source_scalar.hpp','reference/qwenimage21-turbo/qkv_order_scalar.hpp','reference/qwenimage21-turbo/te-softmax-complete-spec-v1/sleef-source/LICENSE.txt','reference/qwenimage21-turbo/te-softmax-complete-spec-v1/sleef-source/Torch-LICENSE','audio/voxcpm2/tests/fixtures/tiny-state.pth','image/paddleocrvl/reference/protocol.json','image/paddleocrvl/reference/protocol-f32.json','reference/paddleocr/weights.json','crates/diffusion/reference/samplers/expected.txt'}
secret_patterns={
 'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
 'provider_token':r'\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{35,})\b',
 'credential_literal':r'(?i)["\'](?:api_key|access_token|refresh_token|password|client_secret)["\']\s*:\s*["\'][A-Za-z0-9_+/=-]{20,}["\']',
}
files=[]; excluded=[]; edits=[]; findings=[]; licenses=[]
for base,dirs,names in os.walk(root):
    dirs[:]=sorted(d for d in dirs if d not in {'target','node_modules','.git','__pycache__','.reference-venv','dist'})
    for name in sorted(names):
        src=Path(base)/name; rel=src.relative_to(root); s=rel.as_posix()
        if src.is_symlink(): excluded.append({'path':s,'reason':'symlink'}); continue
        notice=any(x in name.upper() for x in ('LICENSE','COPYING','NOTICE'))
        allowed=(s in special or s in explicit or (s.startswith('docs/public-release/') and not s.startswith('docs/public-release/assets/')) or (rel.parts[0]=='model-cards' and src.suffix in {'.yml','.yaml','.json','.md'}))
        if rel.parts[0] in roots and not any(x in blocked for x in rel.parts):
            allowed |= (src.suffix in exts or notice or name=='VENDORED' or (name=='build.sh' and rel.parts[0]=='wfe'))
        # Operational inventories/reports are not application source, even when JSON.
        if src.suffix in {'.json','.txt'} and any(x in name.lower() for x in ('manifest','audit','report','timings','profile_results','handoff','verification','preflight','identity','recipe-provenance')) and s not in explicit: allowed=False
        if not allowed: excluded.append({'path':s,'reason':'not public source allowlist'}); continue
        raw=src.read_bytes()
        if len(raw)>16*1024*1024: excluded.append({'path':s,'reason':'oversize; requires manual asset review'}); continue
        try: text=raw.decode('utf-8')
        except UnicodeDecodeError: text=None
        if text is not None:
            for category,pattern in secret_patterns.items():
                if re.search(pattern,text): findings.append({'path':s,'category':category})
            # Machine-specific paths in examples/comments are not portable build inputs.
            new=re.sub(r'/var/home/[^/\s"\'`<>]+', '/home/developer', text)
            new=re.sub(r'/home/'+re.escape(Path.home().name)+r'(?=/|[\s\"\'])', '/home/developer',new)
            for private_name in a.private_name: new=re.sub(r'\b'+re.escape(private_name)+r'\b', 'development-host',new,flags=re.IGNORECASE)
            if s=='docs/public-release/stage.py': new=text
            if new!=text: edits.append({'path':s,'change':'replace machine-specific home/SSH paths'}); raw=new.encode()
        if s=='.gitignore': raw=b'**/target/\n/dist/\n/runtime-build/\n**/node_modules/\n**/__pycache__/\n.env\n.env.*\n!.env.example\n*.pem\n*.key\n*.gguf\n*.safetensors\n*.log\n*.so\n*.o\n*.a\n'
        dest=out/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(raw)
        if src.stat().st_mode&0o111: dest.chmod(0o755)
        files.append({'path':s,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        if notice: licenses.append(s)
# Complete runtime recipe, relocated without modifying its compile/link flags.
src=root/'docs/public-release/build-runtime.sh'
raw=src.read_text()
dest=out/'docs/public-release/build-runtime.sh'; dest.write_text(raw); dest.chmod(0o755)
files=[f for f in files if f['path']!=str(dest.relative_to(out))]
files.append({'path':str(dest.relative_to(out)),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
# Use a tagged, licensed font in public exports; never ship the unproven local font.
font_source=root/'docs/public-release/assets/SymbolsNerdFontMono-Regular.ttf'
if not font_source.is_file(): font_source=root/'wfe/static/nf.ttf'
font_bytes=font_source.read_bytes()
if hashlib.sha256(font_bytes).hexdigest()!='f0f624d9b474bea1662cf7e862d44aebe1ae1f6c7f9cb7a0ca5d0e5ac9561c60':
    raise SystemExit('public font does not match the authoritative tagged source')
old_font=out/'wfe/static/nf.woff2'
if old_font.exists(): old_font.unlink()
files=[f for f in files if f['path']!='wfe/static/nf.woff2']
dest=out/'wfe/static/nf.ttf'; dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(font_bytes)
files=[f for f in files if f['path']!='wfe/static/nf.ttf']
files.append({'path':'wfe/static/nf.ttf','bytes':len(font_bytes),'sha256':hashlib.sha256(font_bytes).hexdigest()})
for entry in files:
    if entry['path'].startswith('wfe/') and entry['path'].endswith('.css'):
        dest=out/entry['path']; old=dest.read_text()
        new=old.replace('nf.woff2) format("woff2")','nf.ttf) format("truetype")')
        if new!=old:
            dest.write_text(new); raw=dest.read_bytes();entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            edits.append({'path':entry['path'],'change':'point font-face to tagged licensed SymbolsNerdFontMono TTF; no other CSS changes'})
# Serve the replacement with its registered media type (public export only).
for entry in files:
    if entry['path'] in {'glue/serve/src/http.rs','glue/serve/src/image.rs','glue/serve/src/home/mod.rs'}:
        dest=out/entry['path'];old=dest.read_text()
        if '"ttf"' not in old:
            marker='Some("woff2") => "font/woff2",' if 'Some("woff2")' in old else '"woff2" => "font/woff2",'
            addition='Some("ttf") => "font/ttf",' if marker.startswith('Some') else '"ttf" => "font/ttf",'
            assert old.count(marker)==1
            new=old.replace(marker,marker+'\n            '+addition)
            dest.write_text(new);raw=dest.read_bytes();entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            edits.append({'path':entry['path'],'change':'serve licensed public TTF replacement as font/ttf; existing MIME mappings unchanged'})
# Private evidence links must not masquerade as shipped public documents.
for entry in files:
    dest=out/entry['path']
    if dest.suffix!='.md': continue
    old=dest.read_text()
    def public_link(m):
        label,target=m.group(1),m.group(2)
        if not re.match(r'https?://|#',target) and any(part in target.split('/') for part in ('reference','evidence')) and not (dest.parent/target.split('#')[0]).exists():
            return label+' (private development evidence is not included)'
        return m.group(0)
    new=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',public_link,old)
    if new!=old:
        dest.write_text(new); raw=dest.read_bytes()
        entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        edits.append({'path':entry['path'],'change':'replace links to excluded private evidence with explicit omission note'})
for key,value in {'files':files,'excluded':excluded,'redactions':edits,'findings':findings,'license-files':licenses}.items():
    (audit/(key+'.json')).write_text(json.dumps(value,indent=2)+'\n')
summary={'candidate':str(out),'files':len(files),'bytes':sum(f['bytes'] for f in files),'excluded_files':len(excluded),'redacted_files':len(edits),'credential_findings':len(findings),'license_files':len(licenses),'published':False,'source_modified':False}
(audit/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary))
if findings: raise SystemExit('credential-pattern findings require review; candidate NOT cleared')
