#!/usr/bin/env python3
"""Serialized AMD GPU job watchdog. No service operations; explicit device cap required."""
import argparse
import fcntl
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

INTERVAL = .10
_spec = importlib.util.spec_from_file_location("memory_sampler", Path(__file__).with_name("measure-process-memory.py"))
memory_sampler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(memory_sampler)
TERM_GRACE = 10.0
KILL_GRACE = 5.0
RESERVE = 8 * 1024**3

def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing')

def group_members(group):
    members = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                members.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return sorted(members)

def process_memory(pid):
    try:
        fields = {}
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                name, value, unit = line.split()
                if unit != 'kB':
                    raise RuntimeError('unexpected proc memory unit')
                fields[name[:-1] + '_bytes'] = int(value) * 1024
        return fields
    except (FileNotFoundError, ProcessLookupError):
        return {'exited_during_sample': True}

def stop_group(proc):
    actions = []
    for sig, grace in ((signal.SIGTERM, TERM_GRACE), (signal.SIGKILL, KILL_GRACE)):
        if not group_members(proc.pid):
            break
        try:
            os.killpg(proc.pid, sig)
            actions.append(signal.Signals(sig).name)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace
        while group_members(proc.pid) and time.monotonic() < deadline:
            proc.poll()
            time.sleep(min(.05, max(0, deadline - time.monotonic())))
    survivors = group_members(proc.pid)
    if survivors:
        raise RuntimeError(f'owned process group still active: {survivors}')
    proc.wait(timeout=KILL_GRACE)
    return actions

def file_sha256(path, reserve=RESERVE):
    digest = hashlib.sha256()
    with Path(path).open('rb') as src:
        while True:
            if memory_available() < reserve:
                raise RuntimeError('host reserve unavailable during hashing')
            chunk = src.read(1024*1024)
            if not chunk: break
            digest.update(chunk)
    return digest.hexdigest()

def amd_allocated(info):
    if not info.get('drm-client-id','').strip():
        raise RuntimeError('AMD client identity unavailable')
    allocated = 0
    for field in ('drm-memory-vram','drm-memory-gtt'):
        parts = info.get(field,'').split()
        if len(parts)!=2 or parts[1] not in ('kB','KiB') or not parts[0].isdigit():
            raise RuntimeError('AMD memory telemetry missing/invalid: '+field)
        allocated += int(parts[0])*1024
    return allocated

def strict_sample(pid):
    """Validate AMD memory fields before using the retained sampler's values."""
    proc = Path('/proc')/str(pid)
    clients = {}
    try:
        for entry in (proc/'fdinfo').iterdir():
            try: text = entry.read_text()
            except FileNotFoundError: continue  # descriptor closed during scan
            info = dict(line.split(':',1) for line in text.splitlines() if ':' in line)
            if info.get('drm-driver','').strip() != 'amdgpu': continue
            clients[(info.get('drm-pdev','').strip(),info.get('drm-client-id','').strip())] = amd_allocated(info)
    except FileNotFoundError:
        if proc.exists(): raise
    row = memory_sampler.sample(pid)
    row.update(drm_clients=len(clients),drm_allocated_bytes=sum(clients.values()),
               drm_devices=sorted({key[0] for key in clients}))
    return row

def unknown_identity(p, error):
    result = {'pid':int(p.name), 'path':error.filename or str(p/'fdinfo'), 'errno':error.errno, 'certified_no_gpu_client':False}
    try:
        result['comm'] = (p/'comm').read_text().strip()
        result['start_ticks'] = (p/'stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError, IndexError): result['identity_unavailable'] = True
    return result

def kfd_inventory():
    root = Path('/sys/class/kfd/kfd/proc')
    entries = sorted(p.name for p in root.iterdir())
    if any(not name.isdigit() for name in entries):
        raise RuntimeError('unexpected KFD process directory entry')
    return entries

def idle_amd_sample(device):
    if not str(device.resolve(strict=True)).startswith('/sys/'):
        raise ValueError('AMD device must resolve inside sysfs')
    if (device/'vendor').read_text().strip().lower() != '0x1002':
        raise ValueError('selected sysfs device is not AMD')
    raw = (device/'gpu_busy_percent').read_text().strip()
    if not raw.isdigit() or not 0 <= int(raw) <= 100:
        raise RuntimeError('invalid GPU busy telemetry')
    return {'monotonic_seconds':time.monotonic(), 'gpu_busy_percent':int(raw), 'device':str(device.resolve())}

def precheck_amd_clients(mode='strict-fdinfo', amd_device=None):
    """Same-UID proc visibility only; parent must ensure system-wide exclusivity."""
    if mode not in ('strict-fdinfo','kfd-compute'):
        raise ValueError('unknown precheck mode')
    observations = []
    if mode == 'kfd-compute':
        if amd_device is None: raise ValueError('kfd-compute requires --amd-device')
        for i in range(3):
            inventory = kfd_inventory()
            if inventory: raise RuntimeError('existing KFD compute contexts: '+json.dumps(inventory))
            sample = idle_amd_sample(Path(amd_device))
            if sample['gpu_busy_percent'] != 0: raise RuntimeError('AMD device not idle')
            observations.append(dict(sample, kfd_pids=inventory))
            if i != 2: time.sleep(.1)
    clients = []
    unknown = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            if p.stat().st_uid != os.getuid(): continue
            for entry in (p/'fdinfo').iterdir():
                info = dict(line.split(':',1) for line in entry.read_text().splitlines() if ':' in line)
                if info.get('drm-driver', '').strip() == 'amdgpu':
                    clients.append({'pid':int(p.name), 'client':info.get('drm-client-id','').strip()})
        except PermissionError as error:
            if mode == 'kfd-compute':
                unknown.append(unknown_identity(p,error))
                continue
            raise RuntimeError(f'AMD precheck unreadable pid={p.name} path={error.filename or str(p / "fdinfo")} errno={error.errno}; exclusivity unverified') from error
        except (FileNotFoundError, ProcessLookupError): pass
    if clients: raise RuntimeError('existing AMD clients: '+json.dumps(clients))
    if mode == 'kfd-compute' and kfd_inventory():
        raise RuntimeError('KFD compute context appeared during precheck')
    return {'mode':mode, 'scope':'same UID readable DRM clients and optional KFD compute enumeration; NOT proof of absent graphics clients; parent exclusive queue required', 'clients':clients, 'inaccessible_pids':unknown, 'idle_samples':observations, 'graphics_exclusivity_verified':False}

def run(out, command, timeout=180, reserve=RESERVE, lock_path=None, inputs=(), precheck_mode="strict-fdinfo", amd_device=None, device_cap=None, expected_pins=None):
    """Return0 only after exit0, reserve checks and complete group cleanup."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite another run.
    report = {'status': 'INCOMPLETE', 'pass': False, 'argv': command,
              'timeout_seconds': timeout, 'reserve_bytes': reserve,
              'interval_seconds': INTERVAL, 'group_verified_empty': False,
              'watchdog_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'GPU process-group watchdog; sampled conservative DRM VRAM+GTT bound, not continuous guarantee or physical-VRAM-only measurement; numeric/drop verdicts separate'}
    proc = None
    lock = None
    received = []
    previous = {}
    started = time.monotonic()
    report['samples'] = 0
    report['device_cap_bytes'] = device_cap
    report['max_drm_allocated_bytes'] = 0
    report['positive_drm_samples'] = 0
    report['numeric_pass'] = None
    report['allocator_drop_pass'] = None
    report['max_child_rss_bytes'] = 0
    report['max_child_hwm_bytes'] = 0
    report['max_group_rss_bytes'] = 0
    def handle(sig, _frame):
        received.append(signal.Signals(sig).name)
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, handle)
        if not command or not Path(command[0]).is_absolute():
            raise ValueError('command must name an absolute direct executable (run wrapper inside container)')
        if not 1 <= timeout <= 1800 or reserve < RESERVE or not isinstance(device_cap, int) or device_cap <= 0:
            raise ValueError('timeout must be1..1800; reserve at least8GiB; positive explicit device cap required')
        if lock_path is None or not Path(lock_path).is_absolute():
            raise ValueError('absolute shared parent queue lock required')
        lock = open(lock_path, 'a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report['queue_lock'] = str(lock_path)
        report['input_sha256'] = {str(Path(p).resolve(strict=True)): file_sha256(p, reserve) for p in inputs}
        executable = Path(command[0]).resolve(strict=True)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError('direct executable missing/not executable')
        report['executable'] = str(executable)
        report['executable_sha256'] = file_sha256(executable, reserve)
        if expected_pins is not None:
            actual_pins = dict(report['input_sha256'])
            actual_pins[str(executable)] = report['executable_sha256']
            if actual_pins != expected_pins:
                raise RuntimeError('locked prelaunch identities differ from authorized job')
            report['expected_pins_verified'] = True
        available = memory_available()
        report['available_before_bytes'] = available
        report['minimum_available_bytes'] = available
        if available < reserve:
            raise RuntimeError('host memory reserve unavailable before launch')
        report['precheck'] = precheck_amd_clients(precheck_mode, amd_device)
        if received:
            raise RuntimeError('signal received before launch')
        with (out/'stdout.log').open('wb') as stdout, (out/'stderr.log').open('wb') as stderr, (out/'samples.jsonl').open('w') as samples:
            proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            report['pid'] = proc.pid
            report['process_group'] = proc.pid
            child_start = time.monotonic()
            while True:
                elapsed = time.monotonic()-child_start
                available = memory_available()
                report['minimum_available_bytes'] = min(available, report['minimum_available_bytes'])
                members = group_members(proc.pid)
                measurements = {str(pid): process_memory(pid) for pid in members}
                child = measurements.get(str(proc.pid), {})
                drm = strict_sample(proc.pid)
                # Sum deduplicated per-process clients conservatively across group members.
                drm_rows = [drm] + [strict_sample(pid) for pid in members if pid != proc.pid]
                if amd_device is not None:
                    expected_pci = Path(amd_device).resolve(strict=True).name.lower()
                    if any(pci.lower() != expected_pci for row in drm_rows for pci in row.get('drm_devices', [])):
                        raise RuntimeError('owned GPU client is on a different PCI device')
                    if any(row['drm_clients'] and not row.get('drm_devices') for row in drm_rows):
                        raise RuntimeError('owned GPU PCI identity unavailable')
                group_drm = sum(row['drm_allocated_bytes'] for row in drm_rows)
                report['max_drm_allocated_bytes'] = max(report['max_drm_allocated_bytes'], group_drm)
                report['positive_drm_samples'] += int(drm['drm_clients'] > 0 and drm['drm_allocated_bytes'] > 0)
                if group_drm > device_cap:
                    raise RuntimeError('sampled group DRM cap exceeded')
                group_rss = sum(v.get('VmRSS_bytes', 0) for v in measurements.values())
                report['max_group_rss_bytes'] = max(report['max_group_rss_bytes'], group_rss)
                report['max_child_rss_bytes'] = max(report['max_child_rss_bytes'], child.get('VmRSS_bytes', 0))
                report['max_child_hwm_bytes'] = max(report['max_child_hwm_bytes'], child.get('VmHWM_bytes', 0))
                samples.write(json.dumps({'seconds': elapsed, 'available_bytes': available, 'members': measurements, 'direct_drm':drm, 'group_drm_bytes':group_drm})+'\n')
                samples.flush()
                report['samples'] += 1
                code = proc.poll()
                report['exit_code'] = code
                if received:
                    raise RuntimeError('watchdog interrupted by '+','.join(received))
                if available < reserve:
                    report['reserve_breached'] = True
                    raise RuntimeError('host memory reserve breached during execution')
                if elapsed >= timeout:
                    report['timed_out'] = True
                    raise RuntimeError('outer GPU job timeout')
                if code is not None:
                    if code != 0:
                        raise RuntimeError('child failed; see logs')
                    if group_members(proc.pid):
                        raise RuntimeError('successful leader left live owned children')
                    if report['positive_drm_samples'] == 0:
                        raise RuntimeError('no positive DRM evidence; fit unmeasured')
                    report['child_success'] = True
                    break
                time.sleep(INTERVAL)
        for name, expected in report['input_sha256'].items():
            if file_sha256(name, reserve) != expected:
                raise RuntimeError('input changed: '+name)
        if file_sha256(executable, reserve) != report['executable_sha256']:
            raise RuntimeError('child executable changed')
    except BaseException as e:
        report['error'] = repr(e)
    finally:
        try:
            if proc is not None:
                report['cleanup_signals'] = stop_group(proc)
                report['exit_code'] = proc.returncode
                report['group_verified_empty'] = not group_members(proc.pid)
            else:
                report['group_verified_empty'] = True
            if received and 'error' not in report:
                report['error'] = 'watchdog interrupted during final verification'
            report['pass'] = report.get('child_success', False) and 'error' not in report and report['group_verified_empty']
            report['status'] = 'PASS' if report['pass'] else 'FAIL'
        except BaseException as e:
            report['cleanup_error'] = repr(e)
            report['manual_recovery_required'] = True
            report['status'] = 'FAIL'
            report['pass'] = False
        finally:
            report['received_signals'] = received
            report['wall_seconds'] = time.monotonic()-started
            try:
                (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            finally:
                if lock is not None: lock.close()
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
    return 0 if report['pass'] else 1

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device-cap-bytes', type=int, required=True)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--reserve-bytes', type=int, default=RESERVE)
    p.add_argument('--precheck-mode', choices=['strict-fdinfo','kfd-compute'], default='strict-fdinfo')
    p.add_argument('--amd-device', type=Path)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--input', type=Path, action='append', default=[])
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    return run(a.out, command, a.timeout, a.reserve_bytes, a.lock, a.input, a.precheck_mode, a.amd_device, a.device_cap_bytes)
if __name__ == '__main__':
    raise SystemExit(main())
