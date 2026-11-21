#include "bpe.h"

#include <clocale>
#include <climits>
#include <cwctype>
#include <fstream>
#include <sstream>

#include "common.h"
#include "gguf.h"

namespace {

void put_utf8(std::string &s, uint32_t cp) {
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xc0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3f));
    } else if (cp < 0x10000) {
        s += (char)(0xe0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3f));
        s += (char)(0x80 | (cp & 0x3f));
    } else {
        s += (char)(0xf0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3f));
        s += (char)(0x80 | ((cp >> 6) & 0x3f));
        s += (char)(0x80 | (cp & 0x3f));
    }
}

// Decode UTF-8 into code points with their byte offsets. Invalid bytes become U+FFFD.
void utf8_cps(const std::string &s, std::vector<uint32_t> &cps, std::vector<size_t> &offs) {
    size_t i = 0;
    while (i < s.size()) {
        uint8_t c = s[i];
        uint32_t cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1f; len = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0f; len = 3; }
        else if ((c >> 3) == 30) { cp = c & 0x07; len = 4; }
        else { cp = 0xfffd; len = 1; }
        if (i + len > s.size()) { cp = 0xfffd; len = 1; }
        for (int j = 1; j < len; j++) cp = (cp << 6) | (s[i + j] & 0x3f);
        cps.push_back(cp);
        offs.push_back(i);
        i += len;
    }
    offs.push_back(s.size());
}

bool is_space(uint32_t c) {
    return c == ' ' || (c >= 9 && c <= 13) || c == 0x85 || c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) ||
           c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}
bool is_letter(uint32_t c) {
    if (c < 0x80) return (c | 32) >= 'a' && (c | 32) <= 'z';
    return std::iswalpha((wint_t)c) != 0;
}
bool is_number(uint32_t c) {
    if (c < 0x80) return c >= '0' && c <= '9';
    return std::iswalnum((wint_t)c) && !std::iswalpha((wint_t)c);
}
bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }
uint32_t lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// Qwen2 pre-tokenizer:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
std::vector<std::string> pretokenize(const std::string &text) {
    std::vector<uint32_t> cp;
    std::vector<size_t> off;
    utf8_cps(text, cp, off);
    size_t n = cp.size();
    std::vector<std::string> out;
    size_t i = 0;
    auto emit = [&](size_t a, size_t b) { out.push_back(text.substr(off[a], off[b] - off[a])); };
    while (i < n) {
        // contractions
        if (cp[i] == '\'' && i + 1 < n) {
            uint32_t a = lower(cp[i + 1]);
            uint32_t b = i + 2 < n ? lower(cp[i + 2]) : 0;
            size_t len = 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') len = 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) len = 3;
            if (len) {
                emit(i, i + len);
                i += len;
                continue;
            }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        {
            size_t j = i;
            if (!is_letter(cp[j]) && !is_nl(cp[j]) && !is_number(cp[j]) && j + 1 < n && is_letter(cp[j + 1])) j++;
            if (is_letter(cp[j])) {
                while (j < n && is_letter(cp[j])) j++;
                emit(i, j);
                i = j;
                continue;
            }
        }
        // \p{N}
        if (is_number(cp[i])) {
            emit(i, i + 1);
            i++;
            continue;
        }
        // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
        {
            size_t j = i;
            if (cp[j] == ' ' && j + 1 < n) j++;
            auto punct = [&](uint32_t c) { return !is_space(c) && !is_letter(c) && !is_number(c); };
            if (punct(cp[j])) {
                while (j < n && punct(cp[j])) j++;
                while (j < n && is_nl(cp[j])) j++;
                emit(i, j);
                i = j;
                continue;
            }
        }
        // whitespace rules
        if (is_space(cp[i])) {
            size_t j = i;
            while (j < n && is_space(cp[j])) j++;
            size_t last_nl = SIZE_MAX;
            for (size_t k = i; k < j; k++)
                if (is_nl(cp[k])) last_nl = k;
            if (last_nl != SIZE_MAX) {  // \s*[\r\n]+
                emit(i, last_nl + 1);
                i = last_nl + 1;
                continue;
            }
            if (j < n && j - i > 1) {  // \s+(?!\S)
                emit(i, j - 1);
                i = j - 1;
                continue;
            }
            emit(i, j);  // \s+
            i = j;
            continue;
        }
        emit(i, i + 1);  // unreachable in practice
        i++;
    }
    return out;
}

std::string read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot read %s", path.c_str());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Parse a JSON string starting at s[i] == '"'; returns the decoded UTF-8 and advances i past the quote.
std::string json_string(const std::string &s, size_t &i) {
    std::string out;
    i++;
    while (i < s.size() && s[i] != '"') {
        char c = s[i++];
        if (c != '\\') {
            out += c;
            continue;
        }
        char e = s[i++];
        switch (e) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'u': {
                uint32_t cp = std::stoul(s.substr(i, 4), nullptr, 16);
                i += 4;
                if (cp >= 0xd800 && cp < 0xdc00 && s[i] == '\\' && s[i + 1] == 'u') {
                    uint32_t lo = std::stoul(s.substr(i + 2, 4), nullptr, 16);
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                    i += 6;
                }
                put_utf8(out, cp);
                break;
            }
            default: out += e;
        }
    }
    i++;
    return out;
}

}  // namespace

void Bpe::finish() {
    std::setlocale(LC_CTYPE, "C.UTF-8");
    // GPT-2 bytes_to_unicode
    std::vector<int> bs, cs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xa1; b <= 0xac; b++) bs.push_back(b);
    for (int b = 0xae; b <= 0xff; b++) bs.push_back(b);
    cs = bs;
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        bool found = false;
        for (int x : bs)
            if (x == b) found = true;
        if (!found) {
            bs.push_back(b);
            cs.push_back(256 + extra++);
        }
    }
    for (size_t k = 0; k < bs.size(); k++) {
        std::string u;
        put_utf8(u, cs[k]);
        byte_enc_[bs[k]] = u;
        byte_dec_[cs[k]] = (uint8_t)bs[k];
    }
    vocab_.clear();
    for (size_t k = 0; k < tokens_.size(); k++)
        if (!tokens_[k].empty()) vocab_[tokens_[k]] = (int)k;
    special_.assign(tokens_.size(), false);
    for (size_t k = 0; k < tokens_.size(); k++) {
        const std::string &t = tokens_[k];
        if (t.size() > 4 && t.compare(0, 2, "<|") == 0 && t.compare(t.size() - 2, 2, "|>") == 0) special_[k] = true;
    }
}

void Bpe::load_gguf(const Gguf &g) {
    tokens_ = g.kv("tokenizer.ggml.tokens").arr_str;
    const auto &merges = g.kv("tokenizer.ggml.merges").arr_str;
    for (size_t k = 0; k < merges.size(); k++) ranks_[merges[k]] = (int)k;
    finish();
}

void Bpe::load_files(const std::string &vocab_json, const std::string &merges_txt) {
    std::string js = read_file(vocab_json);
    size_t i = js.find('{');
    if (i == std::string::npos) fail("%s: not a JSON object", vocab_json.c_str());
    i++;
    while (true) {
        while (i < js.size() && js[i] != '"' && js[i] != '}') i++;
        if (i >= js.size() || js[i] == '}') break;
        std::string key = json_string(js, i);
        while (i < js.size() && (js[i] == ':' || js[i] == ' ')) i++;
        size_t end;
        int id = std::stoi(js.substr(i), &end);
        i += end;
        if (id >= (int)tokens_.size()) tokens_.resize(id + 1);
        tokens_[id] = key;
    }
    // Qwen2 added/special tokens live in tokenizer_config.json; the ones we use are fixed.
    static const char *specials[] = {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>",
                                     "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>",
                                     "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
                                     "<|image_pad|>", "<|video_pad|>"};
    int base = 151643;
    if ((int)tokens_.size() < base + 14) tokens_.resize(base + 14);
    for (int k = 0; k < 14; k++)
        if (tokens_[base + k].empty()) tokens_[base + k] = specials[k];

    std::ifstream mf(merges_txt);
    if (!mf) fail("cannot read %s", merges_txt.c_str());
    std::string line;
    int rank = 0;
    while (std::getline(mf, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.compare(0, 8, "#version") == 0) continue;
        ranks_[line] = rank++;
    }
    finish();
}

int Bpe::id(const std::string &token) const {
    auto it = vocab_.find(token);
    return it == vocab_.end() ? -1 : it->second;
}

void Bpe::bpe_word(const std::string &word, std::vector<int> &out) {
    auto c = cache_.find(word);
    if (c != cache_.end()) {
        out.insert(out.end(), c->second.begin(), c->second.end());
        return;
    }
    std::vector<std::string> sym;
    for (unsigned char b : word) sym.push_back(byte_enc_[b]);
    while (sym.size() > 1) {
        int best = INT_MAX;
        size_t at = 0;
        for (size_t k = 0; k + 1 < sym.size(); k++) {
            auto it = ranks_.find(sym[k] + " " + sym[k + 1]);
            if (it != ranks_.end() && it->second < best) {
                best = it->second;
                at = k;
            }
        }
        if (best == INT_MAX) break;
        std::string a = sym[at], b = sym[at + 1];
        std::vector<std::string> next;
        for (size_t k = 0; k < sym.size(); k++) {
            if (k + 1 < sym.size() && sym[k] == a && sym[k + 1] == b) {
                next.push_back(a + b);
                k++;
            } else {
                next.push_back(sym[k]);
            }
        }
        sym.swap(next);
    }
    std::vector<int> ids;
    for (auto &s : sym) {
        int v = id(s);
        if (v < 0) fail("tokenizer: no id for piece of \"%s\"", word.c_str());
        ids.push_back(v);
    }
    cache_[word] = ids;
    out.insert(out.end(), ids.begin(), ids.end());
}

std::vector<int> Bpe::encode(const std::string &text) {
    std::vector<int> out;
    for (auto &w : pretokenize(text)) bpe_word(w, out);
    return out;
}

std::string Bpe::decode(const std::vector<int> &ids) const {
    std::string out;
    for (int t : ids) {
        if (t < 0 || t >= (int)tokens_.size() || special_[t]) continue;
        std::vector<uint32_t> cps;
        std::vector<size_t> offs;
        utf8_cps(tokens_[t], cps, offs);
        for (uint32_t cp : cps) {
            auto it = byte_dec_.find(cp);
            if (it != byte_dec_.end()) out += (char)it->second;
        }
    }
    return out;
}
