#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

#include "common.h"

size_t GgufTensor::nbytes() const {
    int64_t n = nelements();
    switch (type) {
        case GGML_F32: return n * 4;
        case GGML_F16:
        case GGML_BF16: return n * 2;
        case GGML_Q8_0: return n / 32 * 34;
        default: fail("tensor %s: unsupported ggml type %u", name.c_str(), type);
    }
}

namespace {

struct Reader {
    const uint8_t *p, *end;
    template <class T>
    T get() {
        if (p + sizeof(T) > end) fail("gguf: truncated header");
        T v;
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (p + n > end) fail("gguf: truncated string");
        std::string s(reinterpret_cast<const char *>(p), n);
        p += n;
        return s;
    }
};

// GGUF value types
enum : uint32_t { U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 };

void read_scalar(Reader &r, uint32_t t, GgufValue &v) {
    v.type = t;
    switch (t) {
        case U8: v.u = r.get<uint8_t>(); v.i = (int64_t)v.u; break;
        case I8: v.i = r.get<int8_t>(); v.u = (uint64_t)v.i; break;
        case U16: v.u = r.get<uint16_t>(); v.i = (int64_t)v.u; break;
        case I16: v.i = r.get<int16_t>(); v.u = (uint64_t)v.i; break;
        case U32: v.u = r.get<uint32_t>(); v.i = (int64_t)v.u; break;
        case I32: v.i = r.get<int32_t>(); v.u = (uint64_t)v.i; break;
        case U64: v.u = r.get<uint64_t>(); v.i = (int64_t)v.u; break;
        case I64: v.i = r.get<int64_t>(); v.u = (uint64_t)v.i; break;
        case F32: v.f = r.get<float>(); break;
        case F64: v.f = r.get<double>(); break;
        case BOOL: v.b = r.get<uint8_t>() != 0; v.i = v.b; break;
        case STR: v.s = r.str(); break;
        default: fail("gguf: bad value type %u", t);
    }
    if (t == F32 || t == F64) v.i = (int64_t)v.f;
    else if (t != STR) v.f = (double)v.i;
}

}  // namespace

Gguf::Gguf(const std::string &path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) fail("cannot open %s", path.c_str());
    struct stat st;
    fstat(fd_, &st);
    size_ = st.st_size;
    void *m = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (m == MAP_FAILED) fail("cannot mmap %s", path.c_str());
    map_ = static_cast<const uint8_t *>(m);

    Reader r{map_, map_ + size_};
    if (r.get<uint32_t>() != 0x46554747) fail("%s is not a GGUF file", path.c_str());
    uint32_t version = r.get<uint32_t>();
    if (version < 2) fail("gguf version %u not supported", version);
    uint64_t n_tensors = r.get<uint64_t>();
    uint64_t n_kv = r.get<uint64_t>();
    uint64_t alignment = 32;

    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = r.str();
        uint32_t t = r.get<uint32_t>();
        GgufValue v;
        if (t == ARR) {
            v.type = ARR;
            v.arr_type = r.get<uint32_t>();
            uint64_t n = r.get<uint64_t>();
            if (v.arr_type == STR) {
                v.arr_str.reserve(n);
                for (uint64_t j = 0; j < n; j++) v.arr_str.push_back(r.str());
            } else {
                v.arr_int.reserve(n);
                for (uint64_t j = 0; j < n; j++) {
                    GgufValue e;
                    read_scalar(r, v.arr_type, e);
                    v.arr_int.push_back(e.i);
                }
            }
        } else {
            read_scalar(r, t, v);
        }
        if (key == "general.alignment") alignment = v.u;
        kv_[key] = std::move(v);
    }

    tensors_.resize(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        GgufTensor &t = tensors_[i];
        t.name = r.str();
        uint32_t nd = r.get<uint32_t>();
        t.ne.resize(nd);
        for (uint32_t d = 0; d < nd; d++) t.ne[d] = r.get<int64_t>();
        t.type = r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        index_[t.name] = i;
    }
    size_t header = r.p - map_;
    size_t data_start = (header + alignment - 1) / alignment * alignment;
    for (auto &t : tensors_) {
        t.data = map_ + data_start + t.offset;
        if (t.type == GGML_F32 || t.type == GGML_F16 || t.type == GGML_BF16 || t.type == GGML_Q8_0) {
            if (data_start + t.offset + t.nbytes() > size_) fail("tensor %s extends past end of file", t.name.c_str());
        }
    }
}

Gguf::~Gguf() {
    if (map_) munmap(const_cast<uint8_t *>(map_), size_);
    if (fd_ >= 0) close(fd_);
}

const GgufValue &Gguf::kv(const std::string &key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) fail("gguf: missing key %s", key.c_str());
    return it->second;
}

int64_t Gguf::get_int(const std::string &key, int64_t def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second.i;
}

double Gguf::get_float(const std::string &key, double def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second.f;
}

std::string Gguf::get_str(const std::string &key, const std::string &def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second.s;
}

const GgufTensor *Gguf::find(const std::string &name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

const GgufTensor &Gguf::tensor(const std::string &name) const {
    const GgufTensor *t = find(name);
    if (!t) fail("gguf: missing tensor %s", name.c_str());
    return *t;
}
