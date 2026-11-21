// Minimal read-only GGUF v3 reader over mmap.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

enum GgmlType : uint32_t {
    GGML_F32 = 0,
    GGML_F16 = 1,
    GGML_Q8_0 = 8,
    GGML_BF16 = 30,
};

struct GgufTensor {
    std::string name;
    uint32_t type = 0;
    std::vector<int64_t> ne;  // ne[0] is the innermost (contiguous) dimension
    uint64_t offset = 0;      // relative to the data section
    const uint8_t *data = nullptr;
    int64_t nelements() const {
        int64_t n = 1;
        for (auto d : ne) n *= d;
        return n;
    }
    size_t nbytes() const;
};

struct GgufValue {
    uint32_t type = 0;
    uint64_t u = 0;
    int64_t i = 0;
    double f = 0;
    bool b = false;
    std::string s;
    uint32_t arr_type = 0;
    std::vector<std::string> arr_str;
    std::vector<int64_t> arr_int;
};

class Gguf {
public:
    explicit Gguf(const std::string &path);
    ~Gguf();
    Gguf(const Gguf &) = delete;
    Gguf &operator=(const Gguf &) = delete;

    bool has(const std::string &key) const { return kv_.count(key) != 0; }
    const GgufValue &kv(const std::string &key) const;
    int64_t get_int(const std::string &key, int64_t def) const;
    double get_float(const std::string &key, double def) const;
    std::string get_str(const std::string &key, const std::string &def) const;

    const GgufTensor *find(const std::string &name) const;
    const GgufTensor &tensor(const std::string &name) const;
    const std::vector<GgufTensor> &tensors() const { return tensors_; }

private:
    int fd_ = -1;
    size_t size_ = 0;
    const uint8_t *map_ = nullptr;
    std::map<std::string, GgufValue> kv_;
    std::vector<GgufTensor> tensors_;
    std::map<std::string, size_t> index_;
};
