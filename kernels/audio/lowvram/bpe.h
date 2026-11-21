// Byte-level BPE tokenizer (GPT-2 / Qwen2 style).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class Gguf;

class Bpe {
public:
    void load_gguf(const Gguf &g);
    void load_files(const std::string &vocab_json, const std::string &merges_txt);

    std::vector<int> encode(const std::string &text);
    std::string decode(const std::vector<int> &ids) const;
    int id(const std::string &token) const;  // -1 if unknown
    size_t size() const { return tokens_.size(); }

private:
    void finish();
    void bpe_word(const std::string &word, std::vector<int> &out);

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, int> ranks_;  // "a b" -> rank
    std::string byte_enc_[256];
    std::unordered_map<uint32_t, uint8_t> byte_dec_;
    std::unordered_map<std::string, std::vector<int>> cache_;
    std::vector<bool> special_;
};
