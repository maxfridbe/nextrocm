// WAV file reading (PCM16 / PCM24 / PCM32 / float32, any channel count -> mono) and writing (PCM16 mono).
#pragma once

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

struct Audio {
    int sample_rate = 0;
    std::vector<float> samples;  // mono, [-1, 1]
};

Audio read_wav(const std::string &path);
void write_wav(const std::string &path, const std::vector<float> &samples, int sample_rate);
// Appends samples as raw 16-bit little-endian PCM (same conversion as write_wav) and flushes.
void append_pcm16(FILE *f, const float *x, size_t n);
// Linear-interpolation resample (only used for convenience; the server converts with ffmpeg).
std::vector<float> resample(const std::vector<float> &x, int from, int to);
// Split long audio at quiet points so each piece is at most max_len samples; returns [start, end) pairs.
std::vector<std::pair<size_t, size_t>> split_audio(const std::vector<float> &x, size_t max_len, int sr);
