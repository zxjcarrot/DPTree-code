#pragma once
#include "MurmurHash2.h"
#include "bloom.h"
#include <cassert>
#include <cmath>
#include <cstdint>

template <typename Key>
class bloom1
{
  public:
    using key_type = Key;
    typedef uint64_t word_t;
    static constexpr int bits_per_byte = 8;
    static constexpr int bytes_per_word = 8;
    static constexpr int bits_per_word = bits_per_byte * bytes_per_word;
    static constexpr int member_bit_width = 6;
    static constexpr int member_bit_mask = (1 << member_bit_width) - 1;
    static constexpr int word_hash_bits = 24;
    static constexpr int word_hash_bits_mask = (1ULL << word_hash_bits) - 1;
    static constexpr int k = 3;
    static constexpr int member_hash_bits = std::ceil(k * member_bit_width);
    static constexpr uint64_t hash_seed = 0xdeadbeef12345678ULL;
    bloom1(int entries, double fp_rate)
    {
        this->entries = entries;
        error = fp_rate;

        double num = log(error);
        double denom = 0.480453013918201; // ln(2)^2
        double bpe = -(num / denom);

        double dentries = (double)entries;
        bits_m = (int)(dentries * bpe);

        if (bits_m % bits_per_byte)
        {
            bytes_n = (bits_m / bits_per_byte) + 1;
        }
        else
        {
            bytes_n = bits_m / bits_per_byte;
        }

        if (bytes_n % bytes_per_word)
        {
            words_l = (bytes_n / bytes_per_word) + 1;
        }
        else
        {
            words_l = bytes_n / bytes_per_word;
        }
        hash_bits = word_hash_bits + member_hash_bits;
        // assert(hash_bits <= bits_per_word);
        words = (word_t*)calloc(words_l, sizeof(word_t));
    }

    ~bloom1() { free(words); }

    uint64_t make_word_mask(uint64_t h, int &word_off)
    {
        uint64_t old_h = h;
        uint64_t word_mask = 0;
        word_off = (h & word_hash_bits_mask) % words_l;
        h >>= word_hash_bits;
        int bits_left = member_hash_bits;
        int h_bits = bits_per_word - word_hash_bits;
        while (bits_left)
        {
            if (h_bits < member_bit_width)
            {
                h = MurmurHash64A(&old_h, sizeof(old_h), hash_seed);
                old_h = h;
                h_bits = bits_per_word;
            }
            int member_off = h & member_bit_mask;
            word_mask |= 1ULL << member_off;
            h >>= member_bit_width;
            h_bits -= member_bit_width;
            bits_left -= member_bit_width;
        }
        return word_mask;
    }

    void insert_unsafe(const key_type &key)
    {
        uint64_t h = MurmurHash64A(&key, sizeof(key), hash_seed);
        int word_off = -1;
        uint64_t word_mask = make_word_mask(h, word_off);
        words[word_off] |= word_mask;
    }

    void insert(const key_type &key)
    {
        uint64_t h = MurmurHash64A(&key, sizeof(key), hash_seed);
        int word_off = -1;
        uint64_t word_mask = make_word_mask(h, word_off);
        __sync_fetch_and_or(&words[word_off], word_mask);
    }

    bool check(const key_type &key)
    {
        uint64_t h = MurmurHash64A(&key, sizeof(key), hash_seed);
        int word_off = -1;
        uint64_t word_mask = make_word_mask(h, word_off);
        return (words[word_off] & word_mask) == word_mask;
    }

    int entries;
    int bits_m;
    int bytes_n;
    int words_l;
    int hash_bits;
    word_t *words;
    double error;
};

template <typename Key>
class bloom_opt
{
  public:
    using key_type = Key;
    bloom_opt(int entries, double fp_rate)
    {
        this->entries = entries;
        bloom_init(&bloom, entries, fp_rate);
        this->bits_m = bloom.bits;
        this->bytes_n = this->bits_m / 8;
        this->words_l = this->bytes_n / 8;
        this->k = bloom.hashes;
    }

    ~bloom_opt() { bloom_free(&bloom); }

    void insert_unsafe(const key_type &key)
    {
        bloom_add_nonatomic(&bloom, &key, sizeof(key));
    }

    void insert(const key_type &key) { bloom_add(&bloom, &key, sizeof(key)); }

    bool check(const key_type &key)
    {
        return bloom_check(&bloom, &key, sizeof(key));
    }
    struct bloom bloom;
    int entries;
    int bits_m;
    int bytes_n;
    int words_l;
    int k;
};