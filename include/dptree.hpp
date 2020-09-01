#pragma once

#include <algorithm>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <unistd.h>
#include <set>
#include <vector>
#include <cmath>
#include <atomic>
#include <thread>
#include <cstdlib>
#include <cstdio>
#include <unordered_map>
#include <errno.h>
#include <cstring>
#include <unordered_set>
#include <mutex>
#include <condition_variable>

#include "bloom.hpp"
#include "stx/btree_map.h"
#include "art_tree.hpp"
#include "util.h"
#include "ART.hpp"

#ifdef USE_PAPI
#include <papi.h>
#endif

#include <emmintrin.h> // x86 SSE intrinsics
#include <immintrin.h>

#ifdef HAS_AVX512

#include <avx512fintrin.h>
#include <avx512cdintrin.h>
#include <avx512dqintrin.h>

#endif

extern int nvm_dram_alloc(void **ptr, size_t align, size_t size);

extern void nvm_dram_free(void *ptr, size_t size);

extern double secs_now(void);

extern void cpu_pause();

extern unsigned long long cycles_total;
#define FILL_FACTOR 0.7
#define KEY_CAPACITY 256

namespace dtree {


template<class Key, class Value>
class stxtree_kv_iterator {
public:
    using key_type = Key;
    using value_type = Value;
    using iterator_type = stxtree_kv_iterator<key_type, value_type>;
    typename stx::btree_map<key_type, value_type>::iterator it;

    stxtree_kv_iterator(const typename stx::btree_map<key_type, value_type>::iterator &vit) : it(vit) {}

    key_type key() { return it->first; }

    value_type value() { return it->second; }

    inline bool operator==(const iterator_type &rhs) const { return it == rhs.it; }

    inline bool operator!=(const iterator_type &rhs) const { return !(*this == rhs); }

    inline iterator_type &operator++() { ++it; }

    inline iterator_type operator++(int) {
        auto retval = *this;
        ++(*this);
        return retval;
    }
};


template<typename Key,
        typename Value>
class cvhftree {
public:
    static std::hash<Key> hasher{};
    static constexpr int key_capacity = KEY_CAPACITY;
    static constexpr int probe_limit = 4;
    static constexpr int node_degree = key_capacity / 2;
    static constexpr double bits_per_key_doubled = std::log2(key_capacity + 1);
    static constexpr int bits_per_key = std::ceil(bits_per_key_doubled);
    typedef uint16_t order_type;
    using key_type = Key;
    using value_type = Value;
    using kv_pair = std::pair<key_type, value_type>;
    struct leaf_node;

    static uint64_t hash2(uint64_t x) {
        x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
        x = x ^ (x >> 31);
        return x;
    }

    struct __attribute__((packed)) leaf_node {
        struct bitmap {
            static constexpr int opt_linear_probe_count = 4;//cacheline_size / sizeof(kv_pair);
            static constexpr int bucket_size = 16;
            static constexpr int out_of_capacity_count = bucket_size - 1;
            static constexpr int fp_capacity = key_capacity + out_of_capacity_count;
            static constexpr int num_ints = std::ceil(key_capacity / 32.0);
            uint8_t fp[fp_capacity];
            uint64_t bits[num_ints];

            bitmap() {
                memset(bits, 0, sizeof(bits));
                memset(fp, 0, sizeof(fp));
            }

            enum slot_status {
                empty = 0,
                exist = 1,
                deleted = 2,
                mask = 3
            };

            inline size_t hash_higher_bits(size_t h) {
                return h >> 8;
            }

            inline size_t hash_lower_bits(size_t h) {
                auto ret = h & 0xFF;
                return ret <= 1 ? ret + 2 : ret;
            }

            // dead simple linear probing
            inline int alloc_first_unset_bit(size_t h) {
                int pos = hash_higher_bits(h) % key_capacity;

                while (true) {
                    if (!test_exist(pos)) {
                        set(pos, h);
                        return pos;
                    }
                    pos = (pos + 1) % key_capacity;
                }
                return -1;
            }

            int8_t *get_ith_fp_addr(int i) {
                return &fp[i];
            }

            void clear() {
                memset(fp, 0, sizeof(fp));
                memset(bits, 0, sizeof(bits));
            }

            int popcount() {
                int cnt = 0;
                for (int i = 0; i < key_capacity; ++i) {
                    cnt += test_exist(i);
                }
                return cnt;
            }


            inline int get_status_bits(int i) {
                assert(i >= 0);
                int ipos = i >> 5;
                int idx = (i & 31) * 2;
                assert(ipos < num_ints);
                return (bits[ipos] >> idx) & mask;
            }

            inline void set_status_bits(int i, unsigned long long status) {
                assert(status <= 3);
                assert(i >= 0);
                int ipos = i >> 5;
                int idx = (i & 31) * 2;
                auto others_bits = bits[ipos] & (~(((unsigned long long) mask) << idx));
                bits[ipos] = others_bits | (status << idx);
            }

            inline bool test_fp_match(int i, size_t h) {
                return fp[i] == h;
            }

            inline bool test_exist(int i) {
                return get_status_bits(i) == slot_status::exist;
            }

            inline bool test_empty(int i) {
                return get_status_bits(i) == slot_status::empty;
            }

            inline bool test_deleted(int i) {
                return get_status_bits(i) == slot_status::deleted;
            }

            inline bool test_fp_empty(int i) {
                return fp[i] == 0;
            }

            inline void set_fp(int i, uint8_t fpv) {
                fp[i] = fpv;
                if (i < out_of_capacity_count) {
                    fp[key_capacity + i] = fpv;
                }
            }

            inline void set(int i, size_t h) {
                set_status_bits(i, slot_status::exist);
                set_fp(i, hash_lower_bits(h));
            }

            inline void set_deleted(int i) {
                set_status_bits(i, slot_status::deleted);
                set_fp(i, 1);
            }

            inline void set_deleted(const bitmap &mask) {
                for (int i = 0; i < key_capacity; ++i) {
                    if (mask.test_exist(i)) {
                        set_status_bits(i, slot_status::deleted);
                        set_fp(i, 1);
                    }
                }
            }
        };

        // cacheline-aligned metadata optimized for flush and search
        struct node_meta {
            key_type max_key;
            bitmap bmap;
            struct leaf_node *next;
            order_type count;
            order_type order[key_capacity];
            static constexpr int meta_flush_size = sizeof(bmap.bits) + sizeof(next) + sizeof(max_key);
            uint8_t __padding__[END_PADDING_SIZE(
                    sizeof(count) + sizeof(bmap) + sizeof(order) + sizeof(next) + sizeof(max_key))];

            node_meta() : next(nullptr), count(0) {
                memset(order, 0, sizeof(order));
            }

            leaf_node *next_sibling() const { return next; }

            leaf_node **next_sibling_ref() { return &next; }

            inline int free_cells() const { return key_capacity - count; }

            inline int key_count() const { return count; }

            inline int key_idx(int ith) const { return order[ith]; }

            inline int min_key_idx() const { return order[0]; }

            inline int max_key_idx() const { return order[count - 1]; }

            inline key_type get_max_key() { return max_key; }

            inline void set_max_key(const key_type &max_key) { this->max_key = max_key; }

            inline void set_key_idx(int idx, int key_idx) {
                order[idx] = key_idx;
            }

            inline void append(int key_idx) {
                assert(count<key_capacity);
                order[count++] = key_idx;
            }

            inline void erase(int idx) {
                assert(count > 0);
                memmove(order + idx, order + idx + 1, (count - idx) * sizeof(order_type));
                bmap.set_deleted(idx);
                --count;
            }

            inline void clear_count() {
                count = 0;
            }

            inline void clear() {
                count = 0;
                bmap.clear();
            }

            bitmap &get_bitmap() {
                return bmap;
            }

            // reserve n cells at the front
            void reserve_front(int n) {
                assert(n <= free_cells());
                memmove(order + n, order, count * sizeof(order_type));
                count += n;
            }
        };

        kv_pair pairs[key_capacity];
        node_meta meta[2];
        uint64_t dirty_cacheline_map; // records whether the cacheline relative pairs is dirty
        char padding[END_PADDING_SIZE(sizeof(meta) + sizeof(pairs) + sizeof(dirty_cacheline_map))];

        leaf_node() {
        }

        void clear_dirty_cacheline_map() {
            dirty_cacheline_map = 0;
        }

        void mark_dirty_cacheline(char *addr, size_t len = sizeof(kv_pair)) {
            char *aligned_addr = CACHELINE_ALIGN(addr);
            int off = static_cast<uint64_t>((aligned_addr - reinterpret_cast<char *>(pairs))) / cacheline_size;
            assert(off <= 63);
            dirty_cacheline_map |= (1ULL << off) | (addr + len > aligned_addr + cacheline_size ? 1ULL << (off + 1) : 0);
        }

        int key_count(bool v) const { return meta[v].key_count(); }

        int free_cells(bool v) const { return meta[v].free_cells(); }

        key_type &key(int idx) { return pairs[idx].first; }

        value_type &value(int idx) { return pairs[idx].second; }

        kv_pair ith_kv(int ith, bool v) { return pairs[ith_kv_idx(ith, v)]; }

        int ith_kv_idx(int ith, bool v) { return meta[v].key_idx(ith); }

        key_type &ith_key(int ith, bool v) { return key(meta[v].key_idx(ith)); }

        value_type &ith_value(int ith, bool v) { return value(meta[v].key_idx(ith)); }

        key_type min_key(bool v) { return key(meta[v].min_key_idx()); }

        key_type max_key_via_idx(bool v) { return key(meta[v].max_key_idx()); }

        key_type max_key(bool v) { return meta[v].get_max_key(); }

        void set_max_key(bool v) { meta[v].set_max_key(max_key_via_idx(v)); }

        leaf_node *next_sibling(bool v) { return meta[v].next_sibling(); }

        leaf_node **next_sibling_ref(bool v) { return meta[v].next_sibling_ref(); }

        static leaf_node *new_leaf_node() {
            void *ret;
            nvm_dram_alloc(&ret, cacheline_size, sizeof(leaf_node));
            return new(ret) leaf_node;
        }

        static void delete_leaf_node(leaf_node *l) { nvm_dram_free(l, sizeof(leaf_node)); }

        // extract from `bits` `count` consecutive bits starting at off'th bit
        inline uint64_t extract_bits(uint64_t bits, int off, int count) {
            return bits & (((1L << count) - 1) << off);
        }

        unsigned long long round(unsigned long long value, unsigned int multiple) {
            return (value + multiple - 1) & -multiple;
        }

        void flush(bool nv, int &flushes, int &flushed_node_count) {
            bool cv = !nv;
            bool need_flush_meta = this->dirty_cacheline_map;
            uint64_t t = this->dirty_cacheline_map;
            char * cacheline_addr = reinterpret_cast<char*>(this->pairs);
            while (t) {
                int pos = __builtin_ffsll(t) - 1;
                t &= t - 1;
                clflush(cacheline_addr + pos * cacheline_size);
            }
            if (need_flush_meta) {
                flushes += 1;
                clflush_len(&meta[nv], node_meta::meta_flush_size);
                ++flushed_node_count;
            }
        }

        static constexpr double fill_factor = FILL_FACTOR;
        // When merging a stream of new upsert kvs with a node,
        // any new nodes created should hold at most `max_initial_fill_keys` keys,
        // so there mgiht be enough empty cells to accomadate later upserts.
        static constexpr int max_initial_fill_keys = (int) std::ceil(fill_factor * key_capacity);
        // Nodes with # keys less than this constant are considered under-utilized and need merging.
        static constexpr int merge_node_threshold = key_capacity / 2;

        static std::vector<leaf_node *>
        split_merge_node_with_stream(leaf_node *l, typename std::vector<kv_pair>::iterator &upsert_kvs_sit,
                                     const typename std::vector<kv_pair>::iterator &upsert_kvs_eit, bool cv,
                                     int &split_merges) {
            ++split_merges;
            bool nv = !cv;
            std::vector<leaf_node *> leafs = {new_leaf_node()};
            int l_key_count = l->key_count(cv);
            int i = 0;
            bitmap *node_alloc_bitmap = &leafs.back()->meta[nv].get_bitmap();
            bitmap mask; // records bit positions that should be unmarked due to key moves
            leaf_node *last_leaf = leafs.back();
            auto ensure_last_leaf_capacity = [&last_leaf, &leafs, nv, &node_alloc_bitmap]() {
                if (last_leaf->key_count(nv) >=
                    max_initial_fill_keys) { // last leaf is full, create a new leaf and establish the chain
                    auto new_sibling = new_leaf_node();
                    last_leaf->set_max_key(nv);
                    assert(last_leaf->key_count(nv) == node_alloc_bitmap->popcount());
                    *last_leaf->next_sibling_ref(nv) = new_sibling;
                    leafs.push_back(new_sibling);
                    last_leaf = new_sibling;
                    node_alloc_bitmap = &last_leaf->meta[nv].get_bitmap();
                }
            };
            while (i < l_key_count && upsert_kvs_sit != upsert_kvs_eit) {
                ensure_last_leaf_capacity();
                int cv_key_idx = l->meta[cv].key_idx(i);
                key_type cv_key = l->key(cv_key_idx);
                key_type upsert_key = upsert_kvs_sit->first;
                int free_idx;
                if (cv_key == upsert_key) {
                    free_idx = node_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    last_leaf->pairs[free_idx] = *upsert_kvs_sit;
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    ++i;
                    ++upsert_kvs_sit;
                    mask.set(cv_key_idx, hasher(upsert_key));
                } else if (cv_key < upsert_key) {
                    free_idx = node_alloc_bitmap->alloc_first_unset_bit(hasher(cv_key));
                    last_leaf->pairs[free_idx] = l->pairs[cv_key_idx];
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    ++i;
                } else {
                    free_idx = node_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    last_leaf->pairs[free_idx] = *upsert_kvs_sit;
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    ++upsert_kvs_sit;
                }
                last_leaf->meta[nv].append(free_idx);
                assert(node_alloc_bitmap->test_exist(free_idx) == true);
            }
            while (i < l_key_count) {
                ensure_last_leaf_capacity();
                int cv_key_idx = l->meta[cv].key_idx(i);
                int free_idx = node_alloc_bitmap->alloc_first_unset_bit(hasher(l->pairs[cv_key_idx].first));
                last_leaf->pairs[free_idx] = l->pairs[cv_key_idx];
                last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                last_leaf->meta[nv].append(free_idx);
                assert(node_alloc_bitmap->test_exist(free_idx) == true);
                ++i;
            }
            while (upsert_kvs_sit != upsert_kvs_eit) {
                ensure_last_leaf_capacity();
                int free_idx = node_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_kvs_sit->first));
                last_leaf->pairs[free_idx] = *upsert_kvs_sit;
                last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                last_leaf->meta[nv].append(free_idx);
                assert(node_alloc_bitmap->test_exist(free_idx) == true);
                ++upsert_kvs_sit;
            }
            last_leaf->set_max_key(nv);
            *last_leaf->next_sibling_ref(nv) = l->next_sibling(cv);
            //last_leaf->meta[cv] = last_leaf->meta[nv];
            node_alloc_bitmap->set_deleted(mask);
            assert(leafs.back() == last_leaf);
            assert(upsert_kvs_sit == upsert_kvs_eit);
            assert(last_leaf->key_count(nv) == node_alloc_bitmap->popcount());
            return leafs;
        }

        static std::vector<leaf_node *>
        split_node_with_stream(leaf_node *l, typename std::vector<kv_pair>::iterator &upsert_kvs_sit,
                               const typename std::vector<kv_pair>::iterator &upsert_kvs_eit, bool cv,
                               int &split_merges) {
            ++split_merges;
            bool nv = !cv;
            std::vector<leaf_node *> leafs = {new_leaf_node()};
            int l_key_count = l->key_count(cv);
            int l_free_cells = l->free_cells(cv);
            int i = 0;
            auto hasher = std::hash<key_type>();

            bitmap *node_alloc_bitmap = &leafs.back()->meta[nv].get_bitmap();
            bitmap mask; // records bit positions that should be unmarked due to key moves
            leaf_node *last_leaf = leafs.back();
            auto ensure_last_leaf_capacity = [&last_leaf, &leafs, nv, &node_alloc_bitmap]() {
                if (last_leaf->key_count(nv) >=
                    max_initial_fill_keys) { // last leaf is full, create a new leaf and establish the chain
                    auto new_sibling = new_leaf_node();
                    last_leaf->set_max_key(nv);
                    assert(last_leaf->key_count(nv) == node_alloc_bitmap->popcount());
                    *last_leaf->next_sibling_ref(nv) = new_sibling;
                    leafs.push_back(new_sibling);
                    last_leaf = new_sibling;
                    node_alloc_bitmap = &last_leaf->meta[nv].get_bitmap();
                }
            };
            while (i < l_key_count && upsert_kvs_sit != upsert_kvs_eit) {
                // In order to minimize # flushes, we keep l's content as much and unchanged as we can.
                // We perserve l when the following three conditions hold:
                // 1. # kvs left in stream < # free cells in l.
                // 2. The last leaf created has # keys >= merge_node_threshold. (This prevents this node from being merged again later)
                // 3. (# kvs left in stream + # kvs left after moving keys to previous nodes) >= merge_node_threshold. (The same purpose as 2)
                int sleft = upsert_kvs_eit - upsert_kvs_sit;
                if (sleft <= l_free_cells && last_leaf->key_count(nv) >= merge_node_threshold &&
                    l_key_count - i + sleft >= merge_node_threshold) {
                    goto preserve_l;
                }
                // Otherwise insert kvs merged from stream and l into the newly created node
                ensure_last_leaf_capacity();
                int cv_key_idx = l->meta[cv].key_idx(i);
                key_type cv_key = l->key(cv_key_idx);
                key_type upsert_key = upsert_kvs_sit->first;
                int free_idx;
                if (cv_key == upsert_key) {
                    auto upsert_key_hash = hasher(upsert_key);
                    free_idx = node_alloc_bitmap->alloc_first_unset_bit(upsert_key_hash);
                    last_leaf->pairs[free_idx] = *upsert_kvs_sit;
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    ++i;
                    ++upsert_kvs_sit;
                    mask.set(cv_key_idx, upsert_key_hash); // cv_key_idx is updated and moved to previous nodes
                } else if (cv_key < upsert_key) {
                    auto cv_key_hash = hasher(cv_key);
                    free_idx = node_alloc_bitmap->alloc_first_unset_bit(cv_key_hash);
                    last_leaf->pairs[free_idx] = l->pairs[cv_key_idx];
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    ++i;
                    mask.set(cv_key_idx, cv_key_hash); // cv_key_idx is moved to previous nodes
                } else {
                    free_idx = node_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    last_leaf->pairs[free_idx] = *upsert_kvs_sit;
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    ++upsert_kvs_sit;
                }
                last_leaf->meta[nv].append(free_idx);
                assert(node_alloc_bitmap->test_exist(free_idx) == true);
            }
            assert(upsert_kvs_sit == upsert_kvs_eit);
            while (i < l_key_count) {
                ensure_last_leaf_capacity();
                int cv_key_idx = l->meta[cv].key_idx(i);
                auto cv_key_hash = hasher(l->pairs[cv_key_idx].first);
                int free_idx = node_alloc_bitmap->alloc_first_unset_bit(cv_key_hash);
                last_leaf->pairs[free_idx] = l->pairs[cv_key_idx];
                last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                last_leaf->meta[nv].append(free_idx);
                assert(node_alloc_bitmap->test_exist(free_idx) == true);
                ++i;
                mask.set(cv_key_idx, cv_key_hash); // cv_key_idx is moved to previous nodes
            }
            last_leaf->set_max_key(nv);
            *last_leaf->next_sibling_ref(nv) = l->next_sibling(cv);
            //last_leaf->meta[cv] = last_leaf->meta[nv];
            assert(leafs.back() == last_leaf);
            assert(last_leaf->key_count(nv) == node_alloc_bitmap->popcount());
            return leafs;
            preserve_l:
            // When the conditions hold, we upsert the rest of stream into l.
            node_meta l_new_meta = l->meta[cv];
            l_new_meta.clear_count(); // only clear out count
            bitmap *l_alloc_bitmap = &l_new_meta.get_bitmap();
            while (upsert_kvs_sit != upsert_kvs_eit) {
                int cv_key_idx = l->meta[cv].key_idx(i);
                key_type cv_key = l->key(cv_key_idx);
                key_type upsert_key = upsert_kvs_sit->first;
                assert(i < l_key_count);
                if (cv_key == upsert_key) {
                    auto cv_key_hash = hasher(upsert_key);
                    int free_idx = l_alloc_bitmap->alloc_first_unset_bit(cv_key_hash);
                    l->pairs[free_idx] = *upsert_kvs_sit;
                    l->mark_dirty_cacheline((char *) &l->pairs[free_idx]);
                    ++i;
                    ++upsert_kvs_sit;
                    mask.set(cv_key_idx, cv_key_hash); // cv_key_idx is updated to another cell in the same node
                    l_new_meta.append(free_idx);
                } else if (cv_key < upsert_key) {
                    ++i;
                    l_new_meta.append(cv_key_idx);
                } else {
                    int free_idx = l_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    l->pairs[free_idx] = *upsert_kvs_sit;
                    l->mark_dirty_cacheline((char *) &l->pairs[free_idx]);
                    ++upsert_kvs_sit;
                    l_new_meta.append(free_idx);
                }
            }
            assert(upsert_kvs_sit == upsert_kvs_eit);
            while (i < l_key_count) {
                int cv_key_idx = l->meta[cv].key_idx(i);
                l_new_meta.append(cv_key_idx);
                ++i;
            }

            last_leaf->set_max_key(nv);
            *last_leaf->next_sibling_ref(nv) = l;
            l_alloc_bitmap->set_deleted(mask);
            l->meta[nv] = l_new_meta;
            assert(l->key_count(nv) == l_alloc_bitmap->popcount());
            l->set_max_key(nv);
            leafs.push_back(l);
            return leafs;
        }

        // Upsert kv stream whose keys are <= this->max_key().
        // Return the last leaf node whose keys <= this->max_key() after the kvs are upserted.
        leaf_node *bulk_upsert(leaf_node **prev_ref, typename std::vector<kv_pair>::iterator &upsert_kvs_sit,
                               const typename std::vector<kv_pair>::iterator &upsert_kvs_eit,
                               bool cv, std::unordered_set<leaf_node *> &gc_candidates,
                               int &update_inplace, int &split_merges, int &perserve_l) {
            bool nv = !cv;
            int i = 0;
            int j = 0;
            // copy the small metadata from previous version
            meta[nv] = meta[cv];
            this->clear_dirty_cacheline_map();
            int free_cells = meta[cv].free_cells();
            int in_range_count = upsert_kvs_eit - upsert_kvs_sit;
            std::hash<key_type> hasher;
            assert(key_capacity - free_cells == meta[nv].get_bitmap().popcount());
            if (in_range_count > free_cells) { // not enough free cells to upserts keys <= this->max_key, split instead
                // We merge the stream [upsert_kvs_sit, tit) with this leaf by creating as many leaf nodes as needed.
                auto leafs = leaf_node::split_node_with_stream(this, upsert_kvs_sit, upsert_kvs_eit, cv, split_merges);
                assert(leafs.size() >= 1);
                auto last_leaf = leafs.back();
                if (last_leaf != this) {
                    gc_candidates.insert(this);
                } else {
                    ++perserve_l;
                }

                *prev_ref = leafs[0]; // make sure prev_ref(next version pointer) points to new leaf
                assert(last_leaf->next_sibling(nv) == this->next_sibling(cv));
                return last_leaf;
            }
            if (in_range_count == 0)
                return this;
            ++update_inplace;

            node_meta tmp_meta = meta[nv];
            tmp_meta.clear_count();
            bitmap mask; // records bit positions that should be unmarked due to key moves
            bitmap *node_alloc_bitmap = &tmp_meta.get_bitmap();
            int nv_key_count = meta[nv].key_count();
            int insert_count = 0;
            assert(meta[cv].key_count() == meta[nv].key_count());
            while (i < nv_key_count && upsert_kvs_sit != upsert_kvs_eit) {
                int nv_key_idx = meta[nv].key_idx(i);
                key_type nv_key = key(nv_key_idx);
                key_type upsert_key = upsert_kvs_sit->first;
                if (nv_key < upsert_key) {
                    ++i;
                    tmp_meta.append(nv_key_idx);
                } else if (nv_key > upsert_key) {
                    int free_idx = node_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    pairs[free_idx] = *upsert_kvs_sit;
                    this->mark_dirty_cacheline((char *) &pairs[free_idx]);
                    // insert: place new kv onto a new cell and append the idx to the order array
                    tmp_meta.append(free_idx);
                    ++upsert_kvs_sit;
                    ++insert_count;
                } else { // nv_key == upsert_key
                    // upsert: place updated value onto a new cell and modify the order array
                    auto upsert_key_hash = hasher(upsert_key);
                    int free_idx = node_alloc_bitmap->alloc_first_unset_bit(upsert_key_hash);
                    pairs[free_idx] = *upsert_kvs_sit;
                    this->mark_dirty_cacheline((char *) &pairs[free_idx]);
                    tmp_meta.append(free_idx);
                    ++i;
                    ++upsert_kvs_sit;
                    mask.set(nv_key_idx, upsert_key_hash);
                }
            }

            if (i < nv_key_count) {
                memcpy(tmp_meta.order + tmp_meta.key_count(), meta[nv].order + i,
                       sizeof(order_type) * (nv_key_count - i));
                tmp_meta.count += (nv_key_count - i);
            }

            while (upsert_kvs_sit != upsert_kvs_eit) {
                int free_idx = node_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_kvs_sit->first));
                pairs[free_idx] = *upsert_kvs_sit;
                this->mark_dirty_cacheline((char *) &pairs[free_idx]);
                tmp_meta.append(free_idx);
                ++upsert_kvs_sit;
            }

            assert(upsert_kvs_eit == upsert_kvs_sit);
            assert(nv_key_count + insert_count == tmp_meta.key_count());
            node_alloc_bitmap->set_deleted(mask);
            meta[nv] = tmp_meta;
            this->set_max_key(nv);
            assert(node_alloc_bitmap->popcount() == tmp_meta.key_count());
            return this;
        }

        // Since deletion comes after upsertions, we apply them directly on the new version
        void bulk_delete(const std::vector<key_type> &delete_keys, bool nv) {
            // delete by merging
            int i = 0;
            int j = 0;
            while (i < meta[nv].key_count() && j < delete_keys.size()) {
                int nv_key_idx = meta[nv].key_idx(i);
                key_type nv_key = key(nv_key_idx);
                key_type delete_key = delete_keys[j];
                if (nv_key < delete_key) {
                    ++i;
                } else if (nv_key > delete_key) {
                    ++j;
                } else if (nv_key == delete_key) {
                    meta[nv].erase(i);
                    ++j;
                }
            }
            this->set_max_key(nv);
        }

        static std::vector<leaf_node *>
        merge_multiple_nodes(std::vector<leaf_node *> leafs, const int total_keys, bool nv) {
            const int keys_per_node = std::ceil(total_keys / 2.0);
            std::vector<leaf_node *> merged_leafs = {new_leaf_node()};
            std::hash<key_type> hasher;
            bitmap *node_alloc_bitmap = &merged_leafs.back()->meta[nv].get_bitmap();
            leaf_node *last_leaf = merged_leafs.back();
            auto ensure_last_leaf_capacity = [&last_leaf, &merged_leafs, nv, &node_alloc_bitmap, &keys_per_node]() {
                if (last_leaf->key_count(nv) >=
                    keys_per_node) { // last leaf is full, create a new leaf and establish the chain
                    auto new_sibling = new_leaf_node();
                    last_leaf->set_max_key(nv);
                    assert(last_leaf->key_count(nv) == node_alloc_bitmap->popcount());
                    *last_leaf->next_sibling_ref(nv) = new_sibling;
                    merged_leafs.push_back(new_sibling);
                    last_leaf = new_sibling;
                    node_alloc_bitmap = &last_leaf->meta[nv].get_bitmap();
                }
            };
            for (int i = 0; i < leafs.size(); ++i) {
                leaf_node *l = leafs[i];
                int l_key_count = l->key_count(nv);
                for (int j = 0; j < l_key_count; ++j) {
                    ensure_last_leaf_capacity();
                    int free_idx = node_alloc_bitmap->alloc_first_unset_bit(hasher(l->ith_key(j, nv)));
                    last_leaf->pairs[free_idx] = l->ith_kv(j, nv);
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    last_leaf->meta[nv].append(free_idx);
                }
            }
            merged_leafs.back()->set_max_key(nv);
            assert(last_leaf->key_count(nv) == node_alloc_bitmap->popcount());
            return merged_leafs;
        }

        // Merge policy:
        // 1. If the utilization of l1 >= merge_node_threshold/key_capacity: no merge.
        // 2. Else if l1->key_count() <= l1->next->free_cells(): migrate keys from l1 to l1->next
        // 3. Else: combine keys from l1 onward until # accumulated keys >= key_capcity or the end is reached
        // Return the next node to be examined.
        static leaf_node *merge_underutilized_node(leaf_node **&prev_ref, leaf_node *l1, bool nv,
                                                   std::unordered_set<leaf_node *> &gc_candidates,
                                                   int &node_merges_transfer, int &node_merges_combine) {
            int l1_key_count = l1->key_count(nv);
            leaf_node *l2 = l1->next_sibling(nv);
            if (l1_key_count >= merge_node_threshold || l2 == nullptr) {
                prev_ref = l1->next_sibling_ref(nv);
                return l2;
            }

            if (l1_key_count <= l2->free_cells(nv)) { // l1's keys fit in l2's free cells
                ++node_merges_transfer;
                node_meta l2_tmp_meta = l2->meta[nv];
                bitmap *l2_alloc_bitmap = &l2_tmp_meta.get_bitmap();
                l2_tmp_meta.reserve_front(l1_key_count);
                assert(l2_tmp_meta.key_count() == l1_key_count + l2->key_count(nv));
                for (int i = 0; i < l1_key_count; ++i) {
                    int free_idx = l2_alloc_bitmap->alloc_first_unset_bit(hasher(l1->ith_key(i, nv)));
                    l2->pairs[free_idx] = l1->ith_kv(i, nv);
                    l2->mark_dirty_cacheline((char *) &l2->pairs[free_idx]);
                    l2_tmp_meta.set_key_idx(i, free_idx);
                }
                l2->meta[nv] = l2_tmp_meta;
                l2->set_max_key(nv);
                gc_candidates.insert(*prev_ref);
                *prev_ref = l2;
                assert(l2->key_count(nv) == l2_alloc_bitmap->popcount());
                // TODO: garbage collect l1
                return l2;
            } else { // Otherwise we accumulate keys from l1 onward until # accumulated keys >= key_capcity
                ++node_merges_combine;
                std::vector<leaf_node *> leafs;
                int total_keys = 0;
                auto cur = l1;
                while (cur && total_keys < key_capacity) {
                    total_keys += cur->key_count(nv);
                    gc_candidates.insert(cur);
                    leafs.push_back(cur);
                    cur = cur->next_sibling(nv);
                }
                // TODO: garabge collect leafs
                std::vector<leaf_node *> merged_leafs = merge_multiple_nodes(leafs, total_keys, nv);
                assert(merged_leafs.size() == 2);
                *prev_ref = merged_leafs[0];
                prev_ref = merged_leafs.back()->next_sibling_ref(nv);
                *prev_ref = cur;
                return cur;
            }
        }
    };

    struct stat {
        int kv_count;
        int node_count;

        double avg_fill() {
            return kv_count / (node_count * key_capacity + 1.0);
        }
    };

    stat stats[2];
    leaf_node *head[2];
    Art::Node *art_tree[2];
    bool gv = 0; // globally consistent version on nvm

    cvhftree(double &inner_nodes_build_time) : art_tree_build_time(inner_nodes_build_time) {
        memset(stats, 0, sizeof(stats));
        memset(head, 0, sizeof(head));
        memset(art_tree, 0, sizeof(art_tree));
        printf("sizeof(leaf_node) %d, sizeof(node_meta) %d sizeof(kv_pair) %d\n", sizeof(leaf_node),
               sizeof(typename leaf_node::node_meta), sizeof(kv_pair));
        gv = 0;
    }

    size_t size() { return stats[gv].kv_count; }

    size_t leaf_count() { return stats[gv].node_count; }

    void update_stat(bool v) {
        auto cur = head[v];
        stats[v].kv_count = stats[v].node_count = 0;
        std::unordered_map<int, int> m;
        while (cur) {
            stats[v].kv_count += cur->key_count(v);
            m[cur->key_count(v)]++;
            ++stats[v].node_count;
            cur = cur->next_sibling(v);
        }
        printf("node key count distribution\n");
        for (auto kv : m) {
            printf("%d -> %d\n", kv.first, kv.second);
        }
    }

    stat get_stat() {
        return stats[gv];
    }

    static inline bool is_delete_op(const value_type &v) {
        return v & 1;
    }

    static inline bool is_upsert_op(const value_type &v) {
        return !is_delete_op(v);
    }

    static void prefetch(char *ptr, size_t len) {
        if (ptr == nullptr) return;
        for (char *p = ptr; p < ptr + len; p += cacheline_size) {
            __builtin_prefetch(p);
        }
    }

    struct iterator {
        leaf_node *l;
        int idx;
        bool v;

        explicit iterator(leaf_node *l, int idx, bool v) : l(l), idx(idx), v(v) {}

        inline bool operator==(const iterator &rhs) const { return l == rhs.l && idx == rhs.idx; }

        inline bool operator!=(iterator other) const { return !(*this == other); }

        inline iterator &operator++() {
            if (++idx >= l->key_count(v)) {
                l = l->next_sibling(v);
                ///if (l) prefetch((char*)l->next_sibling(v), sizeof(leaf_node));
                idx = 0;
            }
            return *this;
        }

        inline iterator operator++(int) {
            auto retval = *this;
            ++(*this);
            return retval;
        }

        inline key_type &key() { return l->ith_key(idx, v); }

        inline value_type &value() { return l->ith_value(idx, v); }
    };

    iterator begin() { return iterator(head[gv], 0, gv); }

    iterator end() { return iterator(nullptr, 0, gv); }

    iterator lower_bound(const key_type &key) {
        bool v = gv;
        uint8_t lookup_key[8];
        *reinterpret_cast<uint64_t *>(lookup_key) = __builtin_bswap64(key);
        bool pref = true;
        auto leaf = Art::lowerBound(art_tree[v], lookup_key, 8, 0, 8, pref);
        if (leaf == nullptr) return end();
        auto cur = reinterpret_cast<leaf_node *>(Art::getLeafValue(leaf));
        //prefetch((char*)cur, sizeof(leaf_node));
        while (cur && key > cur->max_key(v)) {
            cur = cur->next_sibling(v);
            //prefetch((char*)cur, sizeof(leaf_node));
        }
        if (cur == nullptr) return end();
        int i = 0;
        int key_count = cur->key_count(v);
        while (i < key_count && key > cur->ith_key(i, v)) {
            ++i;
        }
        return i >= key_count ? end() : iterator(cur, i, v);
    }

    // scan leafs for keys in [start_key, end_key)
    void range_scan(const key_type &start_key, const key_type &end_key, std::vector<value_type> &values,
                    std::function<void(value_type)> processor) {
        assert(start_key <= end_key);
        auto eit = end();
        auto it = lower_bound(start_key);
        while (it != eit && it.key() < end_key) {
            values.push_back(it.value());
            ++it;
        }
    }

    std::atomic<bool> is_in_gc;
    leaf_node empty_node;

    template<class Iterator>
    void merge(Iterator &start, const Iterator &end) {
        bool cv = gv; // currnet version
        bool nv = !gv; // next version
        std::unordered_set<leaf_node *> *gc_candidates = new std::unordered_set<leaf_node *>;
        bool has_del = false;
        int new_kvs = 0;
        {
            std::vector<kv_pair> upsert_kvs;
            auto sit = start;
            auto eit = end;
            auto cur = head[nv] = head[cv];
            leaf_node **prev_ref = &head[nv];
            int update_inplace = 0;
            int split_merges = 0;
            int perserve_l = 0;
            while (cur && sit != eit) {
                key_type cur_high_key = cur->max_key(cv);
                upsert_kvs.clear();
                while (sit != eit && sit.key() <= cur_high_key) {
                    bool is_upsert = is_upsert_op(sit.value());
                    if (is_upsert) {
                        upsert_kvs.push_back(std::make_pair(sit.key(), sit.value()));
                    }
                    has_del |= !is_upsert;
                    ++sit;
                    ++new_kvs;
                }
                auto ustart = upsert_kvs.begin();
                auto uend = upsert_kvs.end();
                auto next = cur->next_sibling(cv);
                //prefetch((char*)next, sizeof(leaf_node));
                cur = cur->bulk_upsert(prev_ref, ustart, uend, cv, *gc_candidates, update_inplace, split_merges,
                                       perserve_l);
                prev_ref = cur->next_sibling_ref(nv);
                assert(*prev_ref == next);
                cur = *prev_ref;
            }
            if (sit != eit) {
                assert(cur == nullptr);
                upsert_kvs.clear();
                while (sit != eit) {
                    if (is_upsert_op(sit.value())) {
                        upsert_kvs.push_back(std::make_pair(sit.key(), sit.value()));
                    }
                    ++sit;
                    ++new_kvs;
                }
                auto ustart = upsert_kvs.begin();
                auto uend = upsert_kvs.end();
                auto leafs = leaf_node::split_merge_node_with_stream(&empty_node, ustart, uend, cv, split_merges);
                assert(leafs.empty() == false);
                assert(leafs.back()->next_sibling(nv) == nullptr);
                assert(ustart == uend);
                *prev_ref = leafs[0];
            } else {
                while (cur) {
                    // copy the metadata from previous version to current version
                    cur->meta[nv] = cur->meta[cv];
                    assert(cur->key_count(nv) == cur->meta[nv].get_bitmap().popcount());
                    cur = cur->next_sibling(cv);
                }
            }
            //printf("update inplace %d, split_merges %d, perserve_l %d\n", update_inplace, split_merges, perserve_l);
        }
        //2. apply deletions in key order on the new version
        {
            if (has_del) {
                auto h = head[nv];
                auto cur = h;
                auto sit = start;
                auto eit = end;
                std::vector<key_type> delete_keys;
                while (cur) {
                    key_type cur_high_key = cur->max_key(nv);
                    delete_keys.clear();
                    while (sit != eit && sit.key() <= cur_high_key) {
                        if (is_delete_op(sit.value())) {
                            delete_keys.push_back(sit.key());
                        }
                        ++sit;
                    }
                    cur->bulk_delete(delete_keys, nv);
                    cur = cur->next_sibling(nv);
                }
            }
        }
        //3. merge under-utilized nodes
        {
            auto h = head[nv];
            auto cur = h;
            leaf_node **prev_ref = &head[nv];
            int node_merges_transfer = 0;
            int node_merges_combine = 0;
            while (cur) {
                //prefetch((char*)cur->next_sibling(nv), sizeof(leaf_node));
                cur = leaf_node::merge_underutilized_node(prev_ref, cur, nv, *gc_candidates, node_merges_transfer,
                                                          node_merges_combine);
            }
            //printf("node_merges_transfer %d, node_merges_combine %d\n", node_merges_transfer, node_merges_combine);
        }
        std::vector<std::pair<key_type, value_type>> kvs;
        kvs.reserve(stats[cv].node_count);
        {
            //4. bulk clflush and update stats
            int flushes = 0, flushed_node = 0;
            auto cur = head[nv];
            stats[nv].kv_count = stats[nv].node_count = 0;
            bool do_wbinvd = new_kvs >= 409600 * 100;
            //bool do_wbinvd = false;
            double flush_latency = 0;
            double start_time = secs_now();
            sfence();
            while (cur) {
                gc_candidates->erase(cur);
                stats[nv].kv_count += cur->meta[nv].key_count();
                ++stats[nv].node_count;
                kvs.push_back(std::make_pair(__builtin_bswap64(cur->max_key(nv)), (uintptr_t) cur));
                auto next = cur->next_sibling(nv);
                //prefetch((char*)next, sizeof(leaf_node));
                if (!do_wbinvd) {
                    cur->flush(nv, flushes, flushed_node);
                }
                cur = next;
            }
            sfence();

            double wbinvd_latency = 0;
            if (do_wbinvd) {
                double t = secs_now();
                wbinvd();
                wbinvd_latency = secs_now() - t;
            }
            flush_latency = secs_now() - start_time;
            //printf("flushes %d, insert kvs: %d flushed nodes: %d, avg flushes per node: %f, gc set size %d, node count %d, kv count %d, flushes per kv: %f, avg fill: %f flush_latency %f, wbinvd_latency %f\n",
            //       flushes, new_kvs, flushed_node, flushes / (flushed_node + 0.1), gc_candidates->size(),
            //       stats[nv].node_count, stats[nv].kv_count, flushes / (new_kvs + 0.1), stats[nv].avg_fill(),
            //       flush_latency, wbinvd_latency);
            is_in_gc.store(true);
            std::thread([this, gc_candidates](Art::Node *old_root) {
                // 6. Now that the new version is persistent, garbage collect nodes not used by the new version
                for (auto node : *gc_candidates) {
                    leaf_node::delete_leaf_node(node);
                }
                Art::destroyNode(old_root);
                delete gc_candidates;
                is_in_gc.store(false);
            }, art_tree[cv]).detach();
        }
        //5. flip the global version
        {
            gv = nv;
            clflush(&gv);
            sfence();
        }
        // 7. build in-memory radix-tree index for leaf nodes
        {
            auto st = secs_now();
            int v = gv;
            auto loadKey = [v](uintptr_t tid, uint8_t key[]) {
                leaf_node *l = reinterpret_cast<leaf_node *>(tid);
                uint64_t max_key = l->max_key(v);
                reinterpret_cast<uint64_t *>(key)[0] = __builtin_bswap64(max_key);
            };
            Art::loadKey = loadKey;
            Art::loadLowerBoundKey = loadKey;
            auto KeyLen = [](const key_type &key) -> int {
                return 8;
            };
            auto KeyExtract = [](const key_type &key, int idx) {
                return ((uint8_t *) &key)[idx];
            };
            art_tree[v] = Art::bulkLoad(kvs, 0, kvs.size() - 1, 0, KeyExtract, KeyLen);
            auto et = secs_now();
            art_tree_build_time += et - st;
        }
        //while (is_in_gc);
    }

    double &art_tree_build_time;
    size_t probes = 0;

    size_t get_probes() {
        return probes;
    }

    size_t clear_probes() {
        probes = 0;
    }

    inline int next(int p) {
        return (p + 1) & (key_capacity - 1);
    }

    std::pair<leaf_node *, int> find_leaf(const key_type &key) {
        bool v = gv;
        uint8_t lookup_key[8];
        *reinterpret_cast<uint64_t *>(lookup_key) = __builtin_bswap64(key);
        bool pref = true;
        auto leaf = Art::lowerBound(art_tree[v], lookup_key, 8, 0, 8, pref);
        if (leaf == nullptr) return make_pair(nullptr, false);
        auto cur = reinterpret_cast<leaf_node *>(Art::getLeafValue(leaf));
        uint8_t search_key_fp = std::hash<key_type>()(key);
        while (cur && key > cur->max_key(v)) {
            cur = cur->next_sibling(v);
        }
        if (cur == nullptr) return make_pair(nullptr, false);
        return std::make_pair(cur, v);
    }

    bool lookup_leaf(leaf_node *cur, bool v, const key_type &key, value_type &value) {
        auto *bmap = &cur->meta[v].get_bitmap();
        auto h = hasher(key);
        int p = bmap->hash_higher_bits(h) % key_capacity;
        int start_p = p;
        auto lower_bits = bmap->hash_lower_bits(h);

        do {
            ++probes;
            if (bmap->test_fp_empty(p))
                return false;
            if (bmap->test_fp_match(p, lower_bits)) {
                if (cur->pairs[p].first == key) {
                    value = cur->pairs[p].second;
                    return true;
                }
            }
            p = next(p);
        } while (p != start_p);
        return false;
    }


    bool lookup_leaf_fp(leaf_node *cur, bool v, const key_type &key, value_type &value) {
        auto *bmap = &cur->meta[v].get_bitmap();
        int p = 0;
        auto lower_bits = bmap->hash_lower_bits(hasher(key));
        do {
            ++probes;
            auto status = bmap->get_status_bits(p);
            if (bmap->test_fp_match(p, lower_bits)) {
                if (cur->pairs[p].first == key) {
                    value = cur->pairs[p].second;
                    return true;
                }
            }
            ++p;
        } while (p < key_capacity);
        return false;
    }

    bool lookup(const key_type &key, value_type &value) {
        bool v = gv;
        uint8_t lookup_key[8];
        *reinterpret_cast<uint64_t *>(lookup_key) = __builtin_bswap64(key);
        bool pref = true;
        auto leaf = Art::lowerBound(art_tree[v], lookup_key, 8, 0, 8, pref);
        if (leaf == nullptr) return false;
        auto cur = reinterpret_cast<leaf_node *>(Art::getLeafValue(leaf));
        uint8_t search_key_fp = std::hash<key_type>()(key);
        while (cur && key > cur->max_key(v)) {
            cur = cur->next_sibling(v);
        }
        if (cur == nullptr) return false;
        auto *bmap = &cur->meta[v].get_bitmap();
        auto h = hasher(key);
        int p = bmap->hash_higher_bits(h) % key_capacity;
        int start_p = p;
        auto lower_bits = bmap->hash_lower_bits(h);

        do {
            ++probes;
            if (bmap->test_fp_empty(p))
                return false;
            if (bmap->test_fp_match(p, lower_bits)) {
                if (cur->pairs[p].first == key) {
                    value = cur->pairs[p].second;
                    return true;
                }
            }
            p = next(p);
        } while (p != start_p);
        return false;
    }

};

template<typename Key,
        typename Value> std::hash<Key> cvhftree<Key, Value>::hasher;

template<typename Key,
        typename Value>
class dpftree {
public:
    using key_type =  Key;
    using value_type = Value;
    using bloom_type = bloom1<key_type>;
    static constexpr double bloom_fp_rate = 0.10;
    static constexpr double merge_threshold = 0.10;

    dpftree() : merges(0), time_spent_merging(0), is_merging(false), merge_time(0), static_tree_size(0),
                expected_dyn_tree_capacity(0) {
        bloom.reset(new bloom_type(1000, bloom_fp_rate));
        dynamic_tree.reset(new stx::btree_map<key_type, value_type>);
        static_tree.reset(new cvhftree<key_type, value_type>(inner_nodes_build_time));
        wal.reset(new struct wal(1000));
    }

    size_t get_buffer_tree_size() {
        return dynamic_tree->size();
    }

    size_t get_buffer_tree_node_count() {
        return dynamic_tree->get_stats().nodes();
    }

    size_t get_wal_entry_count() {
        return wal->entry_count();
    }

    size_t get_base_tree_leaf_count() {
        return static_tree->leaf_count();
    }

    inline uint64_t make_upsert_value(uint64_t v) {
        return (v << 1);
    }

    inline uint64_t strip_upsert_value(uint64_t v) {
        return v >> 1;
    }

    inline uint64_t make_tombstone_value() {
        return make_upsert_value(0) | 1;
    }

    void insert(const key_type &key, const value_type &value, bool check_merge = true) {
        wal->append_and_flush(log_record(log_record::op_type::insertion, key, value));
        dynamic_tree->insert(key, make_upsert_value(value));
        bloom->insert_unsafe(key);
        if (check_merge)
            merge_if_necessary();
    }

    void erase(const key_type &key) {
        wal->append_and_flush(log_record(log_record::op_type::deletion, key));
        dynamic_tree->insert(key, make_tombstone_value());
        bloom->insert_unsafe(key);
        merge_if_necessary();
    }

    int dyn_lookup_count = 0;
    int static_lookup_count = 0;

    bool merging() {
        return is_merging;
    }

    bool lookup(const key_type &key, value_type &value) {
        if (bloom->check(key)) {
            ++dyn_lookup_count;
            auto dyn_it = dynamic_tree->find(key);
            if (dyn_it != dynamic_tree->end()) {
                value = dyn_it->second;
                return true;
            }
        }
        while (is_merging);
        ++static_lookup_count;
        return static_tree->lookup(key, value);
    }

    void lookup_range(const key_type &key_start, const key_type &key_end, std::vector<value_type> &res) {
        while (is_merging);
        auto dyn_it = dynamic_tree->lower_bound(key_start);
        auto dyn_end = dynamic_tree->lower_bound(key_end);
        auto static_it = static_tree->lower_bound(key_start);
        auto static_end = static_tree->lower_bound(key_end);
        while (dyn_it != dyn_end && static_it != static_end) {
            if (dyn_it->first < static_it.key()) {
                res.push_back(dyn_it->second);
                ++dyn_it;
                if (dyn_it->first == static_it.key())
                    ++static_it;
            } else {
                res.push_back(static_it.value());
                ++static_it;
            }
        }
        while (dyn_it != dyn_end) {
            res.push_back(dyn_it->second);
            ++dyn_it;
        }
        while (static_it != static_end) {
            res.push_back(static_it.value());
            ++static_it;
        }
    }

    int get_merges() {
        return merges;
    }

    double get_merge_time() {
        return time_spent_merging;
    }

    double get_real_merge_time() {
        return merge_time;
    }

    double get_inner_node_build_time() {
        return inner_nodes_build_time;
    }

    size_t get_static_size() {
        while (is_merging);
        return static_tree->size();
    }

    void force_merge(bool async = true) {
        auto start = secs_now();
        merge(async);
        time_spent_merging += secs_now() - start;
    }

    int get_bloom_entries() {
        return bloom->entries;
    }

    int get_bloom_bytes() {
        return bloom->bytes_n;
    }

    int get_bloom_hashes() {
        return bloom->k;
    }

    int get_bloom_words() {
        return bloom->words_l;
    }

    size_t get_probes() {
        return static_tree->get_probes();
    }

    void clear_probes() {
        static_tree->clear_probes();
    }

private:
    struct log_record {
        enum class op_type {
            insertion, deletion
        };
        key_type key;
        value_type val;
        size_t h;
        op_type type;
        char cacheline_padding[END_PADDING_SIZE(sizeof(type) - sizeof(key) - sizeof(val) - sizeof(h))];

        log_record(op_type type, const key_type &key, const value_type &val) : type(type), key(key), val(val) {
            h = (std::hash<int>()(static_cast<int>(type)) << 1) ^ (std::hash<key_type>()(key) << 2) ^
                (std::hash<value_type>()(val));
        }

        log_record(op_type type, const key_type &key) : type(type), key(key), val() {
            h = (std::hash<int>()(static_cast<int>(type)) << 1) ^ (std::hash<key_type>()(key) << 2) ^
                (std::hash<value_type>()(val));
        }
    };

    struct wal {
        log_record *records;
        int off;
        int cap;

        wal(size_t size) : records(nullptr), off(0), cap(size * 1.1) {
            auto res = nvm_dram_alloc(&(records), 64, sizeof(log_record) * cap);
            assert(res == 0);
        }

        void ensure_capacity() {
            if (off >= cap) {
                log_record *old_records = records;
                int old_cap = cap;
                cap *= 1.7;
                auto res = nvm_dram_alloc(&(records), 64, sizeof(log_record) * cap);
                assert(res == 0);
                memcpy(records, old_records, off * sizeof(log_record));
                nvm_dram_free(old_records, old_cap * sizeof(log_record));
            }
        }

        void append_and_flush(const struct log_record &r) {
            ensure_capacity();
            records[off] = r;
            clflush_then_sfence(&records[off]);
            ++off;
        }

        size_t entry_count() {
            return off;
        }

        ~wal() { nvm_dram_free(records, sizeof(log_record) * cap); }
    };

    void merge_if_necessary() {
        size_t dyn_size = dynamic_tree->size();
        if (dyn_size >= 1000 && dyn_size >= expected_dyn_tree_capacity) {
            auto start = secs_now();
            merge();
            time_spent_merging += secs_now() - start;
        }
    }

    int expected_dyn_tree_capacity;

    void merge(bool async = true) {
        size_t reserve_count = dynamic_tree->size() + static_tree->size();
        static_tree_size = reserve_count;
        expected_dyn_tree_capacity = (int) std::ceil(reserve_count * (merge_threshold));

        old_dynamic_tree.reset(dynamic_tree.release());
        old_bloom.reset(bloom.release());
        old_wal.reset(wal.release());

        bloom.reset(new bloom_type(std::max(1000, expected_dyn_tree_capacity), bloom_fp_rate));
        dynamic_tree.reset(new stx::btree_map<key_type, value_type>);
        wal.reset(new struct wal(expected_dyn_tree_capacity));
        //printf("new log size: %d\n", expected_dyn_tree_capacity);

        is_merging.store(true);
        //std::thread([this, reserve_count](stx::btree_map<key_type, value_type> * old_dynamic_tree,
        //                                  cvhftree<key_type, value_type> * static_tree) {
        auto start = secs_now();
        stxtree_kv_iterator <key_type, value_type> dyn_sit(old_dynamic_tree->begin());
        stxtree_kv_iterator <key_type, value_type> dyn_eit(old_dynamic_tree->end());

        static_tree->merge(dyn_sit, dyn_eit);

        ++merges;
        merge_time += secs_now() - start;
        this->old_dynamic_tree.reset();
        this->old_wal.reset();
        this->old_bloom.reset();
        is_merging.store(false);
        cv.notify_one();
    }

    std::mutex cvm;
    std::condition_variable cv;

private:
    double inner_nodes_build_time = 0;
    std::atomic_bool is_merging;
    double merge_time;
    double time_spent_merging;
    int merges;
    size_t static_tree_size;
    std::unique_ptr<stx::btree_map<key_type, value_type>> old_dynamic_tree;
    std::unique_ptr<stx::btree_map<key_type, value_type>> dynamic_tree;
    std::unique_ptr<cvhftree<key_type, value_type>> static_tree;
    std::unique_ptr<bloom_type> old_bloom;
    std::unique_ptr<bloom_type> bloom;
    std::unique_ptr<struct wal> old_wal;
    std::unique_ptr<struct wal> wal;
};
}