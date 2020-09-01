#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <mutex>
#include <set>
#include <stdio.h>
#include <sys/resource.h>
#include <utility>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "stx/btree_map.h"
#include "btreeolc.hpp"
#include "ThreadPool.hpp"
#include "art_idx.hpp"
#include "bloom.hpp"
#include "util.h"
#include "ARTOLC/ARTOLC.hpp"

#ifdef USE_PAPI
#include <papi.h>
#endif

#include <emmintrin.h> // x86 SSE intrinsics
#include <immintrin.h>
#ifdef HAS_AVX512
#include <avx512cdintrin.h>
#include <avx512dqintrin.h>
#include <avx512fintrin.h>
#endif

extern int nvm_dram_alloc(void **ptr, size_t align, size_t size);
extern void nvm_dram_free(void *ptr, size_t size);
extern void clflush_1mfence(volatile void *p);
extern void clflush_then_sfence(volatile void *p);
extern void clflush(volatile void *p);
extern void mfence();
extern double secs_now(void);
extern void cpu_pause();
namespace dptree
{

class thread_epoch;
class epoch_manager
{
  public:
    void add_object(thread_epoch *te)
    {
        std::lock_guard<std::mutex> g(mtx);
        thread_epoches.push_back(te);
    }

    void delete_object(thread_epoch *te)
    {
        std::lock_guard<std::mutex> g(mtx);
    restart:
        for (int i = 0; i < thread_epoches.size(); ++i)
        {
            if (thread_epoches[i] == te)
            {
                thread_epoches.erase(thread_epoches.begin() + i);
                goto restart;
            }
        }
    }

    void iterate_thread_epoches(std::function<void(thread_epoch *)> processor)
    {
        std::lock_guard<std::mutex> g(mtx);
        for (int i = 0; i < thread_epoches.size(); ++i)
        {
            processor(thread_epoches[i]);
        }
    }

  private:
    std::vector<thread_epoch *> thread_epoches;
    std::mutex mtx;
};

class thread_epoch
{
  public:
    thread_epoch() : v(0), registered(false), manager(nullptr) {}

    uintptr_t set_value(uintptr_t new_v)
    {
        v = new_v | active();
        return new_v;
    }

    uintptr_t get_value() { return v & (~1); }

    inline void register_to_manager(epoch_manager *manager)
    {
        if (registered)
            return;
        this->manager = manager;
        manager->add_object(this);
        registered = true;
    }

    ~thread_epoch()
    {
        if (manager)
        {
            manager->delete_object(this);
            manager = nullptr;
        }
    }

    void enter() {
        v = v | 1;
    }

    void leave() { v = 0; }

    bool active() { return v & 1; }

  private:
    uintptr_t v;
    bool registered;
    epoch_manager *manager;
};

// RAII-style enter and leave
class epoch_guard {
public:
    epoch_guard(thread_epoch &epoch) : epoch(epoch) {
        epoch.enter();
        mfence();
    }

    ~epoch_guard() { epoch.leave(); }

private:
    thread_epoch &epoch;
};

void spin_until_no_ref(epoch_manager & manager, uintptr_t V) {
    while (true)
    {
        bool might_have_refs = false;
        manager.iterate_thread_epoches(
                [&might_have_refs,
                        V](thread_epoch *te) {
                    auto tev = te->get_value();
                    if (te->active() && (tev == 0 || te->get_value() == V))
                    {
                        might_have_refs = true;
                    }
                });
        if (might_have_refs == false)
            break;
        std::this_thread::yield();
    }
}

template <typename T>
class lockfree_taskqueue
{
  public:
    lockfree_taskqueue(int cap) : consumer_off(0), producer_off(0), max_task_count(std::numeric_limits<int>::max() / 2)
    {
        tasks = new T[cap];
    }

    ~lockfree_taskqueue()
    {
        delete[] tasks;
    }

    void push(const T &task)
    {
        tasks[producer_off.load()] = task;
        ++producer_off;
        cv.notify_one();
    }

    bool pop(int &idx)
    {
        while (true)
        {
            int old_off = -1;
            while (true)
            {

                old_off = consumer_off.load();
                if (old_off >= max_task_count.load())
                    return false;
                if (old_off < producer_off.load())
                {
                    break;
                }
                std::unique_lock<std::mutex> g(mtx);
                cv.wait_for(g, std::chrono::microseconds(10));
            }
            // CAS to get a task
            int new_off = old_off + 1;
            if (consumer_off.compare_exchange_strong(old_off, new_off) == true)
            {
                idx = old_off;
                return true;
            }
        }
    }

    void end_enqueue()
    {
        max_task_count = producer_off.load();
    }

    size_t size()
    {
        return producer_off;
    }

    T &operator[](int i)
    {
        return tasks[i];
    }

  private:
    std::atomic<int> consumer_off;
    std::atomic<int> producer_off;
    std::atomic<int> max_task_count;
    T *tasks;
    std::mutex mtx;
    std::condition_variable cv;
};

template <typename Key, typename Value>
class concur_cvhtree
{
public:
    static constexpr int key_capacity = 256;
    static constexpr int probe_limit = 4;
    static constexpr double bits_per_key_doubled = std::log2(key_capacity + 1);
    static constexpr int bits_per_key = std::ceil(bits_per_key_doubled);
    typedef uint16_t order_type;
    using key_type = Key;
    using value_type = Value;
    using kv_pair = std::pair<key_type, value_type>;
    struct leaf_node;
    static uint64_t hash2(uint64_t x)
    {
        x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
        x = x ^ (x >> 31);
        return x;
    }
    struct alignas(64) leaf_node
    {
        struct bitmap
        {
            static constexpr int ints = std::ceil(key_capacity / 64.0);
            uint64_t bits[ints];
            bitmap() { memset(bits, 0, sizeof(bits)); }
            int popcount()
            {
                int cnt = 0;
                for (int i = 0; i < ints; ++i)
                {
                    cnt += __builtin_popcountll(bits[i]);
                }
                return cnt;
            }
            inline bool test(int i)
            {
                int ipos = i >> 6;
                int idx = i & 63;
                return bits[ipos] & (1UL << idx);
            }

            // dead simple linear probing
            inline int alloc_first_unset_bit(size_t h)
            {
                int pos = h % key_capacity;
                int limit = probe_limit;
                // optimistically probe only limited # entries
                while (limit--)
                {
                    if (!test(pos))
                    {
                        set(pos);
                        return pos;
                    }
                    pos = (pos + 1) % key_capacity;
                }

                pos = hash2(h) % key_capacity;
                while (true)
                {
                    int ipos = pos >> 6;
                    int idx = pos & 63;
                    uint64_t int_map = (bits[ipos] >> idx);
                    uint64_t int_map_reverse = ~int_map;

                    int zidx;
                    if (int_map_reverse == 0 || (zidx = __builtin_ctzll(int_map_reverse) + idx) >= 64)
                    {
                        pos = (pos + 64 - idx) % key_capacity;
                    }
                    else
                    {
                        int zpos = zidx + (ipos << 6);
                        set(zpos);
                        return zpos;
                    }
                }
                return -1;
            }

            inline void set(int i)
            {
                int ipos = i >> 6;
                int idx = i & 63;
                bits[ipos] |= 1UL << idx;
            }

            inline void unset(int i)
            {
                int ipos = i >> 6;
                int idx = i & 63;
                bits[ipos] &= ~(1UL << idx);
            }

            inline void unset(const bitmap &mask)
            {
                for (int i = 0; i < ints; ++i)
                {
                    bits[i] &= ~mask.bits[i];
                }
            }
            void clear() { memset(bits, 0, sizeof(bits)); }

            uint64_t *get_bits() { return bits; }
        };

        // cacheline-aligned metadata optimized for flush and search
        struct node_meta
        {
            order_type order[key_capacity];
            order_type count;
            bitmap bmap;
            key_type max_key;
            struct leaf_node *next;
            uint8_t __padding__[END_PADDING_SIZE(sizeof(count) + sizeof(bmap) +
                                                 sizeof(order) + sizeof(next) +
                                                 sizeof(max_key))];
            static constexpr int meta_flush_size = sizeof(bmap) + sizeof(max_key) + sizeof(next);
            node_meta() : count(0) {}
            leaf_node *next_sibling() const { return next; }
            leaf_node **next_sibling_ref() { return &next; }
            inline int free_cells() const { return key_capacity - count; }
            inline int key_count() const { return count; }
            inline int key_idx(int ith) const { return order[ith]; }
            inline int min_key_idx() const { return order[0]; }
            inline int max_key_idx() const { return order[count - 1]; }
            inline key_type get_max_key() { return max_key; }
            inline void set_max_key(const key_type &max_key)
            {
                this->max_key = max_key;
            }
            inline void set_key_idx(order_type idx, order_type key_idx) { order[idx] = key_idx; }

            inline void append(order_type key_idx)
            {
                //assert(count < key_capacity);
                order[count++] = key_idx;
            }

            inline void erase(int idx)
            {
                assert(count > 0);
                memmove(order + idx, order + idx + 1,
                        (count - idx) * sizeof(order_type));
                bmap.unset(idx);
                --count;
            }

            inline void clear_count() { count = 0; }

            inline void clear()
            {
                count = 0;
                bmap.clear();
            }

            bitmap &get_bitmap() { return bmap; }

            // reserve n cells at the front
            void reserve_front(int n)
            {
                assert(n <= free_cells());
                memmove(order + n, order, count * sizeof(order_type));
                count += n;
            }
        };
        kv_pair pairs[key_capacity];
        node_meta meta[2];
        uint64_t dirty_cacheline_map = 0; // records whether the cacheline relative pairs is dirty

        void clear_dirty_cacheline_map() {
            dirty_cacheline_map = 0;
        }

        void mark_dirty_cacheline(char *addr, size_t len = sizeof(kv_pair)) {
            char *aligned_addr = CACHELINE_ALIGN(addr);
            int off = static_cast<uint64_t>((aligned_addr - reinterpret_cast<char *>(pairs))) / cacheline_size;
            assert(off <= 63);
            dirty_cacheline_map |= (1ULL << off) | (addr + len > aligned_addr + cacheline_size ? 1ULL << (off + 1) : 0);
        }

        leaf_node() { memset(meta, 0, sizeof(meta)); }

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

        static leaf_node *new_leaf_node()
        {
            //return new leaf_node;
            void *ret;
            nvm_dram_alloc(&ret, cacheline_size, sizeof(leaf_node));
            return new (ret) leaf_node;
        }

        static void delete_leaf_node(leaf_node *l)
        {
            //delete l;
            //nvm_dram_free_cacheline_aligned(l);
            nvm_dram_free(l, sizeof(leaf_node));
        }

        // extract from `bits` `count` consecutive bits starting at off'th bit
        inline uint64_t extract_bits(uint64_t bits, int off, int count)
        {
            return bits & (((1L << count) - 1) << off);
        }
        void flush(bool nv, int &flushes, int &flushed_node_count)
        {
            bool cv = !nv;
            auto cv_map = meta[cv].get_bitmap();
            auto nv_map = meta[nv].get_bitmap();
            uint64_t *cv_bits = cv_map.get_bits();
            uint64_t *nv_bits = nv_map.get_bits();
            int cv_key_count = meta[cv].key_count();
            int nv_key_count = meta[nv].key_count();
            constexpr int kvs_per_cacheline = cacheline_size / sizeof(kv_pair);
            bool need_flush_meta = false;
            for (int k = 0; k * 64 < key_capacity; k++)
            {
                uint64_t cv_bitmap = cv_bits[k];
                uint64_t nv_bitmap = nv_bits[k];
                uint64_t xor_bitmap = cv_bitmap ^ nv_bitmap;
                // Only flush the cachelines that contains newly upserted cells
                int bound = std::min(k + 64, (int)key_capacity);
                for (int i = k; i < bound; i += kvs_per_cacheline)
                {
                    uint64_t cacheline_diff_bits =
                        extract_bits(xor_bitmap, i - k, kvs_per_cacheline);
                    need_flush_meta |= (cacheline_diff_bits & nv_bitmap);
                    // metadata needs flush when there are newly
                    // deleted cells on this cachline
                    if (cacheline_diff_bits == 0 ||             // no difference between nv_bitmap and
                                                                // cv_bitmap on this cachline
                        (cacheline_diff_bits & nv_bitmap) == 0) // this cachline only
                                                                // contains deleted cells
                                                                // which do not need
                                                                // flushing
                        continue;
                    ++flushes;
                    need_flush_meta = true;
                    clflush(pairs + i);
                }
            }

            if (need_flush_meta)
            {
                flushes += node_meta::meta_flush_size / cacheline_size;
                clflush_len_no_fence(&meta[nv], node_meta::meta_flush_size);
                ++flushed_node_count;
            }
        }

        static constexpr double fill_factor = 0.70;
        // When merging a stream of new upsert kvs with a node,
        // any new nodes created should hold at most `max_initial_fill_keys` keys,
        // so there mgiht be enough empty cells to accomadate later upserts.
        static constexpr int max_initial_fill_keys =
            (int)std::ceil(fill_factor * key_capacity);
        // Nodes with # keys less than this constant are considered under-utilized
        // and need merging.
        static constexpr int merge_node_threshold = key_capacity / 3;
        static std::vector<leaf_node *> split_merge_node_with_stream(
            leaf_node *l, typename std::vector<kv_pair>::iterator &upsert_kvs_sit,
            const typename std::vector<kv_pair>::iterator &upsert_kvs_eit, bool cv,
            int &split_merges)
        {
            std::hash<key_type> hasher;
            ++split_merges;
            bool nv = !cv;
            std::vector<leaf_node *> leafs = {new_leaf_node()};
            int l_key_count = l->key_count(cv);
            int i = 0;
            bitmap *node_alloc_bitmap = &leafs.back()->meta[nv].get_bitmap();
            bitmap mask; // records bit positions that should be unmarked due to key
                         // moves
            leaf_node *last_leaf = leafs.back();
            auto ensure_last_leaf_capacity = [&last_leaf, &leafs, nv,
                                              &node_alloc_bitmap]() {
                if (last_leaf->key_count(nv) >=
                    max_initial_fill_keys)
                { // last leaf is full, create a new leaf and
                    // establish the chain
                    auto new_sibling = new_leaf_node();
                    last_leaf->set_max_key(nv);
                    assert(last_leaf->key_count(nv) == node_alloc_bitmap->popcount());
                    *last_leaf->next_sibling_ref(nv) = new_sibling;
                    leafs.push_back(new_sibling);
                    last_leaf = new_sibling;
                    node_alloc_bitmap = &last_leaf->meta[nv].get_bitmap();
                }
            };
            while (i < l_key_count && upsert_kvs_sit != upsert_kvs_eit)
            {
                ensure_last_leaf_capacity();
                int cv_key_idx = l->meta[cv].key_idx(i);
                key_type cv_key = l->key(cv_key_idx);
                key_type upsert_key = upsert_kvs_sit->first;
                int free_idx;
                if (cv_key == upsert_key)
                {
                    free_idx =
                        node_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    last_leaf->pairs[free_idx] = *upsert_kvs_sit;
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    ++i;
                    ++upsert_kvs_sit;
                    mask.set(cv_key_idx);
                }
                else if (cv_key < upsert_key)
                {
                    free_idx = node_alloc_bitmap->alloc_first_unset_bit(hasher(cv_key));
                    last_leaf->pairs[free_idx] = l->pairs[cv_key_idx];
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    ++i;
                }
                else
                {
                    free_idx =
                        node_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    last_leaf->pairs[free_idx] = *upsert_kvs_sit;
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    ++upsert_kvs_sit;
                }
                last_leaf->meta[nv].append(free_idx);
                assert(node_alloc_bitmap->test(free_idx) == true);
            }
            while (i < l_key_count)
            {
                ensure_last_leaf_capacity();
                int cv_key_idx = l->meta[cv].key_idx(i);
                int free_idx = node_alloc_bitmap->alloc_first_unset_bit(
                    hasher(l->pairs[cv_key_idx].first));
                last_leaf->pairs[free_idx] = l->pairs[cv_key_idx];
                last_leaf->mark_dirty_cacheline((char*)&last_leaf->pairs[free_idx]);
                last_leaf->meta[nv].append(free_idx);
                assert(node_alloc_bitmap->test(free_idx) == true);
                ++i;
            }
            while (upsert_kvs_sit != upsert_kvs_eit)
            {
                ensure_last_leaf_capacity();
                int free_idx = node_alloc_bitmap->alloc_first_unset_bit(
                    hasher(upsert_kvs_sit->first));
                last_leaf->pairs[free_idx] = *upsert_kvs_sit;
                last_leaf->mark_dirty_cacheline((char*)&last_leaf->pairs[free_idx]);
                last_leaf->meta[nv].append(free_idx);
                assert(node_alloc_bitmap->test(free_idx) == true);
                ++upsert_kvs_sit;
            }
            last_leaf->set_max_key(nv);
            *last_leaf->next_sibling_ref(nv) = l->next_sibling(cv);
            // last_leaf->meta[cv] = last_leaf->meta[nv];
            node_alloc_bitmap->unset(mask);
            assert(leafs.back() == last_leaf);
            assert(upsert_kvs_sit == upsert_kvs_eit);
            assert(last_leaf->key_count(nv) == node_alloc_bitmap->popcount());
            return leafs;
        }

        static std::vector<leaf_node *> split_node_with_stream(
            leaf_node *l, typename std::vector<kv_pair>::iterator &upsert_kvs_sit,
            const typename std::vector<kv_pair>::iterator &upsert_kvs_eit, bool cv,
            int &split_merges)
        {
            ++split_merges;
            bool nv = !cv;
            std::vector<leaf_node *> leafs = {new_leaf_node()};
            int l_key_count = l->key_count(cv);
            int l_free_cells = l->free_cells(cv);
            int i = 0;
            auto hasher = std::hash<key_type>();

            bitmap *node_alloc_bitmap = &leafs.back()->meta[nv].get_bitmap();
            bitmap mask; // records bit positions that should be unmarked due to key
                         // moves
            leaf_node *last_leaf = leafs.back();
            auto ensure_last_leaf_capacity = [&last_leaf, &leafs, nv,
                                              &node_alloc_bitmap]() {
                if (last_leaf->key_count(nv) >=
                    max_initial_fill_keys)
                { // last leaf is full, create a new leaf and
                    // establish the chain
                    auto new_sibling = new_leaf_node();
                    last_leaf->set_max_key(nv);
                    assert(last_leaf->key_count(nv) == node_alloc_bitmap->popcount());
                    *last_leaf->next_sibling_ref(nv) = new_sibling;
                    leafs.push_back(new_sibling);
                    last_leaf = new_sibling;
                    node_alloc_bitmap = &last_leaf->meta[nv].get_bitmap();
                }
            };
            while (i < l_key_count && upsert_kvs_sit != upsert_kvs_eit)
            {
                // In order to minimize # flushes, we keep l's content as much and
                // unchanged as we can.
                // We perserve l when the following three conditions hold:
                // 1. # kvs left in stream < # free cells in l.
                // 2. The last leaf created has # keys >= merge_node_threshold. (This
                // prevents this node from being merged again later)
                // 3. (# kvs left in stream + # kvs left after moving keys to previous
                // nodes) >= merge_node_threshold. (The same purpose as 2)
                int sleft = upsert_kvs_eit - upsert_kvs_sit;
                if (sleft <= l_free_cells &&
                    last_leaf->key_count(nv) >= merge_node_threshold &&
                    l_key_count - i + sleft >= merge_node_threshold)
                {
                    goto preserve_l;
                }
                // Otherwise insert kvs merged from stream and l into the newly created
                // node
                ensure_last_leaf_capacity();
                int cv_key_idx = l->meta[cv].key_idx(i);
                key_type cv_key = l->key(cv_key_idx);
                key_type upsert_key = upsert_kvs_sit->first;
                int free_idx;
                if (cv_key == upsert_key)
                {
                    free_idx =
                        node_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    last_leaf->pairs[free_idx] = *upsert_kvs_sit;
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    ++i;
                    ++upsert_kvs_sit;
                    mask.set(
                        cv_key_idx); // cv_key_idx is updated and moved to previous nodes
                }
                else if (cv_key < upsert_key)
                {
                    free_idx = node_alloc_bitmap->alloc_first_unset_bit(hasher(cv_key));
                    last_leaf->pairs[free_idx] = l->pairs[cv_key_idx];
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    ++i;
                    mask.set(cv_key_idx); // cv_key_idx is moved to previous nodes
                }
                else
                {
                    free_idx =
                        node_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    last_leaf->pairs[free_idx] = *upsert_kvs_sit;
                    last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                    ++upsert_kvs_sit;
                }
                last_leaf->meta[nv].append(free_idx);
                assert(node_alloc_bitmap->test(free_idx) == true);
            }
            assert(upsert_kvs_sit == upsert_kvs_eit);
            while (i < l_key_count)
            {
                ensure_last_leaf_capacity();
                int cv_key_idx = l->meta[cv].key_idx(i);
                int free_idx = node_alloc_bitmap->alloc_first_unset_bit(
                    hasher(l->pairs[cv_key_idx].first));
                last_leaf->pairs[free_idx] = l->pairs[cv_key_idx];
                last_leaf->mark_dirty_cacheline((char *) &last_leaf->pairs[free_idx]);
                last_leaf->meta[nv].append(free_idx);
                assert(node_alloc_bitmap->test(free_idx) == true);
                ++i;
                mask.set(cv_key_idx); // cv_key_idx is moved to previous nodes
            }
            last_leaf->set_max_key(nv);
            *last_leaf->next_sibling_ref(nv) = l->next_sibling(cv);
            // last_leaf->meta[cv] = last_leaf->meta[nv];
            assert(leafs.back() == last_leaf);
            assert(last_leaf->key_count(nv) == node_alloc_bitmap->popcount());
            return leafs;
        preserve_l:
            // When the conditions hold, we upsert the rest of stream into l.
            node_meta &l_new_meta = l->meta[nv];
            l_new_meta.clear_count(); // only clear out count
            bitmap *l_alloc_bitmap = &l_new_meta.get_bitmap();
            while (upsert_kvs_sit != upsert_kvs_eit)
            {
                int cv_key_idx = l->meta[cv].key_idx(i);
                key_type cv_key = l->key(cv_key_idx);
                key_type upsert_key = upsert_kvs_sit->first;
                assert(i < l_key_count);
                if (cv_key == upsert_key)
                {
                    int free_idx =
                        l_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    l->pairs[free_idx] = *upsert_kvs_sit;
                    l->mark_dirty_cacheline((char *) &l->pairs[free_idx]);
                    ++i;
                    ++upsert_kvs_sit;
                    mask.set(cv_key_idx); // cv_key_idx is updated to another cell in the
                                          // same node
                    l_new_meta.append(free_idx);
                }
                else if (cv_key < upsert_key)
                {
                    ++i;
                    l_new_meta.append(cv_key_idx);
                }
                else
                {
                    int free_idx =
                        l_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    l->pairs[free_idx] = *upsert_kvs_sit;
                    l->mark_dirty_cacheline((char *) &l->pairs[free_idx]);
                    ++upsert_kvs_sit;
                    l_new_meta.append(free_idx);
                }
            }
            assert(upsert_kvs_sit == upsert_kvs_eit);
            while (i < l_key_count)
            {
                int cv_key_idx = l->meta[cv].key_idx(i);
                l_new_meta.append(cv_key_idx);
                ++i;
            }

            last_leaf->set_max_key(nv);
            *last_leaf->next_sibling_ref(nv) = l;
            l_alloc_bitmap->unset(mask);
            //l->meta[nv] = l_new_meta;
            assert(l->key_count(nv) == l_alloc_bitmap->popcount());
            l->set_max_key(nv);
            leafs.push_back(l);
            return leafs;
        }

        // Upsert kv stream whose keys are <= this->max_key().
        // Return the last leaf node whose keys <= this->max_key() after the kvs are
        // upserted.
        leaf_node *
        bulk_upsert(leaf_node **prev_ref,
                    typename std::vector<kv_pair>::iterator &upsert_kvs_sit,
                    const typename std::vector<kv_pair>::iterator &upsert_kvs_eit,
                    bool cv, std::unordered_set<leaf_node *> &gc_candidates,
                    int &update_inplace, int &split_merges, int &perserve_l)
        {
            bool nv = !cv;
            int i = 0;
            int j = 0;
            // copy the small metadata from previous version
            meta[nv] = meta[cv];
            int free_cells = meta[cv].free_cells();
            int in_range_count = upsert_kvs_eit - upsert_kvs_sit;
            std::hash<key_type> hasher;
            assert(key_capacity - free_cells == meta[cv].get_bitmap().popcount());
            if (in_range_count > free_cells)
            { // not enough free cells to upserts
                // keys <= this->max_key, split instead
                // We merge the stream [upsert_kvs_sit, tit) with this leaf by creating
                // as many leaf nodes as needed.
                auto leafs = leaf_node::split_node_with_stream(
                    this, upsert_kvs_sit, upsert_kvs_eit, cv, split_merges);
                assert(leafs.size() >= 1);
                auto last_leaf = leafs.back();
                if (last_leaf != this)
                {
                    gc_candidates.insert(this);
                }
                else
                {
                    ++perserve_l;
                }

                *prev_ref = leafs[0]; // make sure prev_ref(next version pointer) points
                                      // to new leaf
                assert(last_leaf->next_sibling(nv) == this->next_sibling(cv));
                return last_leaf;
            }
            if (in_range_count == 0)
                return this;
            ++update_inplace;

            //std::this_thread::sleep_for(std::chrono::microseconds(100));

            this->meta[nv].clear_count();
            bitmap mask; // records bit positions that should be unmarked due to key
                         // moves
            bitmap *node_alloc_bitmap = &this->meta[nv].get_bitmap();
            int cv_key_count = this->meta[cv].key_count();
            int insert_count = 0;
            while (i < cv_key_count && upsert_kvs_sit != upsert_kvs_eit)
            {
                //++false_sharing_cacheline;
                int cv_key_idx = this->meta[cv].key_idx(i);
                key_type cv_key = key(cv_key_idx);
                key_type upsert_key = upsert_kvs_sit->first;
                if (cv_key < upsert_key)
                {
                    ++i;
                    this->meta[nv].append(cv_key_idx);
                }
                else if (cv_key > upsert_key)
                {
                    int free_idx =
                        node_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    this->pairs[free_idx] = *upsert_kvs_sit;
                    this->mark_dirty_cacheline((char *) &this->pairs[free_idx]);
                    // insert: place new kv onto a new cell and append the idx to the
                    // order array
                    this->meta[nv].append(free_idx);
                    ++upsert_kvs_sit;
                    ++insert_count;
                }
                else
                { // cv_key == upsert_key
                    // upsert: place updated value onto a new cell and modify the order
                    // array
                    int free_idx =
                        node_alloc_bitmap->alloc_first_unset_bit(hasher(upsert_key));
                    this->pairs[free_idx] = *upsert_kvs_sit;
                    this->mark_dirty_cacheline((char *) &this->pairs[free_idx]);
                    this->meta[nv].append(free_idx);
                    ++i;
                    ++upsert_kvs_sit;
                    mask.set(cv_key_idx);
                }
            }

            if (i < cv_key_count)
            {
                memcpy(this->meta[nv].order + this->meta[nv].key_count(), this->meta[cv].order + i,
                       sizeof(order_type) * (cv_key_count - i));
                this->meta[nv].count += (cv_key_count - i);
            }

            while (upsert_kvs_sit != upsert_kvs_eit)
            {
                int free_idx = node_alloc_bitmap->alloc_first_unset_bit(
                    hasher(upsert_kvs_sit->first));
                this->pairs[free_idx] = *upsert_kvs_sit;
                this->mark_dirty_cacheline((char *) &pairs[free_idx]);
                this->meta[nv].append(free_idx);
                ++upsert_kvs_sit;
            }

            assert(upsert_kvs_eit == upsert_kvs_sit);
            assert(cv_key_count + insert_count == this->meta[nv].key_count());
            node_alloc_bitmap->unset(mask);
            this->set_max_key(nv);
            assert(node_alloc_bitmap->popcount() == this->meta[nv].key_count());
            return this;
        }

        // Since deletion comes after upsertions, we apply them directly on the new
        // version
        void bulk_delete(const std::vector<key_type> &delete_keys, bool nv)
        {
            // delete by merging
            int i = 0;
            int j = 0;
            while (i < meta[nv].key_count() && j < delete_keys.size())
            {
                int nv_key_idx = meta[nv].key_idx(i);
                key_type nv_key = key(nv_key_idx);
                key_type delete_key = delete_keys[j];
                if (nv_key < delete_key)
                {
                    ++i;
                }
                else if (nv_key > delete_key)
                {
                    ++j;
                }
                else if (nv_key == delete_key)
                {
                    meta[nv].erase(i);
                    ++j;
                }
            }
            this->set_max_key(nv);
        }

        static std::vector<leaf_node *>
        merge_multiple_nodes(std::vector<leaf_node *> leafs, const int total_keys,
                             bool nv)
        {
            const int keys_per_node = std::ceil(total_keys / 2.0);
            std::vector<leaf_node *> merged_leafs = {new_leaf_node()};
            std::hash<key_type> hasher;
            bitmap *node_alloc_bitmap = &merged_leafs.back()->meta[nv].get_bitmap();
            leaf_node *last_leaf = merged_leafs.back();
            auto ensure_last_leaf_capacity = [&last_leaf, &merged_leafs, nv,
                                              &node_alloc_bitmap, &keys_per_node]() {
                if (last_leaf->key_count(nv) >=
                    keys_per_node)
                { // last leaf is full, create a new leaf and
                    // establish the chain
                    auto new_sibling = new_leaf_node();
                    last_leaf->set_max_key(nv);
                    assert(last_leaf->key_count(nv) == node_alloc_bitmap->popcount());
                    *last_leaf->next_sibling_ref(nv) = new_sibling;
                    merged_leafs.push_back(new_sibling);
                    last_leaf = new_sibling;
                    node_alloc_bitmap = &last_leaf->meta[nv].get_bitmap();
                }
            };
            for (int i = 0; i < leafs.size(); ++i)
            {
                leaf_node *l = leafs[i];
                int l_key_count = l->key_count(nv);
                for (int j = 0; j < l_key_count; ++j)
                {
                    ensure_last_leaf_capacity();
                    int free_idx = node_alloc_bitmap->alloc_first_unset_bit(
                        hasher(l->ith_key(j, nv)));
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
        // 1. If the utilization of l1 >= merge_node_threshold/key_capacity: no
        // merge.
        // 2. Else if l1->key_count() <= l1->next->free_cells(): migrate keys from
        // l1 to l1->next
        // 3. Else: combine keys from l1 onward until # accumulated keys >=
        // key_capacity or the end is reached
        // Return the next node to be examined.

        static leaf_node *
        merge_underutilized_node(leaf_node **&prev_ref, leaf_node *l1, bool nv,
                                 std::unordered_set<leaf_node *> &gc_candidates,
                                 int &node_merges_transfer,
                                 int &node_merges_combine,
                                 leaf_node *&last_leaf,
                                 leaf_node *end_leaf = nullptr)
        {
            std::hash<key_type> hasher;
            int l1_key_count = l1->key_count(nv);
            leaf_node *l2 = l1->next_sibling(nv);
            if (l1_key_count >= merge_node_threshold || l2 == end_leaf)
            {
                last_leaf = l1;
                prev_ref = l1->next_sibling_ref(nv);
                return l2;
            }

            if (l1_key_count <= l2->free_cells(nv))
            { // l1's keys fit in l2's free cells
                ++node_merges_transfer;
                node_meta l2_tmp_meta = l2->meta[nv];
                bitmap *l2_alloc_bitmap = &l2_tmp_meta.get_bitmap();
                l2_tmp_meta.reserve_front(l1_key_count);
                assert(l2_tmp_meta.key_count() == l1_key_count + l2->key_count(nv));
                for (int i = 0; i < l1_key_count; ++i)
                {
                    int free_idx = l2_alloc_bitmap->alloc_first_unset_bit(
                        hasher(l1->ith_key(i, nv)));
                    l2->pairs[free_idx] = l1->ith_kv(i, nv);
                    l2->mark_dirty_cacheline((char *) &l2->pairs[free_idx]);
                    l2_tmp_meta.set_key_idx(i, free_idx);
                }
                l2->meta[nv] = l2_tmp_meta;
                l2->set_max_key(nv);
                gc_candidates.insert(*prev_ref);
                *prev_ref = l2;
                last_leaf = l1;
                assert(l2->key_count(nv) == l2_alloc_bitmap->popcount());
                // prev_ref = l2->next_sibling_ref(nv);
                // TODO: garbage collect l1
                return l2;
            }
            else
            { // Otherwise we accumulate keys from l1 onward until #
                // accumulated keys >= key_capacity
                ++node_merges_combine;
                std::vector<leaf_node *> leafs;
                int total_keys = 0;
                auto cur = l1;
                while (cur != end_leaf && total_keys < key_capacity)
                {
                    total_keys += cur->key_count(nv);
                    gc_candidates.insert(cur);
                    leafs.push_back(cur);
                    cur = cur->next_sibling(nv);
                }
                // TODO: garabge collect leafs
                std::vector<leaf_node *> merged_leafs =
                    merge_multiple_nodes(leafs, total_keys, nv);
                assert(merged_leafs.size() == 2);
                *prev_ref = merged_leafs[0];
                last_leaf = merged_leafs.back();
                prev_ref = merged_leafs.back()->next_sibling_ref(nv);
                *prev_ref = cur;
                return cur;
            }
        }
    };

    struct stat
    {
        stat() : kv_count(0), node_count(0) {}
        std::atomic<int> kv_count;
        std::atomic<int> node_count;
        double avg_fill()
        {
            return kv_count.load() / (node_count.load() * key_capacity + 1.0);
        }
    };
    concur_cvhtree(double &inner_nodes_build_time,
                   double &parallel_merge_work_time)
        : art_tree_build_time(inner_nodes_build_time),
          parallel_merge_work_time(parallel_merge_work_time), pool(nullptr)
    {
        memset(stats, 0, sizeof(stats));
        memset(head, 0, sizeof(head));
        memset(art_trees, 0, sizeof(art_trees));
        printf("sizeof(leaf_node) %d, sizeof(node_meta) %d\n", (int)sizeof(leaf_node),
               (int)sizeof(typename leaf_node::node_meta));
        gv = 0;
    }

    size_t size() { return stats[gv].kv_count; }

    void update_stat(bool v)
    {
        auto cur = head[v];
        stats[v].kv_count = stats[v].node_count = 0;
        std::unordered_map<int, int> m;
        while (cur)
        {
            stats[v].kv_count += cur->key_count(v);
            m[cur->key_count(v)]++;
            ++stats[v].node_count;
            cur = cur->next_sibling(v);
        }
        printf("node key count distribution\n");
        for (auto kv : m)
        {
            printf("%d -> %d\n", kv.first, kv.second);
        }
    }
    stat get_stat() { return stats[gv]; }

    static inline bool is_delete_op(const value_type &v) { return v & 1; }

    static inline bool is_upsert_op(const value_type &v)
    {
        return !is_delete_op(v);
    }

    struct iterator
    {
        leaf_node *l;
        int idx;
        bool v;
        explicit iterator(leaf_node *l, int idx, bool v) : l(l), idx(idx), v(v) {}
        inline bool operator==(const iterator &rhs) const
        {
            return l == rhs.l && idx == rhs.idx;
        }
        inline bool operator!=(iterator other) const { return !(*this == other); }

        inline iterator &operator++()
        {
            if (++idx >= l->key_count(v))
            {
                l = l->next_sibling(v);
                //if (l) prefetch((char*)l->next_sibling(v), sizeof(leaf_node));
                idx = 0;
            }
            return *this;
        }
        inline iterator operator++(int)
        {
            auto retval = *this;
            ++(*this);
            return retval;
        }
        inline key_type &key() { return l->ith_key(idx, v); }
        inline value_type &value() { return l->ith_value(idx, v); }
        inline leaf_node *leaf() { return l; }
    };

    iterator begin() { return iterator(head[gv], 0, gv); }
    iterator end() { return iterator(nullptr, 0, gv); }

    leaf_node *lower_bound_leaf(const key_type &key)
    {
        bool v = gv;
        uint8_t lookup_key[sizeof(key_type)];
        // TODO: record reader epoch
        *reinterpret_cast<uint64_t *>(lookup_key) = __builtin_bswap64(key);
        bool pref = true;
        auto tree = art_trees[v];
        if (tree == nullptr)
            return nullptr;
        auto leaf = tree->lowerBound(lookup_key, sizeof(key_type), 0,
                                     sizeof(key_type), pref);
        if (leaf == nullptr)
            return nullptr;
        auto cur = reinterpret_cast<leaf_node *>(ART_IDX::getLeafValue(leaf));
        while (cur && key > cur->max_key(v))
        {
            cur = cur->next_sibling(v);
        }
        return cur;
    }

    iterator lower_bound(const key_type &key)
    {
        bool v = gv;
        uint8_t lookup_key[sizeof(key_type)];
        // TODO: record reader epoch
        *reinterpret_cast<uint64_t *>(lookup_key) = __builtin_bswap64(key);
        bool pref = true;
        auto tree = art_trees[v];
        if (tree == nullptr)
            return end();
        auto leaf = tree->lowerBound(lookup_key, sizeof(key_type), 0,
                                     sizeof(key_type), pref);
        if (leaf == nullptr)
            return end();
        auto cur = reinterpret_cast<leaf_node *>(ART_IDX::getLeafValue(leaf));
        prefetch((char *)cur, sizeof(leaf_node));
        while (cur && key > cur->max_key(v))
        {
            cur = cur->next_sibling(v);
            prefetch((char *)cur, sizeof(leaf_node));
        }
        if (cur == nullptr)
            return end();
        int i = 0;
        int key_count = cur->key_count(v);
        while (i < key_count && key > cur->ith_key(i, v))
        {
            ++i;
        }
        return cur == nullptr ? end() : iterator(cur, i, v);
    }

    // scan leafs for keys in [start_key, end_key)
    void range_scan(const key_type &start_key, const key_type &end_key,
                    std::vector<value_type> &values,
                    std::function<void(value_type)> processor)
    {
        assert(start_key <= end_key);
        auto eit = end();
        auto it = lower_bound(start_key);
        while (it != eit && it.key() < end_key)
        {
            values.push_back(it.value());
            ++it;
        }
    }
    int flush_node_total = 0;
    leaf_node empty_node;
    template <class Iterator>
    void parallel_merge(Iterator &start_it, const Iterator &end_it, int buffer_tree_kv_count, std::function<void()> leaf_merge_done_func,
                        int nworkers = 1)
    {
        if (pool == nullptr)
        {
            pool = new ThreadPool(nworkers);
        }
        auto sanity_check_func = [this](leaf_node *h) {
            int max_key_node_idx = -1;
            uint64_t max_key = 0;
            int node_idx = 0;
            auto prev_mkey = max_key;
            bool sane = true;
            while (h)
            {
                auto mkey = h->max_key(gv);
                // printf("%p max_key %llu idx %d key count %d\n", h, mkey, node_idx,
                // h->key_count(gv));
                if (prev_mkey > mkey)
                {
                    printf("!!! prev_mkey %lu, mkey %lu, mkey_cv %d\n", prev_mkey,
                           mkey, h->key_count(!gv));
                    assert(false);
                }
                sane &= prev_mkey <= mkey;
                if (mkey > max_key)
                {
                    max_key = mkey;
                    max_key_node_idx = node_idx;
                }
                prev_mkey = mkey;
                ++node_idx;
                h = h->next_sibling(gv);
            }
            // printf("max_key %llu, node_idx %d\n", max_key, max_key_node_idx);
            assert(sane);
        };
        // sanity_check_func(head[gv]);
        bool cv = gv;  // currnet version
        bool nv = !gv; // next version
        total_merge_read_data_size += stats[cv].node_count * sizeof(leaf_node);

        int count = buffer_tree_kv_count;
        int range_size = std::min(4 * key_capacity, count / nworkers);
        struct merge_task
        {
            merge_task() : start_leaf(nullptr), end_leaf(nullptr), last_leaf(nullptr), merge_in_kv_count(0) {}

            //[start_leaf, end_leaf) ranges over the leaf nodes being operated
            leaf_node *start_leaf;
            leaf_node *end_leaf;
            // output variable, points to the last node in [start_leaf, end_leaf)
            leaf_node *last_leaf;
            int merge_in_kv_count;
            // [kv_sit, kv_eit) ranges over the key-value entries being merged in
            Iterator kv_sit;
            Iterator kv_eit;
            uint8_t __padding__[END_PADDING_SIZE(
                sizeof(kv_eit) + sizeof(kv_sit) + sizeof(start_leaf) +
                sizeof(end_leaf) + sizeof(last_leaf) + sizeof(merge_in_kv_count))];
        };
        struct merge_worker_stat
        {
            merge_worker_stat()
                : flushes(0), flushed_node(0), update_inplace(0), split_merges(0),
                  perserve_l(0), node_merges_transfer(0), node_merges_combine(0),
                  kv_count(0), merge_time(0), flush_time(0), merge_in_kv_count(0), node_count(0) {}

            //[start_leaf, end_leaf) ranges over the leaf nodes being operated
            std::unordered_set<leaf_node *> gc_candidates;
            std::vector<std::pair<key_type, leaf_node *>> maxkey_node_pairs;
            int flushes;
            int flushed_node;
            int update_inplace;
            int split_merges;
            int perserve_l;
            int node_merges_transfer;
            int node_merges_combine;
            int kv_count;
            int node_count;
            int merge_in_kv_count;
            double merge_time;
            double flush_time;

            uint8_t __padding__[END_PADDING_SIZE(
                sizeof(int) * 10 + sizeof(double) * 2 + sizeof(gc_candidates) +
                +sizeof(maxkey_node_pairs))];
        };

        auto merge_func = [cv, nv, this](merge_task &task, merge_worker_stat &stat) {
            bool has_del = false;
            auto &gc_candidates = stat.gc_candidates;
            auto starttime2 = secs_now();
            stat.merge_in_kv_count += task.merge_in_kv_count;
            // 1. upserts
            {
                auto sit = task.kv_sit;
                auto eit = task.kv_eit;
                auto cur = task.start_leaf;
                leaf_node **prev_ref = &task.start_leaf;
                int &update_inplace = stat.update_inplace;
                int &split_merges = stat.split_merges;
                int &perserve_l = stat.perserve_l;

                while (cur != task.end_leaf && sit != eit)
                {
                    key_type cur_high_key = cur->max_key(cv);
                    std::vector<kv_pair> upsert_kvs;
                    while (sit != eit && sit.key() <= cur_high_key)
                    {
                        bool is_upsert = is_upsert_op(sit.value());
                        if (is_upsert)
                        {
                            upsert_kvs.push_back(std::make_pair(sit.key(), sit.value()));
                        }
                        has_del |= !is_upsert;
                        ++sit;
                    }
                    auto ustart = upsert_kvs.begin();
                    auto uend = upsert_kvs.end();
                    auto next = cur->next_sibling(cv);
                    //if (next != task.end_leaf)
                    prefetch((char *)cur, sizeof(leaf_node));
                    cur = cur->bulk_upsert(prev_ref, ustart, uend, cv, gc_candidates,
                                           update_inplace, split_merges, perserve_l);
                    prev_ref = cur->next_sibling_ref(nv);
                    assert(*prev_ref == next);
                    cur = next;
                }

                if (sit != eit)
                {
                    assert(cur == task.end_leaf);
                    std::vector<kv_pair> upsert_kvs;
                    while (sit != eit)
                    {
                        if (is_upsert_op(sit.value()))
                        {
                            upsert_kvs.push_back(std::make_pair(sit.key(), sit.value()));
                        }
                        ++sit;
                    }
                    auto ustart = upsert_kvs.begin();
                    auto uend = upsert_kvs.end();
                    // auto next = *prev_ref;
                    // assert(next == nullptr);
                    auto leafs = leaf_node::split_merge_node_with_stream(
                        &empty_node, ustart, uend, cv, split_merges);
                    assert(leafs.empty() == false);
                    assert(leafs.back()->next_sibling(nv) == nullptr);
                    assert(ustart == uend);
                    *prev_ref = leafs[0];
                    //*leafs.back()->next_sibling_ref(nv) = task.end_leaf;
                }
                else
                {
                    while (cur != task.end_leaf)
                    {
                        // copy the metadata from previous version to current version
                        cur->meta[nv] = cur->meta[cv];
                        assert(cur->key_count(nv) == cur->meta[nv].get_bitmap().popcount());
                        cur = cur->next_sibling(cv);
                    }
                }
                //printf("update inplace %d, split_merges %d, perserve_l %d\n",
                //      update_inplace, split_merges, perserve_l);
                assert(cur == task.end_leaf);
            }

            // 2.deletions
            {

                if (has_del)
                {
                    auto h = task.start_leaf;
                    auto cur = h;
                    auto sit = task.kv_sit;
                    auto eit = task.kv_eit;
                    std::vector<key_type> delete_keys;
                    while (cur != task.end_leaf)
                    {
                        key_type cur_high_key = cur->max_key(nv);
                        delete_keys.clear();
                        while (sit != eit && sit.key() <= cur_high_key)
                        {
                            if (is_delete_op(sit.value()))
                            {
                                delete_keys.push_back(sit.key());
                            }
                            ++sit;
                        }
                        cur->bulk_delete(delete_keys, nv);
                        cur = cur->next_sibling(nv);
                    }
                }
            }

            // 3.merge under-utilized nodes (consolidation)

            {
                auto h = task.start_leaf;
                auto cur = h;
                leaf_node **prev_ref = &task.start_leaf;
                int &node_merges_transfer = stat.node_merges_transfer;
                int &node_merges_combine = stat.node_merges_combine;
                while (cur != task.end_leaf)
                {
                    //prefetch((char *)cur->next_sibling(nv), sizeof(leaf_node));
                    cur = leaf_node::merge_underutilized_node(
                        prev_ref, cur, nv, gc_candidates, node_merges_transfer,
                        node_merges_combine, task.last_leaf, task.end_leaf);
                }
                // printf("node_merges_transfer %d, node_merges_combine %d\n",
                //        node_merges_transfer, node_merges_combine);
            }
            auto h = task.start_leaf;
            auto cur = task.start_leaf;
            while (cur != task.end_leaf)
            {
                task.last_leaf = cur;
                cur = cur->next_sibling(nv);
            }
            assert(task.last_leaf);
            assert(task.last_leaf->next_sibling(nv) == task.end_leaf);
            stat.merge_time += secs_now() - starttime2;
        };
        auto starttime = secs_now();
        // 4. divide & distribute work

        int max_task_count = buffer_tree_kv_count / range_size + 10;
        lockfree_taskqueue<merge_task> tasks(max_task_count);

        auto merge_worker_func = [&merge_func, &tasks, this](merge_worker_stat &worker_stat) {
            while (true)
            {
                int task_idx = -1;
                auto res = tasks.pop(task_idx);
                if (res == false)
                    break;

                merge_func(tasks[task_idx], worker_stat);
            }
        };
        std::vector<merge_worker_stat> merge_worker_stats;
        std::vector<std::future<void>> merge_worker_futures;
        nworkers = std::min(nworkers, max_task_count);
        merge_worker_stats.resize(nworkers);
        for (size_t i = 0; i < nworkers; ++i)
        {
            merge_worker_futures.emplace_back(
                pool->enqueue(merge_worker_func, std::ref(merge_worker_stats[i])));
        }

        auto sit = start_it;
        leaf_node *start_leaf = head[cv];
        leaf_node *end_leaf = nullptr;
        while (sit != end_it)
        {
            auto orig_sit = sit;
            auto prev_eit = sit;
            int merge_in_kv_count = 0;
            int range_countdown = range_size;
            key_type last_key = sit.key();
            while (sit != end_it && range_countdown > 0)
            {
                int advanced = sit.next_node(last_key);
                range_countdown -= advanced;
                merge_in_kv_count += advanced;
            }
            auto lb_leaf = lower_bound_leaf(last_key);

            if (lb_leaf == nullptr)
            {
                // printf("key %llu found no lower_bound leaf\n", key);
                merge_task task;
                task.start_leaf = start_leaf;
                task.end_leaf = nullptr;
                task.last_leaf = nullptr;
                task.kv_sit = orig_sit;
                task.kv_eit = sit;
                task.merge_in_kv_count = merge_in_kv_count;
                tasks.push(task);
                end_leaf = start_leaf = nullptr;
                // printf("merge task: start_leaf %p, end_leaf %p, max_key %llu\n",
                // tasks.back().start_leaf, tasks.back().end_leaf, prev_eit.key());
            }
            else
            {
                auto max_key = last_key;
                while (sit != end_it && sit.key() <= lb_leaf->max_key(cv))
                {
                    max_key = sit.key();
                    ++sit;
                    ++merge_in_kv_count;
                }
                merge_task task;
                task.start_leaf = start_leaf;
                task.last_leaf = nullptr;
                task.end_leaf = lb_leaf->next_sibling(cv);
                task.kv_sit = orig_sit;
                task.kv_eit = sit;
                task.merge_in_kv_count = merge_in_kv_count;
                tasks.push(task);
                start_leaf = end_leaf = task.end_leaf;
                // printf("merge task: start_leaf %p, end_leaf %p, max_key %llu\n",
                // tasks.back().start_leaf, tasks.back().end_leaf, max_key);
            }
        }
        // If there are leaf nodes not covered by the kv stream, start a worker and
        // copy over the metadata to next version
        if (end_leaf != nullptr)
        {
            merge_task task;
            task.start_leaf = end_leaf;
            task.last_leaf = nullptr;
            task.end_leaf = nullptr;
            task.kv_sit = end_it;
            task.kv_eit = end_it;
            tasks.push(task);
        }
        tasks.end_enqueue();

        // synchronize
        std::for_each(merge_worker_futures.begin(), merge_worker_futures.end(),
                      [](std::future<void> &f) { f.get(); });
        parallel_merge_work_time += secs_now() - starttime;

        starttime = secs_now();
        // connect the lists
        for (size_t i = 1; i < tasks.size(); ++i)
        {
            auto prev_last_leaf = tasks[i - 1].last_leaf;
            auto next_first_leaf = tasks[i].start_leaf;
            *prev_last_leaf->next_sibling_ref(nv) = next_first_leaf;
            tasks[i - 1].end_leaf = next_first_leaf;
        }

        head[nv] = tasks[0].start_leaf;


        // 5. collect stats
        auto gc_func = [cv, nv, this](merge_task &task, merge_worker_stat &stat) {
            auto starttime2 = secs_now();
            auto cur = task.start_leaf;
            auto &gc_candidates = stat.gc_candidates;
            auto &maxkey_node_pairs = stat.maxkey_node_pairs;
            int &flushes = stat.flushes;
            int &flushed_node = stat.flushed_node;
            mfence();
            while (cur != task.end_leaf)
            {
                gc_candidates.erase(cur);
                maxkey_node_pairs.push_back(std::make_pair(cur->max_key(nv), cur));
                stat.kv_count += cur->meta[nv].key_count();
                stats[nv].kv_count += cur->meta[nv].key_count();
                ++stats[nv].node_count;
                auto next = cur->next_sibling(nv);
                cur = next;
            }
            mfence();
            stat.flush_time = secs_now() - starttime2;
        };
        std::atomic<int> gc_work_off(0);
        auto gc_worker_func = [&gc_func, &gc_work_off, &tasks, this](merge_worker_stat &stat) {
            while (true)
            {
                int idx = gc_work_off.fetch_add(1);
                if (idx >= tasks.size())
                    break;
                gc_func(tasks[idx], stat);
            }
        };
        stats[nv].kv_count = stats[nv].node_count = 0;
        std::vector<std::future<void>> gc_worker_futures;
        for (size_t i = 0; i < nworkers; ++i)
        {
            gc_worker_futures.emplace_back(
                pool->enqueue(gc_worker_func, std::ref(merge_worker_stats[i])));
        }


        // synchronize
        std::for_each(gc_worker_futures.begin(), gc_worker_futures.end(),
                      [](std::future<void> &f) { f.get(); });

        parallel_merge_work_time += secs_now() - starttime;

        starttime = secs_now();

        std::vector<leaf_node *> nodes_to_flush;
        for (size_t i = 0; i < nworkers; ++i)
        {
            for (auto p : merge_worker_stats[i].maxkey_node_pairs)
            {
                nodes_to_flush.push_back(p.second);
            }
        }
        //printf("tasks.size() : %d nodes_to_flush.size() : %d\n", (int)tasks.size(), (int)nodes_to_flush.size());
        std::atomic<int> flush_off(0);
        auto flush_worker_func = [&, this](merge_worker_stat &stat) {
            int &flushed_node_count = stat.flushed_node;
            int &flushes = stat.flushes;
            auto starttime2 = secs_now();
            while (true)
            {
                int idx = flush_off.fetch_add(1);
                if (idx >= nodes_to_flush.size())
                    break;
                nodes_to_flush[idx]->flush(nv, flushes, flushed_node_count);
            }
            stat.flush_time = secs_now() - starttime2;
        };

        std::vector<std::future<void>> flush_worker_futures;
        for (size_t i = 0; i < nworkers; ++i)
        {
            flush_worker_futures.emplace_back(
                pool->enqueue(flush_worker_func, std::ref(merge_worker_stats[i])));
        }


        // 6. build in-memory radix-tree index for leaf nodes
        {
            auto st = secs_now();
            auto loadKey = [this, nv](uintptr_t tid, uint8_t key[]) {
                leaf_node *l = reinterpret_cast<leaf_node *>(tid);
                reinterpret_cast<uint64_t *>(key)[0] =
                        __builtin_bswap64(l->max_key(nv));
            };
            art_trees[nv] = new ART_IDX::art_tree(loadKey, loadKey);
            // ART_IDX::art_tree::bulkLoadCreate(loadKey, loadKey, kvs, 0, kvs.size()
            // - 1, 0, KeyExtract, KeyLen);
            auto cur = head[nv];
            int kvs_count = 0;
            for (size_t i = 0; i < nworkers; ++i)
            {
                for (auto p : merge_worker_stats[i].maxkey_node_pairs)
                {
                    uint64_t key = __builtin_bswap64(p.first);
                    uintptr_t value = (uintptr_t)p.second;
                    art_trees[nv]->insert((uint8_t *)&key, value, sizeof(key_type));
                    ++kvs_count;
                }
            }
            auto et = secs_now();
            art_tree_build_time += et - st;
            // printf("node_count %d, kv_count: %d, kvs.size(): %d\n",
            // stats[nv].node_count.load(), stats[nv].kv_count.load(), kvs_count);
        }

        leaf_merge_done_func();

        // synchronize
        std::for_each(flush_worker_futures.begin(), flush_worker_futures.end(),
                      [](std::future<void> &f) { f.get(); });
        total_flush_time += secs_now() - starttime;
        // persist leaf list headers
        clflush_then_sfence((volatile void *) head);

        // 7. flip the global version
        {
            gv = nv;
            clflush_then_sfence(&gv);
        }

        starttime= secs_now();

        for (int i = 0; i < nworkers; ++i)
        {
            auto &stat = merge_worker_stats[i];

//            printf("merge_worker_%d merge_in_kv_count %d kv_count %d node_count %d merge_time %.03f flush_time %.03f "
//                   "flushed_node %d flushes %d update_inplace %d split_merges %d perserve_l %d node_merges_transfer %d  node_merges_combine %d avg_fill %f\n",
//                   i, stat.merge_in_kv_count, stat.kv_count, stat.node_count, stat.merge_time, stat.flush_time,
//                   stat.flushed_node, stat.flushes, stat.update_inplace, stat.split_merges, stat.perserve_l, stat.node_merges_transfer, stat.node_merges_combine, stats[nv].avg_fill());
            flush_node_total += stat.flushed_node;
        }
        //printf("flush_node_total %d\n", flush_node_total);
        starttime = secs_now();
        // 8. Now that the new version is persistent, garbage collect nodes
        // not used by the new version
        {
            ART_IDX::art_tree *old_tree = art_trees[cv];
            // wait for readers operating on previous version to exit
            spin_until_no_ref(reader_manager, (uintptr_t)old_tree);
            // now physically delete all the garbages
            for (size_t i = 0; i < nworkers; ++i)
            {
                for (auto node : merge_worker_stats[i].gc_candidates)
                {
                    leaf_node::delete_leaf_node(node);
                }
            }
            delete old_tree;
        }
        parallel_merge_work_time += secs_now() - starttime;
        // sanity_check_func(head[gv]);
    }

    double &art_tree_build_time;
    double &parallel_merge_work_time;
    double total_flush_time = 0;
    long long total_merge_read_data_size = 0;
    size_t probes = 0;
    size_t get_probes() { return probes; }

    // assume sync word is held
    iterator begin(thread_epoch & e) {
        bool v = gv;
        e.set_value((uint64_t)art_trees[v]);
        return iterator(head[v], 0, v);
    }

    // assume sync word is held
    iterator lower_bound(const key_type &key, thread_epoch & e) {
        std::hash<key_type> hasher;
        bool v = gv;
        ART_IDX::art_tree *tree = (ART_IDX::art_tree *)e.set_value((uint64_t)art_trees[v]);
        if (tree == nullptr)
            return end();
        uint8_t lookup_key[sizeof(key_type)];
        *reinterpret_cast<uint64_t *>(lookup_key) = __builtin_bswap64(key);
        bool pref = true;
        auto leaf = tree->lowerBound(lookup_key, sizeof(key_type), 0,
                                     sizeof(key_type), pref);
        auto cur = reinterpret_cast<leaf_node *>(ART_IDX::getLeafValue(leaf));
        prefetch((char *)cur, sizeof(leaf_node));
        while (cur && key > cur->max_key(v))
        {
            cur = cur->next_sibling(v);
            prefetch((char *)cur, sizeof(leaf_node));
        }
        if (cur == nullptr)
            return end();
        auto lo = 0;
        auto key_count = cur->key_count(v);
        while (lo < key_count && cur->ith_key(lo, v) < key) ++lo;
        if (lo >= key_count) return end();
        return iterator(cur, lo, v);
    }

    bool update_if_found(const key_type & key, const value_type & value) {
        std::hash<key_type> hasher;
        reader_epoch.register_to_manager(&reader_manager);
        epoch_guard g(reader_epoch);
        bool v = gv;
        ART_IDX::art_tree *tree = (ART_IDX::art_tree *)reader_epoch.set_value((uint64_t)art_trees[v]);

        uint8_t lookup_key[sizeof(key_type)];
        *reinterpret_cast<uint64_t *>(lookup_key) = __builtin_bswap64(key);
        bool pref = true;
        auto leaf = tree->lowerBound(lookup_key, sizeof(key_type), 0,
                                     sizeof(key_type), pref);
        if (leaf == nullptr)
            return false;
        auto cur = reinterpret_cast<leaf_node *>(ART_IDX::getLeafValue(leaf));
        uint8_t search_key_fp = std::hash<key_type>()(key);
        while (cur && key > cur->max_key(v))
        {
            cur = cur->next_sibling(v);
        }
        if (cur == nullptr)
            return false;

        size_t h = hasher(key);
        auto *bmap = &cur->meta[v].get_bitmap();
        int p = h % key_capacity;
        int limit = probe_limit;
        while (limit)
        {
            //++probes;
            if (cur->pairs[p].first == key && bmap->test(p))
            {
                cur->pairs[p].second = value;
                clflush_then_sfence(&cur->pairs[p]);
                return true;
            }
            --limit;
            p = (p + 1) % key_capacity;
        }

        p = hash2(h) % key_capacity;
        int start_p = p;
        do
        {
            //++probes;
            if (cur->pairs[p].first == key && bmap->test(p))
            {
                cur->pairs[p].second = value;
                clflush_then_sfence(&cur->pairs[p]);
                return true;
            }
            p = (p + 1) % key_capacity;
        } while (p != start_p);
        return false;
    }

    bool lookup(const key_type &key, value_type &value)
    {
        std::hash<key_type> hasher;
        reader_epoch.register_to_manager(&reader_manager);
        epoch_guard g(reader_epoch);
        bool v = gv;
        ART_IDX::art_tree *tree = (ART_IDX::art_tree *)reader_epoch.set_value((uint64_t)art_trees[v]);
        if (tree == nullptr) {
            return false;
        }
        uint8_t lookup_key[sizeof(key_type)];
        *reinterpret_cast<uint64_t *>(lookup_key) = __builtin_bswap64(key);
        bool pref = true;
        auto leaf = tree->lowerBound(lookup_key, sizeof(key_type), 0,
                                     sizeof(key_type), pref);
        if (leaf == nullptr)
            return false;
        auto cur = reinterpret_cast<leaf_node *>(ART_IDX::getLeafValue(leaf));
        uint8_t search_key_fp = std::hash<key_type>()(key);
        while (cur && key > cur->max_key(v))
        {
            cur = cur->next_sibling(v);
        }
        if (cur == nullptr)
            return false;

        size_t h = hasher(key);
        auto *bmap = &cur->meta[v].get_bitmap();
        int p = h % key_capacity;
        int limit = probe_limit;
        while (limit)
        {
            //++probes;
            if (cur->pairs[p].first == key && bmap->test(p))
            {
                value = cur->pairs[p].second;
                return true;
            }
            --limit;
            p = (p + 1) % key_capacity;
        }

        p = hash2(h) % key_capacity;
        int start_p = p;
        do
        {
            //++probes;
            if (cur->pairs[p].first == key && bmap->test(p))
            {
                value = cur->pairs[p].second;
                return true;
            }
            p = (p + 1) % key_capacity;
        } while (p != start_p);
        return false;
    }

    stat stats[2];
    leaf_node *head[2];
    ART_IDX::art_tree *art_trees[2];
    bool gv = 0; // globally consistent version on nvm
    epoch_manager reader_manager;
    static thread_local thread_epoch reader_epoch;
    ThreadPool *pool;
};

enum class op_type
{
    upsertion,
    deletion
};

template <class Key, class Value>
struct log_record
{
    using key_type = Key;
    using value_type = Value;

    key_type key;
    value_type val;
    size_t h;
    op_type type;
    char cacheline_padding[cacheline_size - sizeof(type) - sizeof(key) -
                           sizeof(val) - sizeof(h)];
    log_record(op_type type, const key_type &key, const value_type &val)
        : type(type), key(key), val(val)
    {
        //h = (std::hash<int>()(static_cast<int>(type)) << 1) ^
        //    (std::hash<key_type>()(key) << 2) ^ (std::hash<value_type>()(val));
    }
    log_record(op_type type, const key_type &key) : type(type), key(key), val()
    {
        //h = (std::hash<int>()(static_cast<int>(type)) << 1) ^
        //    (std::hash<key_type>()(key) << 2) ^ (std::hash<value_type>()(val));
    }
};

template <int PageSize = 64 * 1024>
struct log_page
{
    uint64_t off;
    using log_page_type = log_page<PageSize>;
    log_page_type *next;
    uint8_t __padding__[END_PADDING_SIZE(sizeof(off) + sizeof(next))];
    char storage[];
    log_page() : off(0), next(nullptr) {}
    static constexpr int page_size = PageSize;
    static constexpr int meta_size = sizeof(off) + sizeof(next) + sizeof(__padding__);
    static constexpr int effective_storage_size = page_size - meta_size;
    static log_page_type *new_log_page()
    {
        void *ptr = nullptr;
        auto res = nvm_dram_alloc(&ptr, cacheline_size, PageSize);
        if (res == 0)
        {
            return new (ptr) log_page_type();
        }
        return nullptr;
    }
    static void delete_log_page(log_page_type *p)
    {
        nvm_dram_free(p, PageSize);
    }
};

template <int PageSize = 64 * 1024>
class caching_log_page_allocator
{
  public:
    static constexpr int fill_batch_size = 16;
    caching_log_page_allocator() : freelist(nullptr) {}

    log_page<PageSize> *allocate_page()
    {
        log_page<PageSize> *ret;
        {
            std::lock_guard<std::mutex> g(mtx);
            if (freelist == nullptr)
            {
                for (int i = 0; i < fill_batch_size; ++i)
                {
                    auto page = log_page<PageSize>::new_log_page();
                    page->next = freelist;
                    freelist = page;
                    clflush_then_sfence(&freelist);
                }
            }
            ret = freelist;
            freelist = freelist->next;
        }
        clflush_then_sfence(&freelist);
        return ret;
    }

    void deallocate_page(log_page<PageSize> *p)
    {
        {
            std::lock_guard<std::mutex> g(mtx);
            p->next = freelist;
            freelist = p;
        }
        clflush_then_sfence(&freelist);
    }

    void deallocate_pages(log_page<PageSize> *startp, log_page<PageSize> *endp)
    {
        {
            std::lock_guard<std::mutex> g(mtx);
            endp->next = freelist;
            freelist = startp;
        }
        clflush_then_sfence(&freelist);
    }

    log_page<PageSize> *freelist;
    std::mutex mtx;
};

template <class Key, class Value, int PageSize = 64 * 1024>
class lockfree_pwal
{
  public:
    using key_type = Key;
    using value_type = Value;
    using log_record_type = log_record<key_type, value_type>;
    using log_page_allocator_type = caching_log_page_allocator<PageSize>;
    using log_page_type = log_page<PageSize>;
    log_page_type *tailp;
    log_page_allocator_type *allocator;
    std::atomic_int &record_count_approx;
    std::mutex wal_mtx;
    lockfree_pwal(caching_log_page_allocator<PageSize> *allocator, std::atomic_int &record_count_approx) : tailp(nullptr), allocator(allocator), record_count_approx(record_count_approx) {
        if (tailp == nullptr)
            add_new_tail_page(tailp);
    }
    ~lockfree_pwal()
    {
        auto startp = tailp;
        auto endp = startp;
        while (endp && endp->next)
        {
            endp = endp->next;
        }
        if (startp)
        {
            allocator->deallocate_pages(startp, endp);
        }
    }
    size_t record_count_in_current_page()
    {
        auto tp = tailp;
        if (tp)
        {
            return tp->off / sizeof(log_record_type);
        }
        return 0;
    }
    void add_new_tail_page(log_page_type *tp) {
        auto np = allocator->allocate_page();
        np->off = 0;
        np->next = tp;
        clflush_then_sfence(&np->next);
        if (__sync_bool_compare_and_swap(&tailp, tp, np) == false)
        {
            allocator->deallocate_page(np);
        }
        else
        {
            clflush_then_sfence(&tailp);
            if (tp)
                record_count_approx += log_page_type::effective_storage_size / sizeof(log_record_type);
        }
    }

    void append_and_flush(log_record_type *r)
    {
    restart:
        auto tp = tailp;
        uint64_t off = __sync_fetch_and_add(&tp->off, sizeof(log_record_type));
        if (off + sizeof(log_record_type) <= log_page_type::effective_storage_size)
        {
//            *reinterpret_cast<log_record_type *>(tp->storage + off) = *r;
//            clflush_sfence(tp->storage + off);
            memcpy(tp->storage + off, r, sizeof(*r));
            clflush_then_sfence(tp->storage + off);
        }
        else
        {
            //std::lock_guard<std::mutex> g(wal_mtx);
            add_new_tail_page(tp);
            goto restart;
        }
    }
    template<class F>
    void for_each_record(F f) {
        std::vector<log_page_type*> pages;
        auto p = tailp;
        while (p) {
            pages.push_back(p);
            p = p->next;
        }
        for (int i = (int)pages.size() - 1; i >= 0; --i) {
            log_page_type * p = pages[i];
            size_t off = 0;
            while (off + sizeof(log_record_type) <= log_page_type::effective_storage_size) {
                log_record_type  * rp = reinterpret_cast<log_record_type *>(p->storage + off);
                f(*rp);
                off += sizeof(log_record_type);
            }
        }
    }
};

constexpr int stripes = 60;
constexpr double bloom_err_rate = 0.05;
constexpr double merge_threshold = 0.07;
static thread_local void *log_addr = nullptr;
static thread_local void *cacheline_addr = nullptr;

template <class Key, class Value>
class durable_concur_buffer_btree
{
  public:
    using key_type = Key;
    using value_type = Value;
    using concur_bloom_type = bloom1<key_type>;
    using lock_guard_type = std::lock_guard<std::mutex>;
    using log_type = lockfree_pwal<key_type, value_type>;
    static thread_local log_record<key_type, value_type> *local_record;
    durable_concur_buffer_btree(size_t expected_size, caching_log_page_allocator<> *allocator) : approx_size(0)
    {
        btree = new btreeolc::BTree<key_type, value_type>();
        cap = expected_size = std::max(expected_size, (size_t)1000);
        bloom = new concur_bloom_type(cap, bloom_err_rate);
        logs.resize(stripes);
        for (int i = 0; i < stripes; ++i)
        {
            logs[i] = new log_type(allocator, approx_size);
        }
    }

    ~durable_concur_buffer_btree()
    {
        if (bloom) {
            delete bloom;
            bloom = nullptr;
        }

        if (btree) {
            delete btree;
            btree = nullptr;
        }
        for (int i = 0; i < stripes; ++i)
        {
            delete logs[i];
        }
    }

    inline bool bloom_check(const key_type & key) {
        return bloom->check(key);
    }

    bool lookup(const key_type &key, value_type &value)
    {
        if (bloom_check(key))
        {
            return btree->lookup(key, value);
        }
        return false;
    }

    void insert(const key_type &key, const value_type &value)
    {
        if (local_record == nullptr)
        {
            nvm_dram_alloc((void**)&local_record, cacheline_size, cacheline_size);
        }
        auto btree_leaf_insert_func = [&key, &value, this]() {
            *local_record = log_record<key_type, value_type>(op_type::upsertion, key, value);
            int id = std::hash<key_type>()(key) % stripes;
            logs[id]->append_and_flush(local_record);
            bloom->insert(key);
        };
        btree->insert(key, value, btree_leaf_insert_func);
    }

    template<typename F>
    void update(const key_type &key, const value_type &value, F should_insert_func)
    {
        if (local_record == nullptr)
        {
            nvm_dram_alloc((void**)&local_record, cacheline_size, cacheline_size);
        }
        auto btree_leaf_insert_func = [&key, &value, this]() {
            *local_record = log_record<key_type, value_type>(op_type::upsertion, key, value);
            int id = std::hash<key_type>()(key) % stripes;
            logs[id]->append_and_flush(local_record);
            bloom->insert(key);
        };
        btree->upsert(key, value, should_insert_func, btree_leaf_insert_func);
    }

    typename btreeolc::BTree<key_type, value_type>::unsafe_iterator
    begin_unsafe()
    {
        return btree->begin_unsafe();
    }

    typename btreeolc::BTree<key_type, value_type>::unsafe_iterator end_unsafe()
    {
        return btree->end_unsafe();
    }

    size_t size() { return approx_size.load(std::memory_order_relaxed); }
    size_t capacity() { return cap; }
    size_t real_size()
    {
        size_t c = approx_size;
        for (int i = 0; i < stripes; ++i)
        {
            c += logs[i]->record_count_in_current_page();
        }
        return c;
    }

    typename btreeolc::BTree<key_type, value_type>::range_iterator lookup_range(const key_type & start_key) {
        return btree->lookup_range(start_key);
    }

  private:
    // volatile states
    btreeolc::BTree<key_type, value_type> * btree;
    concur_bloom_type * bloom;
    std::atomic_int approx_size;
    // persistent states
    std::vector<log_type *> logs;
    size_t cap;
};

template <class Key, class Value>
class durable_concur_buffer_art
{
public:
    using key_type = Key;
    using value_type = Value;
    using concur_bloom_type = bloom1<key_type>;
    using lock_guard_type = std::lock_guard<std::mutex>;
    using log_type = lockfree_pwal<key_type, value_type>;
    static thread_local log_record<key_type, value_type> *local_record;
    durable_concur_buffer_art(size_t expected_size, caching_log_page_allocator<> *allocator) : approx_size(0), index(8)
    {
        cap = expected_size = std::max(expected_size, (size_t)1000);
        bloom = new concur_bloom_type(expected_size, bloom_err_rate);
        logs.reserve(stripes);
        for (int i = 0; i < stripes; ++i)
        {
            logs[i] = new log_type(allocator, approx_size);
        }
    }

    void clear_log_pages() {
        for (int i = 0; i < stripes; ++i)
        {
            delete logs[i];
        }
    }
    ~durable_concur_buffer_art()
    {
        delete bloom;
        clear_log_pages();
    }

    bool lookup(const key_type &key, value_type &value)
    {
        if (bloom->check(key))
        {
            return index.find(key, value);
        }
        return false;
    }

    void insert(const key_type &key, const value_type &value)
    {
        if (local_record == nullptr)
        {
            nvm_dram_alloc((void**)&local_record, cacheline_size, cacheline_size);
        }
        auto leaf_insert_func = [&key, &value, this]() {
            *local_record = log_record<key_type, value_type>(op_type::upsertion, key, value);
            int id = std::hash<key_type>()(key) % stripes;
            logs[id]->append_and_flush(local_record);
            bloom->insert(key);
        };
        index.insert(key, value, leaf_insert_func);
    }

    template<typename F>
    void update(const key_type &key, const value_type &value, F should_insert_func)
    {
        if (local_record == nullptr)
        {
            nvm_dram_alloc((void**)&local_record, cacheline_size, cacheline_size);
        }
        auto leaf_insert_func = [&key, &value, this]() {
            *local_record = log_record<key_type, value_type>(op_type::upsertion, key, value);
            int id = std::hash<key_type>()(key) % stripes;
            logs[id]->append_and_flush(local_record);
            bloom->insert(key);
            //++approx_size;
        };
        index.insert(key, value, leaf_insert_func);
    }

    typename ArtOLCIndex<key_type>::iterator
    begin_unsafe()
    {
        return index.begin();
    }

    typename ArtOLCIndex<key_type>::iterator end_unsafe()
    {
        return index.end();
    }

    size_t size() { return approx_size.load(std::memory_order_relaxed); }
    size_t capacity() { return cap; }
    size_t real_size()
    {
        size_t c = approx_size;
        for (int i = 0; i < stripes; ++i)
        {
            c += logs[i]->record_count_in_current_page();
        }
        return c;
    }

    typename ArtOLCIndex<key_type>::iterator lookup_range(const key_type & start_key) {
        return index.lookup_range(start_key);
    }

private:
    // volatile states
    ArtOLCIndex<key_type> index;
    concur_bloom_type *bloom;
    std::atomic_int approx_size;
    size_t cap;
    std::vector<log_type *> logs;
    // persistent states
};

template <class Key, class Value>
thread_local log_record<Key, Value> *durable_concur_buffer_btree<Key, Value>::local_record = nullptr;

template <class Key, class Value>
thread_local log_record<Key, Value> *durable_concur_buffer_art<Key, Value>::local_record = nullptr;

template <class Key, class Value>
class concur_dptree
{
  public:
    using key_type = Key;
    using value_type = Value;
    using buffer_btree_type = durable_concur_buffer_btree<key_type, value_type>;
    using base_tree_type = concur_cvhtree<key_type, value_type>;
    caching_log_page_allocator<> allocator;
    concur_dptree()
        : base_tree_inner_rebuild_time(0), base_tree_parallel_merge_work_time(0),
          merge_state(0), merge_time(0), merge_wait_time(0)
    {
        front_buffer_tree = new buffer_btree_type(1024, &allocator);
        middle_buffer_tree = nullptr;
        rear_base_tree = new base_tree_type(base_tree_inner_rebuild_time,
                                            base_tree_parallel_merge_work_time);
    }
    size_t size() { return front_buffer_tree->real_size() + rear_base_tree->size(); }
    size_t get_buffer_tree_size() { return front_buffer_tree->real_size(); }
    size_t get_probes() { return rear_base_tree->get_probes(); }
    double get_flushtime() { return rear_base_tree->total_flush_time; }
    inline uint64_t make_upsert_value(uint64_t v) { return (v << 1); }
    void insert(const key_type &key, const value_type &value, bool check_merge_threshold = true)
    {
        // register writer
        front_writer_local_epoch.register_to_manager(&front_writer_epoch_manager);
        buffer_btree_type *buffer_tree;
        {
            epoch_guard g(front_writer_local_epoch);
            buffer_tree = reinterpret_cast<buffer_btree_type *>(
                front_writer_local_epoch.set_value((uintptr_t)front_buffer_tree));
            assert(buffer_tree);
            // do stuff
            buffer_tree->insert(key, make_upsert_value(value));
        }
        // printf("key %d value %d\n", key, value);

        if (check_merge_threshold && should_merge())
        {
            start_merge(buffer_tree);
        }
    }
    enum {
        NO_MERGE,
        PRE_MERGE,
        MERGING,
        MERGING_LEAF_DONE,
    };


    bool is_pre_merge(int ms) {
        int state = ms % 4;
        return state == PRE_MERGE;
    }

    bool is_merging () {
        return is_merging(merge_state.load());
    }

    bool is_merging (int ms) {
        int state = ms % 4;
        return state == MERGING;
    }

    bool is_merging_leaf_done (int ms) {
        int state = ms % 4;
        return state == MERGING_LEAF_DONE;
    }

    bool is_no_merge () {
        return is_no_merge(merge_state.load());
    }

    bool is_no_merge(int ms) {
        int state = ms % 4;
        return state == NO_MERGE;
    }

    int become_pre_merge(int ms) {
        assert(ms % 4 == 0);
        return ms + 1;
    }

    int become_merging(int ms) {
        assert(ms % 4 == 0);
        return ms + 2;
    }

    int become_merging_leaf_done(int ms) {
        assert(ms % 4 == 0);
        return ms + 3;
    }

    int become_next_no_merge(int ms) {
        assert(ms % 4 == 0);
        return ms + 4;
    }


    void upsert(const key_type & key, const value_type & value) {
        front_writer_local_epoch.register_to_manager(&front_writer_epoch_manager);
        buffer_btree_type *buffer_tree;
        {
            epoch_guard g(front_writer_local_epoch);
            buffer_tree = reinterpret_cast<buffer_btree_type *>(
                    front_writer_local_epoch.set_value((uintptr_t)front_buffer_tree));
            assert(buffer_tree);
            if (buffer_tree->bloom_check(key) == false) {
                bool updated = rear_base_tree->update_if_found(key, value);
                if (updated == true) {
                    return;
                }
            }

            auto should_insert_f_cpbt = [&key, &value, this](const key_type &key) -> bool {
                return true;
            };
            buffer_tree->update(key, make_upsert_value(value), should_insert_f_cpbt);
        }

        if (should_merge())
        {
            auto start = secs_now();
            do_merge(buffer_tree);
            merge_wait_time = merge_wait_time.load() + secs_now() - start;
        }
    }

    void lookup_range(const key_type &start_key, int scan_size, std::vector<value_type> &values)
    {
        // register reader
        front_reader_local_epoch.register_to_manager(&front_reader_epoch_manager);
        middle_reader_local_epoch.register_to_manager(&middle_reader_epoch_manager);
        rear_base_tree->reader_epoch.register_to_manager(&rear_base_tree->reader_manager);
        values.reserve(scan_size);
        epoch_guard front_g(front_reader_local_epoch);
        epoch_guard middle_g(middle_reader_local_epoch);
        epoch_guard rear_g(rear_base_tree->reader_epoch);

        int scan_cnt = 0;

        auto first_buffer_tree =
            (buffer_btree_type *)front_reader_local_epoch.set_value((uintptr_t)front_buffer_tree);
        auto second_buffer_tree = (buffer_btree_type *)
                                      middle_reader_local_epoch.set_value((uintptr_t)middle_buffer_tree);
        //auto second_buffer_tree = middle_buffer_tree;
        if (second_buffer_tree && first_buffer_tree != second_buffer_tree) {
            auto front_it = first_buffer_tree->lookup_range(start_key);
            auto mid_it = second_buffer_tree->lookup_range(start_key);
            auto rear_it = rear_base_tree->lower_bound(start_key, rear_base_tree->reader_epoch);
            auto rear_end = rear_base_tree->end();

            while (scan_cnt < scan_size && front_it.is_end() == false && mid_it.is_end() == false && rear_it != rear_end) {
                if (front_it.key() < mid_it.key()) {
                    if (front_it.key() <= rear_it.key()) {
                        values.emplace_back(front_it.value());
                        ++front_it;
                        if (front_it.key() == rear_it.key())
                            ++rear_it;
                    } else {
                        values.emplace_back(rear_it.value());
                        ++rear_it;
                    }
                } else if (front_it.key() == mid_it.key()) {
                    if (front_it.key() <= rear_it.key()) {
                        values.emplace_back(front_it.value());
                        ++front_it;
                        ++mid_it;
                        if (front_it.key() == rear_it.key())
                            ++rear_it;
                    } else {
                        values.emplace_back(rear_it.value());
                        ++rear_it;
                    }
                } else {
                    if (mid_it.key() <= rear_it.key()) {
                        values.emplace_back(mid_it.value());
                        ++mid_it;
                        if (mid_it.key() == rear_it.key())
                            ++rear_it;
                    } else {
                        values.emplace_back(rear_it.value());
                        ++rear_it;
                    }
                }
                ++scan_cnt;
            }
            while (scan_cnt < scan_size && front_it.is_end() == false && mid_it.is_end() == false) {
                if (front_it.key() <= mid_it.key()) {
                    values.emplace_back(front_it.value());
                    ++front_it;
                    if (front_it.key() == mid_it.key())
                        ++mid_it;
                } else {
                    values.emplace_back(mid_it.value());
                    ++mid_it;
                }
                ++scan_cnt;
            }
            while (scan_cnt < scan_size && front_it.is_end() == false && rear_it != rear_end) {
                if (front_it.key() <= rear_it.key()) {
                    values.emplace_back(front_it.value());
                    ++front_it;
                    if (front_it.key() == rear_it.key())
                        ++rear_it;
                } else {
                    values.emplace_back(rear_it.value());
                    ++rear_it;
                }
                ++scan_cnt;
            }
            while (scan_cnt < scan_size && mid_it.is_end() == false && rear_it != rear_end) {
                if (mid_it.key() <= rear_it.key()) {
                    values.emplace_back(mid_it.value());
                    ++mid_it;
                    if (mid_it.key() == rear_it.key())
                        ++rear_it;
                } else {
                    values.emplace_back(rear_it.value());
                    ++rear_it;
                }
                ++scan_cnt;
            }
            while (scan_cnt < scan_size && front_it.is_end() == false) {
                values.emplace_back(front_it.value());
                ++front_it;
                ++scan_cnt;
            }
            while (scan_cnt < scan_size && mid_it.is_end() == false) {
                values.emplace_back(mid_it.value());
                ++mid_it;
                ++scan_cnt;
            }
            while (scan_cnt < scan_size && rear_it != rear_end) {
                values.emplace_back(rear_it.value());
                ++rear_it;
                ++scan_cnt;
            }
        } else {
            middle_reader_local_epoch.leave();
            auto first_buffer_tree =
                    (buffer_btree_type *)front_reader_local_epoch.set_value((uintptr_t)front_buffer_tree);
            auto front_it = first_buffer_tree->lookup_range(start_key);
            auto rear_it = rear_base_tree->lower_bound(start_key, rear_base_tree->reader_epoch);
            auto rear_end = rear_base_tree->end();

            while (scan_cnt < scan_size && front_it.is_end() == false && rear_it != rear_end) {
                if (front_it.key() <= rear_it.key()) {
                    values.emplace_back(front_it.value());
                    ++front_it;
                    if (front_it.key() == rear_it.key())
                        ++rear_it;
                } else {
                    values.emplace_back(rear_it.value());
                    ++rear_it;
                }
                ++scan_cnt;
            }
            while (scan_cnt < scan_size && front_it.is_end() == false) {
                values.emplace_back(front_it.value());
                ++front_it;
                ++scan_cnt;
            }
            while (scan_cnt < scan_size && rear_it != rear_end) {
                values.emplace_back(rear_it.value());
                ++rear_it;
                ++scan_cnt;
            }
        }
    }

    bool lookup(const key_type &key, value_type &value)
    {
        // register reader
        front_reader_local_epoch.register_to_manager(&front_reader_epoch_manager);
        middle_reader_local_epoch.register_to_manager(&middle_reader_epoch_manager);

        buffer_btree_type *first_buffer_tree;
        {
            epoch_guard g(front_reader_local_epoch);
            first_buffer_tree =
                    (buffer_btree_type *) front_reader_local_epoch.set_value(
                            (uintptr_t) __atomic_load_n(&front_buffer_tree, __ATOMIC_SEQ_CST));
            //first_buffer_tree = front_buffer_tree;
            if (first_buffer_tree->lookup(key, value))
            {
                return true;
            }
        }
        if (middle_buffer_tree != nullptr)
        {
            epoch_guard g(middle_reader_local_epoch);
            auto second_buffer_tree = reinterpret_cast<buffer_btree_type *>(
                middle_reader_local_epoch.set_value((uintptr_t)middle_buffer_tree));
            //auto second_buffer_tree = middle_buffer_tree;
            if (second_buffer_tree && first_buffer_tree != second_buffer_tree)
            {
                if (second_buffer_tree->lookup(key, value))
                {
                    return true;
                }
            }
        }
        return rear_base_tree->lookup(key, value);
    }

    inline bool should_merge()
    {
        return front_buffer_tree->size() >= front_buffer_tree->capacity();
    }

    void start_merge(buffer_btree_type * buffer_tree) {
        auto start = secs_now();
        do_merge(buffer_tree);
        merge_wait_time = merge_wait_time.load() + secs_now() - start;
    }

    void force_merge() {
        start_merge(front_buffer_tree);
    }

    double get_real_merge_time() { return merge_time; }
    double get_merge_wait_time() { return merge_wait_time; }

    double get_inner_node_build_time() { return base_tree_inner_rebuild_time; }
    double get_merge_work_time() { return base_tree_parallel_merge_work_time; }

    std::atomic<bool> leaf_merge_done{false};

    void do_merge(buffer_btree_type *front_buffer_tree_snapshot)
    {
        int local_ms = merge_state.load();

        auto wait_merging = [this, &local_ms, &front_buffer_tree_snapshot] {
            while ((is_pre_merge(local_ms) || is_merging(local_ms)))
            {
                std::unique_lock<std::mutex> lk(cvm);
                cv.wait_for(lk, std::chrono::milliseconds(10));
                local_ms = merge_state.load();
            }
        };
        wait_merging();

        if (is_no_merge(local_ms)) {
            // CAS to be the merge leader
            if (merge_state.compare_exchange_strong(local_ms, become_pre_merge(local_ms)))
            {
                // becomes pre_merge state
                // one thread becomes the merge leader
                if (front_buffer_tree_snapshot != front_buffer_tree)
                {
                    merge_state.store(become_next_no_merge(local_ms));
                    return;
                }
                // creates a new front buffer tree
                size_t expected_new_base_tree_size =
                        front_buffer_tree->size() + rear_base_tree->size();
                size_t expected_new_buffer_tree_size =
                        merge_threshold * expected_new_base_tree_size;

                auto old_front_buffer_tree = front_buffer_tree;
                auto new_front_buffer_tree =
                        new buffer_btree_type(expected_new_buffer_tree_size, &allocator);
                // make the full front tree as the middle tree for read-only workloads
                assert(middle_buffer_tree == nullptr);
                __atomic_store_n(&middle_buffer_tree, old_front_buffer_tree, __ATOMIC_SEQ_CST);
                // new writers now go to new_front_buffer_tree
                __atomic_store_n(&front_buffer_tree, new_front_buffer_tree, __ATOMIC_SEQ_CST);
                // start a merge worker thread
                std::thread([this, local_ms, old_front_buffer_tree]() { // mfence
                    // wait for old writers to finish
                    // printf("waiting for old writers to finish\n");
                    spin_until_no_ref(front_writer_epoch_manager, (uintptr_t)old_front_buffer_tree);
                    // printf("done waiting for old writers to finish\n");

                    merge_state.store(become_merging(local_ms));
                    // Start merge into the base tree.
                    // We are safe using the unsafe iterator since there are no writers.
                    auto sit = middle_buffer_tree->begin_unsafe();
                    auto send = middle_buffer_tree->end_unsafe();
                    auto start = secs_now();
                    size_t merge_in_kv_count = middle_buffer_tree->real_size();
                    auto when_leaf_merge_done = [this, local_ms]() {
                        merge_state.store(become_merging_leaf_done(local_ms));
                        cv.notify_all();
                    };
                    rear_base_tree->parallel_merge(sit, send, merge_in_kv_count, when_leaf_merge_done, parallel_merge_worker_num);
                    //printf("merge, buffer tree size %d capacity %d, base tree size %d\n", (int)merge_in_kv_count, (int)middle_buffer_tree->capacity(), (int)rear_base_tree->size());
                    merge_time += secs_now() - start;
                    // After the merge is complete, divert new readers to the base_tree
                    auto old_middle_buffer_tree = middle_buffer_tree;
                    __atomic_store_n(&middle_buffer_tree, nullptr, __ATOMIC_RELAXED);
                    //middle_buffer_tree = nullptr;
                    // and wait for old readers to finish reading the middle_buffer_tree.
                    if (old_middle_buffer_tree)
                    {
                        // printf("waiting for old middle readers to finish\n");
                        spin_until_no_ref(middle_reader_epoch_manager, (uintptr_t)old_middle_buffer_tree);
                        // printf("done waiting for old middle readers to finish\n");

                        // printf("waiting for old front readers to finish\n");
                        spin_until_no_ref(front_reader_epoch_manager, (uintptr_t)old_middle_buffer_tree);
                        // printf("done waiting for old front readers to finish\n");
                    }

                    // mark the merge as complete

                    merge_state.store(become_next_no_merge(local_ms));
                    cv.notify_all();
                    delete old_middle_buffer_tree;
                }).detach();
                local_ms = merge_state.load();
            }
        }
        wait_merging();
    }

  private:
    buffer_btree_type *front_buffer_tree;
    buffer_btree_type *middle_buffer_tree;
    base_tree_type *rear_base_tree;
    std::atomic<int> merge_state;
    double base_tree_inner_rebuild_time;
    double base_tree_parallel_merge_work_time;
    double merge_time;
    std::atomic<double> merge_wait_time;
    double flush_time;
    epoch_manager front_writer_epoch_manager;
    epoch_manager front_reader_epoch_manager;
    epoch_manager middle_reader_epoch_manager;
    std::mutex cvm;
    std::condition_variable cv;
    static thread_local thread_epoch front_writer_local_epoch;
    static thread_local thread_epoch front_reader_local_epoch;
    static thread_local thread_epoch middle_reader_local_epoch;
};


template <class Key, class Value>
thread_local thread_epoch concur_dptree<Key, Value>::front_writer_local_epoch;

template <class Key, class Value>
thread_local thread_epoch concur_dptree<Key, Value>::front_reader_local_epoch;

template <class Key, class Value>
thread_local thread_epoch concur_dptree<Key, Value>::middle_reader_local_epoch;

template <class Key, class Value>
thread_local thread_epoch concur_cvhtree<Key, Value>::reader_epoch;

} // namespace dptree