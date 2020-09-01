#pragma once
#include <sys/time.h>
#include <functional>
#include <string>
#include <cstdio>
#include <vector>
#include <thread>
#include <string.h>
#include <immintrin.h>

#ifdef USE_PAPI
#include <papi.h>
#endif
#ifdef NVM_DRAM_MODE
#include "pmalloc.h"
#endif

#define END_PADDING_SIZE(x) ((cacheline_size - ((x) % cacheline_size)) % cacheline_size)
#define ALIGN(addr, alignment) ((char *)((unsigned long)(addr) & ~((alignment) - 1)))
#define CACHELINE_ALIGN(addr) ALIGN(addr, 64)
extern std::vector<uint64_t> keys;
extern std::vector<uint64_t> insert_more_keys;
extern std::vector<uint64_t> lookupKeys;
extern std::vector<uint64_t> sortedKeys;
extern std::vector<uint64_t> notExistKeys;
extern unsigned long write_latency_in_ns;
extern unsigned long CPU_FREQ_MHZ;
extern unsigned long long cycles_total;
extern int parallel_merge_worker_num;
static constexpr int cacheline_size = 64;
unsigned long long cycles_now();
unsigned ctz(unsigned);
void clear_cache();
double secs_now(void);
void print_flush_stat();
void percentile_report(const std::string & name, std::vector<int> & nums);
double measure(std::function<unsigned()> f, const std::string & bench_name, unsigned iteration = 1, bool print_mem = true);
void prepareKeys(int n, const std::string & keyFile, bool sparseKey, bool sorted = true);
void prepareInsertMoreKeys(int n, const std::string &keyFile, bool sparseKey, bool sorted);
void cpu_pause();
void mfence();
void sfence();
void clflush(volatile void *p);
void clflush_then_sfence(volatile void *p);
void clflush_len(volatile void *data, int len);
void clflush_len_no_fence(volatile void *data, int len);
void prefetch(char *ptr, size_t len);
int nvm_dram_alloc(void **ptr, size_t align, size_t size);
void nvm_dram_free(void * ptr, size_t size);

int wbinvd();
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE  64 // 64 byte cache line on x86 and x86-64
#endif

#define INT64S_PER_CACHELINE (CACHE_LINE_SIZE / sizeof(int64_t))
#define INT64S_PER_CACHELINE_SCALE 4

static thread_local bool cpuIdInitialized;
static thread_local uint64_t cpuId;
template<int buckets = 2>
class DistributedCounter2 {
public:
    static_assert(buckets == 0 || (buckets & (buckets - 1)) == 0, "buckets must be a multiple of 2");

    DistributedCounter2(int initVal = 0) {
        countArrayPtr = malloc(buckets * INT64S_PER_CACHELINE * sizeof(int64_t) + CACHE_LINE_SIZE - 1);
        memset(countArrayPtr, 0, buckets * INT64S_PER_CACHELINE * sizeof(int64_t) + CACHE_LINE_SIZE - 1);
        countArray = (int64_t *)(((size_t)countArrayPtr + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1));
        increment(initVal);
    }

    ~DistributedCounter2() {
        free(countArrayPtr);
    }
    inline void increment(int v = 1) {
        __atomic_add_fetch(&countArray[arrayIndex() * INT64S_PER_CACHELINE], v, __ATOMIC_RELAXED);
    }

    inline void decrement(int v = 1) {
        __atomic_sub_fetch(&countArray[arrayIndex() * INT64S_PER_CACHELINE], v, __ATOMIC_RELAXED);
    }

    int64_t get() {
        int64_t val = 0;
        for (int i = 0; i < totalINT64S; i += INT64S_PER_CACHELINE) {
            val += __atomic_load_n(&countArray[i], __ATOMIC_RELAXED);
        }
        return val;
    }

    int64_t get_unsafe() {
        int64_t val = 0;
        for (int i = 0; i < totalINT64S; i += INT64S_PER_CACHELINE) {
            val += countArray[i];
        }
        return val;
    }
private:
    static constexpr int totalINT64S = buckets * INT64S_PER_CACHELINE;
    inline uint64_t getCPUId() {
        if (cpuIdInitialized == false) {
            cpuId = (uint64_t)std::hash<std::thread::id>{}(std::this_thread::get_id());
            cpuIdInitialized = true;
            //printf("cpuid %lu, arrayIndex %d, pointer %p\n", cpuId, arrayIndex(), &countArray[arrayIndex() * INT64S_PER_CACHELINE]);
        }
        return cpuId;
    }

    inline int arrayIndex() {
        return getCPUId() & (buckets  - 1);
    }

    int64_t * countArray;
    void * countArrayPtr;
};


class elided_spin_lock {
private:
    int lockvar;
    char padding[END_PADDING_SIZE(sizeof(lockvar))];
public:
    elided_spin_lock(): lockvar(0) {}
    
    void lock() {
        /* Acquire lock with lock elision */
        while (__atomic_exchange_n(&lockvar, 1, __ATOMIC_ACQUIRE|__ATOMIC_HLE_ACQUIRE))
            _mm_pause(); /* Abort failed transaction */
    }
    
    void unlock() {
        __atomic_store_n(&lockvar, 0, __ATOMIC_RELEASE|__ATOMIC_HLE_RELEASE);
    }
};