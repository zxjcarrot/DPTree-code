
#include "util.h"
#include <unordered_set>
#include <random>
#include <algorithm>
#include <atomic>
#include <cassert>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
using namespace std;

unsigned long write_latency_in_ns;
unsigned long CPU_FREQ_MHZ = 2200;
unsigned long long cycles_total = 0;
atomic<long long> clflush_count(0);
atomic<long long> sfence_count(0);
vector<uint64_t> keys;
vector<uint64_t> lookupKeys;
vector<uint64_t> sortedKeys;
vector<uint64_t> notExistKeys;

void prepareKeys(int n, const string &keyFile, bool sparseKey, bool sorted)
{
    uint64_t end;
    unordered_set<uint64_t> keySet;
    size_t totalKeyCount = n;
    keys.reserve(totalKeyCount);
    lookupKeys.reserve(totalKeyCount);
    std::random_device rd;

    std::mt19937_64 e2(0);

    std::uniform_int_distribution<unsigned long long> dist(0, std::llround(std::pow(2, 63)));

    if (sparseKey == false)
    {
        end = n;
        for (uint64_t i = 0; i < end; ++i)
        {
            keys.push_back(i);
        }
    }
    else
    {
        keySet.reserve(totalKeyCount);
        end = std::numeric_limits<uint64_t>::max();
        while (keySet.size() != totalKeyCount)
        {
            uint64_t key = dist(e2) % end;
            keySet.insert(key);
        }
        auto tset = keySet;
        for (int i = 0; i < n && keySet.empty() == false; ++i)
        {
            uint64_t key = *keySet.begin();
            keySet.erase(key);
            keys.push_back(key);
        }
        for (int i = 0; i < totalKeyCount; ++i) {
            uint64_t key = dist(e2) % end;
            if (tset.find(key) != tset.end()) {
                --i;
                continue;
            }
            notExistKeys.push_back(key);
        }
    }

    if (sorted)
    {
        sort(keys.begin(), keys.end());
    }
    lookupKeys = keys;
    std::random_shuffle(lookupKeys.begin(), lookupKeys.end());
    sortedKeys = keys;
    std::sort(sortedKeys.begin(), sortedKeys.end());
    //print_memory_usage();
    printf("Keys: %d, Lookup Keys: %d, NotExist Keys: %d\n", (int)keys.size(), (int)lookupKeys.size(), (int) notExistKeys.size());
}

inline unsigned ctz(unsigned x) {
    // Count trailing zeros, only defined for x>0
#ifdef __GNUC__
    return __builtin_ctz(x);
#else
    // Adapted from Hacker's Delight
        unsigned n=1;
        if ((x&0xFF)==0) {n+=8; x=x>>8;}
        if ((x&0x0F)==0) {n+=4; x=x>>4;}
        if ((x&0x03)==0) {n+=2; x=x>>2;}
        return n-(x&1);
#endif
}

unsigned long long cycles_now() {
    _mm_lfence();
    unsigned long long v = __rdtsc();
    _mm_lfence();
    return v;
}

double secs_now(void)
{
    struct timeval now_tv;
    gettimeofday(&now_tv, NULL);
    return ((double)now_tv.tv_sec) + ((double)now_tv.tv_usec) / 1000000.0;
}

void clear_cache()
{
    // Remove cache
    int size = 256 * 1024 * 1024;
    char *garbage = new char[size];
    for (int i = 0; i < size; ++i)
        garbage[i] = i;
    for (int i = 100; i < size; ++i)
        garbage[i] += garbage[i - 100];
    delete[] garbage;
}

void print_flush_stat() {
    printf("clfluh count: %lld\n", clflush_count.load());
}
#include <gperftools/malloc_extension.h>
double measure(std::function<unsigned()> f, const string &bench_name, unsigned iteration, bool print_mem)
{
#ifdef USE_PAPI
    const int nevents = 4;
    int papi_events[nevents] = {PAPI_L3_TCM, PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_L3_TCA /*, PAPI_TLB_DM*/}, ret;

    long long values[nevents];
    if ((ret = PAPI_start_counters(papi_events, nevents)) != PAPI_OK)
    {
        fprintf(stderr, "PAPI failed to start counters: %s\n", PAPI_strerror(ret));
        exit(1);
    }
#endif



    clflush_count = 0;
    sfence_count = 0;
    char tcmalloc_stats_buf[1024];
    print_mem = 0;
    if (print_mem) {
        MallocExtension::instance()->GetStats(tcmalloc_stats_buf,
                                              sizeof(tcmalloc_stats_buf));
    }
    if (print_mem)
        printf("%s\n", tcmalloc_stats_buf);

    double start = secs_now();
    unsigned c = 0;
    for (int i = 0; i < iteration; ++i) {
        c += f() + 1;
    }
    double end = secs_now();
    if (print_mem) {
        MallocExtension::instance()->GetStats(tcmalloc_stats_buf,
                                              sizeof(tcmalloc_stats_buf));
    }
    if (print_mem)
        printf("%s\n", tcmalloc_stats_buf);
    double avg = (end - start) * 1000000000L / c;
    printf("Benchmark %s, Elapsed %f secs, %f ns/op, %f mops, %u ops in total, %lld flushes, %f flushes/op, %f sfences/op\n",
           bench_name.c_str(), (end - start), avg,
           c / 1000000.0 / (end - start), c, clflush_count.load(), clflush_count.load() / (c + 0.0), sfence_count.load() / (c + 0.0));
#ifdef USE_PAPI
    if ((ret = PAPI_stop_counters(values, nevents)) != PAPI_OK)
    {
        fprintf(stderr, "PAPI failed to stop counters: %s\n", PAPI_strerror(ret));
        exit(1);
    }

    printf("Level 3 cache misses: %lld\n", values[0]);
    printf("Total cycles: %lld\n", values[1]);
    printf("Instructions completed: %lld\n", values[2]);
    printf("Level 3 accesses: %lld\n", values[3]);
//printf("Data translation lookaside buffer misses: %lld\n", values[4]);
//printf("Cycles stalled on any resource: %lld\n", values[3]);
#endif
    printf("\n");
    return avg;
}

void cpu_pause()
{
    _mm_pause();
}

inline unsigned long read_tsc(void)
{
    unsigned long var;
    unsigned int hi, lo;

    asm volatile("rdtsc"
                 : "=a"(lo), "=d"(hi));
    var = ((unsigned long long int)hi << 32) | lo;

    return var;
}
void mfence()
{
    asm volatile("mfence" ::
                     : "memory");
}

void sfence() {
    asm volatile("sfence" ::
    : "memory");
}

void nontemporal_store_256(void *mem_addr, void *c)
{
    __m256i x = _mm256_load_si256((__m256i const *)c);
    unsigned long etsc = read_tsc() +
                         (unsigned long)(write_latency_in_ns * CPU_FREQ_MHZ / 1000);
    _mm256_stream_si256((__m256i *)mem_addr, x);
    while (read_tsc() < etsc)
        cpu_pause();
    asm volatile("sfence" ::
                     : "memory");
#ifdef COUNT_CLFLUSH
    clflush_count.fetch_add(1);
    sfence_count.fetch_add(1);
#endif
}

void prefetch(char *ptr, size_t len)
{
    if (ptr == nullptr)
        return;
    for (char *p = ptr; p < ptr + len; p += cacheline_size)
    {
        __builtin_prefetch(p);
    }
}

#ifdef HAS_AVX512
void nontemporal_store_512_fenced(void *mem_addr, void *c)
{
    auto t = _mm512_load_si512((const __m512i *)c);
    unsigned long etsc = read_tsc() +
                         (unsigned long)(write_latency_in_ns * CPU_FREQ_MHZ / 1000);
    _mm512_stream_si512((__m512i *)mem_addr, t);
    while (read_tsc() < etsc)
        cpu_pause();
    asm volatile("mfence" ::
                     : "memory");
#ifdef COUNT_CLFLUSH
    clflush_count.fetch_add(1);
#endif
}

void nontemporal_store_512(void *mem_addr, void *c)
{
    auto t = _mm512_load_si512((const __m512i *)c);
    unsigned long etsc = read_tsc() +
                         (unsigned long)(write_latency_in_ns * CPU_FREQ_MHZ / 1000);
    _mm512_stream_si512((__m512i *)mem_addr, t);
    while (read_tsc() < etsc)
        cpu_pause();
#ifdef COUNT_CLFLUSH
    clflush_count.fetch_add(1);
#endif
}

#endif

void clflush_then_sfence(volatile void *p)
{
    volatile char *ptr = CACHELINE_ALIGN(p);
#ifdef COUNT_CLFLUSH
    clflush_count.fetch_add(1);
    sfence_count.fetch_add(1);
#endif
    asm volatile("clwb %0"
    : "+m"(*ptr));
    sfence();
}

void clflush(volatile void *p)
{
    volatile char *ptr = CACHELINE_ALIGN(p);
#ifdef COUNT_CLFLUSH
    clflush_count.fetch_add(1);
#endif
    asm volatile("clwb (%0)" ::"r"(ptr));
}

void clflush_len_no_fence(volatile void *data, int len, std::function<void()> stat_updater) {
    volatile char *ptr = CACHELINE_ALIGN(data);
    for (; ptr < (char *)data + len; ptr += cacheline_size)
    {
#ifdef COUNT_CLFLUSH
        clflush_count.fetch_add(1);
        stat_updater();
#endif
        asm volatile("clwb %0"
        : "+m"(*(volatile char *)ptr));
    }
}


void clflush_len_no_fence(volatile void *data, int len) {
    volatile char *ptr = CACHELINE_ALIGN(data);
    for (; ptr < (char *)data + len; ptr += cacheline_size)
    {
#ifdef COUNT_CLFLUSH
        clflush_count.fetch_add(1);
#endif
        asm volatile("clwb %0"
        : "+m"(*(volatile char *)ptr));
    }
}

void clflush_len(volatile void *data, int len)
{
    volatile char *ptr = CACHELINE_ALIGN(data);
    for (; ptr < (char *)data + len; ptr += cacheline_size)
    {
#ifdef COUNT_CLFLUSH
        clflush_count.fetch_add(1);
#endif
        asm volatile("clwb %0"
                     : "+m"(*(volatile char *)ptr));
    }
#ifdef COUNT_CLFLUSH
    sfence();
    sfence_count.fetch_add(1);
#endif
}

int nvm_dram_alloc(void **ptr, size_t align, size_t size)
{
    assert(size < 1073741824UL);
    int ret = posix_memalign(ptr, align, size);
    return ret;
}

void nvm_dram_free(void *ptr, size_t size)
{
    free(ptr);
}

int nvm_dram_alloc_cacheline_aligned(void **p, size_t size)
{
    assert(size < 1073741824UL);
    assert(size % cacheline_size == 0);
    int ret =  posix_memalign(p, cacheline_size, size);
    return ret;
}

void nvm_dram_free_cacheline_aligned(void *ptr)
{
    assert(((uintptr_t)ptr & (cacheline_size - 1)) == 0);
    free(ptr);
}

void percentile_report(const std::string & name, std::vector<int> & nums) {
    std::sort(nums.begin(), nums.end());
    auto sum = accumulate(nums.begin(), nums.end(), 0LL);
    printf("name %s, total %lld, avg %f, 50p %d, 70p %d, 80p %d, 90p %d, 95p "
           "%d, 99p %d, 99.9p %d\n",
           name.c_str(),sum,
           sum / (nums.size() + 0.1),
           nums[nums.size() * 0.5],
           nums[nums.size() * 0.7],
           nums[nums.size() * 0.8],
           nums[nums.size() * 0.9],
           nums[nums.size() * 0.95],
           nums[nums.size() * 0.99],
           nums[nums.size() * 0.999]);
}

int wbinvd() {
    return system("cat /proc/wbinvd");
}
