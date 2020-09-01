#include <iostream>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <unistd.h>
#include <fstream>
#include <concur_dptree.hpp>
#include <stx/btree_map>
#include <unordered_set>
#include <random>
#include <vector>
#include <util.h>
#include <btreeolc.hpp>

using namespace std;

int parallel_merge_worker_num = 16;

float range_sizes [] = {0.0001,  0.001, 0.01};
const int repeatn = 50000000;
uint64_t cacheline[8];

uint64_t strip_upsert_value(uint64_t v) {
    return v >> 1;
}

uint64_t make_upsert_value(uint64_t v) {
    return (v << 1);
}


void concur_dptree_test(int nworkers, int n) {

    dptree::concur_dptree<uint64_t, uint64_t> index;
    auto insert_func = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            index.insert(keys[i], keys[i]);
        }
    };
    
    measure([&]() -> unsigned {
        std::vector<std::thread> workers;
        int range_size = n / nworkers;
        int start = 0;
        int end = range_size;
        for (;start < n;) {
            workers.emplace_back(std::thread(insert_func, start, end));
            start = end;
            end = std::min(end + range_size, (int)n);
        }
        std::for_each(workers.begin(), workers.end(), [](std::thread & t) { t.join(); });
        // printf("# merges: %d\n", index.get_merges());
        // printf("merge time: %f secs\n", index.get_merge_time());
        printf("real merge time: %f secs\n", index.get_real_merge_time());
        printf("merge wait time: %f secs\n", index.get_merge_wait_time());
        printf("merge work time: %f secs\n", index.get_merge_work_time());
        printf("inner nodes build time: %f secs\n", index.get_inner_node_build_time());
        printf("base tree flush time: %f secs\n", index.get_flushtime());

        return n;
    }, "concur-dptree-insert");
    while (index.is_merging());

    auto lookup_func = [&](int start, int end, std::atomic<int> & count) {
        int repeat = 1;
        int c = 0;
        repeat = repeatn / (end - start) / 2;
        if (repeat < 1)
            repeat = 1;
        for (uint64_t r = 0; r < repeat; ++r) {
            for (size_t i = start; i < end; i++) {
                uint64_t key = lookupKeys[i];
                uint64_t value = 0;
                bool res = index.lookup(key, value);
                assert(res);
                assert(key == strip_upsert_value(value));
                c += 1;
            }
        }
        count += c;
    };
    
    measure([&]() -> unsigned {
        std::vector<std::thread> workers;
        int range_size = n / nworkers;
        int start = 0;
        int end = range_size;
        std::atomic<int> count(0);
        for (;start < n && workers.size() < nworkers;) {
            workers.emplace_back(std::thread(lookup_func, start, end, std::ref(count)));
            start = end;
            end = std::min(end + range_size, (int)n);
        }
        std::for_each(workers.begin(), workers.end(), [](std::thread & t) { t.join(); });
        printf("probes: %lu\n", index.get_probes());
        printf("avg probes per lookup: %f\n", index.get_probes() / (count.load() + 0.1));
        return count.load();
    }, "concur-dptree-search");

}


int main(int argc, char const *argv[]) {
    int n = 10000000;
    bool sparseKey = false;
    int nworkers = 1;
    if (argc > 1)
        n = atoi(argv[1]);
    if (argc > 2)
        nworkers = atoi(argv[2]);
    if (argc > 3)
        parallel_merge_worker_num = atoi(argv[3]);
    else
        parallel_merge_worker_num = nworkers;
    if (argc > 4)
        sparseKey = atoi(argv[4]);
    if (argc > 5)
        write_latency_in_ns = atoi(argv[5]);
    printf("nworkers: %d, parallel_merge_worker_num %d\n", nworkers, parallel_merge_worker_num);
    prepareKeys(n, "", sparseKey, false);
    concur_dptree_test(nworkers, n);
    return 0;
}