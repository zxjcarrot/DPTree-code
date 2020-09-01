#include <dptree.hpp>
#include <fstream>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stx/btree_map>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_set>
#include <util.h>
#include <vector>

using namespace std;


float range_sizes[] = {0.0001, 0.001, 0.01};
const int repeatn = 50000000;
uint64_t cacheline[8];
unsigned lookup_test_iteration = 1;


uint64_t make_upsert_value(uint64_t v) { return (v << 1); }

uint64_t strip_upsert_value(uint64_t v) { return v >> 1; }

void dpftree_cvhftree_test() {
    dtree::dpftree<uint64_t, uint64_t> index;
    int repeat = 1;
    int c = 0;
    measure(
            [&]() -> unsigned {
                size_t i = 0;
                while (i < keys.size()) {
                    index.insert(keys[i], keys[i]);
                    i++;
                }
                printf("# merges: %d\n", index.get_merges());
                printf("merge time: %f secs\n", index.get_merge_time());
                printf("real merge time: %f secs\n", index.get_real_merge_time());
                printf("inner nodes build time: %f secs\n",
                       index.get_inner_node_build_time());
                printf("buffer tree size: %d\n", (int) index.get_buffer_tree_size());
                printf("buffer tree node count: %d\n", (int) index.get_buffer_tree_node_count());
                printf("base tree leaf count: %d\n", (int) index.get_base_tree_leaf_count());
                printf("wal entry count: %d\n", (int) index.get_wal_entry_count());
                printf("bloom entries: %d\n", index.get_bloom_entries());
                printf("bloom bytes: %d\n", index.get_bloom_bytes());
                printf("bloom hashes: %d\n", index.get_bloom_hashes());
                printf("bloom words: %d\n", index.get_bloom_words());
                return keys.size();
            },
            "dptree-cvhtree-insert", 1, true);
    clear_cache();
    while (index.merging());

    double avg_per_search = measure(
            [&]() -> unsigned {
                repeat = repeatn / lookupKeys.size() / 2;
                if (repeat < 1)
                    repeat = 1;
                auto static_lookup_count_save = index.static_lookup_count;
                auto dyn_lookup_count_save = index.dyn_lookup_count;

                for (uint64_t r = 0; r < repeat; ++r) {
                    for (size_t i = 0; i < lookupKeys.size(); i++) {
                        uint64_t key = lookupKeys[i];
                        uint64_t value = 0;
                        bool res = index.lookup(key, value);
                        //probes_vec.push_back(index.get_probes() - probes_save);
                        //assert(res);
                        assert(key == strip_upsert_value(value));
                    }
                }
                auto static_lookup_count =
                        index.static_lookup_count - static_lookup_count_save;
                auto dynamic_lookup_count =
                        index.dyn_lookup_count - dyn_lookup_count_save;
                auto probes = index.get_probes();
                printf("dynamic lookup count: %d\n", dynamic_lookup_count);
                printf("static lookup count: %d\n", static_lookup_count);
                printf("avg probes %f\n", probes / (repeat * lookupKeys.size() + 0.0));
                return repeat * lookupKeys.size();
            },
            "dpftree-cvhftree-search", lookup_test_iteration);
    index.clear_probes();

    uint64_t s = 0;
    uint64_t rc = 0;
    for (int i = 0; i < sizeof(range_sizes) / sizeof(float); ++i) {
        clear_cache();
        int range_size = range_sizes[i] * keys.size();
        printf("range size: %d\n", range_size);
        measure(
                [&]() -> unsigned {
                    repeat = 5;
                    for (int r = 0; r < repeat; ++r) {
                        for (size_t i = 0; i < sortedKeys.size();) {
                            uint64_t startPos = rand() % keys.size();
                            uint64_t endPos =
                                    std::min(startPos + range_size, (uint64_t) keys.size() - 1);
                            uint64_t keyStart = sortedKeys[startPos];
                            uint64_t keyEnd = sortedKeys[endPos];
                            std::vector<uint64_t> res;
                            res.reserve(endPos - startPos);
                            index.lookup_range(keyStart, keyEnd, res);
                            assert(res.size() == endPos - startPos);
                            for (int j = 0; j < res.size(); ++j) {
                                if (strip_upsert_value(res[j]) != sortedKeys[startPos + j]) {
                                    assert(false);
                                }
                                s += res[j];
                                ++rc;
                            }
                            i += endPos - startPos;
                        }
                    }
                    return repeat * sortedKeys.size();
                },
                "dpftree-cvhftree-scan-" + std::to_string(range_sizes[i]));
    }
    printf("%lu %lu\n", s, rc);
}

int main(int argc, char const *argv[])
{
    int n = 10000000;
    bool sparseKey = false;
    string keyFile = "zipf-keys-s1.0.csv";
    if (argc > 1)
        n = atoi(argv[1]);
    if (argc > 2)
        keyFile = argv[2];
    if (argc > 3)
        sparseKey = atoi(argv[3]);
    if (argc > 4)
        write_latency_in_ns = atoi(argv[4]);
    printf("started sparseKey %d\n", sparseKey);
    prepareKeys(n, keyFile, sparseKey, false);
    dpftree_cvhftree_test();
    return 0;
}