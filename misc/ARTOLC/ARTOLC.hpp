#pragma once
#include "Tree.h"
#include "index_key.h"

template <typename KeyType>
class ArtOLCIndex
{
  public:
    ~ArtOLCIndex() { delete idx; }


    static void setKey(Key &k, uint64_t key) { k.setInt(key); }
    static void setKey(Key &k, GenericKey<31> key) { k.set(key.data, 31); }

    template<typename F>
    bool insert(KeyType key, uint64_t value, F f)
    {
        auto t = idx->getThreadInfo();
        Key k;
        setKey(k, key);
        idx->insert(k, value, t, f);
        return true;
    }

    bool find(KeyType key, uint64_t & v)
    {
        auto t = idx->getThreadInfo();
        Key k;
        setKey(k, key);
        uint64_t result = idx->lookup(k, t);
        v = result;
        return true;
    }
    template<typename F>
    bool upsert(KeyType key, uint64_t value, F f)
    {
        auto t = idx->getThreadInfo();
        Key k;
        setKey(k, key);
        idx->insert(k, value, t, f);
        return true;
    }

    bool scan(const KeyType & start_key, int range, KeyType & continue_key, TID results[], size_t & result_count)
    {
        auto t = idx->getThreadInfo();
        Key startKey;
        startKey.setInt(start_key);
        result_count = 0;
        Key continueKey;
        bool has_more = idx->lookupRange(startKey, maxKey, continueKey, results, range, result_count,
                         t);
        continue_key = continueKey.getInt();
        return has_more;
    }


    ArtOLCIndex(uint64_t kt)
    {
        if (sizeof(KeyType) == 8)
        {
            idx = new ART_OLC::Tree([](TID tid, Key &key) {
                key.setInt(*reinterpret_cast<uint64_t *>(tid));
            });
            maxKey.setInt(~0ull);
        }
        else
        {
            idx = new ART_OLC::Tree([](TID tid, Key &key) {
                key.set(reinterpret_cast<char *>(tid), 31);
            });
            uint8_t m[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                           0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                           0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                           0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
            maxKey.set((char *)m, 31);
        }
    }


    struct iterator {
        static const uint64_t scan_unit_size = 64;
        ArtOLCIndex<KeyType> *artindex;
        int count;
        int pos;
        bool nomore;
        KeyType continue_key;
        KeyType start_key;
        KeyType cur_key;
        uint64_t values[scan_unit_size];

        explicit iterator(ArtOLCIndex<KeyType> *artindex=nullptr)
                : artindex(artindex), count(0), pos(0), nomore(false), start_key(0), cur_key(std::numeric_limits<KeyType>::max()), continue_key(0){}

        explicit iterator(ArtOLCIndex<KeyType> *artindex, const KeyType & startKey)
                : artindex(artindex), count(0), pos(0), nomore(false), start_key(startKey), cur_key(std::numeric_limits<KeyType>::max()), continue_key(0){
            while (nomore == false && count == 0)
                fill();
            if (is_end()) {
                cur_key = std::numeric_limits<KeyType>::max();
            } else {
                cur_key = *(KeyType*)values[pos];
            }
        }

        void fill() {
            if (nomore == false) {
                start_key = continue_key;
                size_t result_count = 0;
                nomore = !artindex->scan(start_key, scan_unit_size, continue_key, values, result_count);
                count = result_count;
            } else {
                count = 0;
            }
            pos = 0;
        }

        int next_node(KeyType&last_key) {
            last_key = this->last_key();
            int ret = count;
            fill();
            return ret;
        }

        bool is_end() { return count == 0 && nomore == true; }

        iterator &operator++() {
            if (++pos == count) {
                fill();
            }
            if (is_end()) {
                cur_key = std::numeric_limits<KeyType>::max();
            } else {
                cur_key = *(KeyType*)values[pos];
            }
            return *this;
        }

        bool operator==(const iterator &rhs) {
            return cur_key == rhs.cur_key;
        }

        bool operator!=(const iterator &rhs) {
            return !operator==(rhs);
        }

        KeyType last_key() { return *(KeyType *) values[count - 1]; }

        KeyType key() { return *(KeyType *) values[pos]; }

        uint64_t value() { return values[pos]; }
    };

    iterator begin() {
        return iterator(this, std::numeric_limits<KeyType>::min());
    }
    iterator end() {
        return iterator(this);
    }

    iterator lookup_range(const KeyType & start_key) {
        return iterator(this, start_key);
    }
  private:
    Key maxKey;
    ART_OLC::Tree *idx;
};
