/*
  Adaptive Radix Tree
  Viktor Leis, 2012
  leis@in.tum.de

  Modified by Huanchen Zhang, 2016
  Modified by Xinjing Zhou, 2018
 */

#include <stdlib.h>    // malloc, free
#include <string.h>    // memset, memcpy
#include <stdint.h>    // integer types
#include <emmintrin.h> // x86 SSE intrinsics
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <deque>
#include <string>
#include <functional>
#include <iostream>

using namespace std;

// Constants for the node types
static const int8_t NodeType4=0;
static const int8_t NodeType16=1;
static const int8_t NodeType48=2;
static const int8_t NodeType256=3;

// The maximum prefix length for compressed paths stored in the
// header, if the path is longer it is loaded from the database on
// demand
static const unsigned maxPrefixLength=9;

static const unsigned NodeDItemTHold=227;

// Shared header of all inner nodes
struct Node {
    // length of the compressed path (prefix)
    uint32_t prefixLength;
    // number of non-null children
    uint16_t count;
    // node type
    int8_t type;
    // compressed path (prefix)
    uint8_t prefix[maxPrefixLength];

    Node(int8_t type) : prefixLength(0),count(0),type(type) {}
};

// Node with up to 4 children
struct Node4 : Node {
    uint8_t key[4];
    Node* child[4];

    Node4() : Node(NodeType4) {
	memset(key,0,sizeof(key));
	memset(child,0,sizeof(child));
    }
};

// Node with up to 16 children
struct Node16 : Node {
    uint8_t key[16];
    Node* child[16];

    Node16() : Node(NodeType16) {
	memset(key,0,sizeof(key));
	memset(child,0,sizeof(child));
    }
};

static const uint8_t emptyMarker=48;

// Node with up to 48 children
struct Node48 : Node {
    uint8_t childIndex[256];
    Node* child[48];

    Node48() : Node(NodeType48) {
	memset(childIndex,emptyMarker,sizeof(childIndex));
	memset(child,0,sizeof(child));
    }
};

// Node with up to 256 children
struct Node256 : Node {
    Node* child[256];

    Node256() : Node(NodeType256) {
	memset(child,0,sizeof(child));
    }
};

typedef struct {
    Node* node;
    uint16_t cursor;
} NodeCursor;

class ART;
class ARTIter;


class ART {

public:
    static inline Node* makeLeaf(uintptr_t tid){
        // Create a pseudo-leaf
        return reinterpret_cast<Node*>((tid<<1)|1);
    } 

    static inline uintptr_t getLeafValue(Node* node){
        // The the value stored in the pseudo-leaf
        return reinterpret_cast<uintptr_t>(node)>>1;
    }

    static inline bool isLeaf(Node* node)  {
        // Is the node a leaf?
        return reinterpret_cast<uintptr_t>(node)&1;
    }

    static inline uint8_t flipSign(uint8_t keyByte)  {
        // Flip the sign bit, enables signed SSE comparison of unsigned values, used by Node16
        return keyByte^128;
    }

    //inline void loadKey(uintptr_t tid,uint8_t key[]);
    static inline unsigned ctz(uint16_t x)  {
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

    //****************************************************************

    inline Node** findChild(Node* n,uint8_t keyByte);
    inline Node* minimum(Node* node);
    inline bool leafMatches(Node* leaf,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);
    inline unsigned prefixMismatch(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength);

    Node** lookupRef(Node** nodeRef, Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);
    inline Node* lookup(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);
    inline Node* lookupPessimistic(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);

    //****************************************************************

    inline int CompareToPrefix(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength);
    inline Node* findChild_recordPath(Node* n, uint8_t keyByte, ARTIter* iter);
    inline Node* lower_bound(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength, ARTIter *iter);

    //****************************************************************

    inline unsigned min(unsigned a,unsigned b);
    inline void copyPrefix(Node* src,Node* dst);

    void upsert(Node* node,Node** nodeRef,uint8_t key[],unsigned depth,uintptr_t value,unsigned maxKeyLength, std::function<Node*(uintptr_t)> insertMakeLeaf);
    inline void insert(Node* node,Node** nodeRef,uint8_t key[],unsigned depth,uintptr_t value,unsigned maxKeyLength);

    inline void insertNode4(Node4* node,Node** nodeRef,uint8_t keyByte,Node* child);
    inline void insertNode16(Node16* node,Node** nodeRef,uint8_t keyByte,Node* child);
    inline void insertNode48(Node48* node,Node** nodeRef,uint8_t keyByte,Node* child);
    inline void insertNode256(Node256* node,Node** nodeRef,uint8_t keyByte,Node* child);

    //****************************************************************

    inline void erase(Node* node,Node** nodeRef,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);

    inline void eraseNode4(Node4* node,Node** nodeRef,Node** leafPlace);
    inline void eraseNode16(Node16* node,Node** nodeRef,Node** leafPlace);
    inline void eraseNode48(Node48* node,Node** nodeRef,uint8_t keyByte);
    inline void eraseNode256(Node256* node,Node** nodeRef,uint8_t keyByte);

    ART();
    ART(std::function<void(uintptr_t,uint8_t[])> loadKey, std::function<void(Node**, uintptr_t)> upsertFunc);

    void load(vector<string> &keys, vector<uint64_t> &values, unsigned maxKeyLength);
    void load(vector<uint64_t> &keys, vector<uint64_t> &values);
    void insert(uint8_t key[], uintptr_t value, unsigned maxKeyLength);
    void upsert(uint8_t key[], uintptr_t value, unsigned maxKeyLength, std::function<Node*(uintptr_t)> insertMakeLeaf);
    uint64_t lookup(uint8_t key[], unsigned keyLength, unsigned maxKeyLength);
    uint64_t lookup(uint64_t key64);
    Node** lookupRef(uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength);

    bool lower_bound(uint8_t key[], unsigned keyLength, unsigned maxKeyLength, ARTIter* iter);
    bool lower_bound(uint64_t key64, ARTIter* iter);
    void erase(uint8_t key[], unsigned keyLength, unsigned maxKeyLength);
    uint64_t getMemory();

    friend class ARTIter;

private:
    Node* root;
    unsigned key_length;

    //stats
    uint64_t memory;
    uint64_t num_items;
    uint64_t node4_count;
    uint64_t node16_count;
    uint64_t node48_count;
    uint64_t node256_count;

    // This address is used to communicate that search failed
    Node* nullNode;
    std::function<void(uintptr_t,uint8_t[])> loadKey;
    std::function<void(Node**, uintptr_t)> upsertFunc;
};

class ARTIter {
public:
    inline Node* minimum_recordPath(Node* node);
    inline Node* nextSlot();
    inline Node* currentLeaf();
    inline Node* nextLeaf();

    ARTIter();
    ARTIter(ART* idx);

    uint64_t value();
    uint64_t key() { return key64; }
    bool operator ++ (int);

    friend class ART;

private:
    ART* index;
    std::vector<NodeCursor> node_stack;
    uint64_t val;
    uint64_t key64;
};

