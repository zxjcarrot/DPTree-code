#include <stdlib.h>    // malloc, free
#include <string.h>    // memset, memcpy
#include <stdint.h>    // integer types
#include <emmintrin.h> // x86 SSE intrinsics
#include <stdio.h>
#include <assert.h>
#include <sys/time.h> // gettime
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <functional>

namespace ART_IDX
{
// Constants for the node types
static const int8_t NodeType4 = 0;
static const int8_t NodeType16 = 1;
static const int8_t NodeType48 = 2;
static const int8_t NodeType256 = 3;

// The maximum prefix length for compressed paths stored in the
// header, if the path is longer it is loaded from the database on
// demand
static const unsigned maxPrefixLength = 9;

// Shared header of all inner nodes
struct Node
{
    // length of the compressed path (prefix)
    uint32_t prefixLength;
    // number of non-null children
    uint16_t count;
    // node type
    int8_t type;
    // compressed path (prefix)
    uint8_t prefix[maxPrefixLength];

    Node(int8_t type) : prefixLength(0), count(0), type(type) {}
};

// Node with up to 4 children
struct Node4 : Node
{
    uint8_t key[4];
    Node *child[4];

    Node4() : Node(NodeType4)
    {
        memset(key, 0, sizeof(key));
        memset(child, 0, sizeof(child));
    }
};

// Node with up to 16 children
struct Node16 : Node
{
    uint8_t key[16];
    Node *child[16];

    Node16() : Node(NodeType16)
    {
        memset(key, 0, sizeof(key));
        memset(child, 0, sizeof(child));
    }
};

static const uint8_t emptyMarker = 48;

// Node with up to 48 children
struct Node48 : Node
{
    uint8_t childIndex[256];
    Node *child[48];

    Node48() : Node(NodeType48)
    {
        memset(childIndex, emptyMarker, sizeof(childIndex));
        memset(child, 0, sizeof(child));
    }
};

// Node with up to 256 children
struct Node256 : Node
{
    Node *child[256];

    Node256() : Node(NodeType256)
    {
        memset(child, 0, sizeof(child));
    }
};

// This address is used to communicate that search failed
static Node *nullNode = NULL;

static inline unsigned ctz(uint16_t x)
{
    // Count trailing zeros, only defined for x>0
#ifdef __GNUC__
    return __builtin_ctz(x);
#else
    // Adapted from Hacker's Delight
    unsigned n = 1;
    if ((x & 0xFF) == 0)
    {
        n += 8;
        x = x >> 8;
    }
    if ((x & 0x0F) == 0)
    {
        n += 4;
        x = x >> 4;
    }
    if ((x & 0x03) == 0)
    {
        n += 2;
        x = x >> 2;
    }
    return n - (x & 1);
#endif
}

static inline Node *makeLeaf(uintptr_t tid)
{
    // Create a pseudo-leaf
    return reinterpret_cast<Node *>((tid << 1) | 1);
}

static inline uintptr_t getLeafValue(Node *node)
{
    // The the value stored in the pseudo-leaf
    return reinterpret_cast<uintptr_t>(node) >> 1;
}

static inline bool isLeaf(Node *node)
{
    // Is the node a leaf?
    return reinterpret_cast<uintptr_t>(node) & 1;
}

static uint8_t flipSign(uint8_t keyByte)
{
    // Flip the sign bit, enables signed SSE comparison of unsigned values, used by Node16
    return keyByte ^ 128;
}

static std::function<void(uintptr_t, uint8_t[])> defaultLoadKey = [](uintptr_t tid, uint8_t key[]) {
    // Store the key of the tuple into the key vector
    // Implementation is database specific
    reinterpret_cast<uint64_t *>(key)[0] = __builtin_bswap64(tid);
};

static std::function<void(uintptr_t, uint8_t[])> loadLowerBoundKey;

struct art_tree
{
    std::function<void(uintptr_t, uint8_t[])> loadKey;
    std::function<void(uintptr_t, uint8_t[])> loadLowerBoundKey;
    Node* root;
    art_tree() : loadKey(defaultLoadKey), loadLowerBoundKey(nullptr), root(nullptr) {}
    art_tree(std::function<void(uintptr_t, uint8_t[])> loadKey, std::function<void(uintptr_t, uint8_t[])> loadLowerBoundKey):loadKey(loadKey), loadLowerBoundKey(loadLowerBoundKey), root(nullptr) {}
    ~art_tree() {
        destroyNode(root);
    }
    Node *findChildLowerBound(Node *n, uint8_t keyByte, bool &equal)
    {
        // Find the next child for the keyByte
        switch (n->type)
        {
        case NodeType4:
        {
            Node4 *node = static_cast<Node4 *>(n);
            unsigned i = 0;
            for (; i < node->count && node->key[i] < keyByte; i++)
                ;
            equal = i < node->count && node->key[i] == keyByte;
            return i < node->count ? node->child[i] : nullptr;
        }
        case NodeType16:
        {
            Node16 *node = static_cast<Node16 *>(n);
            __m128i cmp = _mm_cmpgt_epi8(_mm_set1_epi8(flipSign(keyByte)), _mm_loadu_si128(reinterpret_cast<__m128i *>(node->key)));
            unsigned bitfield = (~_mm_movemask_epi8(cmp)) & ((1 << node->count) - 1);
            if (bitfield)
            {
                int idx = ctz(bitfield);
                equal = node->key[idx] == flipSign(keyByte);
                return node->child[idx];
            }
            else
            {
                return nullptr;
            }
        }
        case NodeType48:
        {
            Node48 *node = static_cast<Node48 *>(n);
            equal = node->childIndex[keyByte] != emptyMarker;
            for (uint16_t b = keyByte; b <= 255; ++b)
                if (node->childIndex[b] != emptyMarker)
                    return node->child[node->childIndex[b]];
            return nullptr;
        }
        case NodeType256:
        {
            Node256 *node = static_cast<Node256 *>(n);
            equal = node->child[keyByte];
            for (uint16_t b = keyByte; b <= 255; ++b)
                if (node->child[b])
                    return node->child[b];
            return nullptr;
        }
        }
        throw; // Unreachable
    }
    Node **findChild(Node *n, uint8_t keyByte)
    {
        // Find the next child for the keyByte
        switch (n->type)
        {
        case NodeType4:
        {
            Node4 *node = static_cast<Node4 *>(n);
            for (unsigned i = 0; i < node->count; i++)
                if (node->key[i] == keyByte)
                    return &node->child[i];
            return &nullNode;
        }
        case NodeType16:
        {
            Node16 *node = static_cast<Node16 *>(n);
            __m128i cmp = _mm_cmpeq_epi8(_mm_set1_epi8(flipSign(keyByte)), _mm_loadu_si128(reinterpret_cast<__m128i *>(node->key)));
            unsigned bitfield = _mm_movemask_epi8(cmp) & ((1 << node->count) - 1);
            if (bitfield)
                return &node->child[ctz(bitfield)];
            else
                return &nullNode;
        }
        case NodeType48:
        {
            Node48 *node = static_cast<Node48 *>(n);
            if (node->childIndex[keyByte] != emptyMarker)
                return &node->child[node->childIndex[keyByte]];
            else
                return &nullNode;
        }
        case NodeType256:
        {
            Node256 *node = static_cast<Node256 *>(n);
            return &(node->child[keyByte]);
        }
        }
        throw; // Unreachable
    }

    Node *minimum(Node *node)
    {
        // Find the leaf with smallest key
        if (!node)
            return NULL;

        if (isLeaf(node))
            return node;

        switch (node->type)
        {
        case NodeType4:
        {
            Node4 *n = static_cast<Node4 *>(node);
            return minimum(n->child[0]);
        }
        case NodeType16:
        {
            Node16 *n = static_cast<Node16 *>(node);
            return minimum(n->child[0]);
        }
        case NodeType48:
        {
            Node48 *n = static_cast<Node48 *>(node);
            unsigned pos = 0;
            while (n->childIndex[pos] == emptyMarker)
                pos++;
            return minimum(n->child[n->childIndex[pos]]);
        }
        case NodeType256:
        {
            Node256 *n = static_cast<Node256 *>(node);
            unsigned pos = 0;
            while (!n->child[pos])
                pos++;
            return minimum(n->child[pos]);
        }
        }
        throw; // Unreachable
    }

    Node *maximum(Node *node)
    {
        // Find the leaf with largest key
        if (!node)
            return NULL;

        if (isLeaf(node))
            return node;

        switch (node->type)
        {
        case NodeType4:
        {
            Node4 *n = static_cast<Node4 *>(node);
            return maximum(n->child[n->count - 1]);
        }
        case NodeType16:
        {
            Node16 *n = static_cast<Node16 *>(node);
            return maximum(n->child[n->count - 1]);
        }
        case NodeType48:
        {
            Node48 *n = static_cast<Node48 *>(node);
            unsigned pos = 255;
            while (n->childIndex[pos] == emptyMarker)
                pos--;
            return maximum(n->child[n->childIndex[pos]]);
        }
        case NodeType256:
        {
            Node256 *n = static_cast<Node256 *>(node);
            unsigned pos = 255;
            while (!n->child[pos])
                pos--;
            return maximum(n->child[pos]);
        }
        }
        throw; // Unreachable
    }

    bool leafMatches(Node *leaf, uint8_t key[], unsigned keyLength, unsigned depth, unsigned maxKeyLength)
    {
        // Check if the key of the leaf is equal to the searched key
        if (depth != keyLength)
        {
            uint8_t leafKey[maxKeyLength];
            loadKey(getLeafValue(leaf), leafKey);
            for (unsigned i = depth; i < keyLength; i++)
                if (leafKey[i] != key[i])
                    return false;
        }
        return true;
    }

    unsigned prefixMismatch(Node *node, uint8_t key[], unsigned depth, unsigned maxKeyLength)
    {
        // Compare the key with the prefix of the node, return the number matching bytes
        unsigned pos;
        if (node->prefixLength > maxPrefixLength)
        {
            for (pos = 0; pos < maxPrefixLength; pos++)
                if (key[depth + pos] != node->prefix[pos])
                    return pos;
            uint8_t minKey[maxKeyLength];
            loadKey(getLeafValue(minimum(node)), minKey);
            for (; pos < node->prefixLength; pos++)
                if (key[depth + pos] != minKey[depth + pos])
                    return pos;
        }
        else
        {
            for (pos = 0; pos < node->prefixLength; pos++)
                if (key[depth + pos] != node->prefix[pos])
                    return pos;
        }
        return pos;
    }

    Node **lookupLeafPtr(Node *node, uint8_t key[], unsigned keyLength, unsigned depth, unsigned maxKeyLength)
    {
        // Find the node with a matching key, optimistic version

        bool skippedPrefix = false; // Did we optimistically skip some prefix without checking it?
        Node **leafPtr = &node;
        while (node != NULL)
        {
            if (isLeaf(node))
            {
                break;
                //    if (depth!=keyLength) {
                //       // Check leaf
                //       uint8_t leafKey[maxKeyLength];
                //       loadKey(getLeafValue(node),leafKey);
                //       for (unsigned i=(skippedPrefix?0:depth);i<keyLength;i++)
                //          if (leafKey[i]!=key[i])
                //             return NULL;
                //    }
            }

            if (node->prefixLength)
            {
                if (node->prefixLength < maxPrefixLength)
                {
                    for (unsigned pos = 0; pos < node->prefixLength; pos++)
                        if (key[depth + pos] != node->prefix[pos])
                            return NULL;
                }
                else
                    skippedPrefix = true;
                depth += node->prefixLength;
            }

            leafPtr = findChild(node, key[depth]);
            node = *leafPtr;
            depth++;
        }

        return leafPtr;
    }

    Node *lookup(Node *node, uint8_t key[], unsigned keyLength, unsigned depth, unsigned maxKeyLength)
    {
        // Find the node with a matching key, optimistic version

        bool skippedPrefix = false; // Did we optimistically skip some prefix without checking it?

        while (node != NULL)
        {
            if (isLeaf(node))
            {
                if (!skippedPrefix && depth == keyLength) // No check required
                    return node;

                return node;
            }

            if (node->prefixLength)
            {
                if (node->prefixLength < maxPrefixLength)
                {
                    for (unsigned pos = 0; pos < node->prefixLength; pos++)
                        if (key[depth + pos] != node->prefix[pos])
                            return NULL;
                }
                else
                    skippedPrefix = true;
                depth += node->prefixLength;
            }

            node = *findChild(node, key[depth]);
            depth++;
        }

        return NULL;
    }

    Node *findNextChild(Node *node, uint8_t keyByte)
    {

        return NULL;
    }

    Node *lowerBound(uint8_t key[], unsigned keyLength, unsigned depth, unsigned maxKeyLength, bool &pref) {
        return lowerBound(root, key, keyLength, depth, maxKeyLength, pref);
    }
    
    // return the first leaf whose key is no less than the search key
    Node *lowerBound(Node *node, uint8_t key[], unsigned keyLength, unsigned depth, unsigned maxKeyLength, bool &pref);
    

    Node *lookupPessimistic(Node *node, uint8_t key[], unsigned keyLength, unsigned depth, unsigned maxKeyLength)
    {
        // Find the node with a matching key, alternative pessimistic version

        while (node != NULL)
        {
            if (isLeaf(node))
            {
                if (leafMatches(node, key, keyLength, depth, maxKeyLength))
                    return node;
                return NULL;
            }

            if (prefixMismatch(node, key, depth, maxKeyLength) != node->prefixLength)
                return NULL;
            else
                depth += node->prefixLength;

            node = *findChild(node, key[depth]);
            depth++;
        }

        return NULL;
    }

    unsigned min(unsigned a, unsigned b)
    {
        // Helper function
        return (a < b) ? a : b;
    }

    void copyPrefix(Node *src, Node *dst)
    {
        // Helper function that copies the prefix from the source to the destination node
        dst->prefixLength = src->prefixLength;
        memcpy(dst->prefix, src->prefix, min(src->prefixLength, maxPrefixLength));
    }

    std::function<void(Node **, uintptr_t)> upsertFunc;

    void upsert(Node *node, Node **nodeRef, uint8_t key[], unsigned depth, uintptr_t value, unsigned maxKeyLength, std::function<Node *(uintptr_t)> insertMakeLeaf)
    {
        // Insert the leaf value into the tree

        if (node == NULL)
        {
            *nodeRef = insertMakeLeaf(value);
            return;
        }

        if (isLeaf(node))
        {
            // Replace leaf with Node4 and store both leaves in it
            uint8_t existingKey[maxKeyLength];
            loadKey(getLeafValue(node), existingKey);
            unsigned newPrefixLength = 0;
            while (depth + newPrefixLength < maxKeyLength && existingKey[depth + newPrefixLength] == key[depth + newPrefixLength])
                newPrefixLength++;
            if (depth + newPrefixLength == maxKeyLength)
            {
                upsertFunc(nodeRef, value);
                return;
            }
            Node4 *newNode = new Node4();
            newNode->prefixLength = newPrefixLength;
            memcpy(newNode->prefix, key + depth, min(newPrefixLength, maxPrefixLength));
            *nodeRef = newNode;

            insertNode4(newNode, nodeRef, existingKey[depth + newPrefixLength], node);
            insertNode4(newNode, nodeRef, key[depth + newPrefixLength], insertMakeLeaf(value));
            return;
        }

        // Handle prefix of inner node
        if (node->prefixLength)
        {
            unsigned mismatchPos = prefixMismatch(node, key, depth, maxKeyLength);
            if (mismatchPos != node->prefixLength)
            {
                // Prefix differs, create new node
                Node4 *newNode = new Node4();
                *nodeRef = newNode;
                newNode->prefixLength = mismatchPos;
                memcpy(newNode->prefix, node->prefix, min(mismatchPos, maxPrefixLength));
                // Break up prefix
                if (node->prefixLength < maxPrefixLength)
                {
                    insertNode4(newNode, nodeRef, node->prefix[mismatchPos], node);
                    node->prefixLength -= (mismatchPos + 1);
                    memmove(node->prefix, node->prefix + mismatchPos + 1, min(node->prefixLength, maxPrefixLength));
                }
                else
                {
                    node->prefixLength -= (mismatchPos + 1);
                    uint8_t minKey[maxKeyLength];
                    loadKey(getLeafValue(minimum(node)), minKey);
                    insertNode4(newNode, nodeRef, minKey[depth + mismatchPos], node);
                    memmove(node->prefix, minKey + depth + mismatchPos + 1, min(node->prefixLength, maxPrefixLength));
                }
                insertNode4(newNode, nodeRef, key[depth + mismatchPos], insertMakeLeaf(value));
                return;
            }
            depth += node->prefixLength;
        }

        // Recurse
        Node **child = findChild(node, key[depth]);
        if (*child)
        {
            upsert(*child, child, key, depth + 1, value, maxKeyLength, insertMakeLeaf);
            return;
        }

        // Insert leaf into inner node
        Node *newNode = insertMakeLeaf(value);
        switch (node->type)
        {
        case NodeType4:
            insertNode4(static_cast<Node4 *>(node), nodeRef, key[depth], newNode);
            break;
        case NodeType16:
            insertNode16(static_cast<Node16 *>(node), nodeRef, key[depth], newNode);
            break;
        case NodeType48:
            insertNode48(static_cast<Node48 *>(node), nodeRef, key[depth], newNode);
            break;
        case NodeType256:
            insertNode256(static_cast<Node256 *>(node), nodeRef, key[depth], newNode);
            break;
        }
    }
    void insert(uint8_t key[], uintptr_t value, unsigned maxKeyLength) {
        insert(root, &root, key, 0, value, maxKeyLength);
    }
    void insert(Node *node, Node **nodeRef, uint8_t key[], unsigned depth, uintptr_t value, unsigned maxKeyLength)
    {
        // Insert the leaf value into the tree

        if (node == NULL)
        {
            *nodeRef = makeLeaf(value);
            return;
        }

        if (isLeaf(node))
        {
            // Replace leaf with Node4 and store both leaves in it
            uint8_t existingKey[maxKeyLength];
            loadKey(getLeafValue(node), existingKey);
            unsigned newPrefixLength = 0;
            while (existingKey[depth + newPrefixLength] == key[depth + newPrefixLength])
                newPrefixLength++;

            Node4 *newNode = new Node4();
            newNode->prefixLength = newPrefixLength;
            memcpy(newNode->prefix, key + depth, min(newPrefixLength, maxPrefixLength));
            *nodeRef = newNode;

            insertNode4(newNode, nodeRef, existingKey[depth + newPrefixLength], node);
            insertNode4(newNode, nodeRef, key[depth + newPrefixLength], makeLeaf(value));
            return;
        }

        // Handle prefix of inner node
        if (node->prefixLength)
        {
            unsigned mismatchPos = prefixMismatch(node, key, depth, maxKeyLength);
            if (mismatchPos != node->prefixLength)
            {
                // Prefix differs, create new node
                Node4 *newNode = new Node4();
                *nodeRef = newNode;
                newNode->prefixLength = mismatchPos;
                memcpy(newNode->prefix, node->prefix, min(mismatchPos, maxPrefixLength));
                // Break up prefix
                if (node->prefixLength < maxPrefixLength)
                {
                    insertNode4(newNode, nodeRef, node->prefix[mismatchPos], node);
                    node->prefixLength -= (mismatchPos + 1);
                    memmove(node->prefix, node->prefix + mismatchPos + 1, min(node->prefixLength, maxPrefixLength));
                }
                else
                {
                    node->prefixLength -= (mismatchPos + 1);
                    uint8_t minKey[maxKeyLength];
                    loadKey(getLeafValue(minimum(node)), minKey);
                    insertNode4(newNode, nodeRef, minKey[depth + mismatchPos], node);
                    memmove(node->prefix, minKey + depth + mismatchPos + 1, min(node->prefixLength, maxPrefixLength));
                }
                insertNode4(newNode, nodeRef, key[depth + mismatchPos], makeLeaf(value));
                return;
            }
            depth += node->prefixLength;
        }

        // Recurse
        Node **child = findChild(node, key[depth]);
        if (*child)
        {
            insert(*child, child, key, depth + 1, value, maxKeyLength);
            return;
        }

        // Insert leaf into inner node
        Node *newNode = makeLeaf(value);
        switch (node->type)
        {
        case NodeType4:
            insertNode4(static_cast<Node4 *>(node), nodeRef, key[depth], newNode);
            break;
        case NodeType16:
            insertNode16(static_cast<Node16 *>(node), nodeRef, key[depth], newNode);
            break;
        case NodeType48:
            insertNode48(static_cast<Node48 *>(node), nodeRef, key[depth], newNode);
            break;
        case NodeType256:
            insertNode256(static_cast<Node256 *>(node), nodeRef, key[depth], newNode);
            break;
        }
    }

    void insertNode4(Node4 *node, Node **nodeRef, uint8_t keyByte, Node *child)
    {
        // Insert leaf into inner node
        if (node->count < 4)
        {
            // Insert element
            unsigned pos;
            for (pos = 0; (pos < node->count) && (node->key[pos] < keyByte); pos++)
                ;
            memmove(node->key + pos + 1, node->key + pos, node->count - pos);
            memmove(node->child + pos + 1, node->child + pos, (node->count - pos) * sizeof(uintptr_t));
            node->key[pos] = keyByte;
            node->child[pos] = child;
            node->count++;
        }
        else
        {
            // Grow to Node16
            Node16 *newNode = new Node16();
            *nodeRef = newNode;
            newNode->count = 4;
            copyPrefix(node, newNode);
            for (unsigned i = 0; i < 4; i++)
                newNode->key[i] = flipSign(node->key[i]);
            memcpy(newNode->child, node->child, node->count * sizeof(uintptr_t));
            delete node;
            return insertNode16(newNode, nodeRef, keyByte, child);
        }
    }

    void insertNode16(Node16 *node, Node **nodeRef, uint8_t keyByte, Node *child)
    {
        // Insert leaf into inner node
        if (node->count < 16)
        {
            // Insert element
            uint8_t keyByteFlipped = flipSign(keyByte);
            __m128i cmp = _mm_cmplt_epi8(_mm_set1_epi8(keyByteFlipped), _mm_loadu_si128(reinterpret_cast<__m128i *>(node->key)));
            uint16_t bitfield = _mm_movemask_epi8(cmp) & (0xFFFF >> (16 - node->count));
            unsigned pos = bitfield ? ctz(bitfield) : node->count;
            memmove(node->key + pos + 1, node->key + pos, node->count - pos);
            memmove(node->child + pos + 1, node->child + pos, (node->count - pos) * sizeof(uintptr_t));
            node->key[pos] = keyByteFlipped;
            node->child[pos] = child;
            node->count++;
        }
        else
        {
            // Grow to Node48
            Node48 *newNode = new Node48();
            *nodeRef = newNode;
            memcpy(newNode->child, node->child, node->count * sizeof(uintptr_t));
            for (unsigned i = 0; i < node->count; i++)
                newNode->childIndex[flipSign(node->key[i])] = i;
            copyPrefix(node, newNode);
            newNode->count = node->count;
            delete node;
            return insertNode48(newNode, nodeRef, keyByte, child);
        }
    }

    void insertNode48(Node48 *node, Node **nodeRef, uint8_t keyByte, Node *child)
    {
        // Insert leaf into inner node
        if (node->count < 48)
        {
            // Insert element
            unsigned pos = node->count;
            if (node->child[pos])
                for (pos = 0; node->child[pos] != NULL; pos++)
                    ;
            node->child[pos] = child;
            node->childIndex[keyByte] = pos;
            node->count++;
        }
        else
        {
            // Grow to Node256
            Node256 *newNode = new Node256();
            for (unsigned i = 0; i < 256; i++)
                if (node->childIndex[i] != 48)
                    newNode->child[i] = node->child[node->childIndex[i]];
            newNode->count = node->count;
            copyPrefix(node, newNode);
            *nodeRef = newNode;
            delete node;
            return insertNode256(newNode, nodeRef, keyByte, child);
        }
    }

    void insertNode256(Node256 *node, Node **nodeRef, uint8_t keyByte, Node *child)
    {
        // Insert leaf into inner node
        node->count++;
        node->child[keyByte] = child;
    }
    void destroyNode(Node *node)
    {
        if (node == nullptr || isLeaf(node))
            return;
        switch (node->type)
        {
        case NodeType4:
            destroyNode4(static_cast<Node4 *>(node));
            break;
        case NodeType16:
            destroyNode16(static_cast<Node16 *>(node));
            break;
        case NodeType48:
            destroyNode48(static_cast<Node48 *>(node));
            break;
        case NodeType256:
            destroyNode256(static_cast<Node256 *>(node));
            break;
        }
    }

    void destroyNode4(Node4 *node)
    {
        for (int i = 0; i < node->count; ++i)
        {
            destroyNode(node->child[i]);
        }
        delete node;
    }

    void destroyNode16(Node16 *node)
    {
        for (int i = 0; i < node->count; ++i)
        {
            destroyNode(node->child[i]);
        }
        delete node;
    }

    void destroyNode48(Node48 *node)
    {
        for (int i = 0; i < 256; ++i)
        {
            if (node->childIndex[i] != emptyMarker)
                destroyNode(node->child[node->childIndex[i]]);
        }
        delete node;
    }

    void destroyNode256(Node256 *node)
    {
        for (int i = 0; i < 256; ++i)
        {
            if (node->child[i] != nullptr)
                destroyNode(node->child[i]);
        }
        delete node;
    }
    template <class Key,
              class KeyLenFunc,
              class KeyByteExtractFunc>
    static art_tree* bulkLoadCreate(std::function<void(uintptr_t, uint8_t[])> loadKey, std::function<void(uintptr_t, uint8_t[])> loadLowerBoundKey, std::vector<std::pair<Key, uintptr_t>> &kvs, int lo, int hi, int depth, KeyByteExtractFunc ExtractKeyByte, KeyLenFunc KeyLen) {
        auto tree = new art_tree(loadKey, loadLowerBoundKey);
        tree->root = bulkLoad(kvs, lo, hi, depth, ExtractKeyByte, KeyLen);
        return tree;
    }
    template <class Key,
              class KeyLenFunc,
              class KeyByteExtractFunc>
    static Node *bulkLoad(std::vector<std::pair<Key, uintptr_t>> &kvs, int lo, int hi, int depth, KeyByteExtractFunc ExtractKeyByte, KeyLenFunc KeyLen)
    {
        if (hi - lo + 1 == 1)
            return makeLeaf(kvs[lo].second);
        int childCnt;
        int singlePart;
        std::vector<std::pair<int, int>> parts(256, std::make_pair(-1, -1));
        auto partition = [&]() {
            parts = std::vector<std::pair<int, int>>(256, std::make_pair(-1, -1));
            childCnt = 0;
            singlePart = -1;
            for (int i = lo; i <= hi; ++i)
            {
                auto &kv = kvs[i];
                uint8_t byte = ExtractKeyByte(kv.first, depth);
                if (parts[byte].first == -1)
                {
                    parts[byte].first = parts[byte].second = i;
                    childCnt++;
                    singlePart = byte;
                }
                else
                {
                    parts[byte].second = i;
                }
            }
        };

        partition();

        Node nmeta(0);

        if (childCnt == 1)
        {
            // only one partition, compute the longest common prefix
            int si = parts[singlePart].first, ei = parts[singlePart].second;
            bool diff = false;
            int pos = 0;
            while (diff == false)
            {
                uint8_t byte = ExtractKeyByte(kvs[si].first, depth + pos);
                for (int i = si + 1; i <= ei && diff == false; ++i)
                {
                    if (depth + pos >= KeyLen(kvs[i].first) || ExtractKeyByte(kvs[i].first, depth + pos) != byte)
                    {
                        diff = true;
                    }
                }
                if (diff == false)
                {
                    if (pos < maxPrefixLength)
                    {
                        nmeta.prefix[pos] = byte;
                    }
                    ++pos;
                    ++nmeta.prefixLength;
                }
            }
            if (nmeta.prefixLength > 0)
            {
                depth += nmeta.prefixLength;
                // repartition
                partition();
            }
        }
        Node *n = nullptr;
        if (childCnt <= 4)
        {
            n = new Node4;
        }
        else if (childCnt <= 16)
        {
            n = new Node16;
        }
        else if (childCnt <= 48)
        {
            n = new Node48;
        }
        else
        {
            n = new Node256;
        }
        n->prefixLength = nmeta.prefixLength;
        if (n->prefixLength)
        {
            memcpy(n->prefix, nmeta.prefix, n->prefixLength);
        }

        for (int i = 0; i < 256; ++i)
        {
            uint8_t byte = i;
            if (parts[byte].first == -1)
                continue;
            int partitionSize = parts[byte].second - parts[byte].first + 1;
            Node *child = bulkLoad(kvs, parts[byte].first, parts[byte].second, depth + 1, ExtractKeyByte, KeyLen);
            switch (n->type)
            {
            case NodeType4:
            {
                Node4 *n4 = (Node4 *)n;
                n4->key[n4->count] = byte;
                n4->child[n4->count] = child;
                ++n4->count;
            }
            break;
            case NodeType16:
            {
                Node16 *n16 = (Node16 *)n;
                n16->key[n16->count] = flipSign(byte);
                n16->child[n16->count] = child;
                ++n16->count;
            }
            break;
            case NodeType48:
            {
                Node48 *n48 = (Node48 *)n;
                int pos = n48->count;
                n48->child[pos] = child;
                n48->childIndex[byte] = pos;
                ++n48->count;
            }
            break;
            case NodeType256:
            {
                Node256 *n256 = (Node256 *)n;
                n256->child[byte] = child;
                ++n256->count;
            }
            break;
            }
        }
        return n;
    }
};
} // namespace Art