#include "art_idx.hpp"

namespace ART_IDX {
static thread_local std::vector<Node *> ancestors;
Node *art_tree::lowerBound(Node *node, uint8_t key[], unsigned keyLength, unsigned depth, unsigned maxKeyLength, bool &pref)
{
        // Find the node with a matching key, optimistic version
        ancestors.clear();
        
        while (node != NULL)
        {
            ancestors.push_back(node);
            if (isLeaf(node))
            {
                return node;
            }

            int nodePrefixLen = node->prefixLength;
            if (node->prefixLength)
            {
                unsigned pos = 0;
                if (node->prefixLength <= maxPrefixLength)
                {
                    for (; pos < node->prefixLength && key[depth + pos] == node->prefix[pos]; pos++)
                        ;
                    if (pos < node->prefixLength)
                    {
                        if (node->prefix[pos] < key[depth + pos])
                        {
                            goto backtrack;
                        }
                        else
                        { // node->prefix[pos] > key[depth + pos]
                            return minimum(node);
                        }
                    }
                }
                else
                {
                    for (; pos < maxPrefixLength && key[depth + pos] == node->prefix[pos]; pos++)
                        ;
                    if (pos >= maxPrefixLength)
                    {
                        uint8_t leafKey[maxKeyLength];
                        auto minLeaf = minimum(node);
                        loadLowerBoundKey(getLeafValue(minLeaf), leafKey);
                        for (; pos < node->prefixLength && key[depth + pos] == leafKey[depth + pos]; ++pos)
                            ;
                        if (pos < node->prefixLength)
                        {
                            if (leafKey[depth + pos] < key[depth + pos])
                            {
                                goto backtrack;
                            }
                            else
                            { // leafKey[depth + pos] > key[depth + pos]
                                pref = true;
                                return minLeaf;
                            }
                        }
                    }
                }
                depth += node->prefixLength;
            }

            bool equal = false;
            node = findChildLowerBound(node, key[depth], equal);
            if (node == nullptr)
            {
                depth -= nodePrefixLen;
                goto backtrack;
            }
            if (equal == false)
            {
                return minimum(node);
            }
            depth++;
        }

    backtrack:
        if (ancestors.empty() == false)
        {
            ancestors.pop_back();
        }
        while (ancestors.empty() == false)
        {
            auto node = ancestors.back();
            --depth;
            if (key[depth] < 255)
            {
                bool equal = false;
                auto n = findChildLowerBound(node, key[depth] + 1, equal);
                if (n)
                {
                    return minimum(n);
                }
            }
            depth -= node->prefixLength;
            ancestors.pop_back();
        }
        assert(depth == 0);
        return NULL;
}
}