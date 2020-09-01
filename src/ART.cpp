/*
  Adaptive Radix Tree
  Viktor Leis, 2012
  leis@in.tum.de

  Modified by Huanchen Zhang, 2016
 */

#include "../include/ART.hpp"

#define LOADKEY_INT 1

//********************************************************************
// ARTIter
//********************************************************************


#ifdef LOADKEY_INT
static inline void defaultLoadKey(uintptr_t tid,uint8_t key[]) {
    // Store the key of the tuple into the key vector
    // Implementation is database specific
    reinterpret_cast<uint64_t*>(key)[0]=__builtin_bswap64(tid);
}
#else
static inline void defaultLoadKey(uintptr_t tid,uint8_t key[]) {
    memcpy(reinterpret_cast<void*>(key), (const void*)tid, key_length);
}
#endif

inline Node** ART::findChild(Node* n,uint8_t keyByte) {
    // Find the next child for the keyByte
    switch (n->type) {
    case NodeType4: {
	Node4* node=static_cast<Node4*>(n);
	for (unsigned i=0;i<node->count;i++)
	    if (node->key[i]==keyByte)
		return &node->child[i];
	return &nullNode;
    }
    case NodeType16: {
	Node16* node=static_cast<Node16*>(n);
	__m128i cmp=_mm_cmpeq_epi8(_mm_set1_epi8(flipSign(keyByte)),_mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
	unsigned bitfield=_mm_movemask_epi8(cmp)&((1<<node->count)-1);
	if (bitfield)
	    return &node->child[ctz(bitfield)]; else
	    return &nullNode;
    }
    case NodeType48: {
	Node48* node=static_cast<Node48*>(n);
	if (node->childIndex[keyByte]!=emptyMarker)
	    return &node->child[node->childIndex[keyByte]]; else
	    return &nullNode;
    }
    case NodeType256: {
	Node256* node=static_cast<Node256*>(n);
	return &(node->child[keyByte]);
    }
    }
    throw; // Unreachable
}

inline Node* ART::minimum(Node* node) {
    // Find the leaf with smallest key
    if (!node)
	return NULL;

    if (isLeaf(node))
	return node;

    switch (node->type) {
    case NodeType4: {
	Node4* n=static_cast<Node4*>(node);
	return minimum(n->child[0]);
    }
    case NodeType16: {
	Node16* n=static_cast<Node16*>(node);
	return minimum(n->child[0]);
    }
    case NodeType48: {
	Node48* n=static_cast<Node48*>(node);
	unsigned pos=0;
	while (n->childIndex[pos]==emptyMarker)
	    pos++;
	return minimum(n->child[n->childIndex[pos]]);
    }
    case NodeType256: {
	Node256* n=static_cast<Node256*>(node);
	unsigned pos=0;
	while (!n->child[pos])
	    pos++;
	return minimum(n->child[pos]);
    }
    }
    throw; // Unreachable
}

inline bool ART::leafMatches(Node* leaf,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
    // Check if the key of the leaf is equal to the searched key
    if (depth!=keyLength) {
	uint8_t leafKey[maxKeyLength];
	loadKey(getLeafValue(leaf),leafKey);
	for (unsigned i=depth;i<keyLength;i++)
	    if (leafKey[i]!=key[i])
		return false;
    }
    return true;
}

inline unsigned ART::prefixMismatch(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength) {
    // Compare the key with the prefix of the node, return the number matching bytes
    unsigned pos;
    if (node->prefixLength>maxPrefixLength) {
	for (pos=0;pos<maxPrefixLength;pos++)
	    if (key[depth+pos]!=node->prefix[pos])
		return pos;
	uint8_t minKey[maxKeyLength];
	loadKey(getLeafValue(minimum(node)),minKey);
	for (;pos<node->prefixLength;pos++)
	    if (key[depth+pos]!=minKey[depth+pos])
		return pos;
    } else {
	for (pos=0;pos<node->prefixLength;pos++)
	    if (key[depth+pos]!=node->prefix[pos])
		return pos;
    }
    return pos;
}

Node** ART::lookupRef(Node** nodeRef, Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
	bool skippedPrefix=false; // Did we optimistically skip some prefix without checking it?
    while (node!=NULL) {
	if (isLeaf(node)) {
	    if (!skippedPrefix&&depth==keyLength) // No check required
		return nodeRef;

	    if (depth!=keyLength) {
		// Check leaf
		uint8_t leafKey[maxKeyLength];
		loadKey(getLeafValue(node),leafKey);
		for (unsigned i=(skippedPrefix?0:depth);i<keyLength;i++)
		    if (leafKey[i]!=key[i])
			return NULL;
	    }
	    return nodeRef;
	}

	if (node->prefixLength) {
	    if (node->prefixLength<maxPrefixLength) {
		for (unsigned pos=0;pos<node->prefixLength;pos++)
		    if (key[depth+pos]!=node->prefix[pos])
			return NULL;
	    } else
		skippedPrefix=true;
	    depth+=node->prefixLength;
	}

	nodeRef=findChild(node,key[depth]);
	node = *nodeRef;
	depth++;
    }

    return NULL;
}

inline Node* ART::lookup(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
    // Find the node with a matching key, optimistic version

    bool skippedPrefix=false; // Did we optimistically skip some prefix without checking it?

    while (node!=NULL) {
	if (isLeaf(node)) {
	    if (!skippedPrefix&&depth==keyLength) // No check required
		return node;

	    if (depth!=keyLength) {
		// Check leaf
		uint8_t leafKey[maxKeyLength];
		loadKey(getLeafValue(node),leafKey);
		for (unsigned i=(skippedPrefix?0:depth);i<keyLength;i++)
		    if (leafKey[i]!=key[i])
			return NULL;
	    }
	    return node;
	}

	if (node->prefixLength) {
	    if (node->prefixLength<maxPrefixLength) {
		for (unsigned pos=0;pos<node->prefixLength;pos++)
		    if (key[depth+pos]!=node->prefix[pos])
			return NULL;
	    } else
		skippedPrefix=true;
	    depth+=node->prefixLength;
	}

	node=*findChild(node,key[depth]);
	depth++;
    }

    return NULL;
}

inline Node* ART::lookupPessimistic(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
    // Find the node with a matching key, alternative pessimistic version

    while (node!=NULL) {
	if (isLeaf(node)) {
	    if (leafMatches(node,key,keyLength,depth,maxKeyLength))
		return node;
	    return NULL;
	}

	if (prefixMismatch(node,key,depth,maxKeyLength)!=node->prefixLength)
	    return NULL; else
	    depth+=node->prefixLength;

	node=*findChild(node,key[depth]);
	depth++;
    }

    return NULL;
}

inline unsigned ART::min(unsigned a,unsigned b) {
    // Helper function
    return (a<b)?a:b;
}

inline void ART::copyPrefix(Node* src,Node* dst) {
    // Helper function that copies the prefix from the source to the destination node
    dst->prefixLength=src->prefixLength;
    memcpy(dst->prefix,src->prefix,min(src->prefixLength,maxPrefixLength));
}

void ART::upsert(Node* node,Node** nodeRef,uint8_t key[],unsigned depth,uintptr_t value,unsigned maxKeyLength, std::function<Node*(uintptr_t)> insertMakeLeaf) {
   // Insert the leaf value into the tree

	if (node==NULL) {
		*nodeRef=insertMakeLeaf(value);
		return;
	} 

	if (isLeaf(node)) {
		// Replace leaf with Node4 and store both leaves in it
		uint8_t existingKey[maxKeyLength];
		loadKey(getLeafValue(node),existingKey);
		unsigned newPrefixLength=0;
		while (depth+newPrefixLength < maxKeyLength && existingKey[depth+newPrefixLength]==key[depth+newPrefixLength])
			newPrefixLength++;
		if (depth+newPrefixLength == maxKeyLength) {
			upsertFunc(nodeRef, value);
			return;
		}
		Node4* newNode=new Node4();
		newNode->prefixLength=newPrefixLength;
		memcpy(newNode->prefix,key+depth,min(newPrefixLength,maxPrefixLength));
		*nodeRef=newNode;

		insertNode4(newNode,nodeRef,existingKey[depth+newPrefixLength],node);
		insertNode4(newNode,nodeRef,key[depth+newPrefixLength], insertMakeLeaf(value));
		return;
	}

	// Handle prefix of inner node
	if (node->prefixLength) {
		unsigned mismatchPos=prefixMismatch(node,key,depth,maxKeyLength);
		if (mismatchPos!=node->prefixLength) {
			// Prefix differs, create new node
			Node4* newNode=new Node4();
			*nodeRef=newNode;
			newNode->prefixLength=mismatchPos;
			memcpy(newNode->prefix,node->prefix,min(mismatchPos,maxPrefixLength));
			// Break up prefix
			if (node->prefixLength<maxPrefixLength) {
			insertNode4(newNode,nodeRef,node->prefix[mismatchPos],node);
			node->prefixLength-=(mismatchPos+1);
			memmove(node->prefix,node->prefix+mismatchPos+1,min(node->prefixLength,maxPrefixLength));
			} else {
			node->prefixLength-=(mismatchPos+1);
			uint8_t minKey[maxKeyLength];
			loadKey(getLeafValue(minimum(node)),minKey);
			insertNode4(newNode,nodeRef,minKey[depth+mismatchPos],node);
			memmove(node->prefix,minKey+depth+mismatchPos+1,min(node->prefixLength,maxPrefixLength));
			}
			insertNode4(newNode,nodeRef,key[depth+mismatchPos],insertMakeLeaf(value));
			return;
		}
		depth+=node->prefixLength;
	}

	// Recurse
	Node** child=findChild(node,key[depth]);
	if (*child) {
		upsert(*child,child,key,depth+1,value,maxKeyLength,insertMakeLeaf);
		return;
	}

	// Insert leaf into inner node
	Node* newNode=insertMakeLeaf(value);
	switch (node->type) {
		case NodeType4: insertNode4(static_cast<Node4*>(node),nodeRef,key[depth],newNode); break;
		case NodeType16: insertNode16(static_cast<Node16*>(node),nodeRef,key[depth],newNode); break;
		case NodeType48: insertNode48(static_cast<Node48*>(node),nodeRef,key[depth],newNode); break;
		case NodeType256: insertNode256(static_cast<Node256*>(node),nodeRef,key[depth],newNode); break;
	}
}

inline void ART::insert(Node* node,Node** nodeRef,uint8_t key[],unsigned depth,uintptr_t value,unsigned maxKeyLength) {
    // Insert the leaf value into the tree

    if (node==NULL) {
	*nodeRef=makeLeaf(value);
	return;
    }

    if (isLeaf(node)) {
	// Replace leaf with Node4 and store both leaves in it
	uint8_t existingKey[maxKeyLength];
	loadKey(getLeafValue(node),existingKey);
	unsigned newPrefixLength=0;
	while (existingKey[depth+newPrefixLength]==key[depth+newPrefixLength])
	    newPrefixLength++;

	Node4* newNode=new Node4();
	memory += sizeof(Node4); //h
	node4_count++; //h
	newNode->prefixLength=newPrefixLength;
	memcpy(newNode->prefix,key+depth,min(newPrefixLength,maxPrefixLength));
	*nodeRef=newNode;

	insertNode4(newNode,nodeRef,existingKey[depth+newPrefixLength],node);
	insertNode4(newNode,nodeRef,key[depth+newPrefixLength],makeLeaf(value));
	num_items++; //h
	return;
    }

    // Handle prefix of inner node
    if (node->prefixLength) {
	unsigned mismatchPos=prefixMismatch(node,key,depth,maxKeyLength);
	if (mismatchPos!=node->prefixLength) {
	    // Prefix differs, create new node
	    Node4* newNode=new Node4();
	    memory += sizeof(Node4); //h
	    node4_count++; //h
	    *nodeRef=newNode;
	    newNode->prefixLength=mismatchPos;
	    memcpy(newNode->prefix,node->prefix,min(mismatchPos,maxPrefixLength));
	    // Break up prefix
	    if (node->prefixLength<maxPrefixLength) {
		insertNode4(newNode,nodeRef,node->prefix[mismatchPos],node);
		node->prefixLength-=(mismatchPos+1);
		memmove(node->prefix,node->prefix+mismatchPos+1,min(node->prefixLength,maxPrefixLength));
	    } else {
		node->prefixLength-=(mismatchPos+1);
		uint8_t minKey[maxKeyLength];
		loadKey(getLeafValue(minimum(node)),minKey);
		insertNode4(newNode,nodeRef,minKey[depth+mismatchPos],node);
		memmove(node->prefix,minKey+depth+mismatchPos+1,min(node->prefixLength,maxPrefixLength));
	    }
	    insertNode4(newNode,nodeRef,key[depth+mismatchPos],makeLeaf(value));
	    num_items++; //h
	    return;
	}
	depth+=node->prefixLength;
    }

    // Recurse
    Node** child=findChild(node,key[depth]);
    if (*child) {
	insert(*child,child,key,depth+1,value,maxKeyLength);
	return;
    }

    // Insert leaf into inner node
    Node* newNode=makeLeaf(value);
    switch (node->type) {
    case NodeType4: insertNode4(static_cast<Node4*>(node),nodeRef,key[depth],newNode); break;
    case NodeType16: insertNode16(static_cast<Node16*>(node),nodeRef,key[depth],newNode); break;
    case NodeType48: insertNode48(static_cast<Node48*>(node),nodeRef,key[depth],newNode); break;
    case NodeType256: insertNode256(static_cast<Node256*>(node),nodeRef,key[depth],newNode); break;
    }
    num_items++; //h
}

inline void ART::insertNode4(Node4* node,Node** nodeRef,uint8_t keyByte,Node* child) {
    // Insert leaf into inner node
    if (node->count<4) {
	// Insert element
	unsigned pos;
	for (pos=0;(pos<node->count)&&(node->key[pos]<keyByte);pos++);
	memmove(node->key+pos+1,node->key+pos,node->count-pos);
	memmove(node->child+pos+1,node->child+pos,(node->count-pos)*sizeof(uintptr_t));
	node->key[pos]=keyByte;
	node->child[pos]=child;
	node->count++;
    } else {
	// Grow to Node16
	Node16* newNode=new Node16();
	memory += sizeof(Node16); //h
	node16_count++; //h
	*nodeRef=newNode;
	newNode->count=4;
	copyPrefix(node,newNode);
	for (unsigned i=0;i<4;i++)
	    newNode->key[i]=flipSign(node->key[i]);
	memcpy(newNode->child,node->child,node->count*sizeof(uintptr_t));
	delete node;
	memory -= sizeof(Node4); //h
	node4_count--; //h
	return insertNode16(newNode,nodeRef,keyByte,child);
    }
}

inline void ART::insertNode16(Node16* node,Node** nodeRef,uint8_t keyByte,Node* child) {
    // Insert leaf into inner node
    if (node->count<16) {
	// Insert element
	uint8_t keyByteFlipped=flipSign(keyByte);
	__m128i cmp=_mm_cmplt_epi8(_mm_set1_epi8(keyByteFlipped),_mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
	uint16_t bitfield=_mm_movemask_epi8(cmp)&(0xFFFF>>(16-node->count));
	unsigned pos=bitfield?ctz(bitfield):node->count;
	memmove(node->key+pos+1,node->key+pos,node->count-pos);
	memmove(node->child+pos+1,node->child+pos,(node->count-pos)*sizeof(uintptr_t));
	node->key[pos]=keyByteFlipped;
	node->child[pos]=child;
	node->count++;
    } else {
	// Grow to Node48
	Node48* newNode=new Node48();
	memory += sizeof(Node48); //h
	node48_count++; //h
	*nodeRef=newNode;
	memcpy(newNode->child,node->child,node->count*sizeof(uintptr_t));
	for (unsigned i=0;i<node->count;i++)
	    newNode->childIndex[flipSign(node->key[i])]=i;
	copyPrefix(node,newNode);
	newNode->count=node->count;
	delete node;
	memory -= sizeof(Node16); //h
	node16_count--; //h
	return insertNode48(newNode,nodeRef,keyByte,child);
    }
}

inline void ART::insertNode48(Node48* node,Node** nodeRef,uint8_t keyByte,Node* child) {
    // Insert leaf into inner node
    if (node->count<48) {
	// Insert element
	unsigned pos=node->count;
	if (node->child[pos])
	    for (pos=0;node->child[pos]!=NULL;pos++);
	node->child[pos]=child;
	node->childIndex[keyByte]=pos;
	node->count++;
    } else {
	// Grow to Node256
	Node256* newNode=new Node256();
	memory += sizeof(Node256); //h
	node256_count++; //h
	for (unsigned i=0;i<256;i++)
	    if (node->childIndex[i]!=48)
		newNode->child[i]=node->child[node->childIndex[i]];
	newNode->count=node->count;
	copyPrefix(node,newNode);
	*nodeRef=newNode;
	delete node;
	memory -= sizeof(Node48); //h
	node48_count--; //h
	return insertNode256(newNode,nodeRef,keyByte,child);
    }
}

inline void ART::insertNode256(Node256* node,Node** nodeRef,uint8_t keyByte,Node* child) {
    // Insert leaf into inner node
    node->count++;
    node->child[keyByte]=child;
}


inline void ART::erase(Node* node,Node** nodeRef,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
    // Delete a leaf from a tree

    if (!node)
	return;

    if (isLeaf(node)) {
	// Make sure we have the right leaf
	if (leafMatches(node,key,keyLength,depth,maxKeyLength))
	    *nodeRef=NULL;
	return;
    }

    // Handle prefix
    if (node->prefixLength) {
	if (prefixMismatch(node,key,depth,maxKeyLength)!=node->prefixLength)
	    return;
	depth+=node->prefixLength;
    }

    Node** child=findChild(node,key[depth]);
    if (isLeaf(*child)&&leafMatches(*child,key,keyLength,depth,maxKeyLength)) {
	// Leaf found, delete it in inner node
	switch (node->type) {
	case NodeType4: eraseNode4(static_cast<Node4*>(node),nodeRef,child); break;
	case NodeType16: eraseNode16(static_cast<Node16*>(node),nodeRef,child); break;
	case NodeType48: eraseNode48(static_cast<Node48*>(node),nodeRef,key[depth]); break;
	case NodeType256: eraseNode256(static_cast<Node256*>(node),nodeRef,key[depth]); break;
	}
    } else {
	//Recurse
	erase(*child,child,key,keyLength,depth+1,maxKeyLength);
    }
}

inline void ART::eraseNode4(Node4* node,Node** nodeRef,Node** leafPlace) {
    // Delete leaf from inner node
    unsigned pos=leafPlace-node->child;
    memmove(node->key+pos,node->key+pos+1,node->count-pos-1);
    memmove(node->child+pos,node->child+pos+1,(node->count-pos-1)*sizeof(uintptr_t));
    node->count--;

    if (node->count==1) {
	// Get rid of one-way node
	Node* child=node->child[0];
	if (!isLeaf(child)) {
	    // Concantenate prefixes
	    unsigned l1=node->prefixLength;
	    if (l1<maxPrefixLength) {
		node->prefix[l1]=node->key[0];
		l1++;
	    }
	    if (l1<maxPrefixLength) {
		unsigned l2=min(child->prefixLength,maxPrefixLength-l1);
		memcpy(node->prefix+l1,child->prefix,l2);
		l1+=l2;
	    }
	    // Store concantenated prefix
	    memcpy(child->prefix,node->prefix,min(l1,maxPrefixLength));
	    child->prefixLength+=node->prefixLength+1;
	}
	*nodeRef=child;
	delete node;
	memory -= sizeof(Node4); //h
	node4_count--; //h
    }
}

inline void ART::eraseNode16(Node16* node,Node** nodeRef,Node** leafPlace) {
    // Delete leaf from inner node
    unsigned pos=leafPlace-node->child;
    memmove(node->key+pos,node->key+pos+1,node->count-pos-1);
    memmove(node->child+pos,node->child+pos+1,(node->count-pos-1)*sizeof(uintptr_t));
    node->count--;

    if (node->count==3) {
	// Shrink to Node4
	Node4* newNode=new Node4();
	memory += sizeof(Node4); //h
	node4_count++; //h
	newNode->count=node->count;
	copyPrefix(node,newNode);
	for (unsigned i=0;i<4;i++)
	    newNode->key[i]=flipSign(node->key[i]);
	memcpy(newNode->child,node->child,sizeof(uintptr_t)*4);
	*nodeRef=newNode;
	delete node;
	memory -= sizeof(Node16); //h
	node16_count--; //h
    }
}

inline void ART::eraseNode48(Node48* node,Node** nodeRef,uint8_t keyByte) {
    // Delete leaf from inner node
    node->child[node->childIndex[keyByte]]=NULL;
    node->childIndex[keyByte]=emptyMarker;
    node->count--;

    if (node->count==12) {
	// Shrink to Node16
	Node16 *newNode=new Node16();
	memory += sizeof(Node16); //h
	node16_count++; //h
	*nodeRef=newNode;
	copyPrefix(node,newNode);
	for (unsigned b=0;b<256;b++) {
	    if (node->childIndex[b]!=emptyMarker) {
		newNode->key[newNode->count]=flipSign(b);
		newNode->child[newNode->count]=node->child[node->childIndex[b]];
		newNode->count++;
	    }
	}
	delete node;
	memory -= sizeof(Node48); //h
	node48_count--; //h
    }
}

inline void ART::eraseNode256(Node256* node,Node** nodeRef,uint8_t keyByte) {
    // Delete leaf from inner node
    node->child[keyByte]=NULL;
    node->count--;

    if (node->count==37) {
	// Shrink to Node48
	Node48 *newNode=new Node48();
	memory += sizeof(Node48); //h
	node48_count++; //h
	*nodeRef=newNode;
	copyPrefix(node,newNode);
	for (unsigned b=0;b<256;b++) {
	    if (node->child[b]) {
		newNode->childIndex[b]=newNode->count;
		newNode->child[newNode->count]=node->child[b];
		newNode->count++;
	    }
	}
	delete node;
	memory -= sizeof(Node256); //h
	node256_count--; //h
    }
}

inline int ART::CompareToPrefix(Node* node,uint8_t key[],unsigned depth,unsigned maxKeyLength) {
    unsigned pos;
    if (node->prefixLength>maxPrefixLength) {
	for (pos=0;pos<maxPrefixLength;pos++) {
	    if (key[depth+pos]!=node->prefix[pos]) {
		if (key[depth+pos]>node->prefix[pos])
		    return 1;
		else
		    return -1;
	    }
	}
	uint8_t minKey[maxKeyLength];
	loadKey(getLeafValue(minimum(node)),minKey);
	for (;pos<node->prefixLength;pos++) {
	    if (key[depth+pos]!=minKey[depth+pos]) {
		if (key[depth+pos]>minKey[depth+pos])
		    return 1;
		else
		    return -1;
	    }
	}
    } else {
	for (pos=0;pos<node->prefixLength;pos++) {
	    if (key[depth+pos]!=node->prefix[pos]) {
		if (key[depth+pos]>node->prefix[pos])
		    return 1;
		else
		    return -1;
	    }
	}
    }
    return 0;
}

inline Node* ART::findChild_recordPath(Node* n, uint8_t keyByte, ARTIter* iter) {
    NodeCursor nc;
    nc.node = n;
    switch (n->type) {
    case NodeType4: {
	Node4* node=static_cast<Node4*>(n);
	for (unsigned i=0;i<node->count;i++) {
	    if (node->key[i]>=keyByte) {
		nc.cursor = i;
		iter->node_stack.push_back(nc);
		if (node->key[i]==keyByte)
		    return node->child[i];
		else
		    return iter->minimum_recordPath(node->child[i]);
	    }
	}
	iter->node_stack.pop_back();
	//return iter->minimum_recordPath(iter->nextSlot());
	return iter->nextLeaf();
    }
    case NodeType16: {
	Node16* node=static_cast<Node16*>(n);
	for (unsigned i=0;i<node->count;i++) {
	    //if (node->key[i]>=keyByte) { THE BUGGGGGG!
	    if (node->key[i]>=flipSign(keyByte)) {
		nc.cursor = i;
		iter->node_stack.push_back(nc);
		//if (node->key[i]==keyByte) THE BUGGGGGG!
		if (node->key[i]==flipSign(keyByte))
		    return node->child[i];
		else
		    return iter->minimum_recordPath(node->child[i]);
	    }
	}
	iter->node_stack.pop_back();
	//return iter->minimum_recordPath(iter->nextSlot());
	return iter->nextLeaf();
    }
    case NodeType48: {
	Node48* node=static_cast<Node48*>(n);
	if (node->childIndex[keyByte]!=emptyMarker) {
	    nc.cursor = keyByte;
	    iter->node_stack.push_back(nc);
	    return node->child[node->childIndex[keyByte]]; 
	}
	else {
	    for (unsigned i=keyByte; i<256; i++) {
		if (node->childIndex[i]!=emptyMarker) {
		    nc.cursor = i;
		    iter->node_stack.push_back(nc);
		    return node->child[node->childIndex[i]]; 
		}	  
	    }
	    iter->node_stack.pop_back();
	    //return iter->minimum_recordPath(iter->nextSlot());
	    return iter->nextLeaf();
	}
    }
    case NodeType256: {
	Node256* node=static_cast<Node256*>(n);
	if (node->child[keyByte]!=NULL) {
	    nc.cursor = keyByte;
	    iter->node_stack.push_back(nc);
	    return node->child[keyByte];
	}
	else {
	    for (unsigned i=keyByte; i<256; i++) {
		if (node->child[i]!=NULL) {
		    nc.cursor = i;
		    iter->node_stack.push_back(nc);
		    return node->child[i]; 
		}	  
	    }
	    iter->node_stack.pop_back();
	    //return iter->minimum_recordPath(iter->nextSlot());
	    return iter->nextLeaf();
	}
    }
    }
    throw; // Unreachable
}

inline Node* ART::lower_bound(Node* node,uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength, ARTIter *iter) {
    iter->node_stack.clear();
    while (node!=NULL) {
	if (isLeaf(node)) {
	    return node;
	}

	int ctp = CompareToPrefix(node,key,depth,maxKeyLength);
	depth+=node->prefixLength;

	if (ctp > 0) {
	    iter->node_stack.pop_back();
	    //return iter->minimum_recordPath(iter->nextSlot());
	    return iter->nextLeaf();
	}
	else if (ctp < 0) {
	    return iter->minimum_recordPath(node);
	}

	node = findChild_recordPath(node, key[depth], iter);
	depth++;
    }

    return NULL;
}

ART::ART() 
    : root(NULL), key_length(8), memory(0), num_items(0), 
      node4_count(0), node16_count(0), node48_count(0), node256_count(0), nullNode(NULL), loadKey(defaultLoadKey)
{}

ART::ART(std::function<void(uintptr_t,uint8_t[])> loadKey, std::function<void(Node**, uintptr_t)> upsertFunc) 
    : root(NULL), key_length(8), memory(0), num_items(0), 
      node4_count(0), node16_count(0), node48_count(0), node256_count(0), nullNode(NULL), loadKey(loadKey), upsertFunc(upsertFunc)
{}

void ART::load(vector<string> &keys, vector<uint64_t> &values, unsigned maxKeyLength) {
    uint8_t key[maxKeyLength];
    for (int i = 0; i < keys.size(); i++) {
	loadKey((uintptr_t)keys[i].c_str(), key);
	insert(key, values[i], key_length);
    }
}

void ART::load(vector<uint64_t> &keys, vector<uint64_t> &values) {
    uint8_t key[8];
    for (int i = 0; i < keys.size(); i++) {
	loadKey(keys[i], key);
	insert(key, values[i], key_length);
    }
}

void ART::insert(uint8_t key[], uintptr_t value, unsigned maxKeyLength) {
    insert(root, &root, key, 0, value, maxKeyLength);
}

void ART::upsert(uint8_t key[], uintptr_t value, unsigned maxKeyLength, std::function<Node*(uintptr_t)> insertMakeLeaf) {
	upsert(root, &root, key, 0, value, maxKeyLength, insertMakeLeaf);
}

uint64_t ART::lookup(uint8_t key[], unsigned keyLength, unsigned maxKeyLength) {
    Node* leaf = lookup(root, key, keyLength, 0, maxKeyLength);
    if (isLeaf(leaf))
	return getLeafValue(leaf);
    return (uint64_t)0;
}

uint64_t ART::lookup(uint64_t key64) {
    uint8_t key[8];
    loadKey(key64, key);
    Node* leaf = lookup(root, key, 8, 0, 8);
    if (isLeaf(leaf))
	return getLeafValue(leaf);
    return (uint64_t)0;
}

Node** ART::lookupRef(uint8_t key[],unsigned keyLength,unsigned depth,unsigned maxKeyLength) {
	return lookupRef(&root, root, key, keyLength, 0, maxKeyLength);
}

bool ART::lower_bound(uint8_t key[], unsigned keyLength, unsigned maxKeyLength, ARTIter* iter) {
    Node* leaf = lower_bound(root, key, keyLength, 0, maxKeyLength, iter);
    if (isLeaf(leaf)) {
	iter->val = (uint64_t)getLeafValue(leaf);
	loadKey(iter->val, (uint8_t*)&iter->key64);
	iter->nextLeaf();
	return true;
    }
    else {
	iter->val = 0;
	return false;
    }
}

bool ART::lower_bound(uint64_t key64, ARTIter* iter) {
    uint8_t key[8];
    loadKey(key64, key);
    Node* leaf = lower_bound(root, key, 8, 0, 8, iter);
    if (isLeaf(leaf)) {
	iter->val = (uint64_t)getLeafValue(leaf);
	loadKey(iter->val, (uint8_t*)&iter->key64);
	iter->nextLeaf();
	return true;
    }
    else {
	iter->val = 0;
	return false;
    }
}


void ART::erase(uint8_t key[], unsigned keyLength, unsigned maxKeyLength) {
    erase(root, &root, key, keyLength, 0, maxKeyLength);
}

uint64_t ART::getMemory() {
    return memory;
}


//********************************************************************
// ARTIter
//********************************************************************
ARTIter::ARTIter() {
    index = NULL;
    val = 0;
}

ARTIter::ARTIter(ART* idx) {
    index = idx;
    val = 0;
}


inline Node* ARTIter::minimum_recordPath(Node* node) {
    if (!node)
	return NULL;

    if (index->isLeaf(node))
	return node;

    NodeCursor nc;
    nc.node = node;
    nc.cursor = 0;
    node_stack.push_back(nc);

    switch (node->type) {
    case NodeType4: {
	Node4* n=static_cast<Node4*>(node);
	return minimum_recordPath(n->child[0]);
    }
    case NodeType16: {
	Node16* n=static_cast<Node16*>(node);
	return minimum_recordPath(n->child[0]);
    }
    case NodeType48: {
	Node48* n=static_cast<Node48*>(node);
	unsigned pos=0;
	while (n->childIndex[pos]==emptyMarker)
	    pos++;
	node_stack.back().cursor = pos;
	return minimum_recordPath(n->child[n->childIndex[pos]]);
    }
    case NodeType256: {
	Node256* n=static_cast<Node256*>(node);
	unsigned pos=0;
	while (!n->child[pos])
	    pos++;
	node_stack.back().cursor = pos;
	return minimum_recordPath(n->child[pos]);
    }
    }
    throw; // Unreachable
}

inline Node* ARTIter::nextSlot() {
    //while (node_stack.empty()) { THE BUGGGGGG!
    while (!node_stack.empty()) {
	Node* n = node_stack.back().node;
	uint16_t cursor = node_stack.back().cursor;
	cursor++;
	node_stack.back().cursor = cursor;
	switch (n->type) {
	case NodeType4: {
	    Node4* node=static_cast<Node4*>(n);
	    if (cursor < node->count)
		return node->child[cursor];
	    break;
	}
	case NodeType16: {
	    Node16* node=static_cast<Node16*>(n);
	    if (cursor < node->count)
		return node->child[cursor];
	    break;
	}
	case NodeType48: {
	    Node48* node=static_cast<Node48*>(n);
	    for (unsigned i=cursor; i<256; i++)
		if (node->childIndex[i]!=emptyMarker) {
		    node_stack.back().cursor = i;
		    return node->child[node->childIndex[i]];
		}
	    break;
	}
	case NodeType256: {
	    Node256* node=static_cast<Node256*>(n);
	    for (unsigned i=cursor; i<256; i++)
		if (node->child[i]!=NULL) {
		    node_stack.back().cursor = i;
		    return node->child[i]; 
		}
	    break;
	}
	}
	node_stack.pop_back();
    }
    return NULL;
}

inline Node* ARTIter::currentLeaf() {
    if (node_stack.size() == 0)
	return NULL;

    Node* n = node_stack.back().node;
    uint16_t cursor = node_stack.back().cursor;

    switch (n->type) {
    case NodeType4: {
	Node4* node=static_cast<Node4*>(n);
	return node->child[cursor];
    }
    case NodeType16: {
	Node16* node=static_cast<Node16*>(n);
	return node->child[cursor];
    }
    case NodeType48: {
	Node48* node=static_cast<Node48*>(n);
	return node->child[node->childIndex[cursor]];
    }
    case NodeType256: {
	Node256* node=static_cast<Node256*>(n);
	return node->child[cursor]; 
    }
    }
    return NULL;
}

inline Node* ARTIter::nextLeaf() {
    return minimum_recordPath(nextSlot());
}

uint64_t ARTIter::value() {
    return val;
}

bool ARTIter::operator ++ (int) {
    Node* leaf = currentLeaf();
    if (index->isLeaf(leaf)) {
	val = (uint64_t)index->getLeafValue(leaf);
	index->loadKey(val, (uint8_t*)&key64);
	nextLeaf();
	return true;
    }
    else {
	val = 0;
	return false;
    }
}
