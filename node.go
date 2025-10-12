package comet

import "sync/atomic"

// nodeIDCounter is a package-level counter for auto-incrementing node IDs
var nodeIDCounter uint32

type VectorNode struct {
	id     uint32
	vector []float32
}

// NewVectorNode creates a new Node with an auto-incremented ID.
// The initializer is thread-safe and can be used concurrently
func NewVectorNode(vector []float32) *VectorNode {
	id := atomic.AddUint32(&nodeIDCounter, 1)
	return &VectorNode{
		id:     id,
		vector: vector,
	}
}

// NewVectorNodeWithID creates a new Node with the provided ID.
// Here the ID uniqueness of the node is delegated to the caller.
func NewVectorNodeWithID(id uint32, vector []float32) *VectorNode {
	return &VectorNode{
		id:     id,
		vector: vector,
	}
}

// ID returns the ID of the node
func (n *VectorNode) ID() uint32 {
	return n.id
}

// Vector returns the vector of the node
func (n *VectorNode) Vector() []float32 {
	return n.vector
}
