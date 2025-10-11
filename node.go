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

// isComparable returns true if the node is comparable to another node
// two nodes are comparable if they have the same dimension
func (n *VectorNode) ComparableToNode(other *VectorNode) bool {
	return len(n.vector) == len(other.vector)
}

// isComparableToVector returns true if the node is comparable to a vector
// two nodes are comparable if they have the same dimension
func (n *VectorNode) ComparableToVector(vector []float32) bool {
	return len(n.vector) == len(vector)
}

// Copy returns a copy of the node
func (n *VectorNode) Copy() *VectorNode {
	return &VectorNode{
		id:     n.id,
		vector: append([]float32{}, n.vector...),
	}
}

// Add adds a vector to the node
func (n *VectorNode) Add(vector []float32) {
	for i := range n.vector {
		n.vector[i] += vector[i]
	}
}

// Sub subtracts a vector from the node
func (n *VectorNode) Sub(vector []float32) {
	for i := range n.vector {
		n.vector[i] -= vector[i]
	}
}

// Scale scales the node by a scalar
func (n *VectorNode) Scale(scalar float32) {
	for i := range n.vector {
		n.vector[i] *= scalar
	}
}

// L2Norm returns the Euclidean length (magnitude) of the node's vector.
// This is useful for measuring vector size and preparing for normalization.
//
// Example:
//
//	node := NewVectorNode([]float32{3, 4})
//	length := node.L2Norm()  // Returns 5.0
func (n *VectorNode) L2Norm() float32 {
	return Norm(n.vector)
}

// L2NormSquared returns the squared Euclidean length of the node's vector.
// This is faster than L2Norm() as it avoids the square root operation.
// Use this when you only need to compare magnitudes (ordering is preserved).
func (n *VectorNode) L2NormSquared() float32 {
	return normSquared(n.vector)
}

// NormalizeInPlace normalizes the node's vector to unit length in-place.
// The node's vector is modified directly, making this memory-efficient.
// The node's ID remains unchanged.
//
// After normalization, L2Norm() will return 1.0 (unless the original vector was zero).
//
// Use this when:
//   - You don't need the original vector values
//   - Working with large datasets where memory efficiency matters
//   - Preparing vectors for cosine similarity comparisons
//
// Example:
//
//	node := NewVectorNode([]float32{3, 4})
//	node.NormalizeInPlace()  // node.Vector() is now [0.6, 0.8]
//	node.L2Norm()            // Returns 1.0
func (n *VectorNode) NormalizeInPlace() {
	NormalizeInPlace(n.vector)
}

// Normalize returns a new node with the same ID but with a normalized (unit length) vector.
// The original node is not modified.
//
// After normalization, the returned node's L2Norm() will be 1.0 (unless the original was zero).
//
// Use this when:
//   - You need to keep the original vector intact
//   - Creating normalized copies for comparison
//   - Implementing algorithms that require both normalized and original vectors
//
// Example:
//
//	original := NewVectorNode([]float32{3, 4})
//	normalized := original.Normalize()
//	// original.Vector() is still [3, 4]
//	// normalized.Vector() is [0.6, 0.8]
//	// Both nodes have the same ID
func (n *VectorNode) Normalize() *VectorNode {
	return &VectorNode{
		id:     n.id,
		vector: Normalize(n.vector),
	}
}
