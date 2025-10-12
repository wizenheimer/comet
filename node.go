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

// MetadataNode represents a document with structured metadata fields
type MetadataNode struct {
	id       uint32
	metadata map[string]interface{}
}

// NewMetadataNode creates a new MetadataNode with an auto-incremented ID.
// The initializer is thread-safe and can be used concurrently
func NewMetadataNode(metadata map[string]interface{}) *MetadataNode {
	id := atomic.AddUint32(&nodeIDCounter, 1)
	return &MetadataNode{
		id:       id,
		metadata: metadata,
	}
}

// NewMetadataNodeWithID creates a new MetadataNode with the provided ID.
// Here the ID uniqueness of the node is delegated to the caller.
func NewMetadataNodeWithID(id uint32, metadata map[string]interface{}) *MetadataNode {
	return &MetadataNode{
		id:       id,
		metadata: metadata,
	}
}

// ID returns the ID of the node
func (n *MetadataNode) ID() uint32 {
	return n.id
}

// Metadata returns the metadata of the node
func (n *MetadataNode) Metadata() map[string]interface{} {
	return n.metadata
}
