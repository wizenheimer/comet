package comet

import "sync/atomic"

// nodeIDCounter is a package-level counter for auto-incrementing node IDs.
// It uses atomic operations for thread-safe ID generation across goroutines.
var nodeIDCounter uint32

// VectorNode represents a vector embedding with a unique identifier.
//
// VectorNode is the fundamental data structure for storing vectors in all index types.
// Each node contains:
//   - id: A unique identifier for retrieval and reference
//   - vector: The actual float32 embedding values
//
// The node is immutable after creation - neither the ID nor vector can be modified.
// This immutability ensures thread-safety and prevents accidental corruption of indexed data.
//
// Example:
//
//	// Auto-generated ID
//	vec := []float32{0.1, 0.5, 0.3, 0.8}
//	node := comet.NewVectorNode(vec)
//	fmt.Println(node.ID())      // e.g., 1
//	fmt.Println(node.Vector())  // [0.1, 0.5, 0.3, 0.8]
//
//	// Explicit ID (useful for mapping to external document IDs)
//	node2 := comet.NewVectorNodeWithID(42, vec)
//	fmt.Println(node2.ID())  // 42
type VectorNode struct {
	id     uint32
	vector []float32
}

// NewVectorNode creates a new VectorNode with an auto-incremented ID.
//
// The ID is generated atomically and is guaranteed to be unique across all
// VectorNode and MetadataNode instances created by this package. This makes
// it safe to create nodes concurrently from multiple goroutines.
//
// Parameters:
//   - vector: The embedding vector to store. This should match the dimensionality
//     expected by your index (e.g., 384 for sentence transformers, 768 for BERT).
//
// Returns:
//   - *VectorNode: A new node with auto-generated ID
//
// Thread-safety: This function is safe for concurrent use.
//
// Example:
//
//	embedding := []float32{0.1, 0.2, 0.3, ...}  // 384 dimensions
//	node := comet.NewVectorNode(embedding)
//	index.Add(*node)
func NewVectorNode(vector []float32) *VectorNode {
	id := atomic.AddUint32(&nodeIDCounter, 1)
	return &VectorNode{
		id:     id,
		vector: vector,
	}
}

// NewVectorNodeWithID creates a new VectorNode with an explicit ID.
//
// Use this when you need to control the ID assignment, typically when:
//   - Mapping vectors to existing document IDs from your database
//   - Deserializing indexes from disk
//   - Maintaining consistent IDs across restarts
//
// WARNING: The caller is responsible for ensuring ID uniqueness. Using
// duplicate IDs may cause undefined behavior in the index.
//
// Parameters:
//   - id: The unique identifier for this node
//   - vector: The embedding vector to store
//
// Returns:
//   - *VectorNode: A new node with the specified ID
//
// Example:
//
//	// Map to database primary key
//	dbID := uint32(12345)
//	embedding := []float32{0.1, 0.2, 0.3, ...}
//	node := comet.NewVectorNodeWithID(dbID, embedding)
//	index.Add(*node)
func NewVectorNodeWithID(id uint32, vector []float32) *VectorNode {
	return &VectorNode{
		id:     id,
		vector: vector,
	}
}

// ID returns the unique identifier of this vector node.
//
// Returns:
//   - uint32: The node's ID
func (n *VectorNode) ID() uint32 {
	return n.id
}

// Vector returns the embedding vector stored in this node.
//
// Returns:
//   - []float32: The vector data (not a copy - do not modify)
func (n *VectorNode) Vector() []float32 {
	return n.vector
}

// MetadataNode represents a document with structured metadata attributes.
//
// MetadataNode is used for filtering and structured queries in metadata indexes.
// It associates a unique ID with a map of metadata fields that can be:
//   - Strings: for categorical data (e.g., "category", "status")
//   - Numbers (int, float64): for numeric fields (e.g., "price", "rating")
//   - Booleans: for binary flags (e.g., "in_stock", "featured")
//
// The metadata is stored as map[string]interface{} for flexibility, but
// the metadata index will interpret types and use appropriate data structures:
//   - Categorical values → Roaring Bitmaps for fast set operations
//   - Numeric values → Bit-Sliced Index (BSI) for range queries
//
// Example:
//
//	metadata := map[string]interface{}{
//	    "category": "electronics",
//	    "price": 999.99,
//	    "rating": 4.5,
//	    "in_stock": true,
//	    "tags": []string{"premium", "wireless"},  // Lists also supported
//	}
//	node := comet.NewMetadataNode(metadata)
//	index.Add(*node)
type MetadataNode struct {
	id       uint32
	metadata map[string]interface{}
}

// NewMetadataNode creates a new MetadataNode with an auto-incremented ID.
//
// The ID is generated atomically and is guaranteed to be unique across all
// VectorNode and MetadataNode instances. This makes it safe for concurrent use.
//
// Parameters:
//   - metadata: Map of field names to values. Supported types are:
//   - string: categorical data
//   - int, int32, int64: integer values
//   - float32, float64: floating point values (converted to int64 for BSI)
//   - bool: boolean flags
//   - []string: list of string values (for IN queries)
//
// Returns:
//   - *MetadataNode: A new node with auto-generated ID
//
// Thread-safety: This function is safe for concurrent use.
//
// Example:
//
//	node := comet.NewMetadataNode(map[string]interface{}{
//	    "category": "books",
//	    "price": 29.99,
//	    "in_stock": true,
//	})
//	index.Add(*node)
func NewMetadataNode(metadata map[string]interface{}) *MetadataNode {
	id := atomic.AddUint32(&nodeIDCounter, 1)
	return &MetadataNode{
		id:       id,
		metadata: metadata,
	}
}

// NewMetadataNodeWithID creates a new MetadataNode with an explicit ID.
//
// Use this when you need to control the ID assignment, typically when:
//   - Mapping metadata to existing document IDs from your database
//   - Ensuring consistency with vector node IDs in hybrid indexes
//   - Deserializing indexes from disk
//
// WARNING: The caller is responsible for ensuring ID uniqueness.
//
// Parameters:
//   - id: The unique identifier for this node
//   - metadata: Map of field names to values
//
// Returns:
//   - *MetadataNode: A new node with the specified ID
//
// Example:
//
//	dbID := uint32(12345)
//	node := comet.NewMetadataNodeWithID(dbID, map[string]interface{}{
//	    "category": "electronics",
//	    "price": 999,
//	})
//	index.Add(*node)
func NewMetadataNodeWithID(id uint32, metadata map[string]interface{}) *MetadataNode {
	return &MetadataNode{
		id:       id,
		metadata: metadata,
	}
}

// ID returns the unique identifier of this metadata node.
//
// Returns:
//   - uint32: The node's ID
func (n *MetadataNode) ID() uint32 {
	return n.id
}

// Metadata returns the metadata map for this node.
//
// Returns:
//   - map[string]interface{}: The metadata fields (not a copy - do not modify)
func (n *MetadataNode) Metadata() map[string]interface{} {
	return n.metadata
}
