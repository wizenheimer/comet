package comet

// VectorIndexKind represents the type of indexing strategy used for vector search.
// Different index types offer different tradeoffs between speed, accuracy, and memory usage.
type VectorIndexKind string

var (
	// FlatIndex performs exhaustive search by comparing the query against all vectors.
	// Provides perfect recall but has O(n) time complexity.
	FlatIndexKind VectorIndexKind = "flat"

	// IVFIndex (Inverted File Index) partitions the vector space into clusters.
	// Searches only the nearest clusters, trading accuracy for speed.
	IVFIndexKind VectorIndexKind = "ivf"

	// PQIndex (Product Quantization) compresses vectors using learned codebooks.
	// Significantly reduces memory usage at the cost of some accuracy.
	PQIndexKind VectorIndexKind = "pq"

	// IVFPQIndex combines IVF clustering with PQ compression.
	// Provides fast search with low memory footprint.
	IVFPQIndexKind VectorIndexKind = "ivfpq"

	// HNSWIndex (Hierarchical Navigable Small World) builds a multi-layer graph.
	// Offers excellent query performance with high recall.
	HNSWIndexKind VectorIndexKind = "hnsw"
)

// VectorIndex is the interface for the vector index
type VectorIndex interface {
	// Train the index
	Train(vectors []VectorNode) error

	// Add a new vector to the index
	Add(vector VectorNode) error

	// Remove a vector from the index
	Remove(vector VectorNode) error

	// Flush the vector index
	Flush() error

	// NewSearch creates a new search builder
	NewSearch() VectorSearch

	// Dimensions returns the dimensionality of vectors stored in this index
	Dimensions() int

	// DistanceKind returns the distance metric used for similarity measurement
	DistanceKind() DistanceKind

	// Kind returns the type of vector index
	Kind() VectorIndexKind

	// Trained returns true if the index has been trained
	Trained() bool
}

type TextIndex interface {
	// Add a new text to the index
	Add(id uint32, text string) error

	// Remove a text from the index
	Remove(id uint32) error

	// NewSearch creates a new search builder
	NewSearch() TextSearch

	// Flush the text index
	Flush() error
}

// MetadataIndex is the interface for filtering documents based on metadata
type MetadataIndex interface {
	// Add adds a document with its metadata to the index
	Add(node MetadataNode) error

	// Remove removes a document from the index
	Remove(node MetadataNode) error

	// NewSearch creates a new search builder
	NewSearch() MetadataSearch

	// Flush the metadata index
	Flush() error
}

// HybridSearchIndex provides a unified interface for multi-modal search
type HybridSearchIndex interface {
	// Add adds a document with its vector, text, and metadata to the index
	// The ID is auto-generated and returned
	Add(vector []float32, text string, metadata map[string]interface{}) (uint32, error)

	// AddWithID adds a document with a specific ID
	AddWithID(id uint32, vector []float32, text string, metadata map[string]interface{}) error

	// Remove removes a document from all indexes
	Remove(id uint32) error

	// NewSearch creates a new search builder
	NewSearch() HybridSearch

	// Train trains the vector index (required for some index types like IVF, PQ)
	Train(vectors [][]float32) error

	// Flush flushes all indexes
	Flush() error

	// VectorIndex returns the underlying vector index
	VectorIndex() VectorIndex

	// TextIndex returns the underlying text index
	TextIndex() TextIndex

	// MetadataIndex returns the underlying metadata index
	MetadataIndex() MetadataIndex
}
