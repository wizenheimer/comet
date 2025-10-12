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
}
