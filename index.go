package comet

// VectorIndexKind represents the type of indexing strategy used for vector search.
// Different index types offer different tradeoffs between speed, accuracy, and memory usage.
type VectorIndexKind string

var (
	// FlatIndex performs exhaustive search by comparing the query against all vectors.
	// Provides perfect recall but has O(n) time complexity.
	FlatIndex VectorIndexKind = "flat"

	// IVFIndex (Inverted File Index) partitions the vector space into clusters.
	// Searches only the nearest clusters, trading accuracy for speed.
	IVFIndex VectorIndexKind = "ivf"

	// PQIndex (Product Quantization) compresses vectors using learned codebooks.
	// Significantly reduces memory usage at the cost of some accuracy.
	PQIndex VectorIndexKind = "pq"

	// IVFPQIndex combines IVF clustering with PQ compression.
	// Provides fast search with low memory footprint.
	IVFPQIndex VectorIndexKind = "ivfpq"

	// HNSWIndex (Hierarchical Navigable Small World) builds a multi-layer graph.
	// Offers excellent query performance with high recall.
	HNSWIndex VectorIndexKind = "hnsw"
)

// VectorSearch encapsulates the search context for the vector index
type VectorSearch interface {
	// WithQuery sets the query vector(s) - supports single or batch queries
	WithQuery(queries ...[]float32) VectorSearch

	// WithNode sets the node ID(s) to search from - supports single or batch nodes
	WithNode(nodeIDs ...uint32) VectorSearch

	// WithK sets the number of results to return
	WithK(k int) VectorSearch

	// WithThreshold sets a distance threshold for results (optional)
	WithThreshold(threshold float32) VectorSearch

	// Execute the search and return the results
	Execute() ([]VectorNode, error)
}

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
