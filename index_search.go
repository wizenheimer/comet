package comet

type Result interface {
	GetId() uint32
	GetScore() float32
}

// VectorResult represents a search result for a vector node
type VectorResult struct {
	Node  VectorNode
	Score float32
}

func (r VectorResult) GetId() uint32 {
	return r.Node.ID()
}

func (r VectorResult) GetScore() float32 {
	return r.Score
}

// VectorSearch encapsulates the search context for the vector index
type VectorSearch interface {
	// WithQuery sets the query vector(s) - supports single or batch queries
	WithQuery(queries ...[]float32) VectorSearch

	// WithNode sets the node ID(s) to search from - supports single or batch nodes
	WithNode(nodeIDs ...uint32) VectorSearch

	// WithK sets the number of results to return
	WithK(k int) VectorSearch

	// WithNProbes sets the number of probes to use for the search
	WithNProbes(nProbes int) VectorSearch

	// WithEfSearch sets the efSearch parameter for HNSW search (no-op for other indexes)
	// Allows per-search override of the index's default efSearch value
	WithEfSearch(efSearch int) VectorSearch

	// WithThreshold sets a distance threshold for results (optional)
	WithThreshold(threshold float32) VectorSearch

	// WithScoreAggregation sets the strategy for aggregating scores when the same node
	// appears in results from multiple queries or nodes (defaults to Sum)
	WithScoreAggregation(kind ScoreAggregationKind) VectorSearch

	// WithCutoff sets the autocut parameter for automatically determining result cutoff.
	// A value of -1 (default) disables autocut. Otherwise, specifies number of extrema to find.
	WithCutoff(cutoff int) VectorSearch

	// WithDocumentIDs sets the eligible document IDs for pre-filtering.
	// Only vectors with IDs in this set will be considered as candidates.
	// If empty, all documents are eligible (default behavior).
	WithDocumentIDs(docIDs ...uint32) VectorSearch

	// Execute the search and return the results
	Execute() ([]VectorResult, error)
}

// TextResult represents a search result for a text node
type TextResult struct {
	Id    uint32 // The ID of the text node
	Score float32
}

func (r TextResult) GetId() uint32 {
	return r.Id
}

func (r TextResult) GetScore() float32 {
	return r.Score
}

// TextSearch encapsulates the search context for the text index
type TextSearch interface {
	// WithQuery sets the query text(s) - supports single or batch queries
	WithQuery(queries ...string) TextSearch

	// WithNode sets the node ID(s) to search from - supports single or batch nodes
	WithNode(nodeIDs ...uint32) TextSearch

	// WithK sets the number of results to return
	WithK(k int) TextSearch

	// WithScoreAggregation sets the strategy for aggregating scores when the same node
	// appears in results from multiple queries or nodes (defaults to Sum)
	WithScoreAggregation(kind ScoreAggregationKind) TextSearch

	// WithCutoff sets the autocut parameter for automatically determining result cutoff.
	// A value of -1 (default) disables autocut. Otherwise, specifies number of extrema to find.
	WithCutoff(cutoff int) TextSearch

	// WithDocumentIDs sets the eligible document IDs for pre-filtering.
	// Only documents with IDs in this set will be considered as candidates.
	// If empty, all documents are eligible (default behavior).
	WithDocumentIDs(docIDs ...uint32) TextSearch

	// Execute the search and return the results
	Execute() ([]TextResult, error)
}
