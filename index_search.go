package comet

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

	// WithThreshold sets a distance threshold for results (optional)
	WithThreshold(threshold float32) VectorSearch

	// Execute the search and return the results
	Execute() ([]VectorNode, error)
}
