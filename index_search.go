package comet

// Result is a common interface for all search result types.
//
// This interface enables generic operations like limiting, autocut, and score
// aggregation to work uniformly across different search modalities (vector, text, metadata).
//
// All result types (VectorResult, TextResult, MetadataResult) implement this interface.
type Result interface {
	// GetId returns the unique identifier for this result
	GetId() uint32

	// GetScore returns the relevance score for this result.
	// Interpretation depends on search type:
	//   - Vector search: distance (lower = better)
	//   - Text search: BM25 score (higher = better)
	//   - Metadata search: typically 0 or 1 (binary match)
	GetScore() float32
}

// VectorResult represents a search result from vector similarity search.
//
// Each result contains the matched vector node and a similarity score.
// The score represents distance from the query vector:
//   - Lower scores indicate higher similarity (closer in vector space)
//   - Score interpretation depends on the distance metric:
//   - L2/Euclidean: absolute distance
//   - L2Squared: squared distance (faster, preserves ordering)
//   - Cosine: angular distance (range: [0, 2])
//
// Example:
//
//	results, _ := index.NewSearch().
//	    WithQuery(queryVec).
//	    WithK(10).
//	    Execute()
//
//	for _, result := range results {
//	    fmt.Printf("ID: %d, Distance: %.4f\n",
//	        result.GetId(), result.Score)
//	    fmt.Printf("Vector: %v\n", result.Node.Vector())
//	}
type VectorResult struct {
	// Node is the matched vector node containing ID and vector data
	Node VectorNode

	// Score is the distance from the query vector (lower = more similar)
	Score float32
}

// GetId returns the ID of the matched vector node.
func (r VectorResult) GetId() uint32 {
	return r.Node.ID()
}

// GetScore returns the distance score (lower = more similar).
func (r VectorResult) GetScore() float32 {
	return r.Score
}

// VectorSearch provides a fluent interface for configuring and executing vector searches.
//
// VectorSearch uses the builder pattern to configure search parameters before execution.
// All With* methods return the search instance for method chaining.
//
// Search modes:
//   - Query-based: Search for vectors similar to provided query vectors
//   - Node-based: Find vectors similar to indexed nodes (by ID)
//   - Hybrid: Combine both query and node searches
//
// Advanced features:
//   - Pre-filtering: Restrict search to specific document IDs
//   - Score aggregation: Combine results from multiple queries/nodes
//   - Autocut: Automatically determine optimal result cutoff
//   - Threshold filtering: Exclude results beyond distance threshold
//
// Example - Basic search:
//
//	results, _ := index.NewSearch().
//	    WithQuery(queryVec).
//	    WithK(10).
//	    Execute()
//
// Example - Multi-query with aggregation:
//
//	results, _ := index.NewSearch().
//	    WithQuery(query1, query2, query3).
//	    WithK(20).
//	    WithScoreAggregation(comet.MeanAggregation).
//	    Execute()
//
// Example - Pre-filtered search:
//
//	eligibleIDs := []uint32{1, 5, 10, 15, 20}
//	results, _ := index.NewSearch().
//	    WithQuery(queryVec).
//	    WithDocumentIDs(eligibleIDs...).
//	    WithK(5).
//	    Execute()
type VectorSearch interface {
	// WithQuery sets the query vector(s) for similarity search.
	// Supports single or multiple query vectors for batch search.
	// Results from multiple queries are aggregated using the configured strategy.
	//
	// Parameters:
	//   - queries: One or more query vectors (each []float32 must match index dimension)
	//
	// Returns:
	//   - VectorSearch: The search instance for method chaining
	WithQuery(queries ...[]float32) VectorSearch

	// WithNode sets the node ID(s) to search from (node-based similarity).
	// Finds vectors similar to the specified indexed nodes.
	// Supports single or multiple nodes for batch search.
	//
	// Parameters:
	//   - nodeIDs: One or more node IDs to use as query vectors
	//
	// Returns:
	//   - VectorSearch: The search instance for method chaining
	WithNode(nodeIDs ...uint32) VectorSearch

	// WithK sets the number of nearest neighbors to return.
	//
	// Parameters:
	//   - k: Number of results (default: 10)
	//
	// Returns:
	//   - VectorSearch: The search instance for method chaining
	WithK(k int) VectorSearch

	// WithNProbes sets the number of clusters to probe (IVF/IVFPQ only).
	// Higher values increase recall but reduce speed.
	// This is a no-op for other index types.
	//
	// Typical values:
	//   - 1: Fastest, ~60-70% recall
	//   - 8: Good balance, ~85% recall
	//   - 16: Better recall, ~92% recall
	//   - 32: High recall, ~96% recall
	//
	// Parameters:
	//   - nProbes: Number of clusters to search
	//
	// Returns:
	//   - VectorSearch: The search instance for method chaining
	WithNProbes(nProbes int) VectorSearch

	// WithEfSearch sets the efSearch parameter (HNSW only).
	// Controls the size of the dynamic candidate list during search.
	// Higher values increase recall but reduce speed.
	// This is a no-op for other index types.
	//
	// Typical values:
	//   - 50: Very fast, ~85% recall
	//   - 100: Fast, ~92% recall
	//   - 200: Balanced, ~96% recall (default)
	//   - 400: Slower, ~98% recall
	//
	// Parameters:
	//   - efSearch: Size of candidate list
	//
	// Returns:
	//   - VectorSearch: The search instance for method chaining
	WithEfSearch(efSearch int) VectorSearch

	// WithThreshold sets a distance threshold for filtering results.
	// Only vectors with distance <= threshold are returned.
	// Useful for finding all "sufficiently similar" vectors.
	//
	// Parameters:
	//   - threshold: Maximum distance (results with distance > threshold are excluded)
	//
	// Returns:
	//   - VectorSearch: The search instance for method chaining
	WithThreshold(threshold float32) VectorSearch

	// WithScoreAggregation sets how to combine scores from multiple queries/nodes.
	// Only relevant when using multiple queries or nodes.
	//
	// Available strategies:
	//   - SumAggregation: Sum all scores (emphasizes frequency)
	//   - MaxAggregation: Take maximum distance (conservative)
	//   - MeanAggregation: Average all scores (balanced, default)
	//
	// Parameters:
	//   - kind: The aggregation strategy
	//
	// Returns:
	//   - VectorSearch: The search instance for method chaining
	WithScoreAggregation(kind ScoreAggregationKind) VectorSearch

	// WithCutoff enables automatic result cutoff based on score distribution.
	// Analyzes the score curve to find natural breakpoints.
	//
	// Parameters:
	//   - cutoff: Number of extrema to find before cutting (-1 disables autocut)
	//
	// Returns:
	//   - VectorSearch: The search instance for method chaining
	WithCutoff(cutoff int) VectorSearch

	// WithDocumentIDs pre-filters the search to specific document IDs.
	// Only vectors with IDs in this set will be considered.
	// Useful for combining with metadata filters in hybrid search.
	//
	// Parameters:
	//   - docIDs: Eligible document IDs (empty means all documents)
	//
	// Returns:
	//   - VectorSearch: The search instance for method chaining
	WithDocumentIDs(docIDs ...uint32) VectorSearch

	// Execute performs the configured search and returns results.
	// Results are sorted by distance (ascending - lower is better).
	//
	// Returns:
	//   - []VectorResult: Sorted search results
	//   - error: Error if search fails
	Execute() ([]VectorResult, error)
}

// TextResult represents a search result from full-text (BM25) search.
//
// Each result contains the matched document ID and a relevance score.
// The score represents BM25 relevance:
//   - Higher scores indicate better relevance (more important terms, higher frequency)
//   - Scores are not normalized and can exceed 1.0
//   - Scores depend on:
//   - Term frequency in document
//   - Inverse document frequency
//   - Document length normalization
//
// Note: Unlike vector results where lower is better, text results have
// higher-is-better scores (standard for information retrieval).
//
// Example:
//
//	results, _ := index.NewSearch().
//	    WithQuery("machine learning").
//	    WithK(10).
//	    Execute()
//
//	for _, result := range results {
//	    fmt.Printf("ID: %d, BM25 Score: %.4f\n",
//	        result.Id, result.Score)
//	}
type TextResult struct {
	// Id is the unique identifier of the matched document
	Id uint32

	// Score is the BM25 relevance score (higher = more relevant)
	Score float32
}

// GetId returns the ID of the matched document.
func (r TextResult) GetId() uint32 {
	return r.Id
}

// GetScore returns the BM25 relevance score (higher = more relevant).
func (r TextResult) GetScore() float32 {
	return r.Score
}

// TextSearch provides a fluent interface for configuring and executing text searches.
//
// TextSearch uses the builder pattern for configuration, similar to VectorSearch.
// It supports BM25-based full-text search with tokenization and normalization.
//
// Features:
//   - Single or multi-query search with score aggregation
//   - Pre-filtering by document IDs
//   - Automatic result cutoff (autocut)
//   - Top-K retrieval with heap-based ranking
//
// Example - Basic search:
//
//	results, _ := txtIndex.NewSearch().
//	    WithQuery("machine learning tutorial").
//	    WithK(10).
//	    Execute()
//
// Example - Multi-query search:
//
//	results, _ := txtIndex.NewSearch().
//	    WithQuery("deep learning", "neural networks", "transformers").
//	    WithK(20).
//	    WithScoreAggregation(comet.MaxAggregation).
//	    Execute()
//
// Example - Pre-filtered search:
//
//	eligibleDocs := []uint32{1, 5, 10, 15}
//	results, _ := txtIndex.NewSearch().
//	    WithQuery("tutorial").
//	    WithDocumentIDs(eligibleDocs...).
//	    WithK(5).
//	    Execute()
type TextSearch interface {
	// WithQuery sets the query text(s) for full-text search.
	// Supports single or multiple queries for batch search.
	// Text is tokenized and normalized using UAX#29 word segmentation.
	//
	// Parameters:
	//   - queries: One or more query strings
	//
	// Returns:
	//   - TextSearch: The search instance for method chaining
	WithQuery(queries ...string) TextSearch

	// WithNode sets the node ID(s) to search from (not commonly used for text).
	// This is provided for interface consistency but is rarely needed.
	//
	// Parameters:
	//   - nodeIDs: One or more node IDs
	//
	// Returns:
	//   - TextSearch: The search instance for method chaining
	WithNode(nodeIDs ...uint32) TextSearch

	// WithK sets the number of top results to return.
	//
	// Parameters:
	//   - k: Number of results (default: 10)
	//
	// Returns:
	//   - TextSearch: The search instance for method chaining
	WithK(k int) TextSearch

	// WithScoreAggregation sets how to combine scores from multiple queries.
	// Only relevant when using multiple queries.
	//
	// Available strategies:
	//   - SumAggregation: Sum all scores (default, emphasizes total relevance)
	//   - MaxAggregation: Take maximum score (best match across queries)
	//   - MeanAggregation: Average all scores (balanced)
	//
	// Parameters:
	//   - kind: The aggregation strategy
	//
	// Returns:
	//   - TextSearch: The search instance for method chaining
	WithScoreAggregation(kind ScoreAggregationKind) TextSearch

	// WithCutoff enables automatic result cutoff based on score distribution.
	//
	// Parameters:
	//   - cutoff: Number of extrema to find before cutting (-1 disables)
	//
	// Returns:
	//   - TextSearch: The search instance for method chaining
	WithCutoff(cutoff int) TextSearch

	// WithDocumentIDs pre-filters the search to specific document IDs.
	// Only documents with IDs in this set will be scored.
	//
	// Parameters:
	//   - docIDs: Eligible document IDs (empty means all documents)
	//
	// Returns:
	//   - TextSearch: The search instance for method chaining
	WithDocumentIDs(docIDs ...uint32) TextSearch

	// Execute performs the configured search and returns results.
	// Results are sorted by BM25 score (descending - higher is better).
	//
	// Returns:
	//   - []TextResult: Sorted search results
	//   - error: Error if search fails
	Execute() ([]TextResult, error)
}

// MetadataSearch provides a fluent interface for filtering documents by metadata.
//
// MetadataSearch uses roaring bitmaps and bit-sliced indexes for extremely fast
// filtering on structured attributes. It supports:
//   - Equality and inequality queries
//   - Numeric range queries
//   - Set membership (IN, NOT IN)
//   - Existence checks
//   - Complex boolean logic (AND, OR, NOT)
//
// Filters within a single WithFilters() call are combined with AND logic.
// Filter groups enable OR logic between different filter combinations.
//
// Example - Simple filters (AND logic):
//
//	results, _ := metaIndex.NewSearch().
//	    WithFilters(
//	        comet.Eq("category", "electronics"),
//	        comet.Lte("price", 1000),
//	        comet.Eq("in_stock", true),
//	    ).
//	    Execute()
//
// Example - Complex query with OR:
//
//	// (category=electronics AND price<500) OR (category=books AND rating>=4)
//	group1 := comet.NewFilterGroup().
//	    WithFilters(
//	        comet.Eq("category", "electronics"),
//	        comet.Lt("price", 500),
//	    )
//	group2 := comet.NewFilterGroup().
//	    WithFilters(
//	        comet.Eq("category", "books"),
//	        comet.Gte("rating", 4),
//	    )
//	results, _ := metaIndex.NewSearch().
//	    WithFilterGroups(group1, group2).
//	    Execute()
//
// Example - Set membership:
//
//	results, _ := metaIndex.NewSearch().
//	    WithFilters(
//	        comet.In("category", "electronics", "computers", "phones"),
//	        comet.NotIn("brand", "brandX", "brandY"),
//	    ).
//	    Execute()
type MetadataSearch interface {
	// WithFilters sets the filters to apply with AND logic.
	// All filters must match for a document to be included.
	//
	// Available filter functions:
	//   - Eq(field, value): field == value
	//   - Ne(field, value): field != value
	//   - Lt(field, value): field < value
	//   - Lte(field, value): field <= value
	//   - Gt(field, value): field > value
	//   - Gte(field, value): field >= value
	//   - Between(field, min, max): min <= field <= max
	//   - In(field, ...values): field in values
	//   - NotIn(field, ...values): field not in values
	//   - Exists(field): field exists
	//   - NotExists(field): field doesn't exist
	//
	// Parameters:
	//   - filters: One or more filter conditions (combined with AND)
	//
	// Returns:
	//   - MetadataSearch: The search instance for method chaining
	WithFilters(filters ...Filter) MetadataSearch

	// WithFilterGroups sets complex filter groups with OR logic.
	// Documents matching ANY group are included (OR between groups).
	// Filters within each group are combined with AND.
	//
	// This enables complex boolean expressions like:
	//   (A AND B) OR (C AND D) OR (E AND F)
	//
	// Parameters:
	//   - groups: One or more filter groups (combined with OR)
	//
	// Returns:
	//   - MetadataSearch: The search instance for method chaining
	WithFilterGroups(groups ...*FilterGroup) MetadataSearch

	// Execute performs the filtering and returns matching document IDs.
	// The results contain only document IDs (no scores).
	//
	// Returns:
	//   - []MetadataResult: Matching documents
	//   - error: Error if filtering fails
	Execute() ([]MetadataResult, error)
}
