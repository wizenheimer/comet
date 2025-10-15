package comet

import (
	"fmt"
	"sort"
)

// Compile-time checks to ensure flatIndexSearch implements VectorSearch
var _ VectorSearch = (*flatIndexSearch)(nil)

// flatIndexSearch implements the VectorSearch interface for flat index.
//
// Flat index performs exhaustive search:
//   - Compares query against every vector in the index
//   - Guarantees 100% recall (perfect accuracy)
//   - No training required, no approximation
type flatIndexSearch struct {
	index           *FlatIndex
	queries         [][]float32
	nodeIDs         []uint32
	documentIDs     []uint32
	k               int
	threshold       float32
	aggregationKind ScoreAggregationKind
	cutoff          int
	reranker        Reranker
}

// WithQuery sets the query vector(s) - supports single or batch queries.
// Can be combined with WithNode to search from both direct queries and node-based queries.
func (s *flatIndexSearch) WithQuery(queries ...[]float32) VectorSearch {
	s.queries = queries
	return s
}

// WithNode sets the node ID(s) to search from - supports single or batch nodes.
// Can be combined with WithQuery to search from both direct queries and node-based queries.
func (s *flatIndexSearch) WithNode(nodeIDs ...uint32) VectorSearch {
	s.nodeIDs = nodeIDs
	return s
}

// WithK sets the number of results to return.
// Defaults to all vectors if not set or k exceeds vector count.
func (s *flatIndexSearch) WithK(k int) VectorSearch {
	s.k = k
	return s
}

// WithNProbes is a no-op for flat index (nprobes is used by IVF-based indexes).
// Flat index always performs exhaustive search.
func (s *flatIndexSearch) WithNProbes(nProbes int) VectorSearch {
	return s
}

// WithEfSearch is a no-op for flat index (efSearch is used by HNSW).
// Flat index always performs exhaustive search.
func (s *flatIndexSearch) WithEfSearch(efSearch int) VectorSearch {
	return s
}

// WithThreshold sets a distance threshold for results (optional).
// Only results with distance <= threshold will be returned.
func (s *flatIndexSearch) WithThreshold(threshold float32) VectorSearch {
	s.threshold = threshold
	return s
}

// WithScoreAggregation sets the strategy for aggregating scores when the same node
// appears in results from multiple queries or nodes.
func (s *flatIndexSearch) WithScoreAggregation(kind ScoreAggregationKind) VectorSearch {
	s.aggregationKind = kind
	return s
}

// WithCutoff sets the autocut parameter for automatically determining result cutoff.
// A value of -1 (default) disables autocut. Otherwise, specifies number of extrema to find.
func (s *flatIndexSearch) WithCutoff(cutoff int) VectorSearch {
	s.cutoff = cutoff
	return s
}

// WithDocumentIDs sets the eligible document IDs for pre-filtering.
// Only vectors with IDs in this set will be considered as candidates.
// If empty, all documents are eligible (default behavior).
func (s *flatIndexSearch) WithDocumentIDs(docIDs ...uint32) VectorSearch {
	s.documentIDs = docIDs
	return s
}

// WithReranker sets a custom reranker to reorder search results.
// The reranker is applied after initial search results are obtained.
func (s *flatIndexSearch) WithReranker(reranker Reranker) VectorSearch {
	s.reranker = reranker
	return s
}

// Execute performs the actual search and returns results.
//
// This method validates the search configuration and then executes the search
// using all specified queries (both direct queries and node-based queries).
//
// When multiple queries/nodes are provided, results are aggregated by node ID
// using the configured aggregation strategy (default: Sum).
//
// Returns:
//   - []VectorResult: Search results sorted by distance with scores
//   - error: Returns error if search configuration is invalid
func (s *flatIndexSearch) Execute() ([]VectorResult, error) {
	// Validate that at least one of queries or nodeIDs is set
	if len(s.queries) == 0 && len(s.nodeIDs) == 0 {
		return nil, fmt.Errorf("must specify either queries or node IDs")
	}

	// Set default aggregation kind if not specified
	aggregationKind := s.aggregationKind
	if aggregationKind == "" {
		aggregationKind = SumAggregation
	}

	// Get aggregation instance
	aggregation, err := NewVectorAggregation(aggregationKind)
	if err != nil {
		return nil, err
	}

	// Collect all queries (both direct queries and node-based queries)
	allQueries := make([][]float32, 0, len(s.queries)+len(s.nodeIDs))

	// Add direct queries
	allQueries = append(allQueries, s.queries...)

	// Convert nodes to queries if specified
	if len(s.nodeIDs) > 0 {
		nodeQueries, err := s.lookupNodeVectors()
		if err != nil {
			return nil, err
		}
		allQueries = append(allQueries, nodeQueries...)
	}

	// Execute search with all queries
	var allResults []VectorResult
	for _, query := range allQueries {
		results, err := s.searchSingleQuery(query)
		if err != nil {
			return nil, err
		}
		allResults = append(allResults, results...)
	}

	// Aggregate results (deduplicates by node ID and combines scores)
	aggregatedResults := aggregation.Aggregate(allResults)

	// Apply k limit and autocut
	results := LimitResults(aggregatedResults, s.k)
	results = AutocutResults(results, s.cutoff)

	// Apply reranker if set
	if s.reranker != nil {
		results = s.reranker.Rerank(results)
	}

	return results, nil
}

// lookupNodeVectors converts node IDs to their corresponding vectors.
//
// Searches through the flat index to find vectors by ID.
// Returns error if any node ID is not found.
func (s *flatIndexSearch) lookupNodeVectors() ([][]float32, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	queries := make([][]float32, 0, len(s.nodeIDs))
	for _, nodeID := range s.nodeIDs {
		found := false
		for _, v := range s.index.vectors {
			if v.ID() == nodeID {
				// SOFT DELETE CHECK: Skip deleted nodes
				if s.index.deletedNodes.Contains(nodeID) {
					return nil, fmt.Errorf("node ID %d not found in index (deleted)", nodeID)
				}
				queries = append(queries, v.Vector())
				found = true
				break
			}
		}

		if !found {
			return nil, fmt.Errorf("node ID %d not found in index", nodeID)
		}
	}

	return queries, nil
}

// searchSingleQuery performs exhaustive kNN search for a single query vector.
//
// EXHAUSTIVE SEARCH ALGORITHM:
//  1. Preprocess the query vector (normalize for cosine, no-op for euclidean)
//  2. Calculate distance from preprocessed query to EVERY preprocessed vector in the index
//  3. Filter by document IDs if provided (metadata pre-filtering)
//  4. Filter by threshold if set
//  5. Sort all results by distance (ascending - smaller is more similar)
//  6. Return top k results
//
// OPTIMIZATION FOR COSINE DISTANCE:
// Since both query and stored vectors are normalized during preprocessing,
// the distance calculation is optimized to: 1 - dot(query, vector)
// This eliminates the need for norm calculations and divisions during search.
//
// METADATA PRE-FILTERING:
// If documentIDs are provided via WithDocumentIDs(), only vectors with matching IDs
// are considered as candidates. This enables efficient metadata-based filtering before
// expensive vector similarity calculations.
//
// Time Complexity: O(n × dim + n × log(n)) where:
//   - n is the number of vectors
//   - dim is the vector dimensionality
func (s *flatIndexSearch) searchSingleQuery(query []float32) ([]VectorResult, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	// Validate query dimension
	if len(query) != s.index.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", s.index.dim, len(query))
	}

	// Sanitize k
	k := sanitizeK(s.k, len(s.index.vectors))

	// Preprocess the query according to the distance metric
	// - For cosine: creates normalized copy (returns error if zero vector)
	// - For euclidean: returns query unchanged (always succeeds)
	preprocessedQuery, err := s.index.distance.Preprocess(query)
	if err != nil {
		return nil, err
	}

	// Create document filter for metadata pre-filtering
	docFilter := NewDocumentFilter(s.documentIDs)
	defer ReturnDocumentFilter(docFilter)

	// Calculate distances to all vectors
	// Since all stored vectors are already preprocessed, and we've preprocessed the query,
	// the distance calculation is now optimized (especially for cosine where it's just dot product)
	type result struct {
		vector   VectorNode
		distance float32
	}
	results := make([]result, 0, len(s.index.vectors))

	for _, v := range s.index.vectors {
		// SOFT DELETE CHECK: Skip deleted nodes
		if s.index.deletedNodes.Contains(v.ID()) {
			continue
		}

		// Apply document ID filter if set (metadata pre-filtering)
		if docFilter.ShouldSkip(v.ID()) {
			continue
		}

		// Calculate distance using preprocessed query and preprocessed stored vector
		dist := s.index.distance.Calculate(preprocessedQuery, v.Vector())

		// Apply threshold filter if set
		if s.threshold > 0 && dist > s.threshold {
			continue
		}

		results = append(results, result{vector: v, distance: dist})
	}

	// Sort by distance (ascending - smaller is more similar)
	sort.Slice(results, func(i, j int) bool {
		return results[i].distance < results[j].distance
	})

	// Take top k results
	k = sanitizeK(k, len(results))

	// Convert to VectorResult slice with scores
	finalResults := make([]VectorResult, k)
	for i := 0; i < k; i++ {
		finalResults[i] = VectorResult{
			Node:  results[i].vector,
			Score: results[i].distance,
		}
	}

	return finalResults, nil
}
