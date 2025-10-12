package comet

import (
	"fmt"
	"sort"
)

// Compile-time checks to ensure flatIndexSearch implements VectorSearch
var _ VectorSearch = (*flatIndexSearch)(nil)

// flatIndexSearch implements the VectorSearch interface for flat index.
type flatIndexSearch struct {
	index     *FlatIndex
	queries   [][]float32
	nodeIDs   []uint32
	k         int
	threshold float32
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

// WithK sets the number of results to return
func (s *flatIndexSearch) WithK(k int) VectorSearch {
	s.k = k
	return s
}

// WithNProbes sets the number of probes to use for the search
// This is a no-op for flat index since it doesn't use probes
func (s *flatIndexSearch) WithNProbes(nProbes int) VectorSearch {
	return s
}

// WithThreshold sets a distance threshold for results (optional)
func (s *flatIndexSearch) WithThreshold(threshold float32) VectorSearch {
	s.threshold = threshold
	return s
}

// Execute performs the actual search and returns results.
//
// This method validates the search configuration and then executes the search
// using all specified queries (both direct queries and node-based queries).
//
// Returns:
//   - []VectorNode: Search results sorted by distance
//   - error: Returns error if search configuration is invalid
func (s *flatIndexSearch) Execute() ([]VectorNode, error) {
	// Validate that at least one of queries or nodeIDs is set
	if len(s.queries) == 0 && len(s.nodeIDs) == 0 {
		return nil, fmt.Errorf("must specify either queries or node IDs")
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
	var allResults []VectorNode
	for _, query := range allQueries {
		results, err := s.searchSingleQuery(query)
		if err != nil {
			return nil, err
		}
		allResults = append(allResults, results...)
	}

	return allResults, nil
}

// lookupNodeVectors converts node IDs to their corresponding vectors.
func (s *flatIndexSearch) lookupNodeVectors() ([][]float32, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	queries := make([][]float32, 0, len(s.nodeIDs))
	for _, nodeID := range s.nodeIDs {
		found := false
		for _, v := range s.index.vectors {
			if v.ID() == nodeID {
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

// searchSingleQuery performs the core kNN search for a single query vector.
//
// This is the exhaustive search algorithm:
// 1. Preprocess the query vector (normalize for cosine, no-op for euclidean)
// 2. Calculate distance from preprocessed query to EVERY preprocessed vector in the index
// 3. Sort all results by distance
// 4. Return top k results
//
// For cosine distance, since both query and stored vectors are normalized,
// the distance calculation is just: 1 - dot(query, vector)
// This eliminates the need for norm calculations and divisions during search.
//
// Time Complexity: O(m*n + n*log(n)) where m=dimensionality, n=number of vectors
func (s *flatIndexSearch) searchSingleQuery(query []float32) ([]VectorNode, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	// Validate query dimension
	if len(query) != s.index.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", s.index.dim, len(query))
	}

	// Sanitize k
	k := s.k
	if k <= 0 || k > len(s.index.vectors) {
		k = len(s.index.vectors)
	}

	// Preprocess the query according to the distance metric
	// - For cosine: creates normalized copy (returns error if zero vector)
	// - For euclidean: returns query unchanged (always succeeds)
	preprocessedQuery, err := s.index.distance.Preprocess(query)
	if err != nil {
		return nil, err
	}

	// Calculate distances to all vectors
	// Since all stored vectors are already preprocessed, and we've preprocessed the query,
	// the distance calculation is now optimized (especially for cosine where it's just dot product)
	type result struct {
		vector   VectorNode
		distance float32
	}
	results := make([]result, 0, len(s.index.vectors))

	for _, v := range s.index.vectors {
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
	if k > len(results) {
		k = len(results)
	}

	// Convert to VectorNode slice
	finalResults := make([]VectorNode, k)
	for i := 0; i < k; i++ {
		finalResults[i] = results[i].vector
	}

	return finalResults, nil
}
