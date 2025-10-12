package comet

import (
	"fmt"
	"math"
	"sort"
)

// Compile-time checks to ensure pqIndexSearch implements VectorSearch
var _ VectorSearch = (*pqIndexSearch)(nil)

// pqIndexSearch implements the VectorSearch interface for PQ index.
//
// PQ (Product Quantization) search performs approximate distance computation:
//   - Builds distance tables for query subvectors
//   - Uses table lookups to compute approximate distances
//   - Returns top k from all candidates
type pqIndexSearch struct {
	index     *PQIndex
	queries   [][]float32
	nodeIDs   []uint32
	k         int
	threshold float32
}

// WithQuery sets the query vector(s) - supports single or batch queries.
// Can be combined with WithNode to search from both direct queries and node-based queries.
func (s *pqIndexSearch) WithQuery(queries ...[]float32) VectorSearch {
	s.queries = queries
	return s
}

// WithNode sets the node ID(s) to search from - supports single or batch nodes.
// Can be combined with WithQuery to search from both direct queries and node-based queries.
func (s *pqIndexSearch) WithNode(nodeIDs ...uint32) VectorSearch {
	s.nodeIDs = nodeIDs
	return s
}

// WithK sets the number of results to return.
// Defaults to 10 if not set.
func (s *pqIndexSearch) WithK(k int) VectorSearch {
	s.k = k
	return s
}

// WithNProbes is a no-op for PQ (nprobes is used by IVF-based indexes).
// PQ performs exhaustive search with approximate distances.
func (s *pqIndexSearch) WithNProbes(nprobes int) VectorSearch {
	// PQ doesn't use nprobes, ignored
	return s
}

// WithEfSearch is a no-op for PQ index (efSearch is used by HNSW).
// PQ performs exhaustive search with approximate distances.
func (s *pqIndexSearch) WithEfSearch(efSearch int) VectorSearch {
	return s
}

// WithThreshold sets a distance threshold for results (optional).
// Only results with distance <= threshold will be returned.
func (s *pqIndexSearch) WithThreshold(threshold float32) VectorSearch {
	s.threshold = threshold
	return s
}

// Execute performs the actual search and returns results.
//
// This method validates the search configuration and then executes the search
// using all specified queries (both direct queries and node-based queries).
//
// Returns:
//   - []VectorResult: Search results sorted by distance with scores
//   - error: Returns error if search configuration is invalid
func (s *pqIndexSearch) Execute() ([]VectorResult, error) {
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
	var allResults []VectorResult
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
//
// Returns error if any node ID is not found.
func (s *pqIndexSearch) lookupNodeVectors() ([][]float32, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	queries := make([][]float32, 0, len(s.nodeIDs))
	for _, nodeID := range s.nodeIDs {
		found := false
		for _, v := range s.index.vectorNodes {
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

// searchSingleQuery performs asymmetric PQ distance search for a single query.
//
// ASYMMETRIC DISTANCE:
// - Query: full precision (not quantized)
// - Database vectors: quantized to PQ codes
// - This gives better accuracy than symmetric distance
//
// Algorithm:
//  1. Build distance tables (M × Ksub)
//     - For each subspace m: compute distances from query subvector to all centroids
//  2. For each vector's PQ code: look up pre-computed distances and sum
//  3. Take square root for final L2 distance
//  4. Filter by threshold, sort, and return top k
//
// Time Complexity: O(M × Ksub × dsub + n) where:
//   - M is number of subspaces
//   - Ksub is centroids per subspace
//   - dsub is subspace dimension
//   - n is number of vectors
func (s *pqIndexSearch) searchSingleQuery(query []float32) ([]VectorResult, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	// Validate
	if !s.index.trained {
		return nil, fmt.Errorf("index not trained")
	}

	if len(query) != s.index.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d",
			s.index.dim, len(query))
	}

	if len(s.index.codes) == 0 {
		return []VectorResult{}, nil
	}

	// Preprocess query
	preprocessedQuery, err := s.index.distance.Preprocess(query)
	if err != nil {
		return nil, err
	}

	// Build distance tables
	distTables := make([][]float32, s.index.M)

	for m := 0; m < s.index.M; m++ {
		// Extract query subvector
		start := m * s.index.dsub
		end := start + s.index.dsub
		querySubspace := preprocessedQuery[start:end]

		// Compute squared distances to all centroids
		distTables[m] = make([]float32, s.index.Ksub)
		for ksub := 0; ksub < s.index.Ksub; ksub++ {
			centroid := s.index.codebooks[m][ksub*s.index.dsub : (ksub+1)*s.index.dsub]

			// L2 squared distance
			var dist float32
			for i := range querySubspace {
				diff := querySubspace[i] - centroid[i]
				dist += diff * diff
			}
			distTables[m][ksub] = dist
		}
	}

	// Compute approximate distances using table lookups
	type result struct {
		vector   VectorNode
		distance float32
	}
	results := make([]result, 0, len(s.index.codes))

	for i, code := range s.index.codes {
		// Sum squared distances across subspaces
		dist := float32(0)
		for m := 0; m < s.index.M; m++ {
			dist += distTables[m][code[m]]
		}

		// Take square root for L2 distance
		finalDist := float32(math.Sqrt(float64(dist)))

		// Apply threshold filter
		if s.threshold > 0 && finalDist > s.threshold {
			continue
		}

		results = append(results, result{
			vector:   s.index.vectorNodes[i],
			distance: finalDist,
		})
	}

	// Sort by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].distance < results[j].distance
	})

	// Return top k
	k := s.k
	if k > len(results) {
		k = len(results)
	}

	finalResults := make([]VectorResult, k)
	for i := 0; i < k; i++ {
		finalResults[i] = VectorResult{
			Node:  results[i].vector,
			Score: results[i].distance,
		}
	}

	return finalResults, nil
}
