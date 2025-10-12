package comet

import (
	"fmt"
	"math"
	"sort"
)

// Compile-time checks to ensure pqIndexSearch implements VectorSearch
var _ VectorSearch = (*pqIndexSearch)(nil)

// pqIndexSearch implements the VectorSearch interface for PQ index.
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

func (s *pqIndexSearch) WithK(k int) VectorSearch {
	s.k = k
	return s
}

func (s *pqIndexSearch) WithNProbes(nprobes int) VectorSearch {
	// PQ doesn't use nprobes, ignored
	return s
}

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
//   - []VectorNode: Search results sorted by distance
//   - error: Returns error if search configuration is invalid
func (s *pqIndexSearch) Execute() ([]VectorNode, error) {
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

// searchSingleQuery performs asymmetric PQ distance search.
//
// Algorithm:
//  1. Build distance tables (M × K)
//  2. For each code, look up distances and sum
//  3. Sort and return top k
func (s *pqIndexSearch) searchSingleQuery(query []float32) ([]VectorNode, error) {
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
		return []VectorNode{}, nil
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

	finalResults := make([]VectorNode, k)
	for i := 0; i < k; i++ {
		finalResults[i] = results[i].vector
	}

	return finalResults, nil
}
