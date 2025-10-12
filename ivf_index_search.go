package comet

import (
	"fmt"
	"sort"
)

// Compile-time checks to ensure ivfIndexSearch implements VectorSearch
var _ VectorSearch = (*ivfIndexSearch)(nil)

// ivfIndexSearch implements the VectorSearch interface for IVF index.
//
// IVF (Inverted File) search performs coarse-to-fine search:
//   - Finds nprobes nearest cluster centroids
//   - Searches only vectors in those clusters
//   - Returns top k from candidates
type ivfIndexSearch struct {
	index     *IVFIndex
	queries   [][]float32
	nodeIDs   []uint32
	k         int
	nprobes   int
	threshold float32
}

// WithQuery sets the query vector(s) - supports single or batch queries.
// Can be combined with WithNode to search from both direct queries and node-based queries.
func (s *ivfIndexSearch) WithQuery(queries ...[]float32) VectorSearch {
	s.queries = queries
	return s
}

// WithNode sets the node ID(s) to search from - supports single or batch nodes.
// Can be combined with WithQuery to search from both direct queries and node-based queries.
func (s *ivfIndexSearch) WithNode(nodeIDs ...uint32) VectorSearch {
	s.nodeIDs = nodeIDs
	return s
}

// WithK sets the number of results to return.
// Defaults to 10 if not set.
func (s *ivfIndexSearch) WithK(k int) VectorSearch {
	s.k = k
	return s
}

// WithNProbes sets the number of inverted lists to probe during search.
//
// Higher nprobes = better recall but slower search.
// Typical values: 1 to sqrt(nlist).
// If not set or invalid, defaults to nlist (exhaustive search).
func (s *ivfIndexSearch) WithNProbes(nprobes int) VectorSearch {
	s.nprobes = nprobes
	return s
}

// WithEfSearch is a no-op for IVF index (efSearch is used by HNSW).
// IVF uses nprobes parameter instead.
func (s *ivfIndexSearch) WithEfSearch(efSearch int) VectorSearch {
	return s
}

// WithThreshold sets a distance threshold for results (optional).
// Only results with distance <= threshold will be returned.
func (s *ivfIndexSearch) WithThreshold(threshold float32) VectorSearch {
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
//   - error: Returns error if search configuration is invalid or index not trained
func (s *ivfIndexSearch) Execute() ([]VectorNode, error) {
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
//
// Searches through all inverted lists to find vectors by ID.
// Returns error if any node ID is not found.
func (s *ivfIndexSearch) lookupNodeVectors() ([][]float32, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	queries := make([][]float32, 0, len(s.nodeIDs))
	for _, nodeID := range s.nodeIDs {
		found := false
		for _, list := range s.index.lists {
			for _, v := range list {
				if v.ID() == nodeID {
					queries = append(queries, v.Vector())
					found = true
					break
				}
			}
			if found {
				break
			}
		}

		if !found {
			return nil, fmt.Errorf("node ID %d not found in index", nodeID)
		}
	}

	return queries, nil
}

// searchSingleQuery performs the core IVF search for a single query vector.
//
// Algorithm:
// 1. Find the nprobes nearest centroids to the query
// 2. Collect all candidate vectors from those nprobes inverted lists
// 3. Compute exact distances from query to all candidates
// 4. Sort candidates by distance and return top-k
//
// Time Complexity: O(nlist + nprobes × (n/nlist) × dim + candidates × log(candidates))
func (s *ivfIndexSearch) searchSingleQuery(query []float32) ([]VectorNode, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	// Verify index is trained
	if !s.index.trained {
		return nil, fmt.Errorf("index must be trained before searching")
	}

	// Validate query dimension
	if len(query) != s.index.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d",
			s.index.dim, len(query))
	}

	// Sanitize nprobes
	nprobes := s.nprobes
	if nprobes <= 0 || nprobes > s.index.nlist {
		nprobes = s.index.nlist
	}

	// Preprocess the query according to the distance metric
	preprocessedQuery, err := s.index.distance.Preprocess(query)
	if err != nil {
		return nil, err
	}

	// ═══════════════════════════════════════════════════════════════════════════
	// STEP 1: Find nprobes nearest centroids
	// ═══════════════════════════════════════════════════════════════════════════
	type centroidDist struct {
		index    int
		distance float32
	}

	centroidDistances := make([]centroidDist, len(s.index.centroids))
	for i, centroid := range s.index.centroids {
		dist := s.index.distance.Calculate(preprocessedQuery, centroid)
		centroidDistances[i] = centroidDist{index: i, distance: dist}
	}

	// Sort centroids by distance to query
	sort.Slice(centroidDistances, func(i, j int) bool {
		return centroidDistances[i].distance < centroidDistances[j].distance
	})

	// ═══════════════════════════════════════════════════════════════════════════
	// STEP 2: Collect candidates from nprobes nearest lists
	// ═══════════════════════════════════════════════════════════════════════════
	type candidate struct {
		vector   VectorNode
		distance float32
	}

	candidates := make([]candidate, 0)

	for i := 0; i < nprobes; i++ {
		listIdx := centroidDistances[i].index

		// Search all vectors in this inverted list
		for _, v := range s.index.lists[listIdx] {
			dist := s.index.distance.Calculate(preprocessedQuery, v.Vector())

			// Apply threshold filter if set
			if s.threshold > 0 && dist > s.threshold {
				continue
			}

			candidates = append(candidates, candidate{vector: v, distance: dist})
		}
	}

	// ═══════════════════════════════════════════════════════════════════════════
	// STEP 3: Sort candidates and return top-k
	// ═══════════════════════════════════════════════════════════════════════════
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].distance < candidates[j].distance
	})

	// Take top k results
	k := s.k
	if k > len(candidates) {
		k = len(candidates)
	}

	results := make([]VectorNode, k)
	for i := 0; i < k; i++ {
		results[i] = candidates[i].vector
	}

	return results, nil
}
