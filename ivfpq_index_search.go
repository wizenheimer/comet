package comet

import (
	"fmt"
	"math"
	"sort"
)

// Compile-time checks to ensure ivfpqIndexSearch implements VectorSearch
var _ VectorSearch = (*ivfpqIndexSearch)(nil)

// ivfpqIndexSearch implements VectorSearch for IVFPQ.
type ivfpqIndexSearch struct {
	index     *IVFPQIndex
	queries   [][]float32
	nodeIDs   []uint32
	k         int
	nprobes   int
	threshold float32
}

// WithQuery sets the query vector(s) - supports single or batch queries.
// Can be combined with WithNode to search from both direct queries and node-based queries.
func (s *ivfpqIndexSearch) WithQuery(queries ...[]float32) VectorSearch {
	s.queries = queries
	return s
}

// WithNode sets the node ID(s) to search from - supports single or batch nodes.
// Can be combined with WithQuery to search from both direct queries and node-based queries.
func (s *ivfpqIndexSearch) WithNode(nodeIDs ...uint32) VectorSearch {
	s.nodeIDs = nodeIDs
	return s
}

func (s *ivfpqIndexSearch) WithK(k int) VectorSearch {
	s.k = k
	return s
}

func (s *ivfpqIndexSearch) WithNProbes(nprobes int) VectorSearch {
	s.nprobes = nprobes
	return s
}

func (s *ivfpqIndexSearch) WithThreshold(threshold float32) VectorSearch {
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
func (s *ivfpqIndexSearch) Execute() ([]VectorNode, error) {
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
func (s *ivfpqIndexSearch) lookupNodeVectors() ([][]float32, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	queries := make([][]float32, 0, len(s.nodeIDs))
	for _, nodeID := range s.nodeIDs {
		found := false
		for _, list := range s.index.lists {
			for _, cv := range list {
				if cv.Node.ID() == nodeID {
					queries = append(queries, cv.Node.Vector())
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

// searchSingleQuery performs IVFPQ search with asymmetric distance.
//
// Algorithm:
//  1. Find nprobe nearest IVF centroids
//  2. For each probed cluster:
//     a. Compute query residual = query - centroid
//     b. Build PQ distance table for query residual
//     c. Compute distances using table lookups
//  3. Sort and return top k
func (s *ivfpqIndexSearch) searchSingleQuery(query []float32) ([]VectorNode, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	// Validate
	if !s.index.trained {
		return nil, fmt.Errorf("index must be trained before searching")
	}

	if len(query) != s.index.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d",
			s.index.dim, len(query))
	}

	// Sanitize nprobes
	nprobes := s.nprobes
	if nprobes <= 0 || nprobes > s.index.nlist {
		nprobes = s.index.nlist
	}

	// Preprocess query
	preprocessedQuery, err := s.index.distance.Preprocess(query)
	if err != nil {
		return nil, err
	}

	// STEP 1: Find nearest IVF centroids
	type centroidDist struct {
		index    int
		distance float32
	}

	centroidDistances := make([]centroidDist, len(s.index.centroids))
	for i, centroid := range s.index.centroids {
		dist := s.index.distance.Calculate(preprocessedQuery, centroid)
		centroidDistances[i] = centroidDist{index: i, distance: dist}
	}

	// Sort centroids by distance
	sort.Slice(centroidDistances, func(i, j int) bool {
		return centroidDistances[i].distance < centroidDistances[j].distance
	})

	// STEP 2: Search in nprobe nearest clusters
	type result struct {
		vector   VectorNode
		distance float32
	}
	var results []result

	for i := 0; i < nprobes; i++ {
		listIdx := centroidDistances[i].index
		centroid := s.index.centroids[listIdx]

		// Compute query residual for this cluster
		queryResidual := make([]float32, s.index.dim)
		for d := 0; d < s.index.dim; d++ {
			queryResidual[d] = preprocessedQuery[d] - centroid[d]
		}

		// Build PQ distance table for query residual
		distTables := s.computeDistanceTables(queryResidual)

		// Compute distances for all vectors in this list
		for _, cv := range s.index.lists[listIdx] {
			// Asymmetric distance using table lookups
			dist := s.asymmetricDistance(distTables, cv.Code)

			// Apply threshold filter
			if s.threshold > 0 && dist > s.threshold {
				continue
			}

			results = append(results, result{
				vector:   cv.Node,
				distance: dist,
			})
		}
	}

	// STEP 3: Sort and return top k
	sort.Slice(results, func(i, j int) bool {
		return results[i].distance < results[j].distance
	})

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

// computeDistanceTables builds distance table for query residual.
func (s *ivfpqIndexSearch) computeDistanceTables(queryResidual []float32) [][]float32 {
	distTables := make([][]float32, s.index.M)

	for m := 0; m < s.index.M; m++ {
		// Extract query subspace
		start := m * s.index.dsub
		end := start + s.index.dsub
		querySubspace := queryResidual[start:end]

		// Compute distances to all centroids
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

	return distTables
}

// asymmetricDistance computes distance using PQ code and distance tables.
func (s *ivfpqIndexSearch) asymmetricDistance(distTables [][]float32, code []uint8) float32 {
	dist := float32(0)
	for m := 0; m < s.index.M; m++ {
		dist += distTables[m][code[m]]
	}
	return float32(math.Sqrt(float64(dist)))
}
