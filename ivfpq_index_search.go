package comet

import (
	"fmt"
	"math"
	"sort"
)

// Compile-time checks to ensure ivfpqIndexSearch implements VectorSearch
var _ VectorSearch = (*ivfpqIndexSearch)(nil)

// ivfpqIndexSearch implements VectorSearch for IVFPQ.
//
// IVFPQ combines IVF clustering with PQ compression:
//   - Uses IVF for coarse search (nprobes nearest clusters)
//   - Uses PQ for fast approximate distance computation within clusters
//   - Provides best of both worlds: speed and memory efficiency
type ivfpqIndexSearch struct {
	index           *IVFPQIndex
	queries         [][]float32
	nodeIDs         []uint32
	documentIDs     []uint32
	k               int
	nprobes         int
	threshold       float32
	aggregationKind ScoreAggregationKind
	cutoff          int
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

// WithK sets the number of results to return.
// Defaults to 10 if not set.
func (s *ivfpqIndexSearch) WithK(k int) VectorSearch {
	s.k = k
	return s
}

// WithNProbes sets the number of inverted lists to probe during search.
//
// Higher nprobes = better recall but slower search.
// Typical values: 1 to sqrt(nlist).
// If not set or invalid, defaults to nlist (exhaustive search).
func (s *ivfpqIndexSearch) WithNProbes(nprobes int) VectorSearch {
	s.nprobes = nprobes
	return s
}

// WithEfSearch is a no-op for IVFPQ index (efSearch is used by HNSW).
// IVFPQ uses nprobes parameter instead.
func (s *ivfpqIndexSearch) WithEfSearch(efSearch int) VectorSearch {
	return s
}

// WithThreshold sets a distance threshold for results (optional).
// Only results with distance <= threshold will be returned.
func (s *ivfpqIndexSearch) WithThreshold(threshold float32) VectorSearch {
	s.threshold = threshold
	return s
}

// WithScoreAggregation sets the strategy for aggregating scores when the same node
// appears in results from multiple queries or nodes.
func (s *ivfpqIndexSearch) WithScoreAggregation(kind ScoreAggregationKind) VectorSearch {
	s.aggregationKind = kind
	return s
}

// WithCutoff sets the autocut parameter for automatically determining result cutoff.
// A value of -1 (default) disables autocut. Otherwise, specifies number of extrema to find.
func (s *ivfpqIndexSearch) WithCutoff(cutoff int) VectorSearch {
	s.cutoff = cutoff
	return s
}

// WithDocumentIDs sets the eligible document IDs for pre-filtering.
// Only vectors with IDs in this set will be considered as candidates.
// If empty, all documents are eligible (default behavior).
func (s *ivfpqIndexSearch) WithDocumentIDs(docIDs ...uint32) VectorSearch {
	s.documentIDs = docIDs
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
func (s *ivfpqIndexSearch) Execute() ([]VectorResult, error) {
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

	return results, nil
}

// lookupNodeVectors converts node IDs to their corresponding vectors.
//
// Searches through all inverted lists to find vectors by ID.
// Returns error if any node ID is not found.
func (s *ivfpqIndexSearch) lookupNodeVectors() ([][]float32, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	queries := make([][]float32, 0, len(s.nodeIDs))
	for _, nodeID := range s.nodeIDs {
		found := false
		for _, list := range s.index.lists {
			for _, cv := range list {
				if cv.Node.ID() == nodeID {
					// SOFT DELETE CHECK: Skip deleted nodes
					if s.index.deletedNodes.Contains(nodeID) {
						return nil, fmt.Errorf("node ID %d not found in index (deleted)", nodeID)
					}
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

// searchSingleQuery performs IVFPQ search with asymmetric distance for a single query.
//
// IVFPQ TWO-LEVEL QUANTIZATION:
// - Level 1 (IVF): Coarse quantization using cluster centroids
// - Level 2 (PQ): Fine quantization of residuals within clusters
//
// Algorithm:
//  1. Find nprobes nearest IVF centroids (coarse search)
//  2. For each probed cluster:
//     a. Compute query residual = query - centroid
//     b. Build PQ distance table (M × Ksub) for query residual
//     c. For each vector in cluster: compute distance using table lookups
//  3. Collect all candidates from probed clusters
//  4. Filter by threshold, sort by distance, and return top k
//
// Time Complexity: O(nlist + nprobes × (M × Ksub × dsub + n/nlist)) where:
//   - nlist is number of IVF clusters
//   - nprobes is number of clusters to search
//   - M is number of PQ subspaces
//   - Ksub is centroids per subspace
//   - dsub is subspace dimension
//   - n is total number of vectors
func (s *ivfpqIndexSearch) searchSingleQuery(query []float32) ([]VectorResult, error) {
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
	// Create document filter for metadata pre-filtering
	docFilter := NewDocumentFilter(s.documentIDs)
	defer ReturnDocumentFilter(docFilter)

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
			// SOFT DELETE CHECK: Skip deleted nodes
			if s.index.deletedNodes.Contains(cv.Node.ID()) {
				continue
			}

			// Apply document ID filter if set (metadata pre-filtering)
			if docFilter.ShouldSkip(cv.Node.ID()) {
				continue
			}

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

	k := sanitizeK(s.k, len(results))

	finalResults := make([]VectorResult, k)
	for i := 0; i < k; i++ {
		finalResults[i] = VectorResult{
			Node:  results[i].vector,
			Score: results[i].distance,
		}
	}

	return finalResults, nil
}

// computeDistanceTables builds distance tables for query residual.
//
// For each subspace m:
//   - Extracts the query subvector
//   - Computes L2 squared distances to all Ksub centroids
//
// Returns M × Ksub distance table for fast lookups during search.
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

// asymmetricDistance computes approximate distance using PQ code and distance tables.
//
// For each subspace m:
//   - Uses code[m] to look up pre-computed distance from table
//   - Sums squared distances across all subspaces
//
// Returns L2 distance (square root of sum).
func (s *ivfpqIndexSearch) asymmetricDistance(distTables [][]float32, code []uint8) float32 {
	dist := float32(0)
	for m := 0; m < s.index.M; m++ {
		dist += distTables[m][code[m]]
	}
	return float32(math.Sqrt(float64(dist)))
}
