package comet

import (
	"container/heap"
	"fmt"
	"sort"
	"sync"
)

// Compile-time checks to ensure hnswIndexSearch implements VectorSearch
var _ VectorSearch = (*hnswIndexSearch)(nil)

// ============================================================================
// HEAP POOLS FOR ALLOCATION OPTIMIZATION
// ============================================================================

// minHeapPool is a sync.Pool for min-heaps to reduce allocations during search.
//
// OPTIMIZATION RATIONALE:
// - searchLayer creates new heaps on every call
// - searchLayer is called frequently during insert and search operations
// - Pooling reduces GC pressure and improves throughput
var minHeapPool = sync.Pool{
	New: func() interface{} {
		h := &minHeap{}
		heap.Init(h)
		return h
	},
}

// maxHeapPool is a sync.Pool for max-heaps to reduce allocations during search.
var maxHeapPool = sync.Pool{
	New: func() interface{} {
		h := &maxHeap{}
		heap.Init(h)
		return h
	},
}

// ============================================================================
// SEARCH IMPLEMENTATION
// ============================================================================

// hnswIndexSearch implements VectorSearch for HNSW.
//
// HNSW search performs hierarchical graph traversal:
//   - Starts at the top layer with few nodes
//   - Greedily descends through layers
//   - Performs comprehensive search at layer 0
//   - Returns top k nearest neighbors
type hnswIndexSearch struct {
	index     *HNSWIndex
	queries   [][]float32
	nodeIDs   []uint32
	k         int
	efSearch  int // Per-search override, 0 means use index default
	threshold float32
}

// WithQuery sets the query vector(s) - supports single or batch queries.
// Can be combined with WithNode to search from both direct queries and node-based queries.
func (s *hnswIndexSearch) WithQuery(queries ...[]float32) VectorSearch {
	s.queries = queries
	return s
}

// WithNode sets the node ID(s) to search from - supports single or batch nodes.
// Can be combined with WithQuery to search from both direct queries and node-based queries.
func (s *hnswIndexSearch) WithNode(nodeIDs ...uint32) VectorSearch {
	s.nodeIDs = nodeIDs
	return s
}

// WithK sets the number of results to return.
// Defaults to 10 if not set.
func (s *hnswIndexSearch) WithK(k int) VectorSearch {
	s.k = k
	return s
}

// WithNProbes is a no-op for HNSW (nprobes is used by IVF-based indexes).
// HNSW uses efSearch parameter (set via WithEfSearch) instead.
func (s *hnswIndexSearch) WithNProbes(nprobes int) VectorSearch {
	// HNSW doesn't use nprobes
	return s
}

// WithEfSearch sets the efSearch parameter for this search.
//
// efSearch controls the size of the dynamic candidate list during search.
// Higher efSearch = better recall but slower search.
// If not set (or set to 0), uses the index's default efSearch value.
//
// Typical values: 100-500 for good recall/speed tradeoff.
func (s *hnswIndexSearch) WithEfSearch(efSearch int) VectorSearch {
	s.efSearch = efSearch
	return s
}

// WithThreshold sets a distance threshold for results (optional).
// Only results with distance <= threshold will be returned.
func (s *hnswIndexSearch) WithThreshold(threshold float32) VectorSearch {
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
func (s *hnswIndexSearch) Execute() ([]VectorNode, error) {
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

// lookupNodeVectors retrieves vectors for the specified node IDs.
//
// Returns error if any node ID is not found or has been deleted.
func (s *hnswIndexSearch) lookupNodeVectors() ([][]float32, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	queries := make([][]float32, 0, len(s.nodeIDs))
	for _, nodeID := range s.nodeIDs {
		node, exists := s.index.nodes[nodeID]
		if !exists || s.index.deletedNodes.Contains(nodeID) {
			return nil, fmt.Errorf("node ID %d not found or deleted", nodeID)
		}
		queries = append(queries, node.Vector())
	}

	return queries, nil
}

// searchSingleQuery performs HNSW hierarchical search for a single query.
//
// THE ALGORITHM:
// PHASE 1: Greedy search through upper layers (rough navigation)
//   - Start at entry point (top layer)
//   - For each layer from maxLevel down to 1:
//   - Greedily move to closer neighbors until local minimum found
//
// PHASE 2: Comprehensive search at layer 0 (precise search)
//   - Use searchLayer with efSearch parameter for thorough exploration
//   - Maintains candidate and result heaps for efficiency
//
// PHASE 3: Return top k results
//   - Filter by threshold if set
//   - Sort by distance and return k nearest
//
// Time Complexity: O(M × efSearch × log n) where:
//   - M is the number of connections per node
//   - efSearch is the candidate list size
//   - n is the number of nodes in the index
func (s *hnswIndexSearch) searchSingleQuery(query []float32) ([]VectorNode, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	// Validate
	if len(query) != s.index.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d",
			s.index.dim, len(query))
	}

	if len(s.index.nodes) == 0 || s.index.maxLevel == -1 {
		return []VectorNode{}, nil
	}

	// Preprocess query
	preprocessedQuery, err := s.index.distance.Preprocess(query)
	if err != nil {
		return nil, err
	}

	// ═══════════════════════════════════════════════════════════════════════
	// PHASE 1: GREEDY SEARCH THROUGH UPPER LAYERS
	// ═══════════════════════════════════════════════════════════════════════
	curr := s.index.entryPoint
	currDist := s.index.distance.Calculate(preprocessedQuery, s.index.nodes[curr].Vector())

	for lc := s.index.maxLevel; lc > 0; lc-- {
		changed := true
		for changed {
			changed = false
			node := s.index.nodes[curr]

			if lc < len(node.Edges) {
				for _, neighborID := range node.Edges[lc] {
					// SOFT DELETE CHECK: Skip deleted neighbors
					if s.index.deletedNodes.Contains(neighborID) {
						continue
					}

					d := s.index.distance.Calculate(preprocessedQuery, s.index.nodes[neighborID].Vector())
					if d < currDist {
						currDist = d
						curr = neighborID
						changed = true
					}
				}
			}
		}
	}

	// ═══════════════════════════════════════════════════════════════════════
	// PHASE 2: COMPREHENSIVE SEARCH AT LAYER 0
	// ═══════════════════════════════════════════════════════════════════════
	// Use per-search efSearch if provided, otherwise fall back to index default
	efSearch := s.efSearch
	if efSearch <= 0 {
		efSearch = s.index.efSearch
	}
	candidates := s.index.searchLayer(preprocessedQuery, curr, efSearch, 0)

	// ═══════════════════════════════════════════════════════════════════════
	// PHASE 3: RETURN TOP K
	// ═══════════════════════════════════════════════════════════════════════
	type result struct {
		vector   VectorNode
		distance float32
	}
	results := make([]result, 0, len(candidates))

	for _, c := range candidates {
		if s.threshold > 0 && c.distance > s.threshold {
			continue
		}

		results = append(results, result{
			vector:   s.index.nodes[c.id].VectorNode,
			distance: c.distance,
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

// ============================================================================
// HEAP STRUCTURES FOR EFFICIENT SEARCH
// ============================================================================

// candidate represents a node and its distance during search.
//
// Used in both candidate and result heaps during HNSW search.
type candidate struct {
	id       uint32  // Node ID
	distance float32 // Distance from query
}

// minHeap is a min-heap of candidates (closest on top).
//
// Used for the candidate queue during search - we always want to explore
// the nearest unvisited node next.
type minHeap []candidate

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].distance < h[j].distance }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *minHeap) Push(x interface{}) {
	*h = append(*h, x.(candidate))
}

func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// newMinHeap gets a min-heap from the pool.
//
// IMPORTANT: Caller must call putMinHeap() when done to return to pool.
func newMinHeap() *minHeap {
	return minHeapPool.Get().(*minHeap)
}

// putMinHeap returns a min-heap to the pool after resetting it.
//
// CRITICAL: This must be called when done with the heap to enable pooling.
func putMinHeap(h *minHeap) {
	// Reset heap by truncating to zero length
	// This retains the underlying capacity for reuse
	*h = (*h)[:0]
	minHeapPool.Put(h)
}

// maxHeap is a max-heap of candidates (farthest on top).
//
// Used for the result set during search - we keep the k best candidates
// and can quickly remove the worst one when we find a better candidate.
type maxHeap []candidate

func (h maxHeap) Len() int           { return len(h) }
func (h maxHeap) Less(i, j int) bool { return h[i].distance > h[j].distance }
func (h maxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *maxHeap) Push(x interface{}) {
	*h = append(*h, x.(candidate))
}

func (h *maxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// newMaxHeap gets a max-heap from the pool.
//
// IMPORTANT: Caller must call putMaxHeap() when done to return to pool.
func newMaxHeap() *maxHeap {
	return maxHeapPool.Get().(*maxHeap)
}

// putMaxHeap returns a max-heap to the pool after resetting it.
//
// CRITICAL: This must be called when done with the heap to enable pooling.
func putMaxHeap(h *maxHeap) {
	// Reset heap by truncating to zero length
	// This retains the underlying capacity for reuse
	*h = (*h)[:0]
	maxHeapPool.Put(h)
}
