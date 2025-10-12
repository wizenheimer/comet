// Package comet implements HNSW (Hierarchical Navigable Small World).
//
// WHAT IS HNSW?
// HNSW is a state-of-the-art graph-based algorithm for approximate nearest neighbor
// search. It builds a multi-layered graph where search is O(log n) - incredibly fast!
//
// # HIERARCHICAL SKIP-LIST-LIKE STRUCTURE
//
// Layer 2: Few nodes, long-range connections (highways)
// Layer 1: More nodes, medium-range connections (state roads)
// Layer 0: All nodes, short-range connections (local streets)
//
// Search starts at top layer and descends, getting more refined at each level!
//
// PERFORMANCE:
//   - Search: O(log n) with ~95-99% recall
//   - Memory: 2-3x raw vectors (stores full precision + graph)
//   - Build: O(log n) per insertion
//   - Use case: When speed and accuracy are critical
//
// TIME COMPLEXITY:
//   - Insert: O(M × efConstruction × log n)
//   - Search: O(M × efSearch × log n)
//   - Typical: 10-50 distance calculations for 1M vectors!
package comet

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"sync"

	"github.com/RoaringBitmap/roaring"
)

// Compile-time checks to ensure HNSWIndex implements VectorIndex
var _ VectorIndex = (*HNSWIndex)(nil)

// ============================================================================
// NODE STRUCTURE
// ============================================================================

// hnswNode represents a vertex in the HNSW graph.
//
// Each node exists in layers 0 through Level (inclusive).
type hnswNode struct {
	// Embed VectorNode to avoid duplicate ID field
	VectorNode

	// Level is the maximum layer this node participates in
	Level int

	// Edges stores neighbor IDs at each layer
	// edges[0] = neighbors at layer 0 (2*M neighbors)
	// edges[i] = neighbors at layer i (M neighbors for i > 0)
	Edges [][]uint32
}

// newHnswNode creates a new HNSW node from a VectorNode.
//
// Parameters:
//   - vector: The vector node to wrap
//   - level: The maximum layer this node participates in
//
// Returns:
//   - *hnswNode: Properly initialized node with empty edge arrays
func newHnswNode(vector VectorNode, level int) *hnswNode {
	// Pre-allocate edge arrays for all layers
	edges := make([][]uint32, level+1)
	for i := 0; i <= level; i++ {
		edges[i] = make([]uint32, 0)
	}

	return &hnswNode{
		VectorNode: vector,
		Level:      level,
		Edges:      edges,
	}
}

// ============================================================================
// INDEX CONFIGURATION
// ============================================================================

// DefaultHNSWConfig returns recommended default configuration parameters.
//
// Returns:
//   - m: connections per layer (16) - layer 0 uses 2*M connections
//   - efConstruction: candidate list size during construction (200)
//   - efSearch: candidate list size during search (200)
func DefaultHNSWConfig() (m, efConstruction, efSearch int) {
	return 16, 200, 200
}

// ============================================================================
// INDEX STRUCTURE
// ============================================================================

// HNSWIndex implements HNSW (Hierarchical Navigable Small World).
//
// Memory layout:
//   - Vectors: n × dim × 4 bytes
//   - Graph: n × M × avgLayers × 4 bytes (uint32 IDs)
//   - Total: typically 2-3x raw vector size
//
// Thread-safety: All public methods use read-write mutex.
type HNSWIndex struct {
	// dim is vector dimensionality
	dim int

	// distanceKind specifies the distance metric
	distanceKind DistanceKind

	// distance is the distance calculator
	distance Distance

	// M is connections per layer (except layer 0)
	M int

	// efConstruction is construction-time candidate list size
	efConstruction int

	// efSearch is search-time candidate list size
	efSearch int

	// maxLevel is the highest layer with at least one node
	maxLevel int

	// entryPoint is the node ID to start search from
	entryPoint uint32

	// nodes stores all graph vertices
	// Using map[uint32]*hnswNode for flexibility with IDs
	nodes map[uint32]*hnswNode

	// deletedNodes tracks soft-deleted IDs using roaring bitmap
	// CRITICAL OPTIMIZATION: RoaringBitmap is much more efficient than map[int64]bool
	// - O(log n) membership test vs O(1) but with better memory efficiency
	// - Compressed bitmap representation
	// - Fast iteration for batch operations
	deletedNodes *roaring.Bitmap

	// levelMult is mL = 1/ln(M)
	levelMult float64

	// mu provides thread-safe access
	mu sync.RWMutex

	// nextID is auto-incrementing ID
	nextID uint32
}

// NewHNSWIndex creates a new HNSW index.
//
// Creates an empty multi-layered graph. Vectors can be added immediately
// (no training required unlike IVF/IVFPQ).
//
// Parameters:
//   - dim: Vector dimensionality
//   - distanceKind: Distance metric
//   - m: Connections per layer (except layer 0 which uses 2*M). Typical: 12-48, pass 0 for default (16)
//   - efConstruction: Candidate list size during construction. Higher = better graph, slower build. Typical: 100-500, pass 0 for default (200)
//   - efSearch: Candidate list size during search. Higher = better recall, slower search. Pass 0 for default (200)
//
// Returns:
//   - *HNSWIndex: New empty HNSW index
//   - error: Returns error if parameters invalid
func NewHNSWIndex(dim int, distanceKind DistanceKind, m, efConstruction, efSearch int) (*HNSWIndex, error) {
	// Validate dimension
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}

	// Apply defaults
	if m <= 0 {
		m = 16
	}
	if efConstruction <= 0 {
		efConstruction = 200
	}
	if efSearch <= 0 {
		efSearch = efConstruction
	}

	// Create distance calculator
	distance, err := NewDistance(distanceKind)
	if err != nil {
		return nil, err
	}

	return &HNSWIndex{
		dim:            dim,
		distanceKind:   distanceKind,
		distance:       distance,
		M:              m,
		efConstruction: efConstruction,
		efSearch:       efSearch,
		maxLevel:       -1,                         // -1 for empty graph
		entryPoint:     0,                          // No entry point yet
		nodes:          make(map[uint32]*hnswNode), // Empty graph
		deletedNodes:   roaring.New(),              // CRITICAL: Use roaring bitmap
		levelMult:      1.0 / math.Log(float64(m)), // mL = 1/ln(M)
		nextID:         0,
	}, nil
}

// Train is a no-op for HNSW (no training required).
func (idx *HNSWIndex) Train(vectors []VectorNode) error {
	// HNSW doesn't require training
	return nil
}

// Add adds a vector to the index by inserting into the graph.
//
// CONCURRENCY OPTIMIZATION:
// All expensive operations (validation, level assignment, node preparation)
// are performed OUTSIDE the lock to minimize critical section time.
//
// Algorithm:
//  1. Assign random level (outside lock)
//  2. Prepare node structure (outside lock)
//  3. Critical section: graph modification
//  4. Insert and connect
func (idx *HNSWIndex) Add(vector VectorNode) error {
	// ════════════════════════════════════════════════════════════════════════
	// PHASE 1: VALIDATION (OUTSIDE LOCK)
	// ════════════════════════════════════════════════════════════════════════
	if len(vector.Vector()) != idx.dim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d",
			idx.dim, len(vector.Vector()))
	}

	// Preprocess vector
	if err := idx.distance.PreprocessInPlace(vector.Vector()); err != nil {
		return err
	}

	// ════════════════════════════════════════════════════════════════════════
	// PHASE 2: EXPENSIVE WORK (OUTSIDE LOCK)
	// ════════════════════════════════════════════════════════════════════════

	// Get ID (prepare outside lock, assign inside lock)
	var id uint32
	if vector.ID() != 0 {
		id = vector.ID()
	}

	// Assign random level (outside lock - RNG is expensive)
	level := idx.randomLevel()

	// ════════════════════════════════════════════════════════════════════════
	// PHASE 3: CRITICAL SECTION (WRITE LOCK)
	// ════════════════════════════════════════════════════════════════════════
	idx.mu.Lock()

	// Assign ID if needed (inside lock to ensure uniqueness)
	if id == 0 {
		id = idx.nextID
		idx.nextID++
	}

	// Update max level
	if level > idx.maxLevel {
		idx.maxLevel = level
	}

	// Create node using initializer
	node := newHnswNode(vector, level)

	// Check entryPoint inside write lock to avoid TOCTOU race
	if idx.entryPoint == 0 && len(idx.nodes) == 0 {
		idx.entryPoint = id
		idx.nodes[id] = node
		idx.mu.Unlock()
		return nil
	}

	// Insert into graph
	idx.insertNode(node)
	idx.nodes[id] = node

	idx.mu.Unlock()
	return nil
}

// Remove performs soft delete using roaring bitmap.
//
// CONCURRENCY OPTIMIZATION:
// - Uses read lock first (cheaper) to check if node exists
// - Only acquires write lock for the actual bitmap modification
// - Minimizes write lock contention
//
// SOFT DELETE MECHANISM:
// Instead of immediately removing (expensive O(n × M × L)),
// we mark as deleted in roaring bitmap. Deleted nodes are:
//   - Skipped during search
//   - Still in graph structure
//   - Not counted as active nodes
//
// Call Flush() periodically for actual cleanup.
func (idx *HNSWIndex) Remove(vector VectorNode) error {
	id := vector.ID()

	// ════════════════════════════════════════════════════════════════════════
	// STEP 1: CHECK EXISTENCE (READ LOCK - CHEAPER)
	// ════════════════════════════════════════════════════════════════════════
	idx.mu.RLock()
	_, exists := idx.nodes[id]
	alreadyDeleted := idx.deletedNodes.Contains(id)
	idx.mu.RUnlock()

	// Fast-fail validation outside of write lock
	if !exists {
		return fmt.Errorf("node %d not found", id)
	}
	if alreadyDeleted {
		return fmt.Errorf("node %d already deleted", id)
	}

	// ════════════════════════════════════════════════════════════════════════
	// STEP 2: MARK AS DELETED (WRITE LOCK - ONLY FOR BITMAP UPDATE)
	// ════════════════════════════════════════════════════════════════════════
	idx.mu.Lock()
	idx.deletedNodes.Add(id)
	idx.mu.Unlock()

	return nil
}

// Flush performs hard delete of soft-deleted nodes.
//
// WHEN TO CALL:
//   - After multiple Remove() calls (batch cleanup)
//   - When deleted nodes are significant (e.g., > 10% of index)
//   - During off-peak hours
//
// WHAT IT DOES:
// 1. Removes all edges pointing to deleted nodes
// 2. Deletes nodes from memory
// 3. Updates entry point if needed
// 4. Rebuilds graph structure
//
// COST: O(n × M × L) - expensive, so batch deletions
func (idx *HNSWIndex) Flush() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Quick exit if nothing to flush
	deletedCount := int(idx.deletedNodes.GetCardinality())
	if deletedCount == 0 {
		return nil
	}

	// ═══════════════════════════════════════════════════════════════════════
	// PHASE 1: REMOVE EDGES TO DELETED NODES
	// ═══════════════════════════════════════════════════════════════════════
	for _, otherNode := range idx.nodes {
		if idx.deletedNodes.Contains(otherNode.ID()) {
			continue
		}

		for lc := 0; lc < len(otherNode.Edges); lc++ {
			filtered := make([]uint32, 0, len(otherNode.Edges[lc]))

			for _, nid := range otherNode.Edges[lc] {
				// Keep edge only if target NOT deleted
				// RoaringBitmap Contains() is very fast - O(log n)
				if !idx.deletedNodes.Contains(nid) {
					filtered = append(filtered, nid)
				}
			}

			otherNode.Edges[lc] = filtered
		}
	}

	// ═══════════════════════════════════════════════════════════════════════
	// PHASE 2: UPDATE ENTRY POINT IF NEEDED
	// ═══════════════════════════════════════════════════════════════════════
	if idx.deletedNodes.Contains(idx.entryPoint) {
		// Strategy 1: Find node at current maxLevel
		foundNewEntry := false
		for _, n := range idx.nodes {
			if !idx.deletedNodes.Contains(n.ID()) && n.Level == idx.maxLevel {
				idx.entryPoint = n.ID()
				foundNewEntry = true
				break
			}
		}

		// Strategy 2: Find highest level available
		if !foundNewEntry {
			maxFoundLevel := -1
			for _, n := range idx.nodes {
				if !idx.deletedNodes.Contains(n.ID()) && n.Level > maxFoundLevel {
					maxFoundLevel = n.Level
					idx.entryPoint = n.ID()
				}
			}

			if maxFoundLevel >= 0 {
				idx.maxLevel = maxFoundLevel
			} else {
				// ALL nodes deleted - reset to empty
				idx.entryPoint = 0
				idx.maxLevel = -1
			}
		}
	}

	// ═══════════════════════════════════════════════════════════════════════
	// PHASE 3: FREE MEMORY - HARD DELETE
	// ═══════════════════════════════════════════════════════════════════════
	// Use roaring bitmap's iterator for efficient traversal
	iter := idx.deletedNodes.Iterator()
	for iter.HasNext() {
		id := iter.Next()
		delete(idx.nodes, id)
	}

	// ═══════════════════════════════════════════════════════════════════════
	// PHASE 4: RESET DELETED TRACKING
	// ═══════════════════════════════════════════════════════════════════════
	idx.deletedNodes.Clear()

	return nil
}

// NewSearch creates a new search builder.
func (idx *HNSWIndex) NewSearch() VectorSearch {
	return &hnswIndexSearch{
		index: idx,
		k:     10,
	}
}

// Dimensions returns vector dimensionality.
func (idx *HNSWIndex) Dimensions() int {
	return idx.dim
}

// DistanceKind returns the distance metric.
func (idx *HNSWIndex) DistanceKind() DistanceKind {
	return idx.distanceKind
}

// Kind returns the index type.
func (idx *HNSWIndex) Kind() VectorIndexKind {
	return HNSWIndexKind
}

// Trained always returns true (HNSW doesn't require training).
func (idx *HNSWIndex) Trained() bool {
	return true
}

// SetEfSearch adjusts search-time candidate list size.
func (idx *HNSWIndex) SetEfSearch(ef int) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.efSearch = ef
}

// ============================================================================
// INTERNAL METHODS
// ============================================================================

// randomLevel assigns random level using geometric distribution.
func (idx *HNSWIndex) randomLevel() int {
	probability := 1.0 / float64(idx.M)
	level := 0

	// Cap at 16 to prevent pathological cases
	for level < 16 && rand.Float64() < probability {
		level++
	}

	return level
}

// insertNode inserts node into graph.
//
// INSERTION ALGORITHM:
// PHASE 1: Navigate to insertion point (upper layers)
// PHASE 2: Connect at each layer (from node's level down to 0)
//
// CONCURRENCY: This is an internal helper method. The caller MUST hold the write lock.
func (idx *HNSWIndex) insertNode(node *hnswNode) {
	// Navigate to insertion point
	curr := idx.entryPoint
	currDist := idx.distance.Calculate(node.Vector(), idx.nodes[curr].Vector())

	// Traverse upper layers
	for lc := idx.maxLevel; lc > node.Level; lc-- {
		changed := true
		for changed {
			changed = false
			currNode := idx.nodes[curr]

			if lc < len(currNode.Edges) {
				for _, neighborID := range currNode.Edges[lc] {
					// SOFT DELETE CHECK: Skip deleted neighbors
					if idx.deletedNodes.Contains(neighborID) {
						continue
					}

					d := idx.distance.Calculate(node.Vector(), idx.nodes[neighborID].Vector())
					if d < currDist {
						currDist = d
						curr = neighborID
						changed = true
					}
				}
			}
		}
	}

	// Insert and connect at each layer
	for lc := node.Level; lc >= 0; lc-- {
		candidates := idx.searchLayer(node.Vector(), curr, idx.efConstruction, lc)

		M := idx.M
		if lc == 0 {
			M *= 2
		}

		neighbors := idx.selectNeighbors(candidates, M)

		// Connect bidirectional edges
		for _, neighborID := range neighbors {
			node.Edges[lc] = append(node.Edges[lc], neighborID)

			neighbor := idx.nodes[neighborID]
			if lc <= neighbor.Level {
				neighbor.Edges[lc] = append(neighbor.Edges[lc], node.ID())

				if len(neighbor.Edges[lc]) > M {
					idx.pruneConnections(neighborID, lc, M)
				}
			}
		}

		if len(candidates) > 0 {
			curr = candidates[0].id
		}
	}
}

// searchLayer performs greedy search using heap-based algorithm.
//
// HEAP-BASED OPTIMIZATION:
// Uses two heaps for O(log ef) efficiency:
//   - candidates (min-heap): Nodes to explore, closest first
//   - result (max-heap): Best ef nodes found, worst first
//
// HEAP POOLING OPTIMIZATION:
// Gets heaps from sync.Pool and returns them when done to reduce allocations.
//
// CONCURRENCY: This is an internal helper method. The caller MUST hold at least a read lock.
func (idx *HNSWIndex) searchLayer(query []float32, entryPoint uint32, ef int, layer int) []candidate {
	// Track visited nodes using RoaringBitmap for efficiency
	visited := roaring.New()

	// Candidates heap: nodes to explore (min-heap)
	// Get from pool for allocation optimization
	candidates := newMinHeap()
	defer putMinHeap(candidates) // Return to pool when done

	// Results heap: best ef nodes (max-heap)
	// Get from pool for allocation optimization
	result := newMaxHeap()
	defer putMaxHeap(result) // Return to pool when done

	// Check entry point BEFORE adding to candidates
	if !idx.deletedNodes.Contains(entryPoint) {
		d := idx.distance.Calculate(query, idx.nodes[entryPoint].Vector())
		heap.Push(candidates, candidate{id: entryPoint, distance: d})
		heap.Push(result, candidate{id: entryPoint, distance: d})
	}
	visited.Add(entryPoint)

	// Main search loop
	for candidates.Len() > 0 {
		current := heap.Pop(candidates).(candidate)

		// Early termination check
		if result.Len() >= ef && current.distance > (*result)[0].distance {
			break
		}

		node := idx.nodes[current.id]
		if layer < len(node.Edges) {
			for _, neighborID := range node.Edges[layer] {
				// SOFT DELETE CHECK: Skip deleted neighbors
				if idx.deletedNodes.Contains(neighborID) {
					continue
				}

				if !visited.Contains(neighborID) {
					visited.Add(neighborID)

					d := idx.distance.Calculate(query, idx.nodes[neighborID].Vector())

					if result.Len() < ef || d < (*result)[0].distance {
						heap.Push(candidates, candidate{id: neighborID, distance: d})
						heap.Push(result, candidate{id: neighborID, distance: d})

						if result.Len() > ef {
							heap.Pop(result)
						}
					}
				}
			}
		}
	}

	// Extract results (do this before defer returns heaps to pool)
	finalResults := make([]candidate, result.Len())
	for i := result.Len() - 1; i >= 0; i-- {
		finalResults[i] = heap.Pop(result).(candidate)
	}

	return finalResults
}

// selectNeighbors selects M best neighbors.
//
// SIMPLE HEURISTIC: Pick M nearest neighbors by distance.
// Production systems may use more sophisticated heuristics like RNG
//
// CONCURRENCY: This is an internal helper method. The caller MUST hold the write lock.
func (idx *HNSWIndex) selectNeighbors(candidates []candidate, M int) []uint32 {
	if len(candidates) <= M {
		result := make([]uint32, len(candidates))
		for i, c := range candidates {
			result[i] = c.id
		}
		return result
	}

	// Sort by distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].distance < candidates[j].distance
	})

	result := make([]uint32, M)
	for i := 0; i < M; i++ {
		result[i] = candidates[i].id
	}
	return result
}

// pruneConnections reduces connections to M.
//
// WHY PRUNE?
// Limiting edges to M per node:
//   - Controls memory usage
//   - Prevents over-clustering
//   - Maintains navigability
//
// CONCURRENCY: This is an internal helper method. The caller MUST hold the write lock.
func (idx *HNSWIndex) pruneConnections(nodeID uint32, layer, M int) {
	node := idx.nodes[nodeID]

	// Build candidate list with distances
	candList := make([]candidate, 0, len(node.Edges[layer]))
	for _, nid := range node.Edges[layer] {
		if idx.nodes[nid] == nil {
			continue
		}
		d := idx.distance.Calculate(node.Vector(), idx.nodes[nid].Vector())
		candList = append(candList, candidate{id: nid, distance: d})
	}

	// Sort by distance
	sort.Slice(candList, func(i, j int) bool {
		return candList[i].distance < candList[j].distance
	})

	// Keep M nearest
	numNeighbors := M
	if len(candList) < M {
		numNeighbors = len(candList)
	}
	node.Edges[layer] = make([]uint32, numNeighbors)
	for i := 0; i < numNeighbors; i++ {
		node.Edges[layer][i] = candList[i].id
	}
}
