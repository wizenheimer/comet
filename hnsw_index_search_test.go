package comet

import (
	"container/heap"
	"math"
	"sync"
	"testing"
)

// ============================================================================
// HEAP STRUCTURE TESTS
// ============================================================================

// TestMinHeap tests min-heap operations
func TestMinHeap(t *testing.T) {
	h := newMinHeap()

	// Push candidates
	heap.Push(h, candidate{id: 1, distance: 5.0})
	heap.Push(h, candidate{id: 2, distance: 2.0})
	heap.Push(h, candidate{id: 3, distance: 8.0})
	heap.Push(h, candidate{id: 4, distance: 1.0})

	if h.Len() != 4 {
		t.Errorf("Expected heap length 4, got %d", h.Len())
	}

	// Pop should return minimum (1.0)
	c := heap.Pop(h).(candidate)
	if c.distance != 1.0 {
		t.Errorf("Expected min distance 1.0, got %.1f", c.distance)
	}

	// Next should be 2.0
	c = heap.Pop(h).(candidate)
	if c.distance != 2.0 {
		t.Errorf("Expected distance 2.0, got %.1f", c.distance)
	}

	// Next should be 5.0
	c = heap.Pop(h).(candidate)
	if c.distance != 5.0 {
		t.Errorf("Expected distance 5.0, got %.1f", c.distance)
	}

	// Last should be 8.0
	c = heap.Pop(h).(candidate)
	if c.distance != 8.0 {
		t.Errorf("Expected distance 8.0, got %.1f", c.distance)
	}

	if h.Len() != 0 {
		t.Errorf("Expected empty heap, got length %d", h.Len())
	}
}

// TestMaxHeap tests max-heap operations
func TestMaxHeap(t *testing.T) {
	h := newMaxHeap()

	// Push candidates
	heap.Push(h, candidate{id: 1, distance: 5.0})
	heap.Push(h, candidate{id: 2, distance: 2.0})
	heap.Push(h, candidate{id: 3, distance: 8.0})
	heap.Push(h, candidate{id: 4, distance: 1.0})

	if h.Len() != 4 {
		t.Errorf("Expected heap length 4, got %d", h.Len())
	}

	// Pop should return maximum (8.0)
	c := heap.Pop(h).(candidate)
	if c.distance != 8.0 {
		t.Errorf("Expected max distance 8.0, got %.1f", c.distance)
	}

	// Next should be 5.0
	c = heap.Pop(h).(candidate)
	if c.distance != 5.0 {
		t.Errorf("Expected distance 5.0, got %.1f", c.distance)
	}

	// Next should be 2.0
	c = heap.Pop(h).(candidate)
	if c.distance != 2.0 {
		t.Errorf("Expected distance 2.0, got %.1f", c.distance)
	}

	// Last should be 1.0
	c = heap.Pop(h).(candidate)
	if c.distance != 1.0 {
		t.Errorf("Expected distance 1.0, got %.1f", c.distance)
	}

	if h.Len() != 0 {
		t.Errorf("Expected empty heap, got length %d", h.Len())
	}
}

// TestHeapSwap tests the swap operation
func TestHeapSwap(t *testing.T) {
	h := &minHeap{
		{id: 1, distance: 5.0},
		{id: 2, distance: 2.0},
	}

	h.Swap(0, 1)

	if (*h)[0].id != 2 || (*h)[0].distance != 2.0 {
		t.Error("Swap did not work correctly")
	}

	if (*h)[1].id != 1 || (*h)[1].distance != 5.0 {
		t.Error("Swap did not work correctly")
	}
}

// ============================================================================
// BASIC SEARCH TESTS
// ============================================================================

// TestHNSWIndexSearchSimple tests basic search functionality
func TestHNSWIndexSearchSimple(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{1, 1, 0},
		{2, 0, 0},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Search for nearest to [1, 0, 0]
	query := []float32{1, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// First result should be [1, 0, 0] (exact match)
	if !vectorsAlmostEqual(results[0].Vector(), []float32{1, 0, 0}, 0.001) {
		t.Errorf("Expected first result to be [1, 0, 0], got %v", results[0].Vector())
	}
}

// TestHNSWIndexSearchExactMatch tests finding exact matches
func TestHNSWIndexSearchExactMatch(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Search for exact match
	query := []float32{4, 5, 6}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(1).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	// Should find exact match
	if !vectorsAlmostEqual(results[0].Vector(), []float32{4, 5, 6}, 0.001) {
		t.Errorf("Expected exact match [4, 5, 6], got %v", results[0].Vector())
	}
}

// TestHNSWIndexSearchKGreaterThanSize tests k > index size
func TestHNSWIndexSearchKGreaterThanSize(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add only 3 vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Request more than available
	query := []float32{1, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should return all 3 vectors
	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}
}

// ============================================================================
// THRESHOLD TESTS
// ============================================================================

// TestHNSWIndexSearchWithThreshold tests search with distance threshold
func TestHNSWIndexSearchWithThreshold(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors with known distances from query [1, 0, 0]
	vectors := [][]float32{
		{1, 0, 0},  // distance 0
		{2, 0, 0},  // distance 1
		{4, 0, 0},  // distance 3
		{10, 0, 0}, // distance 9
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Search with threshold of 2.0
	query := []float32{1, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		WithThreshold(2.0).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should only get vectors within distance 2.0
	// [1,0,0] (d=0) and [2,0,0] (d=1)
	if len(results) != 2 {
		t.Errorf("Expected 2 results with threshold, got %d", len(results))
	}

	// Verify all results are within threshold
	for _, result := range results {
		dist := euclideanDistance(query, result.Vector())
		if dist > 2.0 {
			t.Errorf("Result distance %.2f exceeds threshold 2.0", dist)
		}
	}
}

// TestHNSWIndexSearchThresholdStrictFiltering tests strict threshold filtering
func TestHNSWIndexSearchThresholdStrictFiltering(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors far from query
	vectors := [][]float32{
		{10, 10, 10},
		{20, 20, 20},
		{30, 30, 30},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Search with very small threshold
	query := []float32{0, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		WithThreshold(1.0).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should get no results (all too far)
	if len(results) != 0 {
		t.Errorf("Expected 0 results with strict threshold, got %d", len(results))
	}
}

// ============================================================================
// NODE-BASED SEARCH TESTS
// ============================================================================

// TestHNSWIndexSearchByNode tests searching using node IDs
func TestHNSWIndexSearchByNode(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors and remember their IDs
	node1 := NewVectorNode([]float32{1, 0, 0})
	node2 := NewVectorNode([]float32{0, 1, 0})
	node3 := NewVectorNode([]float32{0, 0, 1})
	node4 := NewVectorNode([]float32{2, 0, 0})

	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)
	idx.Add(*node4)

	// Search using node1's ID (should find node4 as nearest)
	results, err := idx.NewSearch().
		WithNode(node1.ID()).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should return 2 results
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// First result should be node1 itself (distance 0)
	if results[0].ID() != node1.ID() {
		t.Errorf("Expected first result to be query node, got ID %d", results[0].ID())
	}
}

// TestHNSWIndexSearchByMultipleNodes tests searching with multiple node IDs
func TestHNSWIndexSearchByMultipleNodes(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	nodes := make([]*VectorNode, 5)
	for i := 0; i < 5; i++ {
		nodes[i] = NewVectorNode([]float32{float32(i), 0, 0})
		idx.Add(*nodes[i])
	}

	// Search using two node IDs
	results, err := idx.NewSearch().
		WithNode(nodes[0].ID(), nodes[4].ID()).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// 2 queries × k=2 = 4 results
	if len(results) != 4 {
		t.Errorf("Expected 4 results (2 per node query), got %d", len(results))
	}
}

// TestHNSWIndexSearchByNonExistentNode tests searching with invalid node ID
func TestHNSWIndexSearchByNonExistentNode(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	node := NewVectorNode([]float32{1, 0, 0})
	idx.Add(*node)

	// Search using non-existent node ID
	_, err = idx.NewSearch().
		WithNode(9999).
		WithK(1).
		Execute()

	if err == nil {
		t.Error("Expected error when searching with non-existent node ID")
	}
}

// TestHNSWIndexSearchByDeletedNode tests searching with deleted node ID
func TestHNSWIndexSearchByDeletedNode(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	node1 := NewVectorNode([]float32{1, 0, 0})
	node2 := NewVectorNode([]float32{2, 0, 0})

	idx.Add(*node1)
	idx.Add(*node2)

	// Delete node1
	idx.Remove(*node1)

	// Search using deleted node ID
	_, err = idx.NewSearch().
		WithNode(node1.ID()).
		WithK(1).
		Execute()

	if err == nil {
		t.Error("Expected error when searching with deleted node ID")
	}
}

// ============================================================================
// MULTIPLE QUERIES TESTS
// ============================================================================

// TestHNSWIndexSearchMultipleQueries tests batch search with multiple queries
func TestHNSWIndexSearchMultipleQueries(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{2, 0, 0},
		{0, 2, 0},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Search with multiple queries
	results, err := idx.NewSearch().
		WithQuery([]float32{1, 0, 0}, []float32{0, 1, 0}).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// 2 queries × k=2 = 4 results
	if len(results) != 4 {
		t.Errorf("Expected 4 results (2 per query), got %d", len(results))
	}
}

// TestHNSWIndexSearchCombinedQueryAndNode tests combined query and node search
func TestHNSWIndexSearchCombinedQueryAndNode(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{2, 0, 0},
	}

	nodes := make([]*VectorNode, len(vectors))
	for i, v := range vectors {
		nodes[i] = NewVectorNode(v)
		idx.Add(*nodes[i])
	}

	// Search using both a direct query and a node ID
	results, err := idx.NewSearch().
		WithQuery([]float32{0, 1, 0}).
		WithNode(nodes[0].ID()).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// 2 queries (1 direct + 1 from node) × k=2 = 4 results
	if len(results) != 4 {
		t.Errorf("Expected 4 results (2 per query), got %d", len(results))
	}
}

// TestHNSWIndexSearchMultipleQueriesAndNodes tests batch search with mixed queries
func TestHNSWIndexSearchMultipleQueriesAndNodes(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{2, 0, 0},
		{0, 2, 0},
		{0, 0, 2},
	}

	nodes := make([]*VectorNode, len(vectors))
	for i, v := range vectors {
		nodes[i] = NewVectorNode(v)
		idx.Add(*nodes[i])
	}

	// Search with 2 direct queries and 2 node IDs
	results, err := idx.NewSearch().
		WithQuery([]float32{1.1, 0, 0}, []float32{0, 1.1, 0}).
		WithNode(nodes[2].ID(), nodes[3].ID()).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// 4 queries (2 direct + 2 from nodes) × k=2 = 8 total results
	if len(results) != 8 {
		t.Errorf("Expected 8 results, got %d", len(results))
	}
}

// ============================================================================
// VALIDATION TESTS
// ============================================================================

// TestHNSWIndexSearchValidation tests validation of search parameters
func TestHNSWIndexSearchValidation(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add a vector
	node := NewVectorNode([]float32{1, 0, 0})
	idx.Add(*node)

	// Test: No query or node specified
	_, err = idx.NewSearch().
		WithK(1).
		Execute()

	if err == nil {
		t.Error("Expected error when no query or node specified")
	}

	// Test: Wrong dimension query
	_, err = idx.NewSearch().
		WithQuery([]float32{1, 2, 3, 4}). // 4D instead of 3D
		WithK(1).
		Execute()

	if err == nil {
		t.Error("Expected error for dimension mismatch")
	}
}

// TestHNSWIndexSearchZeroVectorCosine tests zero vector with cosine distance
func TestHNSWIndexSearchZeroVectorCosine(t *testing.T) {
	idx, err := NewHNSWIndex(3, Cosine, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	node := NewVectorNode([]float32{1, 2, 3})
	idx.Add(*node)

	// Search with zero vector (cannot be normalized)
	_, err = idx.NewSearch().
		WithQuery([]float32{0, 0, 0}).
		WithK(1).
		Execute()

	if err == nil {
		t.Error("Expected error when searching with zero vector using cosine distance")
	}
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

// TestHNSWIndexSearchEmpty tests searching an empty index
func TestHNSWIndexSearchEmpty(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Search empty index
	results, err := idx.NewSearch().
		WithQuery([]float32{1, 2, 3}).
		WithK(10).
		Execute()

	if err != nil {
		t.Errorf("Search() on empty index error: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected 0 results from empty index, got %d", len(results))
	}
}

// TestHNSWIndexSearchAfterAllDeleted tests searching after all nodes deleted
func TestHNSWIndexSearchAfterAllDeleted(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add and delete all nodes
	node1 := NewVectorNode([]float32{1, 0, 0})
	node2 := NewVectorNode([]float32{2, 0, 0})

	idx.Add(*node1)
	idx.Add(*node2)

	idx.Remove(*node1)
	idx.Remove(*node2)
	idx.Flush()

	// Search after all deleted
	results, err := idx.NewSearch().
		WithQuery([]float32{1, 0, 0}).
		WithK(10).
		Execute()

	if err != nil {
		t.Errorf("Search() after flush error: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected 0 results after all deleted, got %d", len(results))
	}
}

// TestHNSWIndexSearchSingleNode tests search with only one node
func TestHNSWIndexSearchSingleNode(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	node := NewVectorNode([]float32{1, 2, 3})
	idx.Add(*node)

	// Search should return the single node
	results, err := idx.NewSearch().
		WithQuery([]float32{1, 2, 3}).
		WithK(1).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}

	if results[0].ID() != node.ID() {
		t.Error("Search should return the single node")
	}
}

// ============================================================================
// DISTANCE METRIC TESTS
// ============================================================================

// TestHNSWIndexSearchDifferentMetrics tests search with different distance metrics
func TestHNSWIndexSearchDifferentMetrics(t *testing.T) {
	tests := []struct {
		name         string
		distanceKind DistanceKind
	}{
		{"Euclidean", Euclidean},
		{"L2Squared", L2Squared},
		{"Cosine", Cosine},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, err := NewHNSWIndex(3, tt.distanceKind, 16, 200, 200)
			if err != nil {
				t.Fatalf("NewHNSWIndex() error: %v", err)
			}

			// Add vectors
			vectors := [][]float32{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
				{1, 1, 0},
			}

			for _, v := range vectors {
				node := NewVectorNode(v)
				err := idx.Add(*node)
				if err != nil {
					t.Fatalf("Add() error: %v", err)
				}
			}

			// Search
			results, err := idx.NewSearch().
				WithQuery([]float32{1, 0, 0}).
				WithK(2).
				Execute()

			if err != nil {
				t.Fatalf("Search() error: %v", err)
			}

			if len(results) == 0 {
				t.Error("Expected some results")
			}
		})
	}
}

// TestHNSWIndexSearchCosine tests cosine distance search specifically
func TestHNSWIndexSearchCosine(t *testing.T) {
	idx, err := NewHNSWIndex(3, Cosine, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{1, 1, 0},
		{0, 1, 0},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Search for nearest to [2, 0, 0] (parallel to [1, 0, 0])
	query := []float32{2, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(1).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}

	// Should find [1, 0, 0] as it's parallel (cosine distance = 0)
	expected := []float32{1, 0, 0}
	if !vectorsAlmostEqual(results[0].Vector(), expected, 0.001) {
		t.Errorf("Expected result %v, got %v", expected, results[0].Vector())
	}
}

// ============================================================================
// CONCURRENCY TESTS
// ============================================================================

// TestHNSWIndexSearchConcurrent tests concurrent searches
func TestHNSWIndexSearchConcurrent(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	for i := 0; i < 50; i++ {
		node := NewVectorNode([]float32{float32(i), 0, 0})
		idx.Add(*node)
	}

	// Perform concurrent searches
	var wg sync.WaitGroup
	numGoroutines := 10

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(offset int) {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				_, err := idx.NewSearch().
					WithQuery([]float32{float32(offset), 0, 0}).
					WithK(5).
					Execute()
				if err != nil {
					t.Errorf("Search() error: %v", err)
				}
			}
		}(i)
	}

	wg.Wait()
}

// TestHNSWIndexSearchConcurrentWithModifications tests concurrent search during modifications
func TestHNSWIndexSearchConcurrentWithModifications(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add initial vectors
	for i := 0; i < 30; i++ {
		node := NewVectorNode([]float32{float32(i), 0, 0})
		idx.Add(*node)
	}

	var wg sync.WaitGroup

	// Concurrent searches
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				_, err := idx.NewSearch().
					WithQuery([]float32{5, 0, 0}).
					WithK(3).
					Execute()
				if err != nil {
					t.Errorf("Search() error: %v", err)
				}
			}
		}()
	}

	// Concurrent adds
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(offset int) {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				node := NewVectorNode([]float32{float32(offset*10 + j + 100), 0, 0})
				idx.Add(*node)
			}
		}(i)
	}

	wg.Wait()
}

// ============================================================================
// ACCURACY TESTS
// ============================================================================

// TestHNSWIndexSearchAccuracy tests search result ordering
func TestHNSWIndexSearchAccuracy(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors with known distances
	vectors := [][]float32{
		{1, 0, 0},   // distance 1 from query
		{2, 0, 0},   // distance 2
		{0.5, 0, 0}, // distance 0.5
		{3, 0, 0},   // distance 3
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Search from origin
	query := []float32{0, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(4).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 4 {
		t.Fatalf("Expected 4 results, got %d", len(results))
	}

	// Results should be ordered by distance
	prevDist := float32(0)
	for _, result := range results {
		dist := euclideanDistance(query, result.Vector())
		if dist < prevDist {
			t.Errorf("Results not ordered by distance: prev=%.2f, curr=%.2f", prevDist, dist)
		}
		prevDist = dist
	}

	// First result should be closest ([0.5, 0, 0])
	if !vectorsAlmostEqual(results[0].Vector(), []float32{0.5, 0, 0}, 0.001) {
		t.Errorf("Expected closest result [0.5, 0, 0], got %v", results[0].Vector())
	}
}

// TestHNSWIndexSearchRecall tests search recall with more data
func TestHNSWIndexSearchRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping recall test in short mode")
	}

	idx, err := NewHNSWIndex(10, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add many vectors
	numVectors := 500
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, 10)
		for j := 0; j < 10; j++ {
			vec[j] = float32((i*10 + j) % 100)
		}
		node := NewVectorNode(vec)
		idx.Add(*node)
	}

	// Search
	query := make([]float32, 10)
	for j := 0; j < 10; j++ {
		query[j] = float32(j % 100)
	}

	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 10 {
		t.Errorf("Expected 10 results, got %d", len(results))
	}

	// Verify results are reasonably close
	for _, result := range results {
		dist := euclideanDistance(query, result.Vector())
		if dist > 500 { // Reasonable bound for this data
			t.Errorf("Result too far: distance %.2f", dist)
		}
	}
}

// ============================================================================
// WITH EF SEARCH TESTS
// ============================================================================

// TestHNSWIndexSearchWithEfSearchDefault tests that search uses index default efSearch when not overridden
func TestHNSWIndexSearchWithEfSearchDefault(t *testing.T) {
	// Create index with default efSearch of 200
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{1, 1, 0},
		{2, 0, 0},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Search without specifying efSearch (should use default of 200)
	query := []float32{1, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// Should find exact match as first result
	if !vectorsAlmostEqual(results[0].Vector(), []float32{1, 0, 0}, 0.001) {
		t.Errorf("Expected first result to be [1, 0, 0], got %v", results[0].Vector())
	}
}

// TestHNSWIndexSearchWithEfSearchOverride tests that WithEfSearch overrides index default
func TestHNSWIndexSearchWithEfSearchOverride(t *testing.T) {
	// Create index with low default efSearch
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 50)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add more vectors for better test
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{1, 1, 0},
		{2, 0, 0},
		{0, 2, 0},
		{0, 0, 2},
		{1, 1, 1},
		{2, 2, 0},
		{0, 2, 2},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	query := []float32{1, 0, 0}

	// Search with higher efSearch override
	results1, err := idx.NewSearch().
		WithQuery(query).
		WithK(5).
		WithEfSearch(200).
		Execute()

	if err != nil {
		t.Fatalf("Search() with efSearch=200 error: %v", err)
	}

	// Search with default (lower) efSearch
	results2, err := idx.NewSearch().
		WithQuery(query).
		WithK(5).
		Execute()

	if err != nil {
		t.Fatalf("Search() with default efSearch error: %v", err)
	}

	// Both should return results (exact behavior may vary, but both should work)
	if len(results1) == 0 {
		t.Errorf("Expected results with efSearch=200, got %d", len(results1))
	}
	if len(results2) == 0 {
		t.Errorf("Expected results with default efSearch, got %d", len(results2))
	}
}

// TestHNSWIndexSearchWithEfSearchRecall tests that higher efSearch improves recall
func TestHNSWIndexSearchWithEfSearchRecall(t *testing.T) {
	// Create index with larger dataset
	dim := 10
	idx, err := NewHNSWIndex(dim, Euclidean, 16, 200, 50)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add 100 random vectors
	numVectors := 100
	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		vectors[i] = vec
		node := NewVectorNode(vec)
		idx.Add(*node)
	}

	// Query for first vector
	query := vectors[0]

	// Search with low efSearch
	resultsLow, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		WithEfSearch(10).
		Execute()

	if err != nil {
		t.Fatalf("Search() with low efSearch error: %v", err)
	}

	// Search with high efSearch
	resultsHigh, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		WithEfSearch(100).
		Execute()

	if err != nil {
		t.Fatalf("Search() with high efSearch error: %v", err)
	}

	// Both should return results
	if len(resultsLow) == 0 {
		t.Errorf("Expected results with low efSearch")
	}
	if len(resultsHigh) == 0 {
		t.Errorf("Expected results with high efSearch")
	}

	// First result should be exact match in both cases
	if !vectorsAlmostEqual(resultsHigh[0].Vector(), query, 0.001) {
		t.Errorf("Expected exact match with high efSearch")
	}
}

// TestHNSWIndexSearchWithEfSearchZero tests that zero efSearch uses default
func TestHNSWIndexSearchWithEfSearchZero(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Search with efSearch=0 (should use default)
	query := []float32{1, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(2).
		WithEfSearch(0).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// Should find exact match
	if !vectorsAlmostEqual(results[0].Vector(), []float32{1, 0, 0}, 0.001) {
		t.Errorf("Expected first result to be [1, 0, 0], got %v", results[0].Vector())
	}
}

// TestHNSWIndexSearchWithEfSearchNegative tests that negative efSearch uses default
func TestHNSWIndexSearchWithEfSearchNegative(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Search with negative efSearch (should use default)
	query := []float32{1, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(2).
		WithEfSearch(-1).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// Should find exact match
	if !vectorsAlmostEqual(results[0].Vector(), []float32{1, 0, 0}, 0.001) {
		t.Errorf("Expected first result to be [1, 0, 0], got %v", results[0].Vector())
	}
}

// TestHNSWIndexSearchWithEfSearchChaining tests that WithEfSearch can be chained with other methods
func TestHNSWIndexSearchWithEfSearchChaining(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{2, 0, 0},
		{3, 0, 0},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Chain WithEfSearch with other methods
	query := []float32{1, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(3).
		WithEfSearch(100).
		WithThreshold(2.5).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should have results filtered by threshold
	if len(results) == 0 {
		t.Errorf("Expected some results")
	}

	// All results should be within threshold distance
	for _, result := range results {
		dist := euclideanDistance(query, result.Vector())
		if dist > 2.5 {
			t.Errorf("Result distance %.2f exceeds threshold 2.5", dist)
		}
	}
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// euclideanDistance calculates Euclidean distance between two vectors
func euclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}
	sum := float32(0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}
