package comet

import (
	"math"
	"sync"
	"testing"
)

// TestFlatIndexSearchSimple tests basic search functionality
func TestFlatIndexSearchSimple(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
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
	if !vectorsEqual(results[0].Vector(), []float32{1, 0, 0}) {
		t.Errorf("Expected first result to be [1, 0, 0], got %v", results[0].Vector())
	}
}

// TestFlatIndexSearchWithThreshold tests search with distance threshold
func TestFlatIndexSearchWithThreshold(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add vectors with known distances
	vectors := [][]float32{
		{1, 0, 0},  // distance 0 from query
		{2, 0, 0},  // distance 1 from query
		{4, 0, 0},  // distance 3 from query
		{10, 0, 0}, // distance 9 from query
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
	if len(results) != 2 {
		t.Errorf("Expected 2 results with threshold, got %d", len(results))
	}
}

// TestFlatIndexSearchCosine tests search with cosine distance
func TestFlatIndexSearchCosine(t *testing.T) {
	idx, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
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

	// Search for nearest to [1, 0, 0]
	query := []float32{2, 0, 0} // Parallel to [1, 0, 0]
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

// TestFlatIndexSearchByNode tests searching using node IDs
func TestFlatIndexSearchByNode(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add vectors and remember their IDs
	node1 := NewVectorNode([]float32{1, 0, 0})
	node2 := NewVectorNode([]float32{0, 1, 0})
	node3 := NewVectorNode([]float32{0, 0, 1})

	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)

	// Search using node1's ID
	results, err := idx.NewSearch().
		WithNode(node1.ID()).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// First result should be the query node itself
	if results[0].ID() != node1.ID() {
		t.Errorf("Expected first result to be query node %d, got %d", node1.ID(), results[0].ID())
	}
}

// TestFlatIndexSearchByMultipleNodes tests batch search using multiple node IDs
func TestFlatIndexSearchByMultipleNodes(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add vectors
	node1 := NewVectorNode([]float32{1, 0, 0})
	node2 := NewVectorNode([]float32{0, 1, 0})
	node3 := NewVectorNode([]float32{0, 0, 1})

	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)

	// Search using multiple nodes
	results, err := idx.NewSearch().
		WithNode(node1.ID(), node2.ID()).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should get 2 results per query = 4 total
	if len(results) != 4 {
		t.Errorf("Expected 4 results, got %d", len(results))
	}
}

// TestFlatIndexSearchByNonExistentNode tests error handling for non-existent node ID
func TestFlatIndexSearchByNonExistentNode(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	node := NewVectorNode([]float32{1, 0, 0})
	idx.Add(*node)

	// Try to search with non-existent node ID
	_, err = idx.NewSearch().
		WithNode(9999).
		WithK(1).
		Execute()

	if err == nil {
		t.Error("Expected error when searching with non-existent node ID")
	}
}

// TestFlatIndexSearchBatchQueries tests batch query search
func TestFlatIndexSearchBatchQueries(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
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

	// Search with multiple queries
	queries := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
	}

	results, err := idx.NewSearch().
		WithQuery(queries...).
		WithK(1).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should get 1 result per query = 2 total
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}
}

// TestFlatIndexSearchValidation tests search parameter validation
func TestFlatIndexSearchValidation(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	node := NewVectorNode([]float32{1, 0, 0})
	idx.Add(*node)

	tests := []struct {
		name    string
		setup   func() VectorSearch
		wantErr bool
	}{
		{
			name: "no query or node",
			setup: func() VectorSearch {
				return idx.NewSearch().WithK(1)
			},
			wantErr: true,
		},
		{
			name: "valid node search",
			setup: func() VectorSearch {
				return idx.NewSearch().
					WithNode(node.ID()).
					WithK(1)
			},
			wantErr: false,
		},
		{
			name: "query dimension mismatch",
			setup: func() VectorSearch {
				return idx.NewSearch().
					WithQuery([]float32{1, 0, 0, 0}). // 4D instead of 3D
					WithK(1)
			},
			wantErr: true,
		},
		{
			name: "zero query with cosine",
			setup: func() VectorSearch {
				cosineIdx, _ := NewFlatIndex(3, Cosine)
				node := NewVectorNode([]float32{1, 0, 0})
				cosineIdx.Add(*node)
				return cosineIdx.NewSearch().
					WithQuery([]float32{0, 0, 0}).
					WithK(1)
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.setup().Execute()
			if tt.wantErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

// TestFlatIndexSearchKBounds tests k parameter edge cases
func TestFlatIndexSearchKBounds(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add 5 vectors
	for i := 0; i < 5; i++ {
		node := NewVectorNode([]float32{float32(i), 0, 0})
		idx.Add(*node)
	}

	tests := []struct {
		name        string
		k           int
		expectedLen int
	}{
		{"k = 0", 0, 5},     // Should return all
		{"k = -1", -1, 5},   // Should return all
		{"k = 3", 3, 3},     // Normal case
		{"k = 5", 5, 5},     // Exact match
		{"k = 100", 100, 5}, // More than available
		{"k = 1", 1, 1},     // Minimum
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().
				WithQuery([]float32{0, 0, 0}).
				WithK(tt.k).
				Execute()

			if err != nil {
				t.Fatalf("Search() error: %v", err)
			}

			if len(results) != tt.expectedLen {
				t.Errorf("Expected %d results, got %d", tt.expectedLen, len(results))
			}
		})
	}
}

// TestFlatIndexConcurrentSearch tests thread-safety of Search operations
func TestFlatIndexConcurrentSearch(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add some vectors
	for i := 0; i < 100; i++ {
		node := NewVectorNode([]float32{float32(i), 0, 0})
		idx.Add(*node)
	}

	const numSearches = 50
	var wg sync.WaitGroup
	wg.Add(numSearches)

	for i := 0; i < numSearches; i++ {
		go func(i int) {
			defer wg.Done()
			query := []float32{float32(i % 10), 0, 0}
			_, err := idx.NewSearch().
				WithQuery(query).
				WithK(10).
				Execute()
			if err != nil {
				t.Errorf("Search() error: %v", err)
			}
		}(i)
	}

	wg.Wait()
}

// TestFlatIndexConcurrentAddAndSearch tests concurrent adds and searches
func TestFlatIndexConcurrentAddAndSearch(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Pre-populate with some vectors
	for i := 0; i < 50; i++ {
		node := NewVectorNode([]float32{float32(i), 0, 0})
		idx.Add(*node)
	}

	var wg sync.WaitGroup
	const numOps = 100

	// Concurrent adds
	wg.Add(numOps)
	for i := 0; i < numOps; i++ {
		go func(i int) {
			defer wg.Done()
			node := NewVectorNode([]float32{float32(i + 50), 0, 0})
			idx.Add(*node)
		}(i)
	}

	// Concurrent searches
	wg.Add(numOps)
	for i := 0; i < numOps; i++ {
		go func(i int) {
			defer wg.Done()
			query := []float32{float32(i % 10), 0, 0}
			idx.NewSearch().
				WithQuery(query).
				WithK(5).
				Execute()
		}(i)
	}

	wg.Wait()
}

// TestFlatIndexEmptySearch tests searching an empty index
func TestFlatIndexEmptySearch(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	query := []float32{1, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected 0 results from empty index, got %d", len(results))
	}
}

// TestFlatIndexSearchResultsOrdered tests that results are properly ordered by distance
func TestFlatIndexSearchResultsOrdered(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add vectors at different distances from origin
	vectors := [][]float32{
		{5, 0, 0},  // distance 5
		{1, 0, 0},  // distance 1
		{10, 0, 0}, // distance 10
		{3, 0, 0},  // distance 3
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	query := []float32{0, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(4).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Check that results are ordered by distance
	expectedOrder := []float32{1, 3, 5, 10}
	for i, expected := range expectedOrder {
		if results[i].Vector()[0] != expected {
			t.Errorf("Result %d: expected distance %f, got %f", i, expected, results[i].Vector()[0])
		}
	}

	// Verify distances are in ascending order
	distance, _ := NewDistance(Euclidean)
	for i := 1; i < len(results); i++ {
		d1 := distance.Calculate(query, results[i-1].Vector())
		d2 := distance.Calculate(query, results[i].Vector())
		if d1 > d2 {
			t.Errorf("Results not properly ordered: distance[%d]=%f > distance[%d]=%f", i-1, d1, i, d2)
		}
	}
}

// TestFlatIndexSearchDifferentDistanceMetrics tests all distance metrics
func TestFlatIndexSearchDifferentDistanceMetrics(t *testing.T) {
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{1, 1, 0},
	}

	metrics := []DistanceKind{Euclidean, Cosine}

	for _, metric := range metrics {
		t.Run(string(metric), func(t *testing.T) {
			idx, err := NewFlatIndex(3, metric)
			if err != nil {
				t.Fatalf("NewFlatIndex() error: %v", err)
			}

			for _, v := range vectors {
				node := NewVectorNode(v)
				err := idx.Add(*node)
				if err != nil {
					t.Fatalf("Add() error: %v", err)
				}
			}

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
		})
	}
}

// Helper functions

func vectorsEqual(v1, v2 []float32) bool {
	if len(v1) != len(v2) {
		return false
	}
	for i := range v1 {
		if v1[i] != v2[i] {
			return false
		}
	}
	return true
}

func vectorsAlmostEqual(v1, v2 []float32, epsilon float32) bool {
	if len(v1) != len(v2) {
		return false
	}
	for i := range v1 {
		if math.Abs(float64(v1[i]-v2[i])) > float64(epsilon) {
			return false
		}
	}
	return true
}
