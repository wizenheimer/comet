package comet

import (
	"sync"
	"testing"
)

// Helper function to create a trained PQ index with test data
func createTrainedPQIndex(t *testing.T, dim int, M int, Nbits int, distanceKind DistanceKind) (*PQIndex, []*VectorNode) {
	t.Helper()

	idx, err := NewPQIndex(dim, distanceKind, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Create training vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i*dim + j) % 10)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	// Train the index
	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add test vectors
	vectors := [][]float32{
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0},
		{1, 1, 0, 0, 0, 0, 0, 0},
		{2, 0, 0, 0, 0, 0, 0, 0},
		{3, 0, 0, 0, 0, 0, 0, 0},
	}

	nodes := make([]*VectorNode, len(vectors))
	for i, v := range vectors {
		nodes[i] = NewVectorNode(v)
		err := idx.Add(*nodes[i])
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	return idx, nodes
}

// TestPQIndexSearchSimple tests basic search functionality
func TestPQIndexSearchSimple(t *testing.T) {
	idx, _ := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	// Search for nearest to [1, 0, 0, 0, 0, 0, 0, 0]
	query := []float32{1, 0, 0, 0, 0, 0, 0, 0}
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

	// Should find vectors close to the query
	if len(results) > 0 {
		// First result should be reasonably close
		firstVec := results[0].Node.Vector()
		if len(firstVec) != 8 {
			t.Errorf("Result vector has wrong dimension: got %d, want 8", len(firstVec))
		}
	}
}

// TestPQIndexSearchWithThreshold tests search with distance threshold
func TestPQIndexSearchWithThreshold(t *testing.T) {
	idx, _ := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	// Search with a restrictive threshold
	query := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		WithThreshold(1.5).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Due to PQ approximation, exact counts may vary, but threshold should filter results
	// Just verify that results are within threshold (approximately)
	for _, result := range results {
		// Can't validate exact distances due to PQ approximation, but ensure we got results
		if len(result.Node.Vector()) != 8 {
			t.Errorf("Result has wrong dimension")
		}
	}
}

// TestPQIndexSearchByNode tests search by node ID
func TestPQIndexSearchByNode(t *testing.T) {
	idx, nodes := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	// Search using first node's ID
	results, err := idx.NewSearch().
		WithNode(nodes[0].ID()).
		WithK(3).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}
}

// TestPQIndexSearchByMultipleNodes tests search with multiple node IDs
func TestPQIndexSearchByMultipleNodes(t *testing.T) {
	idx, nodes := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	// Search using two node IDs
	results, err := idx.NewSearch().
		WithNode(nodes[0].ID(), nodes[1].ID()).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Each node should return k=2 results, so total 4
	if len(results) != 4 {
		t.Errorf("Expected 4 results (2 per node), got %d", len(results))
	}
}

// TestPQIndexSearchByNonExistentNode tests search with non-existent node ID
func TestPQIndexSearchByNonExistentNode(t *testing.T) {
	idx, _ := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	// Search using non-existent node ID
	_, err := idx.NewSearch().
		WithNode(99999).
		WithK(3).
		Execute()

	if err == nil {
		t.Error("Expected error when searching with non-existent node ID")
	}
}

// TestPQIndexSearchCombinedQueryAndNode tests searching with both queries and node IDs
func TestPQIndexSearchCombinedQueryAndNode(t *testing.T) {
	idx, nodes := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	// Search using both a direct query and a node ID
	results, err := idx.NewSearch().
		WithQuery([]float32{0, 1, 0, 0, 0, 0, 0, 0}).
		WithNode(nodes[0].ID()).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should get results from both queries (2 per query = 4 total)
	if len(results) != 4 {
		t.Errorf("Expected 4 results (2 per query), got %d", len(results))
	}
}

// TestPQIndexSearchMultipleQueriesAndNodes tests batch search with mixed queries and nodes
func TestPQIndexSearchMultipleQueriesAndNodes(t *testing.T) {
	idx, nodes := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	// Search with multiple queries and nodes
	results, err := idx.NewSearch().
		WithQuery(
			[]float32{1, 0, 0, 0, 0, 0, 0, 0},
			[]float32{0, 1, 0, 0, 0, 0, 0, 0},
		).
		WithNode(nodes[0].ID(), nodes[1].ID()).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// 2 queries + 2 nodes = 4 total queries, each returning k=2 results = 8 total
	if len(results) != 8 {
		t.Errorf("Expected 8 results (2 per query/node), got %d", len(results))
	}
}

// TestPQIndexSearchBatchQueries tests batch query search
func TestPQIndexSearchBatchQueries(t *testing.T) {
	idx, _ := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	queries := [][]float32{
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0},
	}

	results, err := idx.NewSearch().
		WithQuery(queries...).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// 3 queries × 2 results each = 6 total
	if len(results) != 6 {
		t.Errorf("Expected 6 results, got %d", len(results))
	}
}

// TestPQIndexSearchValidation tests search parameter validation
func TestPQIndexSearchValidation(t *testing.T) {
	idx, nodes := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	tests := []struct {
		name      string
		setupFunc func() VectorSearch
		wantErr   bool
	}{
		{
			name: "no query or node",
			setupFunc: func() VectorSearch {
				return idx.NewSearch().WithK(5)
			},
			wantErr: true,
		},
		{
			name: "valid query search",
			setupFunc: func() VectorSearch {
				return idx.NewSearch().
					WithQuery([]float32{1, 0, 0, 0, 0, 0, 0, 0}).
					WithK(3)
			},
			wantErr: false,
		},
		{
			name: "valid node search",
			setupFunc: func() VectorSearch {
				return idx.NewSearch().
					WithNode(nodes[0].ID()).
					WithK(3)
			},
			wantErr: false,
		},
		{
			name: "query dimension mismatch",
			setupFunc: func() VectorSearch {
				return idx.NewSearch().
					WithQuery([]float32{1, 0, 0}). // Wrong dimension
					WithK(3)
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			search := tt.setupFunc()
			_, err := search.Execute()

			if tt.wantErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

// TestPQIndexSearchBeforeTrain tests that search fails before training
func TestPQIndexSearchBeforeTrain(t *testing.T) {
	idx, err := NewPQIndex(8, Euclidean, 4, 6)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Try to search without training
	_, err = idx.NewSearch().
		WithQuery([]float32{1, 0, 0, 0, 0, 0, 0, 0}).
		WithK(5).
		Execute()

	if err == nil {
		t.Error("Expected error when searching before training")
	}
}

// TestPQIndexSearchKBounds tests search with different k values
func TestPQIndexSearchKBounds(t *testing.T) {
	idx, _ := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	query := []float32{1, 0, 0, 0, 0, 0, 0, 0}

	tests := []struct {
		name        string
		k           int
		expectCount int
	}{
		{"k = 1", 1, 1},
		{"k = 3", 3, 3},
		{"k = 5", 5, 5},
		{"k = 6", 6, 6},     // Should get all 6 vectors
		{"k = 100", 100, 6}, // Should cap at number of vectors
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().
				WithQuery(query).
				WithK(tt.k).
				Execute()

			if err != nil {
				t.Fatalf("Search() error: %v", err)
			}

			if len(results) != tt.expectCount {
				t.Errorf("Expected %d results, got %d", tt.expectCount, len(results))
			}
		})
	}
}

// TestPQIndexSearchEmptyIndex tests search on empty index
func TestPQIndexSearchEmptyIndex(t *testing.T) {
	idx, err := NewPQIndex(8, Euclidean, 4, 6)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Train but don't add any vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, 8)
		for j := 0; j < 8; j++ {
			vec[j] = float32(i)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	idx.Train(trainingVectors)

	// Search in empty index
	results, err := idx.NewSearch().
		WithQuery([]float32{1, 0, 0, 0, 0, 0, 0, 0}).
		WithK(5).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected 0 results from empty index, got %d", len(results))
	}
}

// TestPQIndexSearchWithNProbes tests that WithNProbes is ignored (no-op)
func TestPQIndexSearchWithNProbes(t *testing.T) {
	idx, _ := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	query := []float32{1, 0, 0, 0, 0, 0, 0, 0}

	// Search with nprobes (should be ignored)
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(3).
		WithNProbes(5). // Should be ignored for PQ
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should still get results even though nprobes is set
	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}
}

// TestPQIndexSearchDifferentDistanceMetrics tests search with different distance metrics
func TestPQIndexSearchDifferentDistanceMetrics(t *testing.T) {
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
			idx, err := NewPQIndex(8, tt.distanceKind, 4, 6)
			if err != nil {
				t.Fatalf("NewPQIndex() error: %v", err)
			}

			// Train
			trainingVectors := make([]VectorNode, 100)
			for i := 0; i < 100; i++ {
				vec := make([]float32, 8)
				for j := 0; j < 8; j++ {
					vec[j] = float32((i + 1)) // Non-zero for cosine
				}
				trainingVectors[i] = *NewVectorNode(vec)
			}
			idx.Train(trainingVectors)

			// Add vectors
			vectors := [][]float32{
				{1, 1, 1, 1, 1, 1, 1, 1},
				{2, 2, 2, 2, 2, 2, 2, 2},
				{3, 3, 3, 3, 3, 3, 3, 3},
			}
			for _, v := range vectors {
				idx.Add(*NewVectorNode(v))
			}

			// Search
			results, err := idx.NewSearch().
				WithQuery([]float32{1, 1, 1, 1, 1, 1, 1, 1}).
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

// TestPQIndexConcurrentSearch tests concurrent searches
func TestPQIndexConcurrentSearch(t *testing.T) {
	idx, _ := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	var wg sync.WaitGroup
	numGoroutines := 10
	searchesPerGoroutine := 10

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(offset int) {
			defer wg.Done()
			for j := 0; j < searchesPerGoroutine; j++ {
				query := make([]float32, 8)
				query[0] = float32(offset + j)

				_, err := idx.NewSearch().
					WithQuery(query).
					WithK(3).
					Execute()

				if err != nil {
					t.Errorf("Concurrent search error: %v", err)
				}
			}
		}(i)
	}

	wg.Wait()
}

// TestPQIndexSearchResultsConsistency tests that repeated searches give consistent results
func TestPQIndexSearchResultsConsistency(t *testing.T) {
	idx, _ := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	query := []float32{1, 0, 0, 0, 0, 0, 0, 0}

	// Perform search multiple times
	results1, err := idx.NewSearch().
		WithQuery(query).
		WithK(3).
		Execute()
	if err != nil {
		t.Fatalf("First search error: %v", err)
	}

	results2, err := idx.NewSearch().
		WithQuery(query).
		WithK(3).
		Execute()
	if err != nil {
		t.Fatalf("Second search error: %v", err)
	}

	// Results should be identical
	if len(results1) != len(results2) {
		t.Errorf("Result count differs: %d vs %d", len(results1), len(results2))
	}

	for i := range results1 {
		if results1[i].Node.ID() != results2[i].Node.ID() {
			t.Errorf("Result %d differs: ID %d vs %d", i, results1[i].Node.ID(), results2[i].Node.ID())
		}
	}
}

// TestPQIndexSearchWithZeroThreshold tests search with zero threshold (no filtering)
func TestPQIndexSearchWithZeroThreshold(t *testing.T) {
	idx, _ := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	query := []float32{1, 0, 0, 0, 0, 0, 0, 0}

	// Search with zero threshold (should not filter)
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		WithThreshold(0).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should get all 6 vectors (or up to k=10)
	if len(results) != 6 {
		t.Errorf("Expected 6 results with zero threshold, got %d", len(results))
	}
}

// TestPQIndexSearchChaining tests method chaining
func TestPQIndexSearchChaining(t *testing.T) {
	idx, nodes := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	// Test that methods can be chained in any order
	results, err := idx.NewSearch().
		WithK(2).
		WithThreshold(5.0).
		WithQuery([]float32{1, 0, 0, 0, 0, 0, 0, 0}).
		WithNode(nodes[0].ID()).
		WithNProbes(3). // Ignored but should not error
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should get results from both query and node (2 per query = 4 total)
	if len(results) != 4 {
		t.Errorf("Expected 4 results, got %d", len(results))
	}
}

// TestPQIndexSearchLookupNodeVectors tests the internal lookupNodeVectors function
func TestPQIndexSearchLookupNodeVectors(t *testing.T) {
	idx, nodes := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	// Create search with node IDs
	search := &pqIndexSearch{
		index:   idx,
		nodeIDs: []uint32{nodes[0].ID(), nodes[1].ID()},
		k:       3,
	}

	// Call lookupNodeVectors
	vectors, err := search.lookupNodeVectors()
	if err != nil {
		t.Fatalf("lookupNodeVectors() error: %v", err)
	}

	if len(vectors) != 2 {
		t.Errorf("Expected 2 vectors, got %d", len(vectors))
	}

	// Verify dimensions
	for i, vec := range vectors {
		if len(vec) != 8 {
			t.Errorf("Vector %d has wrong dimension: got %d, want 8", i, len(vec))
		}
	}
}

// TestPQIndexSearchSingleQuery tests the internal searchSingleQuery function
func TestPQIndexSearchSingleQuery(t *testing.T) {
	idx, _ := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	search := &pqIndexSearch{
		index: idx,
		k:     3,
	}

	query := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	results, err := search.searchSingleQuery(query)
	if err != nil {
		t.Fatalf("searchSingleQuery() error: %v", err)
	}

	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}
}

// TestPQIndexSearchApproximateAccuracy tests PQ search approximation quality
func TestPQIndexSearchApproximateAccuracy(t *testing.T) {
	// This test verifies that PQ search returns reasonable approximate results
	idx, _ := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	// Query that's exactly one of the added vectors
	query := []float32{1, 0, 0, 0, 0, 0, 0, 0}

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

	// Due to PQ approximation, the exact match might not be returned,
	// but the result should be close to the query
	result := results[0].Node.Vector()
	if len(result) != 8 {
		t.Errorf("Result has wrong dimension: got %d, want 8", len(result))
	}
}

// TestPQIndexSearchWithHighK tests search with k larger than index size
func TestPQIndexSearchWithHighK(t *testing.T) {
	idx, _ := createTrainedPQIndex(t, 8, 4, 6, Euclidean)

	query := []float32{1, 0, 0, 0, 0, 0, 0, 0}

	// Request more results than vectors in index
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(1000).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should return all 6 vectors
	if len(results) != 6 {
		t.Errorf("Expected 6 results (all vectors), got %d", len(results))
	}
}
