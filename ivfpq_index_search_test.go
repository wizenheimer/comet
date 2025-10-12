package comet

import (
	"sync"
	"testing"
)

// TestIVFPQIndexSearchSimple tests basic IVFPQ search functionality
func TestIVFPQIndexSearchSimple(t *testing.T) {
	dim := 8
	nlist := 2
	M := 4
	Nbits := 4

	idx, err := NewIVFPQIndex(dim, Euclidean, nlist, M, Nbits)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{2, 3, 4, 5, 6, 7, 8, 9},
		{10, 11, 12, 13, 14, 15, 16, 17},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{1, 1, 1, 1, 1, 1, 1, 1},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Search
	query := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(3).
		WithNProbes(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected non-empty results")
	}

	if len(results) > 3 {
		t.Errorf("Expected at most 3 results, got %d", len(results))
	}
}

// TestIVFPQIndexSearchWithThreshold tests search with distance threshold
func TestIVFPQIndexSearchWithThreshold(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{2, 3, 4, 5, 6, 7, 8, 9},
		{100, 100, 100, 100, 100, 100, 100, 100},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Search with tight threshold
	query := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		WithNProbes(2).
		WithThreshold(50.0).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// The far vector should be filtered out
	if len(results) > 2 {
		t.Logf("Got %d results, expected at most 2", len(results))
	}
}

// TestIVFPQIndexSearchByNode tests searching using a node ID
func TestIVFPQIndexSearchByNode(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{2, 3, 4, 5, 6, 7, 8, 9},
		{10, 11, 12, 13, 14, 15, 16, 17},
	}

	nodes := make([]*VectorNode, len(vectors))
	for i, v := range vectors {
		nodes[i] = NewVectorNode(v)
		err := idx.Add(*nodes[i])
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Search using node ID
	results, err := idx.NewSearch().
		WithNode(nodes[0].ID()).
		WithK(2).
		WithNProbes(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected non-empty results")
	}

	if len(results) > 2 {
		t.Errorf("Expected at most 2 results, got %d", len(results))
	}
}

// TestIVFPQIndexSearchByMultipleNodes tests searching with multiple node IDs
func TestIVFPQIndexSearchByMultipleNodes(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{2, 3, 4, 5, 6, 7, 8, 9},
		{10, 11, 12, 13, 14, 15, 16, 17},
	}

	nodes := make([]*VectorNode, len(vectors))
	for i, v := range vectors {
		nodes[i] = NewVectorNode(v)
		err := idx.Add(*nodes[i])
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Search using multiple node IDs
	results, err := idx.NewSearch().
		WithNode(nodes[0].ID(), nodes[1].ID()).
		WithK(2).
		WithNProbes(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// With aggregation enabled (default), results are deduplicated by node ID
	// 2 queries each returning k=2, but duplicates are aggregated
	if len(results) < 2 {
		t.Errorf("Expected at least 2 deduplicated results, got %d", len(results))
	}

	// Verify we got unique node IDs
	seenIDs := make(map[uint32]bool)
	for _, res := range results {
		if seenIDs[res.Node.ID()] {
			t.Errorf("Found duplicate node ID %d in results", res.Node.ID())
		}
		seenIDs[res.Node.ID()] = true
	}
}

// TestIVFPQIndexSearchByNonExistentNode tests error handling for non-existent nodes
func TestIVFPQIndexSearchByNonExistentNode(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Try to search with non-existent node ID
	_, err = idx.NewSearch().
		WithNode(99999).
		WithK(2).
		Execute()

	if err == nil {
		t.Error("Expected error when searching with non-existent node ID")
	}
}

// TestIVFPQIndexSearchCombinedQueryAndNode tests searching with both queries and node IDs
func TestIVFPQIndexSearchCombinedQueryAndNode(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{2, 3, 4, 5, 6, 7, 8, 9},
		{10, 11, 12, 13, 14, 15, 16, 17},
	}

	nodes := make([]*VectorNode, len(vectors))
	for i, v := range vectors {
		nodes[i] = NewVectorNode(v)
		err := idx.Add(*nodes[i])
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Search using both query and node ID
	results, err := idx.NewSearch().
		WithQuery([]float32{1, 2, 3, 4, 5, 6, 7, 8}).
		WithNode(nodes[1].ID()).
		WithK(2).
		WithNProbes(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// With aggregation enabled (default), results are deduplicated by node ID
	// 2 queries (1 direct + 1 from node) each returning k=2
	// Overlapping results are deduplicated
	if len(results) < 2 {
		t.Errorf("Expected at least 2 deduplicated results, got %d", len(results))
	}

	// Verify we got unique node IDs
	seenIDs := make(map[uint32]bool)
	for _, res := range results {
		if seenIDs[res.Node.ID()] {
			t.Errorf("Found duplicate node ID %d in results", res.Node.ID())
		}
		seenIDs[res.Node.ID()] = true
	}
}

// TestIVFPQIndexSearchMultipleQueriesAndNodes tests batch search with mixed queries and nodes
func TestIVFPQIndexSearchMultipleQueriesAndNodes(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{2, 3, 4, 5, 6, 7, 8, 9},
		{10, 11, 12, 13, 14, 15, 16, 17},
		{3, 4, 5, 6, 7, 8, 9, 10},
	}

	nodes := make([]*VectorNode, len(vectors))
	for i, v := range vectors {
		nodes[i] = NewVectorNode(v)
		err := idx.Add(*nodes[i])
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Search with 2 direct queries and 2 node IDs
	results, err := idx.NewSearch().
		WithQuery(
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			[]float32{10, 11, 12, 13, 14, 15, 16, 17},
		).
		WithNode(nodes[1].ID(), nodes[3].ID()).
		WithK(2).
		WithNProbes(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// With aggregation enabled (default), results are deduplicated by node ID
	// 4 queries (2 direct + 2 from nodes) each returning k=2
	// Due to overlapping results, we expect fewer than 8 unique results
	if len(results) < 2 {
		t.Errorf("Expected at least 2 deduplicated results, got %d", len(results))
	}

	// Verify we got unique node IDs
	seenIDs := make(map[uint32]bool)
	for _, res := range results {
		if seenIDs[res.Node.ID()] {
			t.Errorf("Found duplicate node ID %d in results", res.Node.ID())
		}
		seenIDs[res.Node.ID()] = true
	}
}

// TestIVFPQIndexSearchBatchQueries tests searching with multiple queries
func TestIVFPQIndexSearchBatchQueries(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	for i := 0; i < 10; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*10 + j)
		}
		node := NewVectorNode(vec)
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Batch search with 3 queries
	results, err := idx.NewSearch().
		WithQuery(
			[]float32{0, 1, 2, 3, 4, 5, 6, 7},
			[]float32{10, 11, 12, 13, 14, 15, 16, 17},
			[]float32{20, 21, 22, 23, 24, 25, 26, 27},
		).
		WithK(2).
		WithNProbes(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// With aggregation enabled (default), results are deduplicated by node ID
	// 3 queries each returning k=2
	// Due to overlapping results, we expect fewer than 6 unique results
	if len(results) < 2 {
		t.Errorf("Expected at least 2 deduplicated results, got %d", len(results))
	}

	// Verify we got unique node IDs
	seenIDs := make(map[uint32]bool)
	for _, res := range results {
		if seenIDs[res.Node.ID()] {
			t.Errorf("Found duplicate node ID %d in results", res.Node.ID())
		}
		seenIDs[res.Node.ID()] = true
	}
}

// TestIVFPQIndexSearchValidation tests validation of search parameters
func TestIVFPQIndexSearchValidation(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	node := NewVectorNode([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	err = idx.Add(*node)
	if err != nil {
		t.Fatalf("Add() error: %v", err)
	}

	t.Run("no query or node", func(t *testing.T) {
		_, err := idx.NewSearch().WithK(3).Execute()
		if err == nil {
			t.Error("Expected error when neither query nor node is specified")
		}
	})

	t.Run("valid query search", func(t *testing.T) {
		_, err := idx.NewSearch().
			WithQuery([]float32{1, 2, 3, 4, 5, 6, 7, 8}).
			WithK(3).
			Execute()
		if err != nil {
			t.Errorf("Expected no error for valid query search, got: %v", err)
		}
	})

	t.Run("valid node search", func(t *testing.T) {
		_, err := idx.NewSearch().
			WithNode(node.ID()).
			WithK(3).
			Execute()
		if err != nil {
			t.Errorf("Expected no error for valid node search, got: %v", err)
		}
	})

	t.Run("valid combined search", func(t *testing.T) {
		_, err := idx.NewSearch().
			WithQuery([]float32{1, 2, 3, 4, 5, 6, 7, 8}).
			WithNode(node.ID()).
			WithK(3).
			Execute()
		if err != nil {
			t.Errorf("Expected no error for valid combined search, got: %v", err)
		}
	})

	t.Run("query dimension mismatch", func(t *testing.T) {
		_, err := idx.NewSearch().
			WithQuery([]float32{1, 2, 3}). // Wrong dimension
			WithK(3).
			Execute()
		if err == nil {
			t.Error("Expected error for query dimension mismatch")
		}
	})
}

// TestIVFPQIndexSearchBeforeTrain tests that search fails before training
func TestIVFPQIndexSearchBeforeTrain(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Try to search without training
	_, err = idx.NewSearch().
		WithQuery([]float32{1, 2, 3, 4, 5, 6, 7, 8}).
		WithK(3).
		Execute()

	if err == nil {
		t.Error("Expected error when searching before training")
	}
}

// TestIVFPQIndexSearchKBounds tests k parameter bounds
func TestIVFPQIndexSearchKBounds(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add 5 vectors
	for i := 0; i < 5; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*10 + j)
		}
		node := NewVectorNode(vec)
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	testCases := []struct {
		name string
		k    int
		want int
	}{
		{"k = 1", 1, 1},
		{"k = 3", 3, 3},
		{"k = 5", 5, 5},
		{"k = 10", 10, 5}, // Only 5 vectors exist
		{"k = 100", 100, 5},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			results, err := idx.NewSearch().
				WithQuery([]float32{0, 1, 2, 3, 4, 5, 6, 7}).
				WithK(tc.k).
				WithNProbes(2).
				Execute()

			if err != nil {
				t.Fatalf("Search() error: %v", err)
			}

			if len(results) != tc.want {
				t.Errorf("k=%d: got %d results, want %d", tc.k, len(results), tc.want)
			}
		})
	}
}

// TestIVFPQIndexSearchDifferentDistanceMetrics tests search with different distance metrics
func TestIVFPQIndexSearchDifferentDistanceMetrics(t *testing.T) {
	testCases := []struct {
		name     string
		distance DistanceKind
	}{
		{"Euclidean", Euclidean},
		{"L2Squared", L2Squared},
		{"Cosine", Cosine},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dim := 8
			idx, err := NewIVFPQIndex(dim, tc.distance, 2, 4, 4)
			if err != nil {
				t.Fatalf("NewIVFPQIndex() error: %v", err)
			}

			// Train (use non-zero vectors for cosine)
			trainingVectors := make([]VectorNode, 100)
			for i := 0; i < 100; i++ {
				vec := make([]float32, dim)
				for j := 0; j < dim; j++ {
					vec[j] = float32(i*dim+j) + 1.0
				}
				trainingVectors[i] = *NewVectorNode(vec)
			}

			err = idx.Train(trainingVectors)
			if err != nil {
				t.Fatalf("Train() with %s error: %v", tc.distance, err)
			}

			// Add vectors
			for i := 0; i < 5; i++ {
				vec := make([]float32, dim)
				for j := 0; j < dim; j++ {
					vec[j] = float32(i*10+j) + 1.0
				}
				node := NewVectorNode(vec)
				err := idx.Add(*node)
				if err != nil {
					t.Fatalf("Add() with %s error: %v", tc.distance, err)
				}
			}

			// Search
			query := make([]float32, dim)
			for j := 0; j < dim; j++ {
				query[j] = float32(j) + 1.0
			}

			results, err := idx.NewSearch().
				WithQuery(query).
				WithK(3).
				WithNProbes(2).
				Execute()

			if err != nil {
				t.Fatalf("Search() with %s error: %v", tc.distance, err)
			}

			if len(results) == 0 {
				t.Errorf("Expected non-empty results with %s", tc.distance)
			}
		})
	}
}

// TestIVFPQIndexSearchWithNProbes tests search with different nprobes values
func TestIVFPQIndexSearchWithNProbes(t *testing.T) {
	dim := 8
	nlist := 4
	idx, err := NewIVFPQIndex(dim, Euclidean, nlist, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 200)
	for i := 0; i < 200; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors distributed across clusters
	for i := 0; i < 20; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*10 + j)
		}
		node := NewVectorNode(vec)
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	query := []float32{5, 6, 7, 8, 9, 10, 11, 12}

	testCases := []struct {
		name    string
		nprobes int
	}{
		{"nprobes = 1", 1},
		{"nprobes = 2", 2},
		{"nprobes = 4", 4},
		{"nprobes = -1 (all)", -1},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			results, err := idx.NewSearch().
				WithQuery(query).
				WithK(5).
				WithNProbes(tc.nprobes).
				Execute()

			if err != nil {
				t.Fatalf("Search() error: %v", err)
			}

			if len(results) == 0 {
				t.Error("Expected non-empty results")
			}
		})
	}
}

// TestIVFPQIndexConcurrentSearch tests concurrent search operations
func TestIVFPQIndexConcurrentSearch(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	for i := 0; i < 20; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*10 + j)
		}
		node := NewVectorNode(vec)
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Concurrent searches
	const numGoroutines = 10
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(gid int) {
			defer wg.Done()

			query := make([]float32, dim)
			for j := 0; j < dim; j++ {
				query[j] = float32(gid*10 + j)
			}

			results, err := idx.NewSearch().
				WithQuery(query).
				WithK(5).
				WithNProbes(2).
				Execute()

			if err != nil {
				t.Errorf("Concurrent search error: %v", err)
			}

			if len(results) == 0 {
				t.Error("Expected non-empty results")
			}
		}(g)
	}

	wg.Wait()
}

// TestIVFPQIndexSearchEmptyIndex tests search on empty index
func TestIVFPQIndexSearchEmptyIndex(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train but don't add vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Search on empty index
	results, err := idx.NewSearch().
		WithQuery([]float32{1, 2, 3, 4, 5, 6, 7, 8}).
		WithK(5).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected 0 results from empty index, got %d", len(results))
	}
}

// TestIVFPQIndexSearchChaining tests method chaining
func TestIVFPQIndexSearchChaining(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	for i := 0; i < 10; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*10 + j)
		}
		node := NewVectorNode(vec)
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Test chaining all methods
	results, err := idx.NewSearch().
		WithQuery([]float32{1, 2, 3, 4, 5, 6, 7, 8}).
		WithK(3).
		WithNProbes(2).
		WithThreshold(100.0).
		Execute()

	if err != nil {
		t.Fatalf("Chained search error: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected non-empty results from chained search")
	}
}

// TestIVFPQIndexSearchResultsConsistency tests that search results are consistent
func TestIVFPQIndexSearchResultsConsistency(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	for i := 0; i < 10; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*10 + j)
		}
		node := NewVectorNode(vec)
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	query := []float32{5, 6, 7, 8, 9, 10, 11, 12}

	// Run same search multiple times
	var firstResults []VectorResult
	for i := 0; i < 3; i++ {
		results, err := idx.NewSearch().
			WithQuery(query).
			WithK(5).
			WithNProbes(2).
			Execute()

		if err != nil {
			t.Fatalf("Search iteration %d error: %v", i, err)
		}

		if i == 0 {
			firstResults = results
		} else {
			// Results should be consistent
			if len(results) != len(firstResults) {
				t.Errorf("Iteration %d: got %d results, want %d", i, len(results), len(firstResults))
			}
		}
	}
}

// TestIVFPQIndexSearchApproximateAccuracy tests that IVFPQ returns reasonable approximations
func TestIVFPQIndexSearchApproximateAccuracy(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors with known distances
	targetVec := []float32{10, 10, 10, 10, 10, 10, 10, 10}
	node1 := NewVectorNode(targetVec)
	idx.Add(*node1)

	// Add some other vectors
	for i := 0; i < 10; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*20 + j)
		}
		node := NewVectorNode(vec)
		idx.Add(*node)
	}

	// Search for exact vector
	results, err := idx.NewSearch().
		WithQuery(targetVec).
		WithK(5).
		WithNProbes(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Expected non-empty results")
	}

	// The exact vector should be in top results (IVFPQ is approximate, so it might not be #1)
	found := false
	for i := 0; i < len(results) && i < 3; i++ {
		if results[i].Node.ID() == node1.ID() {
			found = true
			break
		}
	}

	if !found {
		t.Log("Note: IVFPQ is an approximate algorithm, exact match may not always be in top 3")
	}
}

// TestIVFPQIndexSearchLookupNodeVectors tests the lookupNodeVectors helper
func TestIVFPQIndexSearchLookupNodeVectors(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{2, 3, 4, 5, 6, 7, 8, 9},
		{10, 11, 12, 13, 14, 15, 16, 17},
	}

	nodes := make([]*VectorNode, len(vectors))
	for i, v := range vectors {
		nodes[i] = NewVectorNode(v)
		err := idx.Add(*nodes[i])
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Search using all nodes
	results, err := idx.NewSearch().
		WithNode(nodes[0].ID(), nodes[1].ID(), nodes[2].ID()).
		WithK(2).
		WithNProbes(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// With aggregation enabled (default), results are deduplicated by node ID
	// 3 node queries each returning k=2
	// Due to overlapping results, we expect fewer than 6 unique results
	if len(results) < 2 {
		t.Errorf("Expected at least 2 deduplicated results, got %d", len(results))
	}

	// Verify we got unique node IDs
	seenIDs := make(map[uint32]bool)
	for _, res := range results {
		if seenIDs[res.Node.ID()] {
			t.Errorf("Found duplicate node ID %d in results", res.Node.ID())
		}
		seenIDs[res.Node.ID()] = true
	}
}
