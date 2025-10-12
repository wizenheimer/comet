package comet

import (
	"testing"
)

// TestIVFIndexSearchCombinedQueryAndNode tests searching with both queries and node IDs
func TestIVFIndexSearchCombinedQueryAndNode(t *testing.T) {
	// Create IVF index
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Create training vectors
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{10, 10, 10}),
		*NewVectorNode([]float32{1, 0, 0}),
		*NewVectorNode([]float32{11, 10, 10}),
	}

	// Train the index
	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{2, 0, 0},
		{10, 10, 10},
		{11, 10, 10},
	}

	nodes := make([]*VectorNode, len(vectors))
	for i, v := range vectors {
		nodes[i] = NewVectorNode(v)
		err := idx.Add(*nodes[i])
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Search using both a direct query and a node ID
	results, err := idx.NewSearch().
		WithQuery([]float32{0, 1, 0}).
		WithNode(nodes[0].ID()).
		WithK(2).
		WithNProbes(2).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should get results from both queries
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

// TestIVFIndexSearchMultipleQueriesAndNodes tests batch search with mixed queries and nodes
func TestIVFIndexSearchMultipleQueriesAndNodes(t *testing.T) {
	// Create IVF index
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Create training vectors
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{10, 10, 10}),
		*NewVectorNode([]float32{1, 0, 0}),
		*NewVectorNode([]float32{11, 10, 10}),
	}

	// Train the index
	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{2, 0, 0},
		{0, 2, 0},
		{10, 10, 10},
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
		WithQuery([]float32{1.1, 0, 0}, []float32{0, 1.1, 0}).
		WithNode(nodes[2].ID(), nodes[3].ID()).
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

// TestIVFIndexSearchCombinedWithThreshold tests combined search with threshold
func TestIVFIndexSearchCombinedWithThreshold(t *testing.T) {
	// Create IVF index
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Create training vectors
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{10, 10, 10}),
		*NewVectorNode([]float32{1, 0, 0}),
		*NewVectorNode([]float32{11, 10, 10}),
	}

	// Train the index
	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{5, 0, 0},
		{0, 5, 0},
	}

	nodes := make([]*VectorNode, len(vectors))
	for i, v := range vectors {
		nodes[i] = NewVectorNode(v)
		err := idx.Add(*nodes[i])
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Search with query and node, but with threshold
	results, err := idx.NewSearch().
		WithQuery([]float32{1, 0, 0}).
		WithNode(nodes[1].ID()). // [0, 1, 0]
		WithK(10).
		WithNProbes(2).
		WithThreshold(2.0).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Both queries should only return vectors within distance 2.0
	if len(results) == 0 {
		t.Error("Expected some results within threshold, got none")
	}
}

// TestIVFIndexSearchValidation tests validation of search parameters
func TestIVFIndexSearchValidation(t *testing.T) {
	// Create IVF index
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Create training vectors
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{10, 10, 10}),
		*NewVectorNode([]float32{1, 0, 0}),
	}

	// Train the index
	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
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
			name: "valid query search",
			setup: func() VectorSearch {
				return idx.NewSearch().
					WithQuery([]float32{1, 0, 0}).
					WithK(1)
			},
			wantErr: false,
		},
		{
			name: "valid combined search",
			setup: func() VectorSearch {
				return idx.NewSearch().
					WithQuery([]float32{1, 0, 0}).
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.setup().Execute()
			if tt.wantErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

// TestIVFIndexSearchBeforeTrain tests that search fails if index is not trained
func TestIVFIndexSearchBeforeTrain(t *testing.T) {
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Try to search without training
	_, err = idx.NewSearch().
		WithQuery([]float32{1, 0, 0}).
		WithK(1).
		Execute()

	if err == nil {
		t.Error("Expected error when searching untrained index, got none")
	}
}
