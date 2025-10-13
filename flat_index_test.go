package comet

import (
	"sync"
	"testing"
)

// TestNewFlatIndex tests the creation of a new flat index
func TestNewFlatIndex(t *testing.T) {
	tests := []struct {
		name         string
		dim          int
		distanceKind DistanceKind
		wantErr      bool
		errMsg       string
	}{
		{
			name:         "valid L2 index",
			dim:          128,
			distanceKind: Euclidean,
			wantErr:      false,
		},
		{
			name:         "valid Cosine index",
			dim:          384,
			distanceKind: Cosine,
			wantErr:      false,
		},
		{
			name:         "zero dimension",
			dim:          0,
			distanceKind: Euclidean,
			wantErr:      true,
			errMsg:       "dimension must be positive",
		},
		{
			name:         "negative dimension",
			dim:          -5,
			distanceKind: Euclidean,
			wantErr:      true,
			errMsg:       "dimension must be positive",
		},
		{
			name:         "invalid distance kind",
			dim:          128,
			distanceKind: "invalid",
			wantErr:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, err := NewFlatIndex(tt.dim, tt.distanceKind)
			if tt.wantErr {
				if err == nil {
					t.Errorf("NewFlatIndex() expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("NewFlatIndex() unexpected error: %v", err)
				return
			}

			if idx == nil {
				t.Fatal("NewFlatIndex() returned nil")
			}

			if idx.Dimensions() != tt.dim {
				t.Errorf("Dimensions() = %d, want %d", idx.Dimensions(), tt.dim)
			}

			if idx.DistanceKind() != tt.distanceKind {
				t.Errorf("DistanceKind() = %v, want %v", idx.DistanceKind(), tt.distanceKind)
			}

			if idx.Kind() != FlatIndexKind {
				t.Errorf("Kind() = %v, want %v", idx.Kind(), FlatIndexKind)
			}
		})
	}
}

// TestFlatIndexAdd tests adding vectors to the index
func TestFlatIndexAdd(t *testing.T) {
	tests := []struct {
		name         string
		dim          int
		distanceKind DistanceKind
		vector       []float32
		wantErr      bool
		errMsg       string
	}{
		{
			name:         "add valid vector L2",
			dim:          3,
			distanceKind: Euclidean,
			vector:       []float32{1, 2, 3},
			wantErr:      false,
		},
		{
			name:         "add valid vector Cosine",
			dim:          3,
			distanceKind: Cosine,
			vector:       []float32{1, 2, 3},
			wantErr:      false,
		},
		{
			name:         "dimension mismatch",
			dim:          3,
			distanceKind: Euclidean,
			vector:       []float32{1, 2, 3, 4},
			wantErr:      true,
			errMsg:       "dimension mismatch",
		},
		{
			name:         "zero vector with cosine",
			dim:          3,
			distanceKind: Cosine,
			vector:       []float32{0, 0, 0},
			wantErr:      true,
			errMsg:       "zero vector not allowed",
		},
		{
			name:         "zero vector with L2",
			dim:          3,
			distanceKind: Euclidean,
			vector:       []float32{0, 0, 0},
			wantErr:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, err := NewFlatIndex(tt.dim, tt.distanceKind)
			if err != nil {
				t.Fatalf("NewFlatIndex() error: %v", err)
			}

			node := NewVectorNode(tt.vector)
			err = idx.Add(*node)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Add() expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Add() unexpected error: %v", err)
			}
		})
	}
}

// TestFlatIndexAddMultiple tests adding multiple vectors
func TestFlatIndexAddMultiple(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	vectors := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		if err := idx.Add(*node); err != nil {
			t.Errorf("Add() error: %v", err)
		}
	}

	// Verify all vectors were added
	if len(idx.vectors) != len(vectors) {
		t.Errorf("Expected %d vectors, got %d", len(vectors), len(idx.vectors))
	}
}

// TestFlatIndexRemove tests removing vectors from the index
func TestFlatIndexRemove(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add some vectors
	node1 := NewVectorNode([]float32{1, 2, 3})
	node2 := NewVectorNode([]float32{4, 5, 6})
	node3 := NewVectorNode([]float32{7, 8, 9})

	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)

	// Remove middle vector (soft delete)
	err = idx.Remove(*node2)
	if err != nil {
		t.Errorf("Remove() error: %v", err)
	}

	// Vectors should still be in slice (soft delete)
	if len(idx.vectors) != 3 {
		t.Errorf("Expected 3 vectors after soft delete, got %d", len(idx.vectors))
	}

	// Call Flush to perform hard delete
	err = idx.Flush()
	if err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	// Now vector should be physically removed
	if len(idx.vectors) != 2 {
		t.Errorf("Expected 2 vectors after flush, got %d", len(idx.vectors))
	}

	// Try to remove already deleted vector
	err = idx.Remove(*node2)
	if err == nil {
		t.Error("Remove() expected error for already deleted vector")
	}

	// Try to remove non-existent vector
	nonExistent := NewVectorNodeWithID(9999, []float32{1, 1, 1})
	err = idx.Remove(*nonExistent)
	if err == nil {
		t.Error("Remove() expected error for non-existent vector")
	}

	// Remove remaining vectors
	idx.Remove(*node1)
	idx.Remove(*node3)

	// Still 2 vectors before flush (soft delete)
	if len(idx.vectors) != 2 {
		t.Errorf("Expected 2 vectors before flush, got %d", len(idx.vectors))
	}

	// Flush to remove all
	idx.Flush()

	if len(idx.vectors) != 0 {
		t.Errorf("Expected 0 vectors after removing all, got %d", len(idx.vectors))
	}
}

// TestFlatIndexFlush tests the Flush method
func TestFlatIndexFlush(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add some vectors
	node1 := NewVectorNode([]float32{1, 2, 3})
	node2 := NewVectorNode([]float32{4, 5, 6})
	node3 := NewVectorNode([]float32{7, 8, 9})

	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)

	// Flush with no deletions should succeed and keep all vectors
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error: %v", err)
	}
	if len(idx.vectors) != 3 {
		t.Errorf("Expected 3 vectors after flush with no deletions, got %d", len(idx.vectors))
	}

	// Soft delete two vectors
	idx.Remove(*node1)
	idx.Remove(*node2)

	// Vectors still in memory before flush
	if len(idx.vectors) != 3 {
		t.Errorf("Expected 3 vectors before flush, got %d", len(idx.vectors))
	}

	// Flush should remove deleted vectors
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	// Only one vector should remain
	if len(idx.vectors) != 1 {
		t.Errorf("Expected 1 vector after flush, got %d", len(idx.vectors))
	}

	// Verify the remaining vector is node3
	if idx.vectors[0].ID() != node3.ID() {
		t.Errorf("Expected remaining vector to be node3, got node with ID %d", idx.vectors[0].ID())
	}

	// Verify deleted bitmap is cleared
	if idx.deletedNodes.GetCardinality() != 0 {
		t.Errorf("Expected deletedNodes bitmap to be empty after flush, got cardinality %d", idx.deletedNodes.GetCardinality())
	}

	// Multiple flushes should be safe
	if err := idx.Flush(); err != nil {
		t.Errorf("Second Flush() error: %v", err)
	}
}

// TestFlatIndexConcurrentAdd tests thread-safety of Add operations
func TestFlatIndexConcurrentAdd(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	const numGoroutines = 100
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(i int) {
			defer wg.Done()
			node := NewVectorNode([]float32{float32(i), 0, 0})
			if err := idx.Add(*node); err != nil {
				t.Errorf("Add() error: %v", err)
			}
		}(i)
	}

	wg.Wait()

	if len(idx.vectors) != numGoroutines {
		t.Errorf("Expected %d vectors, got %d", numGoroutines, len(idx.vectors))
	}
}

// TestFlatIndexSoftDeleteWithSearch tests that soft-deleted nodes are filtered during search
func TestFlatIndexSoftDeleteWithSearch(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add test vectors
	node1 := NewVectorNode([]float32{1, 0, 0})
	node2 := NewVectorNode([]float32{2, 0, 0})
	node3 := NewVectorNode([]float32{3, 0, 0})
	node4 := NewVectorNode([]float32{4, 0, 0})

	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)
	idx.Add(*node4)

	// Search should return all 4 vectors
	query := []float32{1.5, 0, 0}
	results, err := idx.NewSearch().WithQuery(query).WithK(10).Execute()
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}
	if len(results) != 4 {
		t.Errorf("Expected 4 results before deletion, got %d", len(results))
	}

	// Soft delete node2 and node3
	idx.Remove(*node2)
	idx.Remove(*node3)

	// Search should now return only 2 vectors (node1 and node4)
	results, err = idx.NewSearch().WithQuery(query).WithK(10).Execute()
	if err != nil {
		t.Fatalf("Search error after soft delete: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("Expected 2 results after soft delete, got %d", len(results))
	}

	// Verify the correct nodes are returned
	resultIDs := make(map[uint32]bool)
	for _, r := range results {
		resultIDs[r.Node.ID()] = true
	}
	if !resultIDs[node1.ID()] {
		t.Error("Expected node1 in results")
	}
	if !resultIDs[node4.ID()] {
		t.Error("Expected node4 in results")
	}
	if resultIDs[node2.ID()] {
		t.Error("Did not expect node2 (soft deleted) in results")
	}
	if resultIDs[node3.ID()] {
		t.Error("Did not expect node3 (soft deleted) in results")
	}

	// Test search by node ID - should fail for deleted nodes
	_, err = idx.NewSearch().WithNode(node2.ID()).WithK(5).Execute()
	if err == nil {
		t.Error("Expected error when searching by deleted node ID")
	}

	// Test search by node ID - should succeed for non-deleted nodes
	results, err = idx.NewSearch().WithNode(node1.ID()).WithK(5).Execute()
	if err != nil {
		t.Errorf("Search by non-deleted node ID failed: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("Expected 2 results when searching by node1, got %d", len(results))
	}

	// After flush, search should still return 2 vectors
	err = idx.Flush()
	if err != nil {
		t.Fatalf("Flush error: %v", err)
	}

	results, err = idx.NewSearch().WithQuery(query).WithK(10).Execute()
	if err != nil {
		t.Fatalf("Search error after flush: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("Expected 2 results after flush, got %d", len(results))
	}

	// Verify physical removal
	if len(idx.vectors) != 2 {
		t.Errorf("Expected 2 vectors in storage after flush, got %d", len(idx.vectors))
	}
}
