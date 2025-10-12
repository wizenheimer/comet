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

	// Remove middle vector
	err = idx.Remove(*node2)
	if err != nil {
		t.Errorf("Remove() error: %v", err)
	}

	if len(idx.vectors) != 2 {
		t.Errorf("Expected 2 vectors after removal, got %d", len(idx.vectors))
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

	// Flush should always succeed (it's a no-op)
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error: %v", err)
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
