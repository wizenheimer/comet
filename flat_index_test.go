package comet

import (
	"bytes"
	"io"
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

// TestFlatIndexWriteTo tests serialization of the index
func TestFlatIndexWriteTo(t *testing.T) {
	tests := []struct {
		name         string
		dim          int
		distanceKind DistanceKind
		vectors      [][]float32
	}{
		{
			name:         "euclidean index with vectors",
			dim:          3,
			distanceKind: Euclidean,
			vectors: [][]float32{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
		},
		{
			name:         "cosine index with vectors",
			dim:          4,
			distanceKind: Cosine,
			vectors: [][]float32{
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{0, 0, 1, 0},
			},
		},
		{
			name:         "l2squared index with vectors",
			dim:          2,
			distanceKind: L2Squared,
			vectors: [][]float32{
				{1.5, 2.5},
				{3.5, 4.5},
			},
		},
		{
			name:         "empty index",
			dim:          5,
			distanceKind: Euclidean,
			vectors:      [][]float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, err := NewFlatIndex(tt.dim, tt.distanceKind)
			if err != nil {
				t.Fatalf("NewFlatIndex() error: %v", err)
			}

			// Add vectors
			for _, v := range tt.vectors {
				node := NewVectorNode(v)
				if err := idx.Add(*node); err != nil {
					t.Fatalf("Add() error: %v", err)
				}
			}

			// Serialize to buffer
			var buf bytes.Buffer
			n, err := idx.WriteTo(&buf)
			if err != nil {
				t.Fatalf("WriteTo() error: %v", err)
			}

			if n <= 0 {
				t.Errorf("WriteTo() returned %d bytes, expected > 0", n)
			}

			// Verify buffer has data
			if buf.Len() == 0 {
				t.Error("WriteTo() wrote no data to buffer")
			}

			// Verify magic number
			magic := buf.Bytes()[:4]
			if string(magic) != "FLAT" {
				t.Errorf("Invalid magic number: got %s, want FLAT", string(magic))
			}
		})
	}
}

// TestFlatIndexReadFrom tests deserialization of the index
func TestFlatIndexReadFrom(t *testing.T) {
	tests := []struct {
		name         string
		dim          int
		distanceKind DistanceKind
		vectors      [][]float32
	}{
		{
			name:         "euclidean index with vectors",
			dim:          3,
			distanceKind: Euclidean,
			vectors: [][]float32{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
		},
		{
			name:         "cosine index with vectors",
			dim:          4,
			distanceKind: Cosine,
			vectors: [][]float32{
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{0, 0, 1, 0},
			},
		},
		{
			name:         "empty index",
			dim:          5,
			distanceKind: L2Squared,
			vectors:      [][]float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create and populate original index
			original, err := NewFlatIndex(tt.dim, tt.distanceKind)
			if err != nil {
				t.Fatalf("NewFlatIndex() error: %v", err)
			}

			for _, v := range tt.vectors {
				node := NewVectorNode(v)
				if err := original.Add(*node); err != nil {
					t.Fatalf("Add() error: %v", err)
				}
			}

			// Serialize
			var buf bytes.Buffer
			_, err = original.WriteTo(&buf)
			if err != nil {
				t.Fatalf("WriteTo() error: %v", err)
			}

			// Create new index and deserialize
			restored, err := NewFlatIndex(tt.dim, tt.distanceKind)
			if err != nil {
				t.Fatalf("NewFlatIndex() error: %v", err)
			}

			n, err := restored.ReadFrom(&buf)
			if err != nil {
				t.Fatalf("ReadFrom() error: %v", err)
			}

			if n <= 0 {
				t.Errorf("ReadFrom() returned %d bytes, expected > 0", n)
			}

			// Verify restored index matches original
			if restored.Dimensions() != original.Dimensions() {
				t.Errorf("Dimensions mismatch: got %d, want %d", restored.Dimensions(), original.Dimensions())
			}

			if restored.DistanceKind() != original.DistanceKind() {
				t.Errorf("DistanceKind mismatch: got %v, want %v", restored.DistanceKind(), original.DistanceKind())
			}

			if len(restored.vectors) != len(original.vectors) {
				t.Errorf("Vector count mismatch: got %d, want %d", len(restored.vectors), len(original.vectors))
			}

			// Verify vector contents
			for i := range original.vectors {
				origVec := original.vectors[i].Vector()
				restVec := restored.vectors[i].Vector()

				if len(origVec) != len(restVec) {
					t.Errorf("Vector %d dimension mismatch: got %d, want %d", i, len(restVec), len(origVec))
					continue
				}

				for j := range origVec {
					if origVec[j] != restVec[j] {
						t.Errorf("Vector %d component %d mismatch: got %f, want %f", i, j, restVec[j], origVec[j])
					}
				}

				if original.vectors[i].ID() != restored.vectors[i].ID() {
					t.Errorf("Vector %d ID mismatch: got %d, want %d", i, restored.vectors[i].ID(), original.vectors[i].ID())
				}
			}
		})
	}
}

// TestFlatIndexSerializationRoundTrip tests that serialization and deserialization preserve data
func TestFlatIndexSerializationRoundTrip(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		if err := idx.Add(*node); err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Perform a search before serialization
	query := []float32{5, 6, 7}
	resultsBefore, err := idx.NewSearch().WithQuery(query).WithK(2).Execute()
	if err != nil {
		t.Fatalf("Search before serialization error: %v", err)
	}

	// Serialize
	var buf bytes.Buffer
	_, err = idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// Deserialize into new index
	idx2, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Perform same search after deserialization
	resultsAfter, err := idx2.NewSearch().WithQuery(query).WithK(2).Execute()
	if err != nil {
		t.Fatalf("Search after deserialization error: %v", err)
	}

	// Results should be identical
	if len(resultsBefore) != len(resultsAfter) {
		t.Errorf("Result count mismatch: before=%d, after=%d", len(resultsBefore), len(resultsAfter))
	}

	for i := range resultsBefore {
		if resultsBefore[i].Node.ID() != resultsAfter[i].Node.ID() {
			t.Errorf("Result %d ID mismatch: before=%d, after=%d", i, resultsBefore[i].Node.ID(), resultsAfter[i].Node.ID())
		}
		if resultsBefore[i].Score != resultsAfter[i].Score {
			t.Errorf("Result %d score mismatch: before=%f, after=%f", i, resultsBefore[i].Score, resultsAfter[i].Score)
		}
	}
}

// TestFlatIndexSerializationWithDeletions tests serialization with soft-deleted vectors
func TestFlatIndexSerializationWithDeletions(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add vectors
	node1 := NewVectorNode([]float32{1, 2, 3})
	node2 := NewVectorNode([]float32{4, 5, 6})
	node3 := NewVectorNode([]float32{7, 8, 9})
	node4 := NewVectorNode([]float32{10, 11, 12})

	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)
	idx.Add(*node4)

	// Soft delete some vectors
	idx.Remove(*node2)
	idx.Remove(*node4)

	// Verify soft deletes exist before serialization
	if len(idx.vectors) != 4 {
		t.Errorf("Expected 4 vectors before serialization (soft delete), got %d", len(idx.vectors))
	}

	// Serialize (should call Flush automatically)
	var buf bytes.Buffer
	_, err = idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// After WriteTo (which calls Flush), deleted vectors should be removed
	if len(idx.vectors) != 2 {
		t.Errorf("Expected 2 vectors after WriteTo (auto-flush), got %d", len(idx.vectors))
	}

	// Deserialize
	idx2, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Restored index should only have non-deleted vectors
	if len(idx2.vectors) != 2 {
		t.Errorf("Expected 2 vectors in restored index, got %d", len(idx2.vectors))
	}

	// Verify the correct vectors remain
	foundNode1 := false
	foundNode3 := false
	for _, v := range idx2.vectors {
		if v.ID() == node1.ID() {
			foundNode1 = true
		}
		if v.ID() == node3.ID() {
			foundNode3 = true
		}
	}

	if !foundNode1 {
		t.Error("Expected node1 in restored index")
	}
	if !foundNode3 {
		t.Error("Expected node3 in restored index")
	}
}

// TestFlatIndexReadFromInvalidData tests error handling for invalid serialized data
func TestFlatIndexReadFromInvalidData(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() *bytes.Buffer
		wantErr string
	}{
		{
			name: "invalid magic number",
			setup: func() *bytes.Buffer {
				buf := bytes.NewBuffer([]byte("XXXX"))
				return buf
			},
			wantErr: "invalid magic number",
		},
		{
			name: "unsupported version",
			setup: func() *bytes.Buffer {
				var buf bytes.Buffer
				// Write valid magic
				buf.Write([]byte("FLAT"))
				// Write invalid version
				buf.Write([]byte{99, 0, 0, 0}) // version 99
				// Rest of data...
				return &buf
			},
			wantErr: "unsupported version",
		},
		{
			name: "dimension mismatch",
			setup: func() *bytes.Buffer {
				// Create index with dim=3
				idx, _ := NewFlatIndex(3, Euclidean)
				node := NewVectorNode([]float32{1, 2, 3})
				idx.Add(*node)

				var buf bytes.Buffer
				idx.WriteTo(&buf)
				return &buf
			},
			wantErr: "dimension mismatch",
		},
		{
			name: "distance kind mismatch",
			setup: func() *bytes.Buffer {
				// Create index with Euclidean
				idx, _ := NewFlatIndex(5, Euclidean)
				node := NewVectorNode([]float32{1, 2, 3, 4, 5})
				idx.Add(*node)

				var buf bytes.Buffer
				idx.WriteTo(&buf)
				return &buf
			},
			wantErr: "distance kind mismatch",
		},
		{
			name: "truncated data",
			setup: func() *bytes.Buffer {
				buf := bytes.NewBuffer([]byte("FL"))
				return buf
			},
			wantErr: "failed to read magic number",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := tt.setup()

			// For dimension and distance mismatch tests, we need different index params
			var idx *FlatIndex
			var err error

			switch tt.name {
			case "dimension mismatch":
				idx, err = NewFlatIndex(5, Euclidean) // Different dimension
			case "distance kind mismatch":
				idx, err = NewFlatIndex(5, Cosine) // Different distance kind
			default:
				idx, err = NewFlatIndex(3, Euclidean)
			}

			if err != nil {
				t.Fatalf("NewFlatIndex() error: %v", err)
			}

			_, err = idx.ReadFrom(buf)
			if err == nil {
				t.Errorf("ReadFrom() expected error containing '%s', got nil", tt.wantErr)
				return
			}

			// Check if error message contains expected substring
			if tt.wantErr != "" {
				errMsg := err.Error()
				found := false
				// Simple substring check
				for i := 0; i <= len(errMsg)-len(tt.wantErr); i++ {
					if errMsg[i:i+len(tt.wantErr)] == tt.wantErr {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("ReadFrom() error = %v, want error containing '%s'", err, tt.wantErr)
				}
			}
		})
	}
}

// TestFlatIndexSerializationConcurrency tests thread-safety of serialization
func TestFlatIndexSerializationConcurrency(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add some vectors
	for i := 0; i < 100; i++ {
		node := NewVectorNode([]float32{float32(i), float32(i + 1), float32(i + 2)})
		if err := idx.Add(*node); err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Serialize concurrently
	const numGoroutines = 10
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	errors := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			var buf bytes.Buffer
			if _, err := idx.WriteTo(&buf); err != nil {
				errors <- err
			}
		}()
	}

	wg.Wait()
	close(errors)

	// Check for errors
	for err := range errors {
		t.Errorf("Concurrent WriteTo() error: %v", err)
	}
}

// TestFlatIndexSerializationLargeIndex tests serialization with a large number of vectors
func TestFlatIndexSerializationLargeIndex(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large index test in short mode")
	}

	idx, err := NewFlatIndex(128, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add 10,000 vectors
	const numVectors = 10000
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(i*128 + j)
		}
		node := NewVectorNode(vec)
		if err := idx.Add(*node); err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Serialize
	var buf bytes.Buffer
	bytesWritten, err := idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	t.Logf("Serialized %d vectors (%d dimensions) to %d bytes", numVectors, 128, bytesWritten)

	// Deserialize
	idx2, err := NewFlatIndex(128, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	bytesRead, err := idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	if bytesRead != bytesWritten {
		t.Errorf("Bytes read (%d) != bytes written (%d)", bytesRead, bytesWritten)
	}

	// Verify vector count
	if len(idx2.vectors) != numVectors {
		t.Errorf("Expected %d vectors, got %d", numVectors, len(idx2.vectors))
	}
}

// TestFlatIndexWriteToFlushBehavior tests that WriteTo calls Flush
func TestFlatIndexWriteToFlushBehavior(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add vectors
	node1 := NewVectorNode([]float32{1, 2, 3})
	node2 := NewVectorNode([]float32{4, 5, 6})
	node3 := NewVectorNode([]float32{7, 8, 9})

	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)

	// Soft delete one vector
	idx.Remove(*node2)

	// Before WriteTo, should have 3 vectors (soft delete)
	if len(idx.vectors) != 3 {
		t.Errorf("Expected 3 vectors before WriteTo, got %d", len(idx.vectors))
	}

	// Call WriteTo (should flush)
	var buf bytes.Buffer
	_, err = idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// After WriteTo, should have 2 vectors (flush removes soft deletes)
	if len(idx.vectors) != 2 {
		t.Errorf("Expected 2 vectors after WriteTo (auto-flush), got %d", len(idx.vectors))
	}

	// Deleted bitmap should be empty
	if idx.deletedNodes.GetCardinality() != 0 {
		t.Errorf("Expected deletedNodes to be empty after WriteTo, got cardinality %d", idx.deletedNodes.GetCardinality())
	}
}

// TestFlatIndexEmptyIndexSerialization tests serialization of an empty index
func TestFlatIndexEmptyIndexSerialization(t *testing.T) {
	idx, err := NewFlatIndex(10, Cosine)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Serialize empty index
	var buf bytes.Buffer
	n, err := idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	if n <= 0 {
		t.Errorf("WriteTo() returned %d bytes for empty index, expected > 0", n)
	}

	// Deserialize
	idx2, err := NewFlatIndex(10, Cosine)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Verify restored index is also empty
	if len(idx2.vectors) != 0 {
		t.Errorf("Expected 0 vectors in restored index, got %d", len(idx2.vectors))
	}

	// Verify properties match
	if idx2.Dimensions() != idx.Dimensions() {
		t.Errorf("Dimensions mismatch: got %d, want %d", idx2.Dimensions(), idx.Dimensions())
	}

	if idx2.DistanceKind() != idx.DistanceKind() {
		t.Errorf("DistanceKind mismatch: got %v, want %v", idx2.DistanceKind(), idx.DistanceKind())
	}
}

// errorWriter is a writer that always returns an error
type errorWriter struct{}

func (e errorWriter) Write(p []byte) (n int, err error) {
	return 0, io.ErrUnexpectedEOF
}

// TestFlatIndexWriteToError tests error handling during write operations
func TestFlatIndexWriteToError(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	node := NewVectorNode([]float32{1, 2, 3})
	idx.Add(*node)

	// Try to write to an error writer
	var errWriter errorWriter
	_, err = idx.WriteTo(errWriter)
	if err == nil {
		t.Error("WriteTo() expected error when writing to error writer, got nil")
	}
}
