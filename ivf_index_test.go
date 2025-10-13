package comet

import (
	"bytes"
	"io"
	"sync"
	"testing"
)

// TestNewIVFIndex tests IVF index creation with various parameters
func TestNewIVFIndex(t *testing.T) {
	tests := []struct {
		name         string
		dim          int
		nlist        int
		distanceKind DistanceKind
		wantErr      bool
	}{
		{"valid L2 index", 128, 10, Euclidean, false},
		{"valid Cosine index", 384, 20, Cosine, false},
		{"valid L2Squared index", 768, 100, L2Squared, false},
		{"zero dimension", 0, 10, Euclidean, true},
		{"negative dimension", -1, 10, Euclidean, true},
		{"zero nlist", 128, 0, Euclidean, true},
		{"negative nlist", 128, -1, Euclidean, true},
		{"invalid distance kind", 128, 10, DistanceKind("invalid"), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, err := NewIVFIndex(tt.dim, tt.nlist, tt.distanceKind)

			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error but got: %v", err)
				}
				if idx == nil {
					t.Error("Expected non-nil index")
				}
				if idx.Dimensions() != tt.dim {
					t.Errorf("Dimensions() = %d, want %d", idx.Dimensions(), tt.dim)
				}
				if idx.DistanceKind() != tt.distanceKind {
					t.Errorf("DistanceKind() = %v, want %v", idx.DistanceKind(), tt.distanceKind)
				}
				if idx.Kind() != IVFIndexKind {
					t.Errorf("Kind() = %v, want %v", idx.Kind(), IVFIndexKind)
				}
			}
		})
	}
}

// TestIVFIndexTrain tests training functionality
func TestIVFIndexTrain(t *testing.T) {
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Create training vectors - enough for 2 clusters
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{1, 0, 0}),
		*NewVectorNode([]float32{10, 10, 10}),
		*NewVectorNode([]float32{11, 10, 10}),
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Verify index is trained
	if !idx.trained {
		t.Error("Index should be marked as trained")
	}

	// Verify centroids were learned
	if len(idx.centroids) != 2 {
		t.Errorf("Expected 2 centroids, got %d", len(idx.centroids))
	}

	// Verify centroid dimensions
	for i, centroid := range idx.centroids {
		if len(centroid) != 3 {
			t.Errorf("Centroid %d has dimension %d, want 3", i, len(centroid))
		}
	}
}

// TestIVFIndexTrainInsufficientVectors tests training with too few vectors
func TestIVFIndexTrainInsufficientVectors(t *testing.T) {
	idx, err := NewIVFIndex(3, 10, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Only provide 5 vectors for 10 clusters
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{1, 0, 0}),
		*NewVectorNode([]float32{2, 0, 0}),
		*NewVectorNode([]float32{3, 0, 0}),
		*NewVectorNode([]float32{4, 0, 0}),
	}

	err = idx.Train(trainingVectors)
	if err == nil {
		t.Error("Expected error when training with insufficient vectors")
	}
}

// TestIVFIndexAddBeforeTrain tests that add fails before training
func TestIVFIndexAddBeforeTrain(t *testing.T) {
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	node := NewVectorNode([]float32{1, 0, 0})
	err = idx.Add(*node)

	if err == nil {
		t.Error("Expected error when adding before training")
	}
}

// TestIVFIndexAdd tests adding vectors to the index
func TestIVFIndexAdd(t *testing.T) {
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Train first
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{1, 0, 0}),
		*NewVectorNode([]float32{10, 10, 10}),
		*NewVectorNode([]float32{11, 10, 10}),
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{10, 10, 10},
		{11, 11, 11},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		err := idx.Add(*node)
		if err != nil {
			t.Errorf("Add() error: %v", err)
		}
	}

	// Verify vectors were added to inverted lists
	totalVectors := 0
	for _, list := range idx.lists {
		totalVectors += len(list)
	}

	if totalVectors != len(vectors) {
		t.Errorf("Expected %d vectors in index, got %d", len(vectors), totalVectors)
	}
}

// TestIVFIndexAddDimensionMismatch tests adding wrong dimension vectors
func TestIVFIndexAddDimensionMismatch(t *testing.T) {
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Train first
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{10, 10, 10}),
	}
	idx.Train(trainingVectors)

	// Try to add 4D vector to 3D index
	node := NewVectorNode([]float32{1, 0, 0, 0})
	err = idx.Add(*node)

	if err == nil {
		t.Error("Expected error when adding wrong dimension vector")
	}
}

// TestIVFIndexAddZeroVectorCosine tests adding zero vector with cosine distance
func TestIVFIndexAddZeroVectorCosine(t *testing.T) {
	idx, err := NewIVFIndex(3, 2, Cosine)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Train first
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{1, 0, 0}),
		*NewVectorNode([]float32{0, 1, 0}),
	}
	idx.Train(trainingVectors)

	// Try to add zero vector (cannot be normalized)
	node := NewVectorNode([]float32{0, 0, 0})
	err = idx.Add(*node)

	if err == nil {
		t.Error("Expected error when adding zero vector with cosine distance")
	}
}

// TestIVFIndexRemove tests removing vectors from the index
func TestIVFIndexRemove(t *testing.T) {
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Train and add vectors
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{10, 10, 10}),
	}
	idx.Train(trainingVectors)

	node1 := NewVectorNode([]float32{1, 0, 0})
	node2 := NewVectorNode([]float32{2, 0, 0})
	idx.Add(*node1)
	idx.Add(*node2)

	// Remove first node (soft delete)
	err = idx.Remove(*node1)
	if err != nil {
		t.Errorf("Remove() error: %v", err)
	}

	// Count vectors - should still be 2 (soft delete)
	totalVectors := 0
	for _, list := range idx.lists {
		totalVectors += len(list)
	}

	if totalVectors != 2 {
		t.Errorf("Expected 2 vectors after soft delete, got %d", totalVectors)
	}

	// Call Flush to perform hard delete
	err = idx.Flush()
	if err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	// Count vectors after flush - should be 1
	totalVectors = 0
	for _, list := range idx.lists {
		totalVectors += len(list)
	}

	if totalVectors != 1 {
		t.Errorf("Expected 1 vector after flush, got %d", totalVectors)
	}

	// Try to remove already deleted vector
	err = idx.Remove(*node1)
	if err == nil {
		t.Error("Remove() expected error for already deleted vector")
	}
}

// TestIVFIndexRemoveNonExistent tests removing non-existent vector
func TestIVFIndexRemoveNonExistent(t *testing.T) {
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Train first
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{10, 10, 10}),
	}
	idx.Train(trainingVectors)

	// Try to remove a node that was never added
	node := NewVectorNode([]float32{1, 0, 0})
	err = idx.Remove(*node)

	if err == nil {
		t.Error("Expected error when removing non-existent vector")
	}
}

// TestIVFIndexFlush tests the flush method
func TestIVFIndexFlush(t *testing.T) {
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Train and add vectors
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{10, 10, 10}),
	}
	idx.Train(trainingVectors)

	node1 := NewVectorNode([]float32{1, 0, 0})
	node2 := NewVectorNode([]float32{2, 0, 0})
	node3 := NewVectorNode([]float32{3, 0, 0})
	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)

	// Flush with no deletions should succeed and keep all vectors
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	totalVectors := 0
	for _, list := range idx.lists {
		totalVectors += len(list)
	}
	if totalVectors != 3 {
		t.Errorf("Expected 3 vectors after flush with no deletions, got %d", totalVectors)
	}

	// Soft delete two vectors
	idx.Remove(*node1)
	idx.Remove(*node2)

	// Vectors still in memory before flush
	totalVectors = 0
	for _, list := range idx.lists {
		totalVectors += len(list)
	}
	if totalVectors != 3 {
		t.Errorf("Expected 3 vectors before flush, got %d", totalVectors)
	}

	// Flush should remove deleted vectors
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	// Only one vector should remain
	totalVectors = 0
	for _, list := range idx.lists {
		totalVectors += len(list)
	}
	if totalVectors != 1 {
		t.Errorf("Expected 1 vector after flush, got %d", totalVectors)
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

// TestIVFIndexConcurrentAdd tests concurrent additions to the index
func TestIVFIndexConcurrentAdd(t *testing.T) {
	idx, err := NewIVFIndex(3, 4, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Train first
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{5, 5, 5}),
		*NewVectorNode([]float32{10, 10, 10}),
		*NewVectorNode([]float32{15, 15, 15}),
	}
	idx.Train(trainingVectors)

	// Add vectors concurrently
	var wg sync.WaitGroup
	numGoroutines := 10
	vectorsPerGoroutine := 10

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(offset int) {
			defer wg.Done()
			for j := 0; j < vectorsPerGoroutine; j++ {
				node := NewVectorNode([]float32{
					float32(offset*10 + j),
					float32(offset*10 + j),
					float32(offset*10 + j),
				})
				err := idx.Add(*node)
				if err != nil {
					t.Errorf("Add() error: %v", err)
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify all vectors were added
	totalVectors := 0
	for _, list := range idx.lists {
		totalVectors += len(list)
	}

	expected := numGoroutines * vectorsPerGoroutine
	if totalVectors != expected {
		t.Errorf("Expected %d vectors after concurrent adds, got %d", expected, totalVectors)
	}
}

// TestIVFIndexConcurrentAddAndSearch tests concurrent adds and searches
func TestIVFIndexConcurrentAddAndSearch(t *testing.T) {
	idx, err := NewIVFIndex(3, 4, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Train first
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{5, 5, 5}),
		*NewVectorNode([]float32{10, 10, 10}),
		*NewVectorNode([]float32{15, 15, 15}),
	}
	idx.Train(trainingVectors)

	// Add some initial vectors
	for i := 0; i < 20; i++ {
		node := NewVectorNode([]float32{float32(i), float32(i), float32(i)})
		idx.Add(*node)
	}

	var wg sync.WaitGroup

	// Concurrent adds
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(offset int) {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				node := NewVectorNode([]float32{
					float32(offset*10 + j + 20),
					float32(offset*10 + j + 20),
					float32(offset*10 + j + 20),
				})
				idx.Add(*node)
			}
		}(i)
	}

	// Concurrent searches
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				_, err := idx.NewSearch().
					WithQuery([]float32{5, 5, 5}).
					WithK(3).
					Execute()
				if err != nil {
					t.Errorf("Search() error: %v", err)
				}
			}
		}()
	}

	wg.Wait()
}

// TestIVFIndexMultipleClusters tests proper distribution across multiple clusters
func TestIVFIndexMultipleClusters(t *testing.T) {
	idx, err := NewIVFIndex(3, 3, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Train with well-separated clusters
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{1, 0, 0}),
		*NewVectorNode([]float32{10, 10, 10}),
		*NewVectorNode([]float32{11, 10, 10}),
		*NewVectorNode([]float32{20, 20, 20}),
		*NewVectorNode([]float32{21, 20, 20}),
	}
	idx.Train(trainingVectors)

	// Add vectors close to each centroid
	vectors := [][]float32{
		{0.5, 0, 0},    // Cluster 0
		{10.5, 10, 10}, // Cluster 1
		{20.5, 20, 20}, // Cluster 2
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		idx.Add(*node)
	}

	// Verify vectors are distributed across different clusters
	nonEmptyLists := 0
	for _, list := range idx.lists {
		if len(list) > 0 {
			nonEmptyLists++
		}
	}

	if nonEmptyLists < 2 {
		t.Errorf("Expected vectors in at least 2 clusters, got %d", nonEmptyLists)
	}
}

// TestIVFIndexDifferentDistanceMetrics tests IVF with different distance metrics
func TestIVFIndexDifferentDistanceMetrics(t *testing.T) {
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
			idx, err := NewIVFIndex(3, 2, tt.distanceKind)
			if err != nil {
				t.Fatalf("NewIVFIndex() error: %v", err)
			}

			// Train
			trainingVectors := []VectorNode{
				*NewVectorNode([]float32{1, 0, 0}),
				*NewVectorNode([]float32{0, 1, 0}),
				*NewVectorNode([]float32{10, 10, 10}),
			}
			err = idx.Train(trainingVectors)
			if err != nil {
				t.Fatalf("Train() error: %v", err)
			}

			// Add and search
			node := NewVectorNode([]float32{1, 1, 0})
			err = idx.Add(*node)
			if err != nil {
				t.Fatalf("Add() error: %v", err)
			}

			results, err := idx.NewSearch().
				WithQuery([]float32{1, 1, 0}).
				WithK(1).
				Execute()

			if err != nil {
				t.Fatalf("Search() error: %v", err)
			}

			if len(results) != 1 {
				t.Errorf("Expected 1 result, got %d", len(results))
			}
		})
	}
}

// TestIVFIndexGetters tests getter methods
func TestIVFIndexGetters(t *testing.T) {
	dim := 384
	nlist := 100
	distanceKind := Cosine

	idx, err := NewIVFIndex(dim, nlist, distanceKind)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	if idx.Dimensions() != dim {
		t.Errorf("Dimensions() = %d, want %d", idx.Dimensions(), dim)
	}

	if idx.DistanceKind() != distanceKind {
		t.Errorf("DistanceKind() = %v, want %v", idx.DistanceKind(), distanceKind)
	}

	if idx.Kind() != IVFIndexKind {
		t.Errorf("Kind() = %v, want %v", idx.Kind(), IVFIndexKind)
	}
}

// TestIVFIndexNewSearch tests search builder creation
func TestIVFIndexNewSearch(t *testing.T) {
	idx, err := NewIVFIndex(3, 16, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	search := idx.NewSearch()
	if search == nil {
		t.Error("NewSearch() returned nil")
	}

	// Verify it's the correct type
	ivfSearch, ok := search.(*ivfIndexSearch)
	if !ok {
		t.Error("NewSearch() did not return *ivfIndexSearch")
	}

	// Verify defaults
	if ivfSearch.k != 10 {
		t.Errorf("Default k = %d, want 10", ivfSearch.k)
	}

	// nprobes should default to sqrt(nlist) = sqrt(16) = 4
	if ivfSearch.nprobes != 4 {
		t.Errorf("Default nprobes = %d, want 4", ivfSearch.nprobes)
	}
}

// TestIVFIndexLargeScale tests IVF with more realistic scale
func TestIVFIndexLargeScale(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large scale test in short mode")
	}

	dim := 128
	nlist := 10
	numVectors := 1000

	idx, err := NewIVFIndex(dim, nlist, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i*dim + j) % 100)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	// Train
	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i*dim + j) % 100)
		}
		node := NewVectorNode(vec)
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Verify vectors were added
	totalVectors := 0
	for _, list := range idx.lists {
		totalVectors += len(list)
	}

	if totalVectors != numVectors {
		t.Errorf("Expected %d vectors, got %d", numVectors, totalVectors)
	}

	// Search
	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = float32(j % 100)
	}

	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		WithNProbes(3).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 10 {
		t.Errorf("Expected 10 results, got %d", len(results))
	}
}

// TestIVFIndexSoftDeleteWithSearch tests that soft-deleted nodes are filtered during search
func TestIVFIndexSoftDeleteWithSearch(t *testing.T) {
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Train the index
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{10, 10, 10}),
	}
	idx.Train(trainingVectors)

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
	totalVectors := 0
	for _, list := range idx.lists {
		totalVectors += len(list)
	}
	if totalVectors != 2 {
		t.Errorf("Expected 2 vectors in storage after flush, got %d", totalVectors)
	}
}

// TestIVFIndexWriteTo tests serialization of the IVF index
func TestIVFIndexWriteTo(t *testing.T) {
	// Create and train index
	dim := 32
	nlist := 4
	idx, err := NewIVFIndex(dim, nlist, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	if err := idx.Train(trainingVectors); err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	for i := 0; i < 10; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i+100)*dim + j)
		}
		node := NewVectorNode(vec)
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
	if string(magic) != "IVFX" {
		t.Errorf("Invalid magic number: got %s, want IVFX", string(magic))
	}
}

// TestIVFIndexReadFrom tests deserialization of the IVF index
func TestIVFIndexReadFrom(t *testing.T) {
	// Create and train original index
	dim := 32
	nlist := 4
	original, err := NewIVFIndex(dim, nlist, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	if err := original.Train(trainingVectors); err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	for i := 0; i < 10; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i+100)*dim + j)
		}
		node := NewVectorNode(vec)
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
	restored, err := NewIVFIndex(dim, nlist, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
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

	if restored.nlist != original.nlist {
		t.Errorf("nlist mismatch: got %d, want %d", restored.nlist, original.nlist)
	}

	if restored.Trained() != original.Trained() {
		t.Errorf("Trained mismatch: got %v, want %v", restored.Trained(), original.Trained())
	}

	// Verify centroids
	if len(restored.centroids) != len(original.centroids) {
		t.Errorf("Centroids count mismatch: got %d, want %d", len(restored.centroids), len(original.centroids))
	}

	// Verify inverted lists count
	if len(restored.lists) != len(original.lists) {
		t.Errorf("Lists count mismatch: got %d, want %d", len(restored.lists), len(original.lists))
	}

	// Verify total vectors
	originalTotal := 0
	for _, list := range original.lists {
		originalTotal += len(list)
	}
	restoredTotal := 0
	for _, list := range restored.lists {
		restoredTotal += len(list)
	}
	if restoredTotal != originalTotal {
		t.Errorf("Total vectors mismatch: got %d, want %d", restoredTotal, originalTotal)
	}
}

// TestIVFIndexSerializationRoundTrip tests that serialization and deserialization preserve data
func TestIVFIndexSerializationRoundTrip(t *testing.T) {
	// Create and train index
	dim := 32
	nlist := 4
	idx, err := NewIVFIndex(dim, nlist, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	if err := idx.Train(trainingVectors); err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	for i := 0; i < 10; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i+100)*dim + j)
		}
		node := NewVectorNode(vec)
		if err := idx.Add(*node); err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Perform a search before serialization
	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = float32(105*dim + j)
	}
	resultsBefore, err := idx.NewSearch().WithQuery(query).WithK(3).Execute()
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
	idx2, err := NewIVFIndex(dim, nlist, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Perform same search after deserialization
	resultsAfter, err := idx2.NewSearch().WithQuery(query).WithK(3).Execute()
	if err != nil {
		t.Fatalf("Search after deserialization error: %v", err)
	}

	// Results should match in count
	if len(resultsBefore) != len(resultsAfter) {
		t.Errorf("Result count mismatch: before=%d, after=%d", len(resultsBefore), len(resultsAfter))
	}

	// IVF is approximate (searches only nprobe partitions), so exact ordering may vary for similar distances
	// Verify that the same set of IDs is returned
	idsBefore := make(map[uint32]bool)
	for _, r := range resultsBefore {
		idsBefore[r.Node.ID()] = true
	}

	idsAfter := make(map[uint32]bool)
	for _, r := range resultsAfter {
		idsAfter[r.Node.ID()] = true
	}

	// Check that all IDs from before are in after
	for id := range idsBefore {
		if !idsAfter[id] {
			t.Errorf("ID %d present before serialization but missing after", id)
		}
	}

	// Check that all IDs from after are in before
	for id := range idsAfter {
		if !idsBefore[id] {
			t.Errorf("ID %d present after serialization but missing before", id)
		}
	}
}

// TestIVFIndexSerializationWithDeletions tests serialization with soft-deleted vectors
func TestIVFIndexSerializationWithDeletions(t *testing.T) {
	// Create and train index
	dim := 32
	nlist := 4
	idx, err := NewIVFIndex(dim, nlist, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	if err := idx.Train(trainingVectors); err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	nodes := make([]*VectorNode, 4)
	for i := 0; i < 4; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i+100)*dim + j)
		}
		node := NewVectorNode(vec)
		nodes[i] = node
		if err := idx.Add(*node); err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Soft delete some vectors
	idx.Remove(*nodes[1])
	idx.Remove(*nodes[3])

	// Verify soft deletes exist before serialization
	initialTotal := 0
	for _, list := range idx.lists {
		initialTotal += len(list)
	}
	if initialTotal != 4 {
		t.Errorf("Expected 4 vectors before serialization (soft delete), got %d", initialTotal)
	}

	// Serialize (should call Flush automatically)
	var buf bytes.Buffer
	_, err = idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// After WriteTo (which calls Flush), deleted vectors should be removed
	afterFlushTotal := 0
	for _, list := range idx.lists {
		afterFlushTotal += len(list)
	}
	if afterFlushTotal != 2 {
		t.Errorf("Expected 2 vectors after WriteTo (auto-flush), got %d", afterFlushTotal)
	}

	// Deserialize
	idx2, err := NewIVFIndex(dim, nlist, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Restored index should only have non-deleted vectors
	restoredTotal := 0
	for _, list := range idx2.lists {
		restoredTotal += len(list)
	}
	if restoredTotal != 2 {
		t.Errorf("Expected 2 vectors in restored index, got %d", restoredTotal)
	}
}

// TestIVFIndexReadFromInvalidData tests error handling for invalid serialized data
func TestIVFIndexReadFromInvalidData(t *testing.T) {
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
				buf.Write([]byte("IVFX"))
				// Write invalid version
				buf.Write([]byte{99, 0, 0, 0}) // version 99
				return &buf
			},
			wantErr: "unsupported version",
		},
		{
			name: "truncated data",
			setup: func() *bytes.Buffer {
				buf := bytes.NewBuffer([]byte("IV"))
				return buf
			},
			wantErr: "failed to read magic number",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := tt.setup()

			idx, err := NewIVFIndex(32, 4, Euclidean)
			if err != nil {
				t.Fatalf("NewIVFIndex() error: %v", err)
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

// TestIVFIndexSerializationUntrained tests serialization of untrained index
func TestIVFIndexSerializationUntrained(t *testing.T) {
	// Create untrained index
	idx, err := NewIVFIndex(32, 4, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Serialize untrained index
	var buf bytes.Buffer
	n, err := idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	if n <= 0 {
		t.Errorf("WriteTo() returned %d bytes for untrained index, expected > 0", n)
	}

	// Deserialize
	idx2, err := NewIVFIndex(32, 4, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Verify restored index is also untrained
	if idx2.Trained() {
		t.Error("Expected restored index to be untrained")
	}

	// Verify centroids are nil/empty
	if len(idx2.centroids) > 0 {
		t.Error("Expected no centroids in untrained restored index")
	}

	// Verify lists have no vectors (lists slice may be allocated, but should be empty)
	totalVectors := 0
	for _, list := range idx2.lists {
		totalVectors += len(list)
	}
	if totalVectors > 0 {
		t.Errorf("Expected no vectors in untrained restored index, got %d", totalVectors)
	}
}

// TestIVFIndexWriteToFlushBehavior tests that WriteTo calls Flush
func TestIVFIndexWriteToFlushBehavior(t *testing.T) {
	// Create and train index
	dim := 32
	nlist := 4
	idx, err := NewIVFIndex(dim, nlist, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	if err := idx.Train(trainingVectors); err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors
	nodes := make([]*VectorNode, 3)
	for i := 0; i < 3; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i+100)*dim + j)
		}
		node := NewVectorNode(vec)
		nodes[i] = node
		if err := idx.Add(*node); err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Soft delete one vector
	idx.Remove(*nodes[1])

	// Before WriteTo, should have 3 vectors (soft delete)
	totalVectors := 0
	for _, list := range idx.lists {
		totalVectors += len(list)
	}
	if totalVectors != 3 {
		t.Errorf("Expected 3 vectors before WriteTo, got %d", totalVectors)
	}

	// Call WriteTo (should flush)
	var buf bytes.Buffer
	_, err = idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// After WriteTo, should have 2 vectors (flush removes soft deletes)
	totalVectors = 0
	for _, list := range idx.lists {
		totalVectors += len(list)
	}
	if totalVectors != 2 {
		t.Errorf("Expected 2 vectors after WriteTo (auto-flush), got %d", totalVectors)
	}

	// Deleted bitmap should be empty
	if idx.deletedNodes.GetCardinality() != 0 {
		t.Errorf("Expected deletedNodes to be empty after WriteTo, got cardinality %d", idx.deletedNodes.GetCardinality())
	}
}

// errorWriterIVF is a writer that always returns an error
type errorWriterIVF struct{}

func (e errorWriterIVF) Write(p []byte) (n int, err error) {
	return 0, io.ErrUnexpectedEOF
}

// TestIVFIndexWriteToError tests error handling during write operations
func TestIVFIndexWriteToError(t *testing.T) {
	// Create and train index
	dim := 32
	nlist := 4
	idx, err := NewIVFIndex(dim, nlist, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	if err := idx.Train(trainingVectors); err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	vec := make([]float32, dim)
	for j := 0; j < dim; j++ {
		vec[j] = float32(100*dim + j)
	}
	node := NewVectorNode(vec)
	idx.Add(*node)

	// Try to write to an error writer
	var errWriter errorWriterIVF
	_, err = idx.WriteTo(errWriter)
	if err == nil {
		t.Error("WriteTo() expected error when writing to error writer, got nil")
	}
}
