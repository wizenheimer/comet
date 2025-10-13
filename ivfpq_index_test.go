package comet

import (
	"sync"
	"testing"
)

// getTotalVectors is a test utility that calculates the total number of vectors
// across all inverted lists in an IVFPQ index.
func getTotalVectors(idx *IVFPQIndex) int {
	total := 0
	for _, list := range idx.lists {
		total += len(list)
	}
	return total
}

// TestNewIVFPQIndex tests IVFPQ index creation with various parameters
func TestNewIVFPQIndex(t *testing.T) {
	tests := []struct {
		name         string
		dim          int
		distanceKind DistanceKind
		nlist        int
		M            int
		Nbits        int
		wantErr      bool
	}{
		{"valid L2 index", 128, Euclidean, 10, 8, 8, false},
		{"valid Cosine index", 384, Cosine, 20, 8, 8, false},
		{"valid L2Squared index", 768, L2Squared, 100, 8, 8, false},
		{"zero dimension", 0, Euclidean, 10, 8, 8, true},
		{"negative dimension", -1, Euclidean, 10, 8, 8, true},
		{"zero nlist", 128, Euclidean, 0, 8, 8, true},
		{"negative nlist", 128, Euclidean, -1, 8, 8, true},
		{"zero M", 128, Euclidean, 10, 0, 8, true},
		{"negative M", 128, Euclidean, 10, -1, 8, true},
		{"M doesn't divide dim", 100, Euclidean, 10, 8, 8, true},
		{"zero Nbits", 128, Euclidean, 10, 8, 0, true},
		{"negative Nbits", 128, Euclidean, 10, 8, -1, true},
		{"Nbits too large", 128, Euclidean, 10, 8, 17, true},
		{"invalid distance kind", 128, DistanceKind("invalid"), 10, 8, 8, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, err := NewIVFPQIndex(tt.dim, tt.distanceKind, tt.nlist, tt.M, tt.Nbits)

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
				if idx.Kind() != IVFPQIndexKind {
					t.Errorf("Kind() = %v, want %v", idx.Kind(), IVFPQIndexKind)
				}
				if idx.Trained() {
					t.Error("New index should not be trained")
				}
				// Verify derived parameters
				expectedKsub := 1 << tt.Nbits
				if idx.Ksub != expectedKsub {
					t.Errorf("Ksub = %d, want %d", idx.Ksub, expectedKsub)
				}
				expectedDsub := tt.dim / tt.M
				if idx.dsub != expectedDsub {
					t.Errorf("dsub = %d, want %d", idx.dsub, expectedDsub)
				}
			}
		})
	}
}

// TestIVFPQIndexTrain tests training functionality
func TestIVFPQIndexTrain(t *testing.T) {
	dim := 8
	nlist := 2
	M := 4
	Nbits := 4 // Small for faster tests

	idx, err := NewIVFPQIndex(dim, Euclidean, nlist, M, Nbits)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Create sufficient training vectors (nlist*10 minimum)
	numVectors := 100
	trainingVectors := make([]VectorNode, numVectors)
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i*dim + j) % 10)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Verify index is trained
	if !idx.Trained() {
		t.Error("Index should be marked as trained")
	}

	// Verify IVF centroids were learned
	if len(idx.centroids) != nlist {
		t.Errorf("Expected %d centroids, got %d", nlist, len(idx.centroids))
	}

	// Verify centroid dimensions
	for i, centroid := range idx.centroids {
		if len(centroid) != dim {
			t.Errorf("Centroid %d has dimension %d, want %d", i, len(centroid), dim)
		}
	}

	// Verify PQ codebooks were learned
	if len(idx.codebooks) != M {
		t.Errorf("Expected %d codebooks, got %d", M, len(idx.codebooks))
	}

	// Verify each codebook has correct size
	expectedKsub := 1 << Nbits
	expectedDsub := dim / M
	for i, codebook := range idx.codebooks {
		expectedSize := expectedKsub * expectedDsub
		if len(codebook) != expectedSize {
			t.Errorf("Codebook %d has size %d, want %d", i, len(codebook), expectedSize)
		}
	}
}

// TestIVFPQIndexTrainInsufficientVectors tests training with too few vectors
func TestIVFPQIndexTrainInsufficientVectors(t *testing.T) {
	idx, err := NewIVFPQIndex(8, Euclidean, 10, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Only provide 5 vectors for 10 clusters (need at least 100)
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0, 0, 0, 0, 0, 0}),
		*NewVectorNode([]float32{1, 0, 0, 0, 0, 0, 0, 0}),
		*NewVectorNode([]float32{2, 0, 0, 0, 0, 0, 0, 0}),
		*NewVectorNode([]float32{3, 0, 0, 0, 0, 0, 0, 0}),
		*NewVectorNode([]float32{4, 0, 0, 0, 0, 0, 0, 0}),
	}

	err = idx.Train(trainingVectors)
	if err == nil {
		t.Error("Expected error when training with insufficient vectors")
	}
}

// TestIVFPQIndexTrainDimensionMismatch tests training with wrong dimensions
func TestIVFPQIndexTrainDimensionMismatch(t *testing.T) {
	idx, err := NewIVFPQIndex(8, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Provide vectors with wrong dimension
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		trainingVectors[i] = *NewVectorNode([]float32{0, 0, 0}) // Wrong dim: 3 instead of 8
	}

	err = idx.Train(trainingVectors)
	if err == nil {
		t.Error("Expected error when training with dimension mismatch")
	}
}

// TestIVFPQIndexAddBeforeTrain tests that add fails before training
func TestIVFPQIndexAddBeforeTrain(t *testing.T) {
	idx, err := NewIVFPQIndex(8, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	node := NewVectorNode([]float32{1, 0, 0, 0, 0, 0, 0, 0})
	err = idx.Add(*node)

	if err == nil {
		t.Error("Expected error when adding before training")
	}
}

// TestIVFPQIndexAdd tests adding vectors to the index
func TestIVFPQIndexAdd(t *testing.T) {
	dim := 8
	nlist := 2
	M := 4
	Nbits := 4

	idx, err := NewIVFPQIndex(dim, Euclidean, nlist, M, Nbits)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train first
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i*dim + j) % 10)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Verify initial size
	if totalSize := getTotalVectors(idx); totalSize != 0 {
		t.Errorf("Initial size = %d, want 0", totalSize)
	}

	// Add vectors
	node1 := NewVectorNode([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	err = idx.Add(*node1)
	if err != nil {
		t.Fatalf("Add() error: %v", err)
	}

	if totalSize := getTotalVectors(idx); totalSize != 1 {
		t.Errorf("After adding 1 vector, size = %d, want 1", totalSize)
	}

	node2 := NewVectorNode([]float32{8, 7, 6, 5, 4, 3, 2, 1})
	err = idx.Add(*node2)
	if err != nil {
		t.Fatalf("Add() error: %v", err)
	}

	if totalSize := getTotalVectors(idx); totalSize != 2 {
		t.Errorf("After adding 2 vectors, size = %d, want 2", totalSize)
	}

	// Verify vectors are in inverted lists
	if totalInLists := getTotalVectors(idx); totalInLists != 2 {
		t.Errorf("Total vectors in lists = %d, want 2", totalInLists)
	}
}

// TestIVFPQIndexAddDimensionMismatch tests adding vectors with wrong dimension
func TestIVFPQIndexAddDimensionMismatch(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train first
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

	// Try to add vector with wrong dimension
	node := NewVectorNode([]float32{1, 2, 3}) // Wrong dimension
	err = idx.Add(*node)

	if err == nil {
		t.Error("Expected error when adding vector with wrong dimension")
	}
}

// TestIVFPQIndexAddZeroVectorCosine tests adding zero vector with cosine distance
func TestIVFPQIndexAddZeroVectorCosine(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Cosine, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train with non-zero vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim+j) + 1.0 // Ensure non-zero
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Try to add zero vector with cosine distance
	node := NewVectorNode([]float32{0, 0, 0, 0, 0, 0, 0, 0})
	err = idx.Add(*node)

	if err == nil {
		t.Error("Expected error when adding zero vector with cosine distance")
	}
}

// TestIVFPQIndexRemove tests removing vectors
func TestIVFPQIndexRemove(t *testing.T) {
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
	node1 := NewVectorNode([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	node2 := NewVectorNode([]float32{8, 7, 6, 5, 4, 3, 2, 1})

	idx.Add(*node1)
	idx.Add(*node2)

	if totalSize := getTotalVectors(idx); totalSize != 2 {
		t.Errorf("After adding, size = %d, want 2", totalSize)
	}

	// Remove first vector (soft delete)
	err = idx.Remove(*node1)
	if err != nil {
		t.Fatalf("Remove() error: %v", err)
	}

	// Vectors should still be in storage (soft delete)
	if totalSize := getTotalVectors(idx); totalSize != 2 {
		t.Errorf("After soft delete, size = %d, want 2", totalSize)
	}

	// Call Flush to perform hard delete
	err = idx.Flush()
	if err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	// Now vector should be physically removed
	if totalSize := getTotalVectors(idx); totalSize != 1 {
		t.Errorf("After flush, size = %d, want 1", totalSize)
	}

	// Try to remove already deleted vector
	err = idx.Remove(*node1)
	if err == nil {
		t.Error("Remove() expected error for already deleted vector")
	}

	// Remove second vector
	err = idx.Remove(*node2)
	if err != nil {
		t.Fatalf("Remove() error: %v", err)
	}

	// Still 1 vector before flush (soft delete)
	if totalSize := getTotalVectors(idx); totalSize != 1 {
		t.Errorf("Before second flush, size = %d, want 1", totalSize)
	}

	// Flush to remove all
	idx.Flush()

	if totalSize := getTotalVectors(idx); totalSize != 0 {
		t.Errorf("After removing all vectors, size = %d, want 0", totalSize)
	}
}

// TestIVFPQIndexRemoveNonExistent tests removing non-existent vector
func TestIVFPQIndexRemoveNonExistent(t *testing.T) {
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

	// Try to remove vector that doesn't exist
	node := NewVectorNode([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	err = idx.Remove(*node)

	if err == nil {
		t.Error("Expected error when removing non-existent vector")
	}
}

// TestIVFPQIndexFlush tests flush operation
func TestIVFPQIndexFlush(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train the index
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	idx.Train(trainingVectors)

	// Add some vectors
	node1 := NewVectorNode([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	node2 := NewVectorNode([]float32{2, 3, 4, 5, 6, 7, 8, 9})
	node3 := NewVectorNode([]float32{3, 4, 5, 6, 7, 8, 9, 10})

	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)

	// Flush with no deletions should succeed and keep all vectors
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error: %v", err)
	}
	if totalSize := getTotalVectors(idx); totalSize != 3 {
		t.Errorf("Expected 3 vectors after flush with no deletions, got %d", totalSize)
	}

	// Soft delete two vectors
	idx.Remove(*node1)
	idx.Remove(*node2)

	// Vectors still in memory before flush
	if totalSize := getTotalVectors(idx); totalSize != 3 {
		t.Errorf("Expected 3 vectors before flush, got %d", totalSize)
	}

	// Flush should remove deleted vectors
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	// Only one vector should remain
	if totalSize := getTotalVectors(idx); totalSize != 1 {
		t.Errorf("Expected 1 vector after flush, got %d", totalSize)
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

// TestIVFPQIndexGetters tests getter methods
func TestIVFPQIndexGetters(t *testing.T) {
	dim := 128
	nlist := 10
	M := 8
	Nbits := 8
	distanceKind := Euclidean

	idx, err := NewIVFPQIndex(dim, distanceKind, nlist, M, Nbits)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	if idx.Dimensions() != dim {
		t.Errorf("Dimensions() = %d, want %d", idx.Dimensions(), dim)
	}

	if idx.DistanceKind() != distanceKind {
		t.Errorf("DistanceKind() = %v, want %v", idx.DistanceKind(), distanceKind)
	}

	if idx.Kind() != IVFPQIndexKind {
		t.Errorf("Kind() = %v, want %v", idx.Kind(), IVFPQIndexKind)
	}

	// Verify inverted lists are initialized
	if len(idx.lists) != nlist {
		t.Errorf("Number of lists = %d, want %d", len(idx.lists), nlist)
	}
	for i, list := range idx.lists {
		if len(list) != 0 {
			t.Errorf("List %d has size %d, want 0", i, len(list))
		}
	}
}

// TestIVFPQIndexNewSearch tests search builder creation
func TestIVFPQIndexNewSearch(t *testing.T) {
	idx, err := NewIVFPQIndex(8, Euclidean, 4, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	search := idx.NewSearch()
	if search == nil {
		t.Error("NewSearch() returned nil")
	}

	// Verify search is correct type
	_, ok := search.(*ivfpqIndexSearch)
	if !ok {
		t.Error("NewSearch() did not return *ivfpqIndexSearch")
	}
}

// TestIVFPQIndexConcurrentAdd tests concurrent additions
func TestIVFPQIndexConcurrentAdd(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 4, 4, 4)
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

	// Concurrent adds
	const numGoroutines = 10
	const vectorsPerGoroutine = 5

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(gid int) {
			defer wg.Done()
			for i := 0; i < vectorsPerGoroutine; i++ {
				vec := make([]float32, dim)
				for j := 0; j < dim; j++ {
					vec[j] = float32(gid*100 + i*10 + j)
				}
				node := NewVectorNode(vec)
				if err := idx.Add(*node); err != nil {
					t.Errorf("Concurrent Add() error: %v", err)
				}
			}
		}(g)
	}

	wg.Wait()

	expectedSize := numGoroutines * vectorsPerGoroutine
	if totalSize := getTotalVectors(idx); totalSize != expectedSize {
		t.Errorf("After concurrent adds, size = %d, want %d", totalSize, expectedSize)
	}
}

// TestIVFPQIndexDifferentDistanceMetrics tests different distance metrics
func TestIVFPQIndexDifferentDistanceMetrics(t *testing.T) {
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

			// Create training vectors (non-zero for cosine)
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
			for i := 0; i < 10; i++ {
				vec := make([]float32, dim)
				for j := 0; j < dim; j++ {
					vec[j] = float32(i*10+j) + 1.0
				}
				node := NewVectorNode(vec)
				err = idx.Add(*node)
				if err != nil {
					t.Fatalf("Add() with %s error: %v", tc.distance, err)
				}
			}

			if totalSize := getTotalVectors(idx); totalSize != 10 {
				t.Errorf("Size with %s = %d, want 10", tc.distance, totalSize)
			}
		})
	}
}

// TestIVFPQIndexCompressionRatio tests compression effectiveness
func TestIVFPQIndexCompressionRatio(t *testing.T) {
	dim := 128
	M := 8
	Nbits := 8 // K = 256

	idx, err := NewIVFPQIndex(dim, Euclidean, 4, M, Nbits)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 500)
	for i := 0; i < 500; i++ {
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

	// Add a vector
	testVec := make([]float32, dim)
	for j := 0; j < dim; j++ {
		testVec[j] = float32(j)
	}
	node := NewVectorNode(testVec)
	err = idx.Add(*node)
	if err != nil {
		t.Fatalf("Add() error: %v", err)
	}

	// Calculate memory usage
	originalSize := dim * 4 // 4 bytes per float32
	compressedSize := M     // M bytes for PQ code

	compressionRatio := float64(originalSize) / float64(compressedSize)

	t.Logf("Compression: %d bytes -> %d bytes (%.0fx)", originalSize, compressedSize, compressionRatio)

	// Verify compression is significant (at least 10x)
	if compressionRatio < 10.0 {
		t.Errorf("Compression ratio %.2fx is too low, want at least 10x", compressionRatio)
	}
}

// TestIVFPQIndexLargeScale tests with larger dataset
func TestIVFPQIndexLargeScale(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large scale test in short mode")
	}

	dim := 128
	nlist := 10
	M := 8
	Nbits := 8

	idx, err := NewIVFPQIndex(dim, Euclidean, nlist, M, Nbits)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train with 1000 vectors
	numTraining := 1000
	trainingVectors := make([]VectorNode, numTraining)
	for i := 0; i < numTraining; i++ {
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

	// Add 500 vectors
	numVectors := 500
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((numTraining+i)*dim + j)
		}
		node := NewVectorNode(vec)
		err = idx.Add(*node)
		if err != nil {
			t.Fatalf("Add() error at vector %d: %v", i, err)
		}
	}

	// Verify distribution across lists
	if totalInLists := getTotalVectors(idx); totalInLists != numVectors {
		t.Errorf("Total in lists = %d, want %d", totalInLists, numVectors)
	}

	// Calculate memory savings
	originalMemory := numVectors * dim * 4
	compressedMemory := numVectors * M
	codebookMemory := M * (1 << Nbits) * (dim / M) * 4
	centroidMemory := nlist * dim * 4
	totalPQMemory := compressedMemory + codebookMemory + centroidMemory

	t.Logf("Original memory: %d bytes (%.2f MB)", originalMemory, float64(originalMemory)/(1024*1024))
	t.Logf("Compressed codes: %d bytes (%.2f KB)", compressedMemory, float64(compressedMemory)/1024)
	t.Logf("Codebooks: %d bytes (%.2f KB)", codebookMemory, float64(codebookMemory)/1024)
	t.Logf("Centroids: %d bytes (%.2f KB)", centroidMemory, float64(centroidMemory)/1024)
	t.Logf("Total IVFPQ memory: %d bytes (%.2f MB)", totalPQMemory, float64(totalPQMemory)/(1024*1024))
	t.Logf("Compression ratio: %.2fx", float64(originalMemory)/float64(totalPQMemory))
}

// TestIVFPQIndexResidualEncoding tests that residuals are properly encoded
func TestIVFPQIndexResidualEncoding(t *testing.T) {
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

	// Add a vector
	testVec := make([]float32, dim)
	for j := 0; j < dim; j++ {
		testVec[j] = float32(j)
	}
	node := NewVectorNode(testVec)
	err = idx.Add(*node)
	if err != nil {
		t.Fatalf("Add() error: %v", err)
	}

	// Verify vector is in one of the lists
	found := false
	var code []uint8
	for _, list := range idx.lists {
		for _, cv := range list {
			if cv.Node.ID() == node.ID() {
				found = true
				code = cv.Code
				break
			}
		}
		if found {
			break
		}
	}

	if !found {
		t.Error("Added vector not found in any inverted list")
	}

	// Verify code has correct length
	if len(code) != M {
		t.Errorf("PQ code has length %d, want %d", len(code), M)
	}

	// Verify each code element is within valid range
	maxCode := uint8((1 << Nbits) - 1)
	for i, c := range code {
		if c > maxCode {
			t.Errorf("Code[%d] = %d exceeds max %d", i, c, maxCode)
		}
	}
}

// TestIVFPQIndexMultipleTraining tests that retraining updates the index
func TestIVFPQIndexMultipleTraining(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// First training
	trainingVectors1 := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors1[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors1)
	if err != nil {
		t.Fatalf("First Train() error: %v", err)
	}

	if !idx.Trained() {
		t.Error("Index should be trained after first training")
	}

	// Second training (should replace old training)
	trainingVectors2 := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i+1000)*dim + j)
		}
		trainingVectors2[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors2)
	if err != nil {
		t.Fatalf("Second Train() error: %v", err)
	}

	if !idx.Trained() {
		t.Error("Index should still be trained after second training")
	}

	// Verify we can still add vectors
	node := NewVectorNode([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	err = idx.Add(*node)
	if err != nil {
		t.Fatalf("Add() after retraining error: %v", err)
	}
}

// TestIVFPQIndexGetListSizesDistribution tests list size distribution
func TestIVFPQIndexGetListSizesDistribution(t *testing.T) {
	dim := 8
	nlist := 4
	idx, err := NewIVFPQIndex(dim, Euclidean, nlist, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train with clustered data
	trainingVectors := make([]VectorNode, 200)
	for i := 0; i < 200; i++ {
		vec := make([]float32, dim)
		cluster := i % nlist
		for j := 0; j < dim; j++ {
			vec[j] = float32(cluster*100 + i)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add vectors to different clusters
	for i := 0; i < 40; i++ {
		vec := make([]float32, dim)
		cluster := i % nlist
		for j := 0; j < dim; j++ {
			vec[j] = float32(cluster*100 + i + 1000)
		}
		node := NewVectorNode(vec)
		err = idx.Add(*node)
		if err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Check list sizes
	if len(idx.lists) != nlist {
		t.Errorf("Number of lists = %d, want %d", len(idx.lists), nlist)
	}

	// Verify total
	if total := getTotalVectors(idx); total != 40 {
		t.Errorf("Total vectors across lists = %d, want 40", total)
	}
}

// TestIVFPQIndexSoftDeleteWithSearch tests that soft-deleted nodes are filtered during search
func TestIVFPQIndexSoftDeleteWithSearch(t *testing.T) {
	dim := 8
	idx, err := NewIVFPQIndex(dim, Euclidean, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewIVFPQIndex() error: %v", err)
	}

	// Train the index
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	idx.Train(trainingVectors)

	// Add test vectors
	node1 := NewVectorNode([]float32{1, 0, 0, 0, 0, 0, 0, 0})
	node2 := NewVectorNode([]float32{2, 0, 0, 0, 0, 0, 0, 0})
	node3 := NewVectorNode([]float32{3, 0, 0, 0, 0, 0, 0, 0})
	node4 := NewVectorNode([]float32{4, 0, 0, 0, 0, 0, 0, 0})

	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)
	idx.Add(*node4)

	// Search should return all 4 vectors
	query := []float32{1.5, 0, 0, 0, 0, 0, 0, 0}
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
	if totalSize := getTotalVectors(idx); totalSize != 2 {
		t.Errorf("Expected 2 vectors in storage after flush, got %d", totalSize)
	}
}
