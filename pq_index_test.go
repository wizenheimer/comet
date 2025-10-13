package comet

import (
	"bytes"
	"io"
	"sync"
	"testing"
)

// TestCalculatePQParams tests PQ parameter calculation
func TestCalculatePQParams(t *testing.T) {
	tests := []struct {
		name      string
		dim       int
		wantM     int
		wantNbits int
	}{
		{"dimension 768", 768, 8, 8},
		{"dimension 384", 384, 8, 8},
		{"dimension 128", 128, 8, 8},
		{"dimension 64", 64, 8, 8},
		{"dimension 32", 32, 8, 8},
		{"dimension 16", 16, 8, 8},
		{"dimension 100", 100, 10, 8}, // 100 divisible by 10
		{"dimension 17", 17, 17, 8},   // 17 only divisible by itself and 1
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			M, Nbits := CalculatePQParams(tt.dim)
			if M != tt.wantM {
				t.Errorf("CalculatePQParams(%d) M = %d, want %d", tt.dim, M, tt.wantM)
			}
			if Nbits != tt.wantNbits {
				t.Errorf("CalculatePQParams(%d) Nbits = %d, want %d", tt.dim, Nbits, tt.wantNbits)
			}
			// Verify M divides dim
			if tt.dim%M != 0 {
				t.Errorf("CalculatePQParams(%d) returned M=%d which doesn't divide dimension", tt.dim, M)
			}
		})
	}
}

// TestNewPQIndex tests PQ index creation with various parameters
func TestNewPQIndex(t *testing.T) {
	tests := []struct {
		name         string
		dim          int
		distanceKind DistanceKind
		M            int
		Nbits        int
		wantErr      bool
	}{
		{"valid L2 index", 128, Euclidean, 8, 8, false},
		{"valid Cosine index", 384, Cosine, 8, 8, false},
		{"valid L2Squared index", 768, L2Squared, 8, 8, false},
		{"zero dimension", 0, Euclidean, 8, 8, true},
		{"negative dimension", -1, Euclidean, 8, 8, true},
		{"zero M", 128, Euclidean, 0, 8, true},
		{"negative M", 128, Euclidean, -1, 8, true},
		{"M doesn't divide dim", 100, Euclidean, 8, 8, true},
		{"zero Nbits", 128, Euclidean, 8, 0, true},
		{"negative Nbits", 128, Euclidean, 8, -1, true},
		{"Nbits too large", 128, Euclidean, 8, 17, true},
		{"invalid distance kind", 128, DistanceKind("invalid"), 8, 8, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, err := NewPQIndex(tt.dim, tt.distanceKind, tt.M, tt.Nbits)

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
				if idx.Kind() != PQIndexKind {
					t.Errorf("Kind() = %v, want %v", idx.Kind(), PQIndexKind)
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

// TestPQIndexTrain tests training functionality
func TestPQIndexTrain(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 8

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Create sufficient training vectors
	numVectors := 300 // More than Ksub=256
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

	// Verify codebooks were learned
	if len(idx.codebooks) != M {
		t.Errorf("Expected %d codebooks, got %d", M, len(idx.codebooks))
	}

	// Verify each codebook has correct size
	expectedKsub := 1 << Nbits // 256
	expectedDsub := dim / M
	for i, codebook := range idx.codebooks {
		expectedSize := expectedKsub * expectedDsub
		if len(codebook) != expectedSize {
			t.Errorf("Codebook %d has size %d, want %d", i, len(codebook), expectedSize)
		}
	}
}

// TestPQIndexTrainInsufficientVectors tests training with too few vectors
func TestPQIndexTrainInsufficientVectors(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 8 // Ksub = 256

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Only provide 100 vectors when 256 are required
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err == nil {
		t.Error("Expected error when training with insufficient vectors")
	}
}

// TestPQIndexTrainDimensionMismatch tests training with wrong dimension vectors
func TestPQIndexTrainDimensionMismatch(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 8

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Create vectors with wrong dimension
	trainingVectors := make([]VectorNode, 300)
	for i := 0; i < 300; i++ {
		// Wrong dimension: 10 instead of 8
		vec := make([]float32, 10)
		for j := 0; j < 10; j++ {
			vec[j] = float32(i)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}

	err = idx.Train(trainingVectors)
	if err == nil {
		t.Error("Expected error when training with wrong dimension vectors")
	}
}

// TestPQIndexAddBeforeTrain tests that add fails before training
func TestPQIndexAddBeforeTrain(t *testing.T) {
	idx, err := NewPQIndex(8, Euclidean, 4, 8)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	node := NewVectorNode([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	err = idx.Add(*node)

	if err == nil {
		t.Error("Expected error when adding before training")
	}
}

// TestPQIndexAdd tests adding vectors to the index
func TestPQIndexAdd(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 6 // Ksub = 64, more manageable for testing

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
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

	// Add vectors
	vectors := [][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{2, 3, 4, 5, 6, 7, 8, 9},
		{3, 4, 5, 6, 7, 8, 9, 10},
		{4, 5, 6, 7, 8, 9, 10, 11},
	}

	for _, v := range vectors {
		node := NewVectorNode(v)
		err := idx.Add(*node)
		if err != nil {
			t.Errorf("Add() error: %v", err)
		}
	}

	// Verify vectors were added
	if len(idx.codes) != len(vectors) {
		t.Errorf("Expected %d codes, got %d", len(vectors), len(idx.codes))
	}

	if len(idx.vectorNodes) != len(vectors) {
		t.Errorf("Expected %d vector nodes, got %d", len(vectors), len(idx.vectorNodes))
	}

	// Verify each code has M bytes
	for i, code := range idx.codes {
		if len(code) != M {
			t.Errorf("Code %d has length %d, want %d", i, len(code), M)
		}
	}
}

// TestPQIndexAddDimensionMismatch tests adding wrong dimension vectors
func TestPQIndexAddDimensionMismatch(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 6

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Train first
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	idx.Train(trainingVectors)

	// Try to add 10D vector to 8D index
	node := NewVectorNode([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	err = idx.Add(*node)

	if err == nil {
		t.Error("Expected error when adding wrong dimension vector")
	}
}

// TestPQIndexAddZeroVectorCosine tests adding zero vector with cosine distance
func TestPQIndexAddZeroVectorCosine(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 6

	idx, err := NewPQIndex(dim, Cosine, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Train first with non-zero vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i + 1) // Non-zero
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	idx.Train(trainingVectors)

	// Try to add zero vector (cannot be normalized)
	node := NewVectorNode([]float32{0, 0, 0, 0, 0, 0, 0, 0})
	err = idx.Add(*node)

	if err == nil {
		t.Error("Expected error when adding zero vector with cosine distance")
	}
}

// TestPQIndexRemove tests removing vectors from the index
func TestPQIndexRemove(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 6

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Train and add vectors
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	idx.Train(trainingVectors)

	node1 := NewVectorNode([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	node2 := NewVectorNode([]float32{2, 3, 4, 5, 6, 7, 8, 9})
	node3 := NewVectorNode([]float32{3, 4, 5, 6, 7, 8, 9, 10})
	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)

	// Remove middle node (soft delete)
	err = idx.Remove(*node2)
	if err != nil {
		t.Errorf("Remove() error: %v", err)
	}

	// Vectors should still be in slices (soft delete)
	if len(idx.codes) != 3 {
		t.Errorf("Expected 3 codes after soft delete, got %d", len(idx.codes))
	}

	if len(idx.vectorNodes) != 3 {
		t.Errorf("Expected 3 vector nodes after soft delete, got %d", len(idx.vectorNodes))
	}

	// Call Flush to perform hard delete
	err = idx.Flush()
	if err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	// Now vectors should be physically removed
	if len(idx.codes) != 2 {
		t.Errorf("Expected 2 codes after flush, got %d", len(idx.codes))
	}

	if len(idx.vectorNodes) != 2 {
		t.Errorf("Expected 2 vector nodes after flush, got %d", len(idx.vectorNodes))
	}

	// Verify correct nodes remain
	remainingIDs := []uint32{node1.ID(), node3.ID()}
	for i, node := range idx.vectorNodes {
		if node.ID() != remainingIDs[i] {
			t.Errorf("Node %d has ID %d, want %d", i, node.ID(), remainingIDs[i])
		}
	}

	// Try to remove already deleted vector
	err = idx.Remove(*node2)
	if err == nil {
		t.Error("Remove() expected error for already deleted vector")
	}
}

// TestPQIndexRemoveNonExistent tests removing non-existent vector
func TestPQIndexRemoveNonExistent(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 6

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Train first
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	idx.Train(trainingVectors)

	// Try to remove a node that was never added
	node := NewVectorNode([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	err = idx.Remove(*node)

	if err == nil {
		t.Error("Expected error when removing non-existent vector")
	}
}

// TestPQIndexFlush tests the flush method
func TestPQIndexFlush(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 6

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Train the index
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i)
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
	if len(idx.codes) != 3 {
		t.Errorf("Expected 3 codes after flush with no deletions, got %d", len(idx.codes))
	}

	// Soft delete two vectors
	idx.Remove(*node1)
	idx.Remove(*node2)

	// Vectors still in memory before flush
	if len(idx.codes) != 3 {
		t.Errorf("Expected 3 codes before flush, got %d", len(idx.codes))
	}

	// Flush should remove deleted vectors
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	// Only one vector should remain
	if len(idx.codes) != 1 {
		t.Errorf("Expected 1 code after flush, got %d", len(idx.codes))
	}

	if len(idx.vectorNodes) != 1 {
		t.Errorf("Expected 1 vector node after flush, got %d", len(idx.vectorNodes))
	}

	// Verify the remaining vector is node3
	if idx.vectorNodes[0].ID() != node3.ID() {
		t.Errorf("Expected remaining vector to be node3, got node with ID %d", idx.vectorNodes[0].ID())
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

// TestPQIndexGetters tests getter methods
func TestPQIndexGetters(t *testing.T) {
	dim := 384
	M := 8
	Nbits := 8
	distanceKind := Cosine

	idx, err := NewPQIndex(dim, distanceKind, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	if idx.Dimensions() != dim {
		t.Errorf("Dimensions() = %d, want %d", idx.Dimensions(), dim)
	}

	if idx.DistanceKind() != distanceKind {
		t.Errorf("DistanceKind() = %v, want %v", idx.DistanceKind(), distanceKind)
	}

	if idx.Kind() != PQIndexKind {
		t.Errorf("Kind() = %v, want %v", idx.Kind(), PQIndexKind)
	}

	if idx.Trained() {
		t.Error("New index should not be trained")
	}

	// Train and check again
	trainingVectors := make([]VectorNode, 300)
	for i := 0; i < 300; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i + 1) % 10)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	idx.Train(trainingVectors)

	if !idx.Trained() {
		t.Error("Trained index should return true for Trained()")
	}
}

// TestPQIndexNewSearch tests search builder creation
func TestPQIndexNewSearch(t *testing.T) {
	idx, err := NewPQIndex(8, Euclidean, 4, 6)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	search := idx.NewSearch()
	if search == nil {
		t.Error("NewSearch() returned nil")
	}

	// Verify it's the correct type
	pqSearch, ok := search.(*pqIndexSearch)
	if !ok {
		t.Error("NewSearch() did not return *pqIndexSearch")
	}

	// Verify defaults
	if pqSearch.k != 10 {
		t.Errorf("Default k = %d, want 10", pqSearch.k)
	}
}

// TestPQIndexConcurrentAdd tests concurrent additions to the index
func TestPQIndexConcurrentAdd(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 6

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
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
				vec := make([]float32, dim)
				for k := 0; k < dim; k++ {
					vec[k] = float32(offset*100 + j*10 + k)
				}
				node := NewVectorNode(vec)
				err := idx.Add(*node)
				if err != nil {
					t.Errorf("Add() error: %v", err)
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify all vectors were added
	expected := numGoroutines * vectorsPerGoroutine
	if len(idx.codes) != expected {
		t.Errorf("Expected %d codes after concurrent adds, got %d", expected, len(idx.codes))
	}
	if len(idx.vectorNodes) != expected {
		t.Errorf("Expected %d vector nodes after concurrent adds, got %d", expected, len(idx.vectorNodes))
	}
}

// TestPQIndexEncode tests the encode function
func TestPQIndexEncode(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 6

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
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
	idx.Train(trainingVectors)

	// Test encode
	testVec := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	code := idx.encode(testVec)

	// Verify code length
	if len(code) != M {
		t.Errorf("Code length = %d, want %d", len(code), M)
	}

	// Verify each code element is within valid range
	Ksub := 1 << Nbits
	for i, c := range code {
		if int(c) >= Ksub {
			t.Errorf("Code[%d] = %d, which is >= Ksub=%d", i, c, Ksub)
		}
	}

	// Encode same vector twice, should get same code
	code2 := idx.encode(testVec)
	if len(code) != len(code2) {
		t.Error("Encoding same vector twice gave different length codes")
	}
	for i := range code {
		if code[i] != code2[i] {
			t.Errorf("Encoding same vector twice gave different codes at position %d: %d vs %d", i, code[i], code2[i])
		}
	}
}

// TestPQIndexDifferentDistanceMetrics tests PQ with different distance metrics
func TestPQIndexDifferentDistanceMetrics(t *testing.T) {
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
			dim := 8
			M := 4
			Nbits := 6

			idx, err := NewPQIndex(dim, tt.distanceKind, M, Nbits)
			if err != nil {
				t.Fatalf("NewPQIndex() error: %v", err)
			}

			// Train
			trainingVectors := make([]VectorNode, 100)
			for i := 0; i < 100; i++ {
				vec := make([]float32, dim)
				for j := 0; j < dim; j++ {
					vec[j] = float32(i + 1) // Non-zero for cosine
				}
				trainingVectors[i] = *NewVectorNode(vec)
			}
			err = idx.Train(trainingVectors)
			if err != nil {
				t.Fatalf("Train() error: %v", err)
			}

			// Add and verify
			node := NewVectorNode([]float32{1, 2, 3, 4, 5, 6, 7, 8})
			err = idx.Add(*node)
			if err != nil {
				t.Fatalf("Add() error: %v", err)
			}

			if len(idx.codes) != 1 {
				t.Errorf("Expected 1 code, got %d", len(idx.codes))
			}
		})
	}
}

// TestPQIndexCompressionRatio tests that PQ achieves significant compression
func TestPQIndexCompressionRatio(t *testing.T) {
	dim := 768
	M := 8
	Nbits := 8

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 300)
	for i := 0; i < 300; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i*dim + j) % 100)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	idx.Train(trainingVectors)

	// Add vector
	node := NewVectorNode(make([]float32, dim))
	idx.Add(*node)

	// Check compression
	// Original: dim * 4 bytes = 768 * 4 = 3072 bytes
	// Compressed: M bytes = 8 bytes
	originalSize := dim * 4
	compressedSize := M
	compressionRatio := float64(originalSize) / float64(compressedSize)

	if compressionRatio < 100 {
		t.Errorf("Expected compression ratio > 100x, got %.2fx", compressionRatio)
	}

	t.Logf("Compression: %d bytes -> %d bytes (%.0fx)", originalSize, compressedSize, compressionRatio)
}

// TestPQIndexLargeScale tests PQ with more realistic scale
func TestPQIndexLargeScale(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large scale test in short mode")
	}

	dim := 128
	M := 8
	Nbits := 8
	numVectors := 1000

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 300)
	for i := 0; i < 300; i++ {
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
	if len(idx.codes) != numVectors {
		t.Errorf("Expected %d codes, got %d", numVectors, len(idx.codes))
	}

	// Calculate memory usage
	originalMemory := numVectors * dim * 4 // float32 = 4 bytes
	compressedMemory := numVectors * M
	codebookMemory := M * (1 << Nbits) * (dim / M) * 4
	totalMemory := compressedMemory + codebookMemory

	t.Logf("Original memory: %d bytes (%.2f MB)", originalMemory, float64(originalMemory)/(1024*1024))
	t.Logf("Compressed codes: %d bytes (%.2f KB)", compressedMemory, float64(compressedMemory)/1024)
	t.Logf("Codebooks: %d bytes (%.2f KB)", codebookMemory, float64(codebookMemory)/1024)
	t.Logf("Total PQ memory: %d bytes (%.2f MB)", totalMemory, float64(totalMemory)/(1024*1024))
	t.Logf("Compression ratio: %.2fx", float64(originalMemory)/float64(totalMemory))
}

// TestPQIndexCodebookStructure tests that codebooks have correct structure
func TestPQIndexCodebookStructure(t *testing.T) {
	dim := 16
	M := 4
	Nbits := 6
	Ksub := 1 << Nbits // 64
	dsub := dim / M    // 4

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Train
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i*dim + j) % 20)
		}
		trainingVectors[i] = *NewVectorNode(vec)
	}
	idx.Train(trainingVectors)

	// Verify codebook structure
	if len(idx.codebooks) != M {
		t.Fatalf("Expected %d codebooks, got %d", M, len(idx.codebooks))
	}

	for m := 0; m < M; m++ {
		expectedSize := Ksub * dsub
		if len(idx.codebooks[m]) != expectedSize {
			t.Errorf("Codebook %d has size %d, want %d", m, len(idx.codebooks[m]), expectedSize)
		}

		// Verify we can access each centroid
		for k := 0; k < Ksub; k++ {
			start := k * dsub
			end := start + dsub
			centroid := idx.codebooks[m][start:end]
			if len(centroid) != dsub {
				t.Errorf("Centroid %d in codebook %d has length %d, want %d", k, m, len(centroid), dsub)
			}
		}
	}
}

// TestPQIndexMultipleTraining tests that training can be called multiple times
func TestPQIndexMultipleTraining(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 6

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// First training
	trainingVectors1 := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i)
		}
		trainingVectors1[i] = *NewVectorNode(vec)
	}
	err = idx.Train(trainingVectors1)
	if err != nil {
		t.Fatalf("First Train() error: %v", err)
	}

	// Second training (retraining)
	trainingVectors2 := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i + 100)
		}
		trainingVectors2[i] = *NewVectorNode(vec)
	}
	err = idx.Train(trainingVectors2)
	if err != nil {
		t.Fatalf("Second Train() error: %v", err)
	}

	// Should still be trained
	if !idx.Trained() {
		t.Error("Index should be trained after retraining")
	}
}

// TestPQIndexSoftDeleteWithSearch tests that soft-deleted nodes are filtered during search
func TestPQIndexSoftDeleteWithSearch(t *testing.T) {
	dim := 8
	M := 4
	Nbits := 6

	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Train the index
	trainingVectors := make([]VectorNode, 100)
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i)
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
	if len(idx.codes) != 2 {
		t.Errorf("Expected 2 codes in storage after flush, got %d", len(idx.codes))
	}
	if len(idx.vectorNodes) != 2 {
		t.Errorf("Expected 2 vector nodes in storage after flush, got %d", len(idx.vectorNodes))
	}
}

// TestPQIndexWriteTo tests serialization of the PQ index
func TestPQIndexWriteTo(t *testing.T) {
	// Create and train index
	dim := 32
	M, Nbits := 8, 8
	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 256)
	for i := 0; i < 256; i++ {
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
			vec[j] = float32((i+256)*dim + j)
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
	if string(magic) != "PQIX" {
		t.Errorf("Invalid magic number: got %s, want PQIX", string(magic))
	}
}

// TestPQIndexReadFrom tests deserialization of the PQ index
func TestPQIndexReadFrom(t *testing.T) {
	// Create and train original index
	dim := 32
	M, Nbits := 8, 8
	original, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 256)
	for i := 0; i < 256; i++ {
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
			vec[j] = float32((i+256)*dim + j)
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
	restored, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
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

	if restored.M != original.M {
		t.Errorf("M mismatch: got %d, want %d", restored.M, original.M)
	}

	if restored.Trained() != original.Trained() {
		t.Errorf("Trained mismatch: got %v, want %v", restored.Trained(), original.Trained())
	}

	if len(restored.vectorNodes) != len(original.vectorNodes) {
		t.Errorf("Vector count mismatch: got %d, want %d", len(restored.vectorNodes), len(original.vectorNodes))
	}

	if len(restored.codes) != len(original.codes) {
		t.Errorf("Codes count mismatch: got %d, want %d", len(restored.codes), len(original.codes))
	}

	// Verify codebook dimensions
	if len(restored.codebooks) != len(original.codebooks) {
		t.Errorf("Codebooks count mismatch: got %d, want %d", len(restored.codebooks), len(original.codebooks))
	}
}

// TestPQIndexSerializationRoundTrip tests that serialization and deserialization preserve data
func TestPQIndexSerializationRoundTrip(t *testing.T) {
	// Create and train index
	dim := 32
	M, Nbits := 8, 8
	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 256)
	for i := 0; i < 256; i++ {
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
	testVectors := make([]VectorNode, 10)
	for i := 0; i < 10; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32((i+256)*dim + j)
		}
		node := NewVectorNode(vec)
		testVectors[i] = *node
		if err := idx.Add(*node); err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Perform a search before serialization
	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = float32(260*dim + j)
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
	idx2, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
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

	// Results should have same count (PQ is lossy, so exact ordering may differ slightly)
	if len(resultsBefore) != len(resultsAfter) {
		t.Errorf("Result count mismatch: before=%d, after=%d", len(resultsBefore), len(resultsAfter))
	}

	// Verify all result IDs are from our test vectors
	// (PQ is approximate, so we don't require exact ordering match)
	for i, result := range resultsAfter {
		found := false
		for _, tv := range testVectors {
			if result.Node.ID() == tv.ID() {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Result %d has unexpected ID %d", i, result.Node.ID())
		}
	}
}

// TestPQIndexSerializationWithDeletions tests serialization with soft-deleted vectors
func TestPQIndexSerializationWithDeletions(t *testing.T) {
	// Create and train index
	dim := 32
	M, Nbits := 8, 8
	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 256)
	for i := 0; i < 256; i++ {
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
			vec[j] = float32((i+256)*dim + j)
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
	if len(idx.vectorNodes) != 4 {
		t.Errorf("Expected 4 vectors before serialization (soft delete), got %d", len(idx.vectorNodes))
	}

	// Serialize (should call Flush automatically)
	var buf bytes.Buffer
	_, err = idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// After WriteTo (which calls Flush), deleted vectors should be removed
	if len(idx.vectorNodes) != 2 {
		t.Errorf("Expected 2 vectors after WriteTo (auto-flush), got %d", len(idx.vectorNodes))
	}

	// Deserialize
	idx2, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Restored index should only have non-deleted vectors
	if len(idx2.vectorNodes) != 2 {
		t.Errorf("Expected 2 vectors in restored index, got %d", len(idx2.vectorNodes))
	}

	// Verify the correct vectors remain
	foundNode0 := false
	foundNode2 := false
	for _, v := range idx2.vectorNodes {
		if v.ID() == nodes[0].ID() {
			foundNode0 = true
		}
		if v.ID() == nodes[2].ID() {
			foundNode2 = true
		}
	}

	if !foundNode0 {
		t.Error("Expected node 0 in restored index")
	}
	if !foundNode2 {
		t.Error("Expected node 2 in restored index")
	}
}

// TestPQIndexReadFromInvalidData tests error handling for invalid serialized data
func TestPQIndexReadFromInvalidData(t *testing.T) {
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
				buf.Write([]byte("PQIX"))
				// Write invalid version
				buf.Write([]byte{99, 0, 0, 0}) // version 99
				return &buf
			},
			wantErr: "unsupported version",
		},
		{
			name: "truncated data",
			setup: func() *bytes.Buffer {
				buf := bytes.NewBuffer([]byte("PQ"))
				return buf
			},
			wantErr: "failed to read magic number",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := tt.setup()

			idx, err := NewPQIndex(32, Euclidean, 8, 8)
			if err != nil {
				t.Fatalf("NewPQIndex() error: %v", err)
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

// TestPQIndexSerializationUntrained tests serialization of untrained index
func TestPQIndexSerializationUntrained(t *testing.T) {
	// Create untrained index
	idx, err := NewPQIndex(32, Euclidean, 8, 8)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
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
	idx2, err := NewPQIndex(32, Euclidean, 8, 8)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Verify restored index is also untrained
	if idx2.Trained() {
		t.Error("Expected restored index to be untrained")
	}

	// Verify codebooks are nil/empty
	if len(idx2.codebooks) > 0 {
		t.Error("Expected no codebooks in untrained restored index")
	}
}

// TestPQIndexWriteToFlushBehavior tests that WriteTo calls Flush
func TestPQIndexWriteToFlushBehavior(t *testing.T) {
	// Create and train index
	dim := 32
	M, Nbits := 8, 8
	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 256)
	for i := 0; i < 256; i++ {
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
			vec[j] = float32((i+256)*dim + j)
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
	if len(idx.vectorNodes) != 3 {
		t.Errorf("Expected 3 vectors before WriteTo, got %d", len(idx.vectorNodes))
	}

	// Call WriteTo (should flush)
	var buf bytes.Buffer
	_, err = idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// After WriteTo, should have 2 vectors (flush removes soft deletes)
	if len(idx.vectorNodes) != 2 {
		t.Errorf("Expected 2 vectors after WriteTo (auto-flush), got %d", len(idx.vectorNodes))
	}

	// Deleted bitmap should be empty
	if idx.deletedNodes.GetCardinality() != 0 {
		t.Errorf("Expected deletedNodes to be empty after WriteTo, got cardinality %d", idx.deletedNodes.GetCardinality())
	}
}

// errorWriter is a writer that always returns an error
type errorWriterPQ struct{}

func (e errorWriterPQ) Write(p []byte) (n int, err error) {
	return 0, io.ErrUnexpectedEOF
}

// TestPQIndexWriteToError tests error handling during write operations
func TestPQIndexWriteToError(t *testing.T) {
	// Create and train index
	dim := 32
	M, Nbits := 8, 8
	idx, err := NewPQIndex(dim, Euclidean, M, Nbits)
	if err != nil {
		t.Fatalf("NewPQIndex() error: %v", err)
	}

	// Generate training vectors
	trainingVectors := make([]VectorNode, 256)
	for i := 0; i < 256; i++ {
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
		vec[j] = float32(256*dim + j)
	}
	node := NewVectorNode(vec)
	idx.Add(*node)

	// Try to write to an error writer
	var errWriter errorWriterPQ
	_, err = idx.WriteTo(errWriter)
	if err == nil {
		t.Error("WriteTo() expected error when writing to error writer, got nil")
	}
}
