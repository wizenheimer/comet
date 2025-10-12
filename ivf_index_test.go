package comet

import (
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

	// Remove first node
	err = idx.Remove(*node1)
	if err != nil {
		t.Errorf("Remove() error: %v", err)
	}

	// Count remaining vectors
	totalVectors := 0
	for _, list := range idx.lists {
		totalVectors += len(list)
	}

	if totalVectors != 1 {
		t.Errorf("Expected 1 vector after removal, got %d", totalVectors)
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

	// Flush should be a no-op and return no error
	err = idx.Flush()
	if err != nil {
		t.Errorf("Flush() error: %v", err)
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
