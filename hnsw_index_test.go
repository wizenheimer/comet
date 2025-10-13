package comet

import (
	"bytes"
	"io"
	"sync"
	"testing"
)

// ============================================================================
// CONSTRUCTOR TESTS
// ============================================================================

// TestNewHNSWIndex tests HNSW index creation with various parameters
func TestNewHNSWIndex(t *testing.T) {
	tests := []struct {
		name             string
		dim              int
		distanceKind     DistanceKind
		m                int
		efConstruction   int
		efSearch         int
		wantErr          bool
		expectedM        int
		expectedEfConst  int
		expectedEfSearch int
	}{
		{
			name:             "valid L2 index with explicit params",
			dim:              128,
			distanceKind:     Euclidean,
			m:                16,
			efConstruction:   200,
			efSearch:         200,
			wantErr:          false,
			expectedM:        16,
			expectedEfConst:  200,
			expectedEfSearch: 200,
		},
		{
			name:             "valid Cosine index",
			dim:              384,
			distanceKind:     Cosine,
			m:                32,
			efConstruction:   400,
			efSearch:         300,
			wantErr:          false,
			expectedM:        32,
			expectedEfConst:  400,
			expectedEfSearch: 300,
		},
		{
			name:             "valid L2Squared index",
			dim:              768,
			distanceKind:     L2Squared,
			m:                12,
			efConstruction:   100,
			efSearch:         100,
			wantErr:          false,
			expectedM:        12,
			expectedEfConst:  100,
			expectedEfSearch: 100,
		},
		{
			name:             "default M (0 becomes 16)",
			dim:              128,
			distanceKind:     Euclidean,
			m:                0,
			efConstruction:   200,
			efSearch:         200,
			wantErr:          false,
			expectedM:        16,
			expectedEfConst:  200,
			expectedEfSearch: 200,
		},
		{
			name:             "default efConstruction (0 becomes 200)",
			dim:              128,
			distanceKind:     Euclidean,
			m:                16,
			efConstruction:   0,
			efSearch:         200,
			wantErr:          false,
			expectedM:        16,
			expectedEfConst:  200,
			expectedEfSearch: 200,
		},
		{
			name:             "default efSearch (0 becomes efConstruction)",
			dim:              128,
			distanceKind:     Euclidean,
			m:                16,
			efConstruction:   200,
			efSearch:         0,
			wantErr:          false,
			expectedM:        16,
			expectedEfConst:  200,
			expectedEfSearch: 200,
		},
		{
			name:           "zero dimension",
			dim:            0,
			distanceKind:   Euclidean,
			m:              16,
			efConstruction: 200,
			efSearch:       200,
			wantErr:        true,
		},
		{
			name:           "negative dimension",
			dim:            -5,
			distanceKind:   Euclidean,
			m:              16,
			efConstruction: 200,
			efSearch:       200,
			wantErr:        true,
		},
		{
			name:           "invalid distance kind",
			dim:            128,
			distanceKind:   DistanceKind("invalid"),
			m:              16,
			efConstruction: 200,
			efSearch:       200,
			wantErr:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, err := NewHNSWIndex(tt.dim, tt.distanceKind, tt.m, tt.efConstruction, tt.efSearch)

			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Expected no error but got: %v", err)
				return
			}

			if idx == nil {
				t.Fatal("Expected non-nil index")
			}

			if idx.Dimensions() != tt.dim {
				t.Errorf("Dimensions() = %d, want %d", idx.Dimensions(), tt.dim)
			}

			if idx.DistanceKind() != tt.distanceKind {
				t.Errorf("DistanceKind() = %v, want %v", idx.DistanceKind(), tt.distanceKind)
			}

			if idx.Kind() != HNSWIndexKind {
				t.Errorf("Kind() = %v, want %v", idx.Kind(), HNSWIndexKind)
			}

			if idx.M != tt.expectedM {
				t.Errorf("M = %d, want %d", idx.M, tt.expectedM)
			}

			if idx.efConstruction != tt.expectedEfConst {
				t.Errorf("efConstruction = %d, want %d", idx.efConstruction, tt.expectedEfConst)
			}

			if idx.efSearch != tt.expectedEfSearch {
				t.Errorf("efSearch = %d, want %d", idx.efSearch, tt.expectedEfSearch)
			}

			// Verify initial state
			if idx.maxLevel != -1 {
				t.Errorf("maxLevel = %d, want -1 for empty index", idx.maxLevel)
			}

			if idx.entryPoint != 0 {
				t.Errorf("entryPoint = %d, want 0 for empty index", idx.entryPoint)
			}

			if len(idx.nodes) != 0 {
				t.Errorf("nodes map should be empty, got %d nodes", len(idx.nodes))
			}

			if idx.deletedNodes.GetCardinality() != 0 {
				t.Errorf("deletedNodes should be empty, got %d deleted", idx.deletedNodes.GetCardinality())
			}
		})
	}
}

// TestDefaultHNSWConfig tests the default configuration helper
func TestDefaultHNSWConfig(t *testing.T) {
	m, efConstruction, efSearch := DefaultHNSWConfig()

	if m != 16 {
		t.Errorf("Default M = %d, want 16", m)
	}

	if efConstruction != 200 {
		t.Errorf("Default efConstruction = %d, want 200", efConstruction)
	}

	if efSearch != 200 {
		t.Errorf("Default efSearch = %d, want 200", efSearch)
	}
}

// ============================================================================
// TRAIN TESTS
// ============================================================================

// TestHNSWIndexTrain tests that training is a no-op
func TestHNSWIndexTrain(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// HNSW doesn't require training
	trainingVectors := []VectorNode{
		*NewVectorNode([]float32{0, 0, 0}),
		*NewVectorNode([]float32{1, 0, 0}),
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Errorf("Train() should not error: %v", err)
	}

	// Index should still be marked as trained
	if !idx.Trained() {
		t.Error("Trained() should return true")
	}
}

// ============================================================================
// ADD TESTS
// ============================================================================

// TestHNSWIndexAddSingle tests adding a single vector
func TestHNSWIndexAddSingle(t *testing.T) {
	tests := []struct {
		name         string
		dim          int
		distanceKind DistanceKind
		vector       []float32
		wantErr      bool
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
		},
		{
			name:         "zero vector with cosine",
			dim:          3,
			distanceKind: Cosine,
			vector:       []float32{0, 0, 0},
			wantErr:      true,
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
			idx, err := NewHNSWIndex(tt.dim, tt.distanceKind, 16, 200, 200)
			if err != nil {
				t.Fatalf("NewHNSWIndex() error: %v", err)
			}

			node := NewVectorNode(tt.vector)
			err = idx.Add(*node)

			if tt.wantErr {
				if err == nil {
					t.Error("Add() expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Add() unexpected error: %v", err)
				return
			}

			// Verify node was added
			idx.mu.RLock()
			if len(idx.nodes) != 1 {
				t.Errorf("Expected 1 node, got %d", len(idx.nodes))
			}

			if idx.entryPoint == 0 {
				t.Error("Entry point should be set after first add")
			}

			if idx.maxLevel < 0 {
				t.Error("maxLevel should be >= 0 after adding a node")
			}
			idx.mu.RUnlock()
		})
	}
}

// TestHNSWIndexAddMultiple tests adding multiple vectors
func TestHNSWIndexAddMultiple(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	vectors := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12},
		{13, 14, 15},
	}

	for i, v := range vectors {
		node := NewVectorNode(v)
		if err := idx.Add(*node); err != nil {
			t.Errorf("Add() error at index %d: %v", i, err)
		}
	}

	// Verify all vectors were added
	idx.mu.RLock()
	if len(idx.nodes) != len(vectors) {
		t.Errorf("Expected %d nodes, got %d", len(vectors), len(idx.nodes))
	}

	// Verify entry point is set
	if idx.entryPoint == 0 {
		t.Error("Entry point should be set")
	}

	// Verify nodes have edges (except possibly the first one)
	totalEdges := 0
	for _, node := range idx.nodes {
		for layer := 0; layer <= node.Level; layer++ {
			totalEdges += len(node.Edges[layer])
		}
	}

	if totalEdges == 0 {
		t.Error("Expected some edges to be created in the graph")
	}
	idx.mu.RUnlock()
}

// TestHNSWIndexAddWithCustomID tests adding vectors with custom IDs
func TestHNSWIndexAddWithCustomID(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	customID := uint32(1000)
	node := NewVectorNodeWithID(customID, []float32{1, 2, 3})

	err = idx.Add(*node)
	if err != nil {
		t.Fatalf("Add() error: %v", err)
	}

	idx.mu.RLock()
	_, exists := idx.nodes[customID]
	idx.mu.RUnlock()

	if !exists {
		t.Errorf("Node with custom ID %d not found", customID)
	}
}

// TestHNSWIndexRandomLevel tests the level assignment distribution
func TestHNSWIndexRandomLevel(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Generate many levels and check distribution
	levels := make(map[int]int)
	numSamples := 1000

	for i := 0; i < numSamples; i++ {
		level := idx.randomLevel()
		levels[level]++

		// Verify level is within reasonable bounds
		if level < 0 || level > 16 {
			t.Errorf("randomLevel() = %d, should be in [0, 16]", level)
		}
	}

	// Most nodes should be at level 0
	if levels[0] < numSamples/2 {
		t.Errorf("Expected most nodes at level 0, got %d out of %d", levels[0], numSamples)
	}

	// Should have some nodes at higher levels
	higherLevels := 0
	for level, count := range levels {
		if level > 0 {
			higherLevels += count
		}
	}

	if higherLevels == 0 {
		t.Error("Expected some nodes at higher levels")
	}
}

// ============================================================================
// REMOVE AND FLUSH TESTS
// ============================================================================

// TestHNSWIndexRemove tests soft deletion
func TestHNSWIndexRemove(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
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

	// Verify soft delete
	idx.mu.RLock()
	if len(idx.nodes) != 3 {
		t.Errorf("Expected 3 nodes (soft delete), got %d", len(idx.nodes))
	}

	if !idx.deletedNodes.Contains(node2.ID()) {
		t.Error("Node should be marked as deleted")
	}

	if idx.deletedNodes.GetCardinality() != 1 {
		t.Errorf("Expected 1 deleted node, got %d", idx.deletedNodes.GetCardinality())
	}
	idx.mu.RUnlock()
}

// TestHNSWIndexRemoveNonExistent tests removing non-existent vector
func TestHNSWIndexRemoveNonExistent(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Try to remove a node that was never added
	node := NewVectorNode([]float32{1, 0, 0})
	err = idx.Remove(*node)

	if err == nil {
		t.Error("Expected error when removing non-existent vector")
	}
}

// TestHNSWIndexRemoveTwice tests removing the same vector twice
func TestHNSWIndexRemoveTwice(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	node := NewVectorNode([]float32{1, 2, 3})
	idx.Add(*node)

	// First removal should succeed
	err = idx.Remove(*node)
	if err != nil {
		t.Errorf("First Remove() error: %v", err)
	}

	// Second removal should fail
	err = idx.Remove(*node)
	if err == nil {
		t.Error("Expected error when removing already deleted vector")
	}
}

// TestHNSWIndexFlush tests hard deletion
func TestHNSWIndexFlush(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	node1 := NewVectorNode([]float32{1, 2, 3})
	node2 := NewVectorNode([]float32{4, 5, 6})
	node3 := NewVectorNode([]float32{7, 8, 9})

	idx.Add(*node1)
	idx.Add(*node2)
	idx.Add(*node3)

	// Soft delete one
	idx.Remove(*node2)

	// Flush (hard delete)
	err = idx.Flush()
	if err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	// Verify hard delete
	idx.mu.RLock()
	if len(idx.nodes) != 2 {
		t.Errorf("Expected 2 nodes after flush, got %d", len(idx.nodes))
	}

	if idx.deletedNodes.GetCardinality() != 0 {
		t.Errorf("Expected 0 deleted nodes after flush, got %d", idx.deletedNodes.GetCardinality())
	}

	_, exists := idx.nodes[node2.ID()]
	if exists {
		t.Error("Flushed node should not exist")
	}
	idx.mu.RUnlock()
}

// TestHNSWIndexFlushEmpty tests flushing when nothing is deleted
func TestHNSWIndexFlushEmpty(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors without deleting any
	node := NewVectorNode([]float32{1, 2, 3})
	idx.Add(*node)

	// Flush should succeed (no-op)
	err = idx.Flush()
	if err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	idx.mu.RLock()
	if len(idx.nodes) != 1 {
		t.Errorf("Expected 1 node after flush, got %d", len(idx.nodes))
	}
	idx.mu.RUnlock()
}

// TestHNSWIndexFlushEntryPoint tests entry point update during flush
func TestHNSWIndexFlushEntryPoint(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add several vectors
	nodes := make([]*VectorNode, 5)
	for i := 0; i < 5; i++ {
		nodes[i] = NewVectorNode([]float32{float32(i), 0, 0})
		idx.Add(*nodes[i])
	}

	// Get entry point
	idx.mu.RLock()
	entryID := idx.entryPoint
	idx.mu.RUnlock()

	// Remove entry point
	for _, n := range nodes {
		if n.ID() == entryID {
			idx.Remove(*n)
			break
		}
	}

	// Flush
	err = idx.Flush()
	if err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	// Verify new entry point was selected
	idx.mu.RLock()
	if idx.entryPoint == entryID {
		t.Error("Entry point should have been updated")
	}

	if idx.entryPoint == 0 && len(idx.nodes) > 0 {
		t.Error("Entry point should be set when nodes exist")
	}
	idx.mu.RUnlock()
}

// TestHNSWIndexFlushAll tests flushing all nodes
func TestHNSWIndexFlushAll(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add and delete all vectors
	node1 := NewVectorNode([]float32{1, 2, 3})
	node2 := NewVectorNode([]float32{4, 5, 6})

	idx.Add(*node1)
	idx.Add(*node2)

	idx.Remove(*node1)
	idx.Remove(*node2)

	// Flush
	err = idx.Flush()
	if err != nil {
		t.Errorf("Flush() error: %v", err)
	}

	// Verify complete cleanup
	idx.mu.RLock()
	if len(idx.nodes) != 0 {
		t.Errorf("Expected 0 nodes after flushing all, got %d", len(idx.nodes))
	}

	if idx.entryPoint != 0 {
		t.Error("Entry point should be 0 when all nodes deleted")
	}

	if idx.maxLevel != -1 {
		t.Error("maxLevel should be -1 when all nodes deleted")
	}
	idx.mu.RUnlock()
}

// ============================================================================
// GETTER TESTS
// ============================================================================

// TestHNSWIndexGetters tests getter methods
func TestHNSWIndexGetters(t *testing.T) {
	dim := 384
	distanceKind := Cosine
	m := 32
	efConstruction := 400
	efSearch := 300

	idx, err := NewHNSWIndex(dim, distanceKind, m, efConstruction, efSearch)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	if idx.Dimensions() != dim {
		t.Errorf("Dimensions() = %d, want %d", idx.Dimensions(), dim)
	}

	if idx.DistanceKind() != distanceKind {
		t.Errorf("DistanceKind() = %v, want %v", idx.DistanceKind(), distanceKind)
	}

	if idx.Kind() != HNSWIndexKind {
		t.Errorf("Kind() = %v, want %v", idx.Kind(), HNSWIndexKind)
	}

	if !idx.Trained() {
		t.Error("Trained() should return true for HNSW")
	}
}

// TestHNSWIndexSetEfSearch tests setting efSearch
func TestHNSWIndexSetEfSearch(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Set new efSearch
	newEfSearch := 500
	idx.SetEfSearch(newEfSearch)

	idx.mu.RLock()
	if idx.efSearch != newEfSearch {
		t.Errorf("efSearch = %d, want %d", idx.efSearch, newEfSearch)
	}
	idx.mu.RUnlock()
}

// TestHNSWIndexNewSearch tests search builder creation
func TestHNSWIndexNewSearch(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	search := idx.NewSearch()
	if search == nil {
		t.Fatal("NewSearch() returned nil")
	}

	// Verify it's the correct type
	hnswSearch, ok := search.(*hnswIndexSearch)
	if !ok {
		t.Error("NewSearch() did not return *hnswIndexSearch")
	}

	// Verify defaults
	if hnswSearch.k != 10 {
		t.Errorf("Default k = %d, want 10", hnswSearch.k)
	}
}

// ============================================================================
// CONCURRENCY TESTS
// ============================================================================

// TestHNSWIndexConcurrentAdd tests concurrent additions
func TestHNSWIndexConcurrentAdd(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	const numGoroutines = 50
	const vectorsPerGoroutine = 10

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(offset int) {
			defer wg.Done()
			for j := 0; j < vectorsPerGoroutine; j++ {
				node := NewVectorNode([]float32{
					float32(offset*10 + j),
					float32(offset*10 + j + 1),
					float32(offset*10 + j + 2),
				})
				if err := idx.Add(*node); err != nil {
					t.Errorf("Add() error: %v", err)
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify all vectors were added
	idx.mu.RLock()
	expected := numGoroutines * vectorsPerGoroutine
	if len(idx.nodes) != expected {
		t.Errorf("Expected %d nodes, got %d", expected, len(idx.nodes))
	}
	idx.mu.RUnlock()
}

// TestHNSWIndexConcurrentRemove tests concurrent removals
func TestHNSWIndexConcurrentRemove(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	const numVectors = 100
	nodes := make([]*VectorNode, numVectors)
	for i := 0; i < numVectors; i++ {
		nodes[i] = NewVectorNode([]float32{float32(i), 0, 0})
		idx.Add(*nodes[i])
	}

	// Remove concurrently
	var wg sync.WaitGroup
	wg.Add(numVectors)

	for i := 0; i < numVectors; i++ {
		go func(index int) {
			defer wg.Done()
			idx.Remove(*nodes[index])
		}(i)
	}

	wg.Wait()

	// Verify all marked as deleted
	idx.mu.RLock()
	deletedCount := idx.deletedNodes.GetCardinality()
	idx.mu.RUnlock()

	if int(deletedCount) != numVectors {
		t.Errorf("Expected %d deleted nodes, got %d", numVectors, deletedCount)
	}
}

// TestHNSWIndexConcurrentAddAndSearch tests concurrent adds and searches
func TestHNSWIndexConcurrentAddAndSearch(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

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

// TestHNSWIndexConcurrentAddRemoveFlush tests concurrent operations
func TestHNSWIndexConcurrentAddRemoveFlush(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add initial vectors
	initialNodes := make([]*VectorNode, 50)
	for i := 0; i < 50; i++ {
		initialNodes[i] = NewVectorNode([]float32{float32(i), 0, 0})
		idx.Add(*initialNodes[i])
	}

	var wg sync.WaitGroup

	// Concurrent adds
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			node := NewVectorNode([]float32{float32(i + 100), 0, 0})
			idx.Add(*node)
		}
	}()

	// Concurrent removes
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			idx.Remove(*initialNodes[i])
		}
	}()

	// Concurrent flush (should not panic)
	wg.Add(1)
	go func() {
		defer wg.Done()
		idx.Flush()
	}()

	wg.Wait()

	// Just verify no panics and index is in consistent state
	idx.mu.RLock()
	nodeCount := len(idx.nodes)
	idx.mu.RUnlock()

	if nodeCount < 0 {
		t.Error("Node count should not be negative")
	}
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

// TestHNSWIndexEmptySearch tests searching an empty index
func TestHNSWIndexEmptySearch(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Search empty index
	results, err := idx.NewSearch().
		WithQuery([]float32{1, 2, 3}).
		WithK(10).
		Execute()

	if err != nil {
		t.Errorf("Search() on empty index error: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected 0 results from empty index, got %d", len(results))
	}
}

// TestHNSWIndexSingleNode tests operations with a single node
func TestHNSWIndexSingleNode(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	node := NewVectorNode([]float32{1, 2, 3})
	idx.Add(*node)

	// Search should return the single node
	results, err := idx.NewSearch().
		WithQuery([]float32{1, 2, 3}).
		WithK(1).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}

	if len(results) > 0 && results[0].Node.ID() != node.ID() {
		t.Error("Search should return the single node")
	}
}

// TestHNSWIndexDifferentDistanceMetrics tests HNSW with different metrics
func TestHNSWIndexDifferentDistanceMetrics(t *testing.T) {
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
			idx, err := NewHNSWIndex(3, tt.distanceKind, 16, 200, 200)
			if err != nil {
				t.Fatalf("NewHNSWIndex() error: %v", err)
			}

			// Add vectors
			vectors := [][]float32{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
				{1, 1, 0},
				{1, 0, 1},
			}

			for _, v := range vectors {
				node := NewVectorNode(v)
				err := idx.Add(*node)
				if err != nil {
					t.Fatalf("Add() error: %v", err)
				}
			}

			// Search
			results, err := idx.NewSearch().
				WithQuery([]float32{1, 0, 0}).
				WithK(3).
				Execute()

			if err != nil {
				t.Fatalf("Search() error: %v", err)
			}

			if len(results) == 0 {
				t.Error("Expected some results")
			}
		})
	}
}

// TestHNSWIndexLargeScale tests HNSW with more vectors
func TestHNSWIndexLargeScale(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large scale test in short mode")
	}

	dim := 128
	numVectors := 1000

	idx, err := NewHNSWIndex(dim, Euclidean, 16, 200, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
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
			t.Fatalf("Add() error at index %d: %v", i, err)
		}
	}

	// Verify all vectors were added
	idx.mu.RLock()
	nodeCount := len(idx.nodes)
	idx.mu.RUnlock()

	if nodeCount != numVectors {
		t.Errorf("Expected %d nodes, got %d", numVectors, nodeCount)
	}

	// Search
	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = float32(j % 100)
	}

	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if len(results) != 10 {
		t.Errorf("Expected 10 results, got %d", len(results))
	}
}

// ============================================================================
// NODE STRUCTURE TESTS
// ============================================================================

// TestNewHnswNode tests node initialization
func TestNewHnswNode(t *testing.T) {
	vectorNode := *NewVectorNode([]float32{1, 2, 3})
	level := 2

	node := newHnswNode(vectorNode, level)

	if node.Level != level {
		t.Errorf("Level = %d, want %d", node.Level, level)
	}

	if node.ID() != vectorNode.ID() {
		t.Errorf("ID mismatch: got %d, want %d", node.ID(), vectorNode.ID())
	}

	if len(node.Edges) != level+1 {
		t.Errorf("Expected %d edge layers, got %d", level+1, len(node.Edges))
	}

	// Verify all edge layers are initialized
	for i := 0; i <= level; i++ {
		if node.Edges[i] == nil {
			t.Errorf("Edge layer %d not initialized", i)
		}
	}
}

// ============================================================================
// SERIALIZATION TESTS
// ============================================================================

// TestHNSWIndexWriteTo tests serialization of the HNSW index
func TestHNSWIndexWriteTo(t *testing.T) {
	dim := 32
	m, efConstruction, efSearch := 8, 100, 100
	idx, err := NewHNSWIndex(dim, Euclidean, m, efConstruction, efSearch)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	for i := 0; i < 10; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
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
	if string(magic) != "HNSW" {
		t.Errorf("Invalid magic number: got %s, want HNSW", string(magic))
	}
}

// TestHNSWIndexReadFrom tests deserialization of the HNSW index
func TestHNSWIndexReadFrom(t *testing.T) {
	dim := 32
	m, efConstruction, efSearch := 8, 100, 100
	original, err := NewHNSWIndex(dim, Euclidean, m, efConstruction, efSearch)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	for i := 0; i < 10; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
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
	restored, err := NewHNSWIndex(dim, Euclidean, m, efConstruction, efSearch)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
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

	if restored.maxLevel != original.maxLevel {
		t.Errorf("maxLevel mismatch: got %d, want %d", restored.maxLevel, original.maxLevel)
	}

	if restored.entryPoint != original.entryPoint {
		t.Errorf("entryPoint mismatch: got %d, want %d", restored.entryPoint, original.entryPoint)
	}

	// Verify node count
	if len(restored.nodes) != len(original.nodes) {
		t.Errorf("Node count mismatch: got %d, want %d", len(restored.nodes), len(original.nodes))
	}
}

// TestHNSWIndexSerializationRoundTrip tests that serialization and deserialization preserve data
func TestHNSWIndexSerializationRoundTrip(t *testing.T) {
	dim := 32
	m, efConstruction, efSearch := 8, 100, 100
	idx, err := NewHNSWIndex(dim, Euclidean, m, efConstruction, efSearch)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	for i := 0; i < 20; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		node := NewVectorNode(vec)
		if err := idx.Add(*node); err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Perform a search before serialization
	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = float32(10*dim + j)
	}
	resultsBefore, err := idx.NewSearch().WithQuery(query).WithK(5).Execute()
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
	idx2, err := NewHNSWIndex(dim, Euclidean, m, efConstruction, efSearch)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Perform same search after deserialization
	resultsAfter, err := idx2.NewSearch().WithQuery(query).WithK(5).Execute()
	if err != nil {
		t.Fatalf("Search after deserialization error: %v", err)
	}

	// Results should match in count
	if len(resultsBefore) != len(resultsAfter) {
		t.Errorf("Result count mismatch: before=%d, after=%d", len(resultsBefore), len(resultsAfter))
	}

	// HNSW is approximate, so exact ordering may vary for vectors with similar distances
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

// TestHNSWIndexSerializationWithDeletions tests serialization with soft-deleted nodes
func TestHNSWIndexSerializationWithDeletions(t *testing.T) {
	dim := 32
	m, efConstruction, efSearch := 8, 100, 100
	idx, err := NewHNSWIndex(dim, Euclidean, m, efConstruction, efSearch)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	nodes := make([]*VectorNode, 6)
	for i := 0; i < 6; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		node := NewVectorNode(vec)
		nodes[i] = node
		if err := idx.Add(*node); err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Soft delete some nodes
	idx.Remove(*nodes[1])
	idx.Remove(*nodes[3])
	idx.Remove(*nodes[5])

	// Verify soft deletes exist before serialization
	initialNodeCount := len(idx.nodes)
	if initialNodeCount != 6 {
		t.Errorf("Expected 6 nodes before serialization (soft delete), got %d", initialNodeCount)
	}

	// Serialize (should call Flush automatically)
	var buf bytes.Buffer
	_, err = idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// After WriteTo (which calls Flush), deleted nodes should be removed
	afterFlushCount := len(idx.nodes)
	if afterFlushCount != 3 {
		t.Errorf("Expected 3 nodes after WriteTo (auto-flush), got %d", afterFlushCount)
	}

	// Deserialize
	idx2, err := NewHNSWIndex(dim, Euclidean, m, efConstruction, efSearch)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Restored index should only have non-deleted nodes
	restoredCount := len(idx2.nodes)
	if restoredCount != 3 {
		t.Errorf("Expected 3 nodes in restored index, got %d", restoredCount)
	}

	// Verify only nodes 0, 2, 4 exist
	for _, expectedID := range []uint32{nodes[0].ID(), nodes[2].ID(), nodes[4].ID()} {
		if _, exists := idx2.nodes[expectedID]; !exists {
			t.Errorf("Expected node %d to exist in restored index", expectedID)
		}
	}
	for _, deletedID := range []uint32{nodes[1].ID(), nodes[3].ID(), nodes[5].ID()} {
		if _, exists := idx2.nodes[deletedID]; exists {
			t.Errorf("Expected node %d to NOT exist in restored index (deleted)", deletedID)
		}
	}
}

// TestHNSWIndexReadFromInvalidData tests error handling for invalid serialized data
func TestHNSWIndexReadFromInvalidData(t *testing.T) {
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
				buf.Write([]byte("HNSW"))
				// Write invalid version
				buf.Write([]byte{99, 0, 0, 0}) // version 99
				return &buf
			},
			wantErr: "unsupported version",
		},
		{
			name: "truncated data",
			setup: func() *bytes.Buffer {
				buf := bytes.NewBuffer([]byte("HN"))
				return buf
			},
			wantErr: "failed to read magic number",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := tt.setup()

			m, efC, efS := DefaultHNSWConfig()
			idx, err := NewHNSWIndex(32, Euclidean, m, efC, efS)
			if err != nil {
				t.Fatalf("NewHNSWIndex() error: %v", err)
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

// TestHNSWIndexSerializationEmpty tests serialization of an empty index
func TestHNSWIndexSerializationEmpty(t *testing.T) {
	m, efC, efS := DefaultHNSWConfig()
	idx, err := NewHNSWIndex(32, Euclidean, m, efC, efS)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
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
	idx2, err := NewHNSWIndex(32, Euclidean, m, efC, efS)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Verify restored index is also empty
	if len(idx2.nodes) != 0 {
		t.Errorf("Expected 0 nodes in restored empty index, got %d", len(idx2.nodes))
	}

	if idx2.maxLevel != -1 {
		t.Errorf("Expected maxLevel -1 in empty index, got %d", idx2.maxLevel)
	}
}

// TestHNSWIndexWriteToFlushBehavior tests that WriteTo calls Flush
func TestHNSWIndexWriteToFlushBehavior(t *testing.T) {
	dim := 32
	m, efConstruction, efSearch := 8, 100, 100
	idx, err := NewHNSWIndex(dim, Euclidean, m, efConstruction, efSearch)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add vectors
	nodes := make([]*VectorNode, 4)
	for i := 0; i < 4; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float32(i*dim + j)
		}
		node := NewVectorNode(vec)
		nodes[i] = node
		if err := idx.Add(*node); err != nil {
			t.Fatalf("Add() error: %v", err)
		}
	}

	// Soft delete one node
	idx.Remove(*nodes[2])

	// Before WriteTo, should have 4 nodes (soft delete)
	if len(idx.nodes) != 4 {
		t.Errorf("Expected 4 nodes before WriteTo, got %d", len(idx.nodes))
	}

	// Call WriteTo (should flush)
	var buf bytes.Buffer
	_, err = idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// After WriteTo, should have 3 nodes (flush removes soft deletes)
	if len(idx.nodes) != 3 {
		t.Errorf("Expected 3 nodes after WriteTo (auto-flush), got %d", len(idx.nodes))
	}

	// Deleted bitmap should be empty
	if idx.deletedNodes.GetCardinality() != 0 {
		t.Errorf("Expected deletedNodes to be empty after WriteTo, got cardinality %d", idx.deletedNodes.GetCardinality())
	}
}

// errorWriterHNSW is a writer that always returns an error
type errorWriterHNSW struct{}

func (e errorWriterHNSW) Write(p []byte) (n int, err error) {
	return 0, io.ErrUnexpectedEOF
}

// TestHNSWIndexWriteToError tests error handling during write operations
func TestHNSWIndexWriteToError(t *testing.T) {
	dim := 32
	m, efC, efS := DefaultHNSWConfig()
	idx, err := NewHNSWIndex(dim, Euclidean, m, efC, efS)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	vec := make([]float32, dim)
	for j := 0; j < dim; j++ {
		vec[j] = float32(j)
	}
	node := NewVectorNode(vec)
	idx.Add(*node)

	// Try to write to an error writer
	var errWriter errorWriterHNSW
	_, err = idx.WriteTo(errWriter)
	if err == nil {
		t.Error("WriteTo() expected error when writing to error writer, got nil")
	}
}
