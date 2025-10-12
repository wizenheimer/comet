package comet

import (
	"reflect"
	"sort"
	"testing"
)

// TestHNSWIndex_WithDocumentIDs tests document filtering for HNSW index
func TestHNSWIndex_WithDocumentIDs(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 100)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add test vectors
	testVectors := []struct {
		id     uint32
		vector []float32
	}{
		{1, []float32{1, 0, 0}},
		{2, []float32{0, 1, 0}},
		{3, []float32{0, 0, 1}},
		{4, []float32{1, 1, 0}},
		{5, []float32{0, 1, 1}},
		{6, []float32{1, 0, 1}},
	}

	for _, tv := range testVectors {
		node := NewVectorNodeWithID(tv.id, tv.vector)
		idx.Add(*node)
	}

	query := []float32{1, 0, 0}

	tests := []struct {
		name        string
		documentIDs []uint32
		expectedIDs []uint32
	}{
		{
			name:        "No filter - all documents",
			documentIDs: nil,
			expectedIDs: []uint32{1, 2, 3, 4, 5, 6},
		},
		{
			name:        "Filter to subset",
			documentIDs: []uint32{1, 3, 5},
			expectedIDs: []uint32{1, 3, 5},
		},
		{
			name:        "Filter to single document",
			documentIDs: []uint32{2},
			expectedIDs: []uint32{2},
		},
		{
			name:        "Filter to non-existent documents",
			documentIDs: []uint32{100, 200},
			expectedIDs: []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().
				WithQuery(query).
				WithK(10).
				WithDocumentIDs(tt.documentIDs...).
				Execute()

			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			gotIDs := make([]uint32, len(results))
			for i, r := range results {
				gotIDs[i] = r.Node.ID()
			}
			sort.Slice(gotIDs, func(i, j int) bool { return gotIDs[i] < gotIDs[j] })

			if !reflect.DeepEqual(gotIDs, tt.expectedIDs) {
				t.Errorf("Expected IDs %v, got %v", tt.expectedIDs, gotIDs)
			}
		})
	}
}

// TestHNSWIndex_WithDocumentIDs_WithEfSearch tests filtering with custom efSearch
func TestHNSWIndex_WithDocumentIDs_WithEfSearch(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 100)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add test vectors
	for i := uint32(1); i <= 20; i++ {
		vec := make([]float32, 3)
		vec[i%3] = float32(i)
		node := NewVectorNodeWithID(i, vec)
		idx.Add(*node)
	}

	query := []float32{1, 0, 0}

	// Filter to subset with custom efSearch
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(5).
		WithEfSearch(50).
		WithDocumentIDs(2, 4, 6, 8, 10, 12).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Verify only filtered IDs are in results
	for _, r := range results {
		id := r.Node.ID()
		if id != 2 && id != 4 && id != 6 && id != 8 && id != 10 && id != 12 {
			t.Errorf("Got unexpected ID %d, expected only 2, 4, 6, 8, 10, 12", id)
		}
	}
}

// TestHNSWIndex_WithDocumentIDs_AfterDeletion tests filtering after soft deletion
func TestHNSWIndex_WithDocumentIDs_AfterDeletion(t *testing.T) {
	idx, err := NewHNSWIndex(3, Euclidean, 16, 200, 100)
	if err != nil {
		t.Fatalf("NewHNSWIndex() error: %v", err)
	}

	// Add test vectors
	testVectors := []struct {
		id     uint32
		vector []float32
	}{
		{1, []float32{1, 0, 0}},
		{2, []float32{0, 1, 0}},
		{3, []float32{0, 0, 1}},
		{4, []float32{1, 1, 0}},
	}

	for _, tv := range testVectors {
		node := NewVectorNodeWithID(tv.id, tv.vector)
		idx.Add(*node)
	}

	// Delete node 2
	node2 := NewVectorNodeWithID(2, []float32{0, 1, 0})
	idx.Remove(*node2)

	query := []float32{1, 0, 0}

	// Filter should respect both deletion and document IDs
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		WithDocumentIDs(1, 2, 3). // Include deleted node 2
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Should only get IDs 1 and 3 (node 2 is deleted)
	expectedIDs := []uint32{1, 3}
	gotIDs := make([]uint32, len(results))
	for i, r := range results {
		gotIDs[i] = r.Node.ID()
	}
	sort.Slice(gotIDs, func(i, j int) bool { return gotIDs[i] < gotIDs[j] })

	if !reflect.DeepEqual(gotIDs, expectedIDs) {
		t.Errorf("Expected IDs %v, got %v", expectedIDs, gotIDs)
	}
}

// BenchmarkHNSWIndex_WithDocumentFilter benchmarks filtering performance
func BenchmarkHNSWIndex_WithDocumentFilter(b *testing.B) {
	idx, _ := NewHNSWIndex(128, Euclidean, 16, 200, 100)

	// Add 10,000 vectors
	for i := uint32(1); i <= 10000; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(i % 100)
		}
		node := NewVectorNodeWithID(i, vec)
		idx.Add(*node)
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = 1.0
	}

	b.Run("Without filter (10K docs)", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = idx.NewSearch().
				WithQuery(query).
				WithK(10).
				Execute()
		}
	})

	b.Run("With filter (100 docs)", func(b *testing.B) {
		docIDs := make([]uint32, 100)
		for i := range docIDs {
			docIDs[i] = uint32(i + 1)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = idx.NewSearch().
				WithQuery(query).
				WithK(10).
				WithDocumentIDs(docIDs...).
				Execute()
		}
	})
}
