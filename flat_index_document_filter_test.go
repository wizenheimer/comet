package comet

import (
	"reflect"
	"sort"
	"testing"
)

// TestFlatIndex_WithDocumentIDs tests document filtering for flat index
func TestFlatIndex_WithDocumentIDs(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
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
		{
			name:        "Empty filter list - same as no filter",
			documentIDs: []uint32{},
			expectedIDs: []uint32{1, 2, 3, 4, 5, 6},
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

// TestFlatIndex_WithDocumentIDs_MultipleQueries tests filtering with batch queries
func TestFlatIndex_WithDocumentIDs_MultipleQueries(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add test vectors
	for i := uint32(1); i <= 10; i++ {
		vec := make([]float32, 3)
		vec[i%3] = float32(i)
		node := NewVectorNodeWithID(i, vec)
		idx.Add(*node)
	}

	// Multiple queries with document filter
	queries := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
	}

	results, err := idx.NewSearch().
		WithQuery(queries...).
		WithK(10).
		WithDocumentIDs(2, 4, 6, 8). // Only even IDs
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Verify only filtered IDs are in results
	for _, r := range results {
		id := r.Node.ID()
		if id != 2 && id != 4 && id != 6 && id != 8 {
			t.Errorf("Got unexpected ID %d, expected only 2, 4, 6, 8", id)
		}
	}
}

// TestFlatIndex_WithDocumentIDs_WithThreshold tests filtering combined with threshold
func TestFlatIndex_WithDocumentIDs_WithThreshold(t *testing.T) {
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add test vectors
	testVectors := []struct {
		id     uint32
		vector []float32
	}{
		{1, []float32{1, 0, 0}},  // Distance 0 from query
		{2, []float32{2, 0, 0}},  // Distance 1 from query
		{3, []float32{3, 0, 0}},  // Distance 2 from query
		{4, []float32{10, 0, 0}}, // Distance 9 from query
	}

	for _, tv := range testVectors {
		node := NewVectorNodeWithID(tv.id, tv.vector)
		idx.Add(*node)
	}

	query := []float32{1, 0, 0}

	// Filter to IDs 1, 2, 3 with threshold 1.5
	results, err := idx.NewSearch().
		WithQuery(query).
		WithK(10).
		WithDocumentIDs(1, 2, 3).
		WithThreshold(1.5).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Should only get IDs 1 and 2 (within threshold and in filter)
	expectedIDs := []uint32{1, 2}
	gotIDs := make([]uint32, len(results))
	for i, r := range results {
		gotIDs[i] = r.Node.ID()
	}
	sort.Slice(gotIDs, func(i, j int) bool { return gotIDs[i] < gotIDs[j] })

	if !reflect.DeepEqual(gotIDs, expectedIDs) {
		t.Errorf("Expected IDs %v, got %v", expectedIDs, gotIDs)
	}
}

// BenchmarkFlatIndex_WithDocumentFilter benchmarks filtering performance
func BenchmarkFlatIndex_WithDocumentFilter(b *testing.B) {
	idx, _ := NewFlatIndex(128, Euclidean)

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

	b.Run("With filter (10 docs)", func(b *testing.B) {
		docIDs := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

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
