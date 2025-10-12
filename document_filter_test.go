package comet

import (
	"reflect"
	"sort"
	"testing"
)

func TestDocumentFilter_Basic(t *testing.T) {
	tests := []struct {
		name     string
		docIDs   []uint32
		testID   uint32
		eligible bool
	}{
		{
			name:     "Empty filter - all eligible",
			docIDs:   []uint32{},
			testID:   100,
			eligible: true,
		},
		{
			name:     "ID in filter",
			docIDs:   []uint32{1, 2, 3, 4, 5},
			testID:   3,
			eligible: true,
		},
		{
			name:     "ID not in filter",
			docIDs:   []uint32{1, 2, 3, 4, 5},
			testID:   10,
			eligible: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewDocumentFilter(tt.docIDs)

			if filter.IsEligible(tt.testID) != tt.eligible {
				t.Errorf("IsEligible(%d) = %v, want %v", tt.testID, !tt.eligible, tt.eligible)
			}

			if filter.ShouldSkip(tt.testID) == tt.eligible {
				t.Errorf("ShouldSkip(%d) = %v, want %v", tt.testID, tt.eligible, !tt.eligible)
			}
		})
	}
}

func TestVectorSearch_WithDocumentIDs(t *testing.T) {
	// Create index with test data
	idx, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("NewFlatIndex() error: %v", err)
	}

	// Add vectors with specific IDs
	testVectors := []struct {
		id     uint32
		vector []float32
	}{
		{1, []float32{1, 0, 0}},
		{2, []float32{0, 1, 0}},
		{3, []float32{0, 0, 1}},
		{4, []float32{1, 1, 0}},
		{5, []float32{0, 1, 1}},
	}

	for _, tv := range testVectors {
		node := NewVectorNodeWithID(tv.id, tv.vector)
		idx.Add(*node)
	}

	t.Run("No filter - all documents eligible", func(t *testing.T) {
		query := []float32{1, 0, 0}
		results, err := idx.NewSearch().
			WithQuery(query).
			WithK(10).
			Execute()

		if err != nil {
			t.Fatalf("Search() error: %v", err)
		}

		if len(results) != 5 {
			t.Errorf("Expected 5 results, got %d", len(results))
		}
	})

	t.Run("With document IDs filter - subset eligible", func(t *testing.T) {
		query := []float32{1, 0, 0}

		// Only search documents 1, 3, and 5
		results, err := idx.NewSearch().
			WithQuery(query).
			WithK(10).
			WithDocumentIDs(1, 3, 5).
			Execute()

		if err != nil {
			t.Fatalf("Search() error: %v", err)
		}

		if len(results) != 3 {
			t.Errorf("Expected 3 results, got %d", len(results))
		}

		// Verify only IDs 1, 3, 5 are returned
		gotIDs := make([]uint32, len(results))
		for i, r := range results {
			gotIDs[i] = r.Node.ID()
		}
		sort.Slice(gotIDs, func(i, j int) bool { return gotIDs[i] < gotIDs[j] })

		expectedIDs := []uint32{1, 3, 5}
		if !reflect.DeepEqual(gotIDs, expectedIDs) {
			t.Errorf("Expected IDs %v, got %v", expectedIDs, gotIDs)
		}
	})

	t.Run("With empty document IDs - same as no filter", func(t *testing.T) {
		query := []float32{1, 0, 0}

		results, err := idx.NewSearch().
			WithQuery(query).
			WithK(10).
			WithDocumentIDs(). // Empty list
			Execute()

		if err != nil {
			t.Fatalf("Search() error: %v", err)
		}

		if len(results) != 5 {
			t.Errorf("Expected 5 results (no filtering), got %d", len(results))
		}
	})

	t.Run("With single document ID", func(t *testing.T) {
		query := []float32{0, 1, 0}

		results, err := idx.NewSearch().
			WithQuery(query).
			WithK(10).
			WithDocumentIDs(2). // Only document 2
			Execute()

		if err != nil {
			t.Fatalf("Search() error: %v", err)
		}

		if len(results) != 1 {
			t.Errorf("Expected 1 result, got %d", len(results))
		}

		if results[0].Node.ID() != 2 {
			t.Errorf("Expected ID 2, got %d", results[0].Node.ID())
		}
	})

	t.Run("With non-existent document IDs", func(t *testing.T) {
		query := []float32{1, 0, 0}

		// Filter to non-existent documents
		results, err := idx.NewSearch().
			WithQuery(query).
			WithK(10).
			WithDocumentIDs(100, 200, 300).
			Execute()

		if err != nil {
			t.Fatalf("Search() error: %v", err)
		}

		if len(results) != 0 {
			t.Errorf("Expected 0 results, got %d", len(results))
		}
	})
}

func TestDocumentFilter_Count(t *testing.T) {
	tests := []struct {
		name          string
		docIDs        []uint32
		expectedCount uint64
	}{
		{
			name:          "Empty filter",
			docIDs:        []uint32{},
			expectedCount: 0, // Nil filter returns 0 (all docs eligible)
		},
		{
			name:          "Single document",
			docIDs:        []uint32{1},
			expectedCount: 1,
		},
		{
			name:          "Multiple documents",
			docIDs:        []uint32{1, 2, 3, 4, 5},
			expectedCount: 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewDocumentFilter(tt.docIDs)
			count := filter.Count()

			if count != tt.expectedCount {
				t.Errorf("Count() = %d, want %d", count, tt.expectedCount)
			}
		})
	}
}

func TestDocumentFilter_IsEmpty(t *testing.T) {
	tests := []struct {
		name        string
		docIDs      []uint32
		expectEmpty bool
	}{
		{
			name:        "Empty filter is not empty (all docs eligible)",
			docIDs:      []uint32{},
			expectEmpty: false,
		},
		{
			name:        "Filter with documents is not empty",
			docIDs:      []uint32{1, 2, 3},
			expectEmpty: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewDocumentFilter(tt.docIDs)
			isEmpty := filter.IsEmpty()

			if isEmpty != tt.expectEmpty {
				t.Errorf("IsEmpty() = %v, want %v", isEmpty, tt.expectEmpty)
			}
		})
	}
}

// Benchmark to show the performance benefit of document filtering
func BenchmarkVectorSearch_WithDocumentFilter(b *testing.B) {
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
		// Filter to only 100 documents
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
