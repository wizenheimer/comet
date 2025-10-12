package comet

import (
	"testing"
)

// TestIVFIndex_WithDocumentIDs tests document filtering for IVF index
func TestIVFIndex_WithDocumentIDs(t *testing.T) {
	idx, err := NewIVFIndex(3, 2, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Add training vectors
	trainingVectors := []VectorNode{
		*NewVectorNodeWithID(101, []float32{1, 0, 0}),
		*NewVectorNodeWithID(102, []float32{0, 1, 0}),
		*NewVectorNodeWithID(103, []float32{0, 0, 1}),
		*NewVectorNodeWithID(104, []float32{1, 1, 0}),
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
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
		wantInSet   []uint32
	}{
		{
			name:        "No filter - all documents",
			documentIDs: nil,
			wantInSet:   []uint32{1, 2, 3, 4, 5, 6},
		},
		{
			name:        "Filter to subset",
			documentIDs: []uint32{1, 3, 5},
			wantInSet:   []uint32{1, 3, 5},
		},
		{
			name:        "Filter to single document",
			documentIDs: []uint32{2},
			wantInSet:   []uint32{2},
		},
		{
			name:        "Filter to non-existent documents",
			documentIDs: []uint32{100, 200},
			wantInSet:   []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().
				WithQuery(query).
				WithK(10).
				WithNProbes(2).
				WithDocumentIDs(tt.documentIDs...).
				Execute()

			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			gotIDs := make([]uint32, len(results))
			for i, r := range results {
				gotIDs[i] = r.Node.ID()
			}

			// Check that all returned IDs are in the expected set
			for _, gotID := range gotIDs {
				found := false
				for _, wantID := range tt.wantInSet {
					if gotID == wantID {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Got unexpected ID %d, want only IDs from %v", gotID, tt.wantInSet)
				}
			}

			// For empty expected set, ensure no results
			if len(tt.wantInSet) == 0 && len(gotIDs) != 0 {
				t.Errorf("Expected no results, got %v", gotIDs)
			}
		})
	}
}

// TestIVFIndex_WithDocumentIDs_WithNProbes tests filtering with different nprobes
func TestIVFIndex_WithDocumentIDs_WithNProbes(t *testing.T) {
	idx, err := NewIVFIndex(3, 4, Euclidean)
	if err != nil {
		t.Fatalf("NewIVFIndex() error: %v", err)
	}

	// Add training vectors (need at least nlist vectors)
	trainingVectors := make([]VectorNode, 12)
	for i := 0; i < 12; i++ {
		vec := make([]float32, 3)
		vec[i%3] = float32(i)
		trainingVectors[i] = *NewVectorNodeWithID(uint32(100+i), vec)
	}

	err = idx.Train(trainingVectors)
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	// Add test vectors
	for i := uint32(1); i <= 20; i++ {
		vec := make([]float32, 3)
		vec[i%3] = float32(i)
		node := NewVectorNodeWithID(i, vec)
		idx.Add(*node)
	}

	query := []float32{1, 0, 0}

	// Test with different nprobes values
	tests := []struct {
		name    string
		nprobes int
		docIDs  []uint32
	}{
		{
			name:    "nprobes=1 with filter",
			nprobes: 1,
			docIDs:  []uint32{1, 2, 3, 4, 5},
		},
		{
			name:    "nprobes=2 with filter",
			nprobes: 2,
			docIDs:  []uint32{1, 2, 3, 4, 5},
		},
		{
			name:    "nprobes=4 with filter",
			nprobes: 4,
			docIDs:  []uint32{1, 2, 3, 4, 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().
				WithQuery(query).
				WithK(10).
				WithNProbes(tt.nprobes).
				WithDocumentIDs(tt.docIDs...).
				Execute()

			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Verify only filtered IDs are in results
			for _, r := range results {
				id := r.Node.ID()
				found := false
				for _, docID := range tt.docIDs {
					if id == docID {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Got unexpected ID %d, expected only IDs from %v", id, tt.docIDs)
				}
			}
		})
	}
}

// BenchmarkIVFIndex_WithDocumentFilter benchmarks filtering performance
func BenchmarkIVFIndex_WithDocumentFilter(b *testing.B) {
	idx, _ := NewIVFIndex(128, 100, Euclidean)

	// Create training vectors
	trainingVectors := make([]VectorNode, 1000)
	for i := 0; i < 1000; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(i % 100)
		}
		trainingVectors[i] = *NewVectorNodeWithID(uint32(10000+i), vec)
	}
	idx.Train(trainingVectors)

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
				WithNProbes(10).
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
				WithNProbes(10).
				WithDocumentIDs(docIDs...).
				Execute()
		}
	})
}
