package comet

import (
	"testing"
)

// TestMergeResults tests result merging and deduplication.
func TestMergeResults(t *testing.T) {
	tests := []struct {
		name     string
		input    []HybridSearchResult
		expected map[uint32]float64 // ID -> expected score
	}{
		{
			name: "no duplicates",
			input: []HybridSearchResult{
				{ID: 1, Score: 0.5},
				{ID: 2, Score: 0.8},
				{ID: 3, Score: 0.3},
			},
			expected: map[uint32]float64{
				1: 0.5,
				2: 0.8,
				3: 0.3,
			},
		},
		{
			name: "duplicates with different scores",
			input: []HybridSearchResult{
				{ID: 1, Score: 0.5},
				{ID: 2, Score: 0.8},
				{ID: 1, Score: 0.9}, // Higher score for ID 1
				{ID: 3, Score: 0.3},
				{ID: 2, Score: 0.6}, // Lower score for ID 2
			},
			expected: map[uint32]float64{
				1: 0.9, // Highest score kept
				2: 0.8, // Highest score kept
				3: 0.3,
			},
		},
		{
			name: "multiple duplicates",
			input: []HybridSearchResult{
				{ID: 1, Score: 0.1},
				{ID: 1, Score: 0.5},
				{ID: 1, Score: 0.9},
				{ID: 1, Score: 0.3},
			},
			expected: map[uint32]float64{
				1: 0.9, // Highest of all
			},
		},
		{
			name:     "empty input",
			input:    []HybridSearchResult{},
			expected: map[uint32]float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			merged := mergeResults(tt.input)

			if len(merged) != len(tt.expected) {
				t.Errorf("expected %d unique results, got %d", len(tt.expected), len(merged))
			}

			// Verify highest scores are kept
			scoreMap := make(map[uint32]float64)
			for _, result := range merged {
				scoreMap[result.ID] = result.Score
			}

			for id, expectedScore := range tt.expected {
				actualScore, found := scoreMap[id]
				if !found {
					t.Errorf("expected to find ID %d in merged results", id)
					continue
				}
				if actualScore != expectedScore {
					t.Errorf("for ID %d: expected score %f, got %f", id, expectedScore, actualScore)
				}
			}
		})
	}
}

// TestMergeResults_Nil tests merging nil/empty results.
func TestMergeResults_Nil(t *testing.T) {
	merged := mergeResults(nil)
	if merged != nil {
		t.Error("merging nil should return nil")
	}

	merged = mergeResults([]HybridSearchResult{})
	if merged != nil {
		t.Error("merging empty slice should return nil")
	}
}

// TestSortResultsByScore tests result sorting.
func TestSortResultsByScore(t *testing.T) {
	tests := []struct {
		name     string
		input    []HybridSearchResult
		expected []uint32 // Expected order of IDs (highest score first)
	}{
		{
			name: "unsorted results",
			input: []HybridSearchResult{
				{ID: 1, Score: 0.5},
				{ID: 2, Score: 0.9},
				{ID: 3, Score: 0.3},
				{ID: 4, Score: 0.7},
			},
			expected: []uint32{2, 4, 1, 3},
		},
		{
			name: "already sorted",
			input: []HybridSearchResult{
				{ID: 1, Score: 0.9},
				{ID: 2, Score: 0.7},
				{ID: 3, Score: 0.5},
			},
			expected: []uint32{1, 2, 3},
		},
		{
			name: "reverse sorted",
			input: []HybridSearchResult{
				{ID: 1, Score: 0.1},
				{ID: 2, Score: 0.5},
				{ID: 3, Score: 0.9},
			},
			expected: []uint32{3, 2, 1},
		},
		{
			name: "equal scores",
			input: []HybridSearchResult{
				{ID: 1, Score: 0.5},
				{ID: 2, Score: 0.5},
				{ID: 3, Score: 0.5},
			},
			expected: []uint32{1, 2, 3}, // Order doesn't matter for equal scores
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results := make([]HybridSearchResult, len(tt.input))
			copy(results, tt.input)

			sortResultsByScore(results)

			// Verify descending order
			for i := 1; i < len(results); i++ {
				if results[i-1].Score < results[i].Score {
					t.Errorf("results not sorted: %f should be >= %f at positions %d, %d",
						results[i-1].Score, results[i].Score, i-1, i)
				}
			}

			// Verify expected order (for non-equal scores)
			if tt.name != "equal scores" {
				for i, expectedID := range tt.expected {
					if results[i].ID != expectedID {
						t.Errorf("at position %d: expected ID %d, got %d", i, expectedID, results[i].ID)
					}
				}
			}
		})
	}
}

// TestSortResultsByScore_Empty tests sorting empty results.
func TestSortResultsByScore_Empty(t *testing.T) {
	results := []HybridSearchResult{}
	sortResultsByScore(results)
	// Should not panic
}

// TestSortResultsByScore_Single tests sorting single result.
func TestSortResultsByScore_Single(t *testing.T) {
	results := []HybridSearchResult{{ID: 1, Score: 0.5}}
	sortResultsByScore(results)

	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
	if results[0].ID != 1 {
		t.Errorf("expected ID 1, got %d", results[0].ID)
	}
}

// BenchmarkMergeResults benchmarks result merging.
func BenchmarkMergeResults(b *testing.B) {
	// Create test data with duplicates
	results := make([]HybridSearchResult, 1000)
	for i := 0; i < 1000; i++ {
		results[i] = HybridSearchResult{
			ID:    uint32(i % 100), // 10x duplication
			Score: float64(i) / 1000.0,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mergeResults(results)
	}
}

// BenchmarkSortResultsByScore benchmarks result sorting.
func BenchmarkSortResultsByScore(b *testing.B) {
	// Create test data
	results := make([]HybridSearchResult, 1000)
	for i := 0; i < 1000; i++ {
		results[i] = HybridSearchResult{
			ID:    uint32(i),
			Score: float64(1000-i) / 1000.0,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Make a copy for each iteration
		testResults := make([]HybridSearchResult, len(results))
		copy(testResults, results)
		sortResultsByScore(testResults)
	}
}
