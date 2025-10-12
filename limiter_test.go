package comet

import (
	"testing"
)

func TestSanitizeK(t *testing.T) {
	tests := []struct {
		name       string
		k          int
		maxResults int
		want       int
	}{
		{
			name:       "k is zero",
			k:          0,
			maxResults: 10,
			want:       10,
		},
		{
			name:       "k is negative",
			k:          -5,
			maxResults: 10,
			want:       10,
		},
		{
			name:       "k exceeds maxResults",
			k:          100,
			maxResults: 10,
			want:       10,
		},
		{
			name:       "k is within bounds",
			k:          5,
			maxResults: 10,
			want:       5,
		},
		{
			name:       "k equals maxResults",
			k:          10,
			maxResults: 10,
			want:       10,
		},
		{
			name:       "maxResults is zero",
			k:          5,
			maxResults: 0,
			want:       0,
		},
		{
			name:       "both zero",
			k:          0,
			maxResults: 0,
			want:       0,
		},
		{
			name:       "k is 1",
			k:          1,
			maxResults: 10,
			want:       1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sanitizeK(tt.k, tt.maxResults)
			if got != tt.want {
				t.Errorf("sanitizeK(%d, %d) = %d, want %d",
					tt.k, tt.maxResults, got, tt.want)
			}
		})
	}
}

func TestLimitResults(t *testing.T) {
	// Create test results
	createResults := func(count int) []VectorResult {
		results := make([]VectorResult, count)
		for i := 0; i < count; i++ {
			results[i] = VectorResult{
				Node:  *NewVectorNodeWithID(uint32(i), []float32{float32(i)}),
				Score: float32(i),
			}
		}
		return results
	}

	tests := []struct {
		name        string
		resultsSize int
		k           int
		wantSize    int
	}{
		{
			name:        "k is zero - returns all",
			resultsSize: 10,
			k:           0,
			wantSize:    10,
		},
		{
			name:        "k is negative - returns all",
			resultsSize: 10,
			k:           -5,
			wantSize:    10,
		},
		{
			name:        "k exceeds results - returns all",
			resultsSize: 5,
			k:           10,
			wantSize:    5,
		},
		{
			name:        "k within bounds - returns k",
			resultsSize: 10,
			k:           5,
			wantSize:    5,
		},
		{
			name:        "k equals results size",
			resultsSize: 10,
			k:           10,
			wantSize:    10,
		},
		{
			name:        "empty results",
			resultsSize: 0,
			k:           5,
			wantSize:    0,
		},
		{
			name:        "k is 1",
			resultsSize: 10,
			k:           1,
			wantSize:    1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results := createResults(tt.resultsSize)
			got := limitResults(results, tt.k)

			if len(got) != tt.wantSize {
				t.Errorf("limitResults() returned %d results, want %d",
					len(got), tt.wantSize)
			}

			// Verify that the returned results are the first k elements
			for i := 0; i < len(got); i++ {
				if got[i].Node.ID() != uint32(i) {
					t.Errorf("limitResults()[%d].Node.ID() = %d, want %d",
						i, got[i].Node.ID(), uint32(i))
				}
			}
		})
	}
}

func TestLimitResultsPreservesOrder(t *testing.T) {
	// Create test results with specific IDs
	results := []VectorResult{
		{Node: *NewVectorNodeWithID(100, []float32{1.0}), Score: 1.0},
		{Node: *NewVectorNodeWithID(200, []float32{2.0}), Score: 2.0},
		{Node: *NewVectorNodeWithID(300, []float32{3.0}), Score: 3.0},
		{Node: *NewVectorNodeWithID(400, []float32{4.0}), Score: 4.0},
		{Node: *NewVectorNodeWithID(500, []float32{5.0}), Score: 5.0},
	}

	limited := limitResults(results, 3)

	// Verify we got exactly 3 results
	if len(limited) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(limited))
	}

	// Verify order is preserved
	expectedIDs := []uint32{100, 200, 300}
	for i, result := range limited {
		if result.Node.ID() != expectedIDs[i] {
			t.Errorf("Result[%d] ID = %d, want %d", i, result.Node.ID(), expectedIDs[i])
		}
	}
}

func TestAutocut(t *testing.T) {
	tests := []struct {
		name     string
		scores   []float32
		cutoff   int
		expected int
	}{
		{
			name:     "empty slice",
			scores:   []float32{},
			cutoff:   1,
			expected: 0,
		},
		{
			name:     "single element",
			scores:   []float32{1.0},
			cutoff:   1,
			expected: 1,
		},
		{
			name:     "two elements",
			scores:   []float32{1.0, 2.0},
			cutoff:   1,
			expected: 2,
		},
		{
			name:     "linear distribution - no clear cutoff",
			scores:   []float32{0.1, 0.2, 0.3, 0.4, 0.5},
			cutoff:   1,
			expected: 2, // Algorithm finds extremum at index 2
		},
		{
			name:     "clear gap after first few results",
			scores:   []float32{0.1, 0.15, 0.2, 0.5, 0.6, 0.7, 0.8},
			cutoff:   1,
			expected: 3, // Should cut after the gap
		},
		{
			name:     "cluster with outliers",
			scores:   []float32{0.1, 0.12, 0.13, 0.14, 0.15, 0.8, 0.9, 1.0},
			cutoff:   1,
			expected: 5, // Should keep the tight cluster
		},
		{
			name:     "cutoff 2 - find second extremum",
			scores:   []float32{0.1, 0.2, 0.4, 0.45, 0.7, 0.75, 0.9, 1.0},
			cutoff:   2,
			expected: 4, // Should find second extremum
		},
		{
			name:     "cutoff higher than extrema count",
			scores:   []float32{0.1, 0.2, 0.5, 0.6},
			cutoff:   5,
			expected: 4, // Returns all when not enough extrema
		},
		{
			name:     "all same values",
			scores:   []float32{0.5, 0.5, 0.5, 0.5, 0.5},
			cutoff:   1,
			expected: 5, // No extrema in flat distribution
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Autocut(tt.scores, tt.cutoff)
			if got != tt.expected {
				t.Errorf("Autocut() = %d, want %d", got, tt.expected)
			}
		})
	}
}

func TestAutocutResults(t *testing.T) {
	// Helper to create results
	createResults := func(scores []float32) []VectorResult {
		results := make([]VectorResult, len(scores))
		for i, score := range scores {
			results[i] = VectorResult{
				Node:  *NewVectorNodeWithID(uint32(i), []float32{score}),
				Score: score,
			}
		}
		return results
	}

	tests := []struct {
		name         string
		scores       []float32
		cutoff       int
		expectedSize int
	}{
		{
			name:         "cutoff -1 returns all (no-op)",
			scores:       []float32{0.1, 0.2, 0.3, 0.4, 0.5},
			cutoff:       -1,
			expectedSize: 5,
		},
		{
			name:         "cutoff -1 with clear gap (still no-op)",
			scores:       []float32{0.1, 0.15, 0.2, 0.9, 1.0},
			cutoff:       -1,
			expectedSize: 5,
		},
		{
			name:         "empty results with cutoff -1",
			scores:       []float32{},
			cutoff:       -1,
			expectedSize: 0,
		},
		{
			name:         "empty results with cutoff 1",
			scores:       []float32{},
			cutoff:       1,
			expectedSize: 0,
		},
		{
			name:         "cutoff 1 finds gap",
			scores:       []float32{0.1, 0.15, 0.2, 0.8, 0.9, 1.0},
			cutoff:       1,
			expectedSize: 3,
		},
		{
			name:         "cutoff 1 with tight cluster",
			scores:       []float32{0.1, 0.11, 0.12, 0.13, 0.14, 0.9},
			cutoff:       1,
			expectedSize: 5,
		},
		{
			name:         "single result",
			scores:       []float32{0.5},
			cutoff:       1,
			expectedSize: 1,
		},
		{
			name:         "cutoff 2 finds second extremum",
			scores:       []float32{0.1, 0.2, 0.4, 0.45, 0.7, 0.75, 0.9, 1.0},
			cutoff:       2,
			expectedSize: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results := createResults(tt.scores)
			got := autocutResults(results, tt.cutoff)

			if len(got) != tt.expectedSize {
				t.Errorf("autocutResults() returned %d results, want %d",
					len(got), tt.expectedSize)
			}

			// Verify that results are preserved in order
			for i := 0; i < len(got); i++ {
				if got[i].Node.ID() != uint32(i) {
					t.Errorf("autocutResults()[%d].Node.ID() = %d, want %d",
						i, got[i].Node.ID(), uint32(i))
				}
				if got[i].Score != tt.scores[i] {
					t.Errorf("autocutResults()[%d].Score = %f, want %f",
						i, got[i].Score, tt.scores[i])
				}
			}
		})
	}
}

func TestAutocutResultsPreservesOrder(t *testing.T) {
	// Create results with specific IDs to verify order preservation
	results := []VectorResult{
		{Node: *NewVectorNodeWithID(100, []float32{0.1}), Score: 0.1},
		{Node: *NewVectorNodeWithID(200, []float32{0.15}), Score: 0.15},
		{Node: *NewVectorNodeWithID(300, []float32{0.2}), Score: 0.2},
		{Node: *NewVectorNodeWithID(400, []float32{0.8}), Score: 0.8},
		{Node: *NewVectorNodeWithID(500, []float32{0.9}), Score: 0.9},
	}

	cut := autocutResults(results, 1)

	// Should cut after the gap (first 3 results)
	if len(cut) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(cut))
	}

	// Verify order and IDs are preserved
	expectedIDs := []uint32{100, 200, 300}
	expectedScores := []float32{0.1, 0.15, 0.2}

	for i, result := range cut {
		if result.Node.ID() != expectedIDs[i] {
			t.Errorf("Result[%d] ID = %d, want %d", i, result.Node.ID(), expectedIDs[i])
		}
		if result.Score != expectedScores[i] {
			t.Errorf("Result[%d] Score = %f, want %f", i, result.Score, expectedScores[i])
		}
	}
}

func TestAutocutWithRealWorldScores(t *testing.T) {
	// Simulate real-world distance scores where closer items have lower scores
	// and there's a natural cluster boundary
	tests := []struct {
		name        string
		scores      []float32
		cutoff      int
		expectedMin int
		expectedMax int
		description string
	}{
		{
			name: "tight cluster then outliers",
			scores: []float32{
				0.05, 0.06, 0.07, 0.08, 0.09, // tight cluster of 5
				0.5, 0.6, 0.7, 0.8, 0.9, // outliers
			},
			cutoff:      1,
			expectedMin: 8,
			expectedMax: 10,
			description: "finds extremum in distribution",
		},
		{
			name: "gradual increase",
			scores: []float32{
				0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
			},
			cutoff:      1,
			expectedMin: 2,
			expectedMax: 3,
			description: "finds first extremum",
		},
		{
			name: "two clusters",
			scores: []float32{
				0.1, 0.12, 0.14, // first cluster
				0.5, 0.52, 0.54, // second cluster
				0.9, 0.92, // third cluster
			},
			cutoff:      1,
			expectedMin: 3,
			expectedMax: 4,
			description: "cut after first cluster",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Autocut(tt.scores, tt.cutoff)
			if got < tt.expectedMin || got > tt.expectedMax {
				t.Errorf("Autocut() = %d, want between %d and %d (%s)",
					got, tt.expectedMin, tt.expectedMax, tt.description)
			}
		})
	}
}
