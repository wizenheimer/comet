package comet

import (
	"testing"
)

func TestSumAggregation(t *testing.T) {
	// Create test nodes
	node1 := NewVectorNodeWithID(1, []float32{1.0, 2.0})
	node2 := NewVectorNodeWithID(2, []float32{3.0, 4.0})
	node3 := NewVectorNodeWithID(3, []float32{5.0, 6.0})

	// Create results with duplicates
	results := []VectorResult{
		{Node: *node1, Score: 0.1},
		{Node: *node2, Score: 0.2},
		{Node: *node1, Score: 0.15}, // Duplicate node1
		{Node: *node3, Score: 0.3},
		{Node: *node1, Score: 0.05}, // Another duplicate node1
	}

	agg, _ := GetScoreAggregation(SumAggregation)
	aggregated := agg.Aggregate(results)

	// Should have 3 unique nodes
	if len(aggregated) != 3 {
		t.Errorf("Expected 3 unique nodes, got %d", len(aggregated))
	}

	// Verify node1 has summed score: 0.1 + 0.15 + 0.05 = 0.3
	found := false
	for _, result := range aggregated {
		if result.Node.ID() == 1 {
			found = true
			expected := float32(0.3)
			if result.Score != expected {
				t.Errorf("Expected node 1 score to be %f, got %f", expected, result.Score)
			}
		}
	}
	if !found {
		t.Error("Node 1 not found in aggregated results")
	}

	// Verify results are sorted by score (ascending)
	for i := 1; i < len(aggregated); i++ {
		if aggregated[i].Score < aggregated[i-1].Score {
			t.Error("Results are not sorted by score")
		}
	}
}

func TestMaxAggregation(t *testing.T) {
	// Create test nodes
	node1 := NewVectorNodeWithID(1, []float32{1.0, 2.0})
	node2 := NewVectorNodeWithID(2, []float32{3.0, 4.0})

	// Create results with duplicates
	results := []VectorResult{
		{Node: *node1, Score: 0.1},
		{Node: *node2, Score: 0.2},
		{Node: *node1, Score: 0.5}, // Max for node1
		{Node: *node1, Score: 0.15},
	}

	agg, _ := GetScoreAggregation(MaxAggregation)
	aggregated := agg.Aggregate(results)

	// Should have 2 unique nodes
	if len(aggregated) != 2 {
		t.Errorf("Expected 2 unique nodes, got %d", len(aggregated))
	}

	// Verify node1 has max score: 0.5
	for _, result := range aggregated {
		if result.Node.ID() == 1 {
			expected := float32(0.5)
			if result.Score != expected {
				t.Errorf("Expected node 1 score to be %f, got %f", expected, result.Score)
			}
		}
	}
}

func TestMeanAggregation(t *testing.T) {
	// Create test nodes
	node1 := NewVectorNodeWithID(1, []float32{1.0, 2.0})
	node2 := NewVectorNodeWithID(2, []float32{3.0, 4.0})

	// Create results with duplicates
	results := []VectorResult{
		{Node: *node1, Score: 0.1},
		{Node: *node2, Score: 0.2},
		{Node: *node1, Score: 0.2},
		{Node: *node1, Score: 0.3},
	}

	agg, _ := GetScoreAggregation(MeanAggregation)
	aggregated := agg.Aggregate(results)

	// Should have 2 unique nodes
	if len(aggregated) != 2 {
		t.Errorf("Expected 2 unique nodes, got %d", len(aggregated))
	}

	// Verify node1 has average score: (0.1 + 0.2 + 0.3) / 3 = 0.2
	for _, result := range aggregated {
		if result.Node.ID() == 1 {
			expected := float32(0.2)
			if result.Score != expected {
				t.Errorf("Expected node 1 score to be %f, got %f", expected, result.Score)
			}
		}
	}
}

func TestAggregationEmptyResults(t *testing.T) {
	results := []VectorResult{}

	kinds := []ScoreAggregationKind{
		SumAggregation,
		MaxAggregation,
		MeanAggregation,
	}

	for _, kind := range kinds {
		agg, _ := GetScoreAggregation(kind)
		aggregated := agg.Aggregate(results)
		if len(aggregated) != 0 {
			t.Errorf("%s: Expected empty results for empty input", kind)
		}
	}
}

func TestAggregationSingleResult(t *testing.T) {
	node1 := NewVectorNodeWithID(1, []float32{1.0, 2.0})
	results := []VectorResult{
		{Node: *node1, Score: 0.5},
	}

	kinds := []ScoreAggregationKind{
		SumAggregation,
		MaxAggregation,
		MeanAggregation,
	}

	for _, kind := range kinds {
		agg, _ := GetScoreAggregation(kind)
		aggregated := agg.Aggregate(results)
		if len(aggregated) != 1 {
			t.Errorf("%s: Expected 1 result", kind)
		}
		if aggregated[0].Score != 0.5 {
			t.Errorf("%s: Expected score 0.5, got %f", kind, aggregated[0].Score)
		}
	}
}

func TestGetScoreAggregation(t *testing.T) {
	// Test valid kinds
	agg, err := GetScoreAggregation(SumAggregation)
	if err != nil || agg == nil {
		t.Error("Expected valid SumAggregation")
	}
	if agg.Kind() != SumAggregation {
		t.Errorf("Expected SumAggregation, got %s", agg.Kind())
	}

	agg, err = GetScoreAggregation(MaxAggregation)
	if err != nil || agg == nil {
		t.Error("Expected valid MaxAggregation")
	}

	agg, err = GetScoreAggregation(MeanAggregation)
	if err != nil || agg == nil {
		t.Error("Expected valid MeanAggregation")
	}

	// Test invalid kind
	agg, err = GetScoreAggregation("invalid")
	if err == nil {
		t.Error("Expected error for invalid aggregation kind")
	}
	if agg != nil {
		t.Error("Expected nil aggregation for invalid kind")
	}
}

func TestDefaultScoreAggregation(t *testing.T) {
	agg := DefaultScoreAggregation()
	if agg == nil {
		t.Error("Expected non-nil default aggregation")
	}
	if agg.Kind() != SumAggregation {
		t.Errorf("Expected default to be SumAggregation, got %s", agg.Kind())
	}
}

func TestNewAggregation(t *testing.T) {
	// Test NewAggregation function
	agg, err := NewAggregation(SumAggregation)
	if err != nil || agg == nil {
		t.Error("Expected valid SumAggregation from NewAggregation")
	}
	if agg.Kind() != SumAggregation {
		t.Errorf("Expected SumAggregation, got %s", agg.Kind())
	}

	agg, err = NewAggregation(MaxAggregation)
	if err != nil || agg == nil {
		t.Error("Expected valid MaxAggregation from NewAggregation")
	}

	agg, err = NewAggregation(MeanAggregation)
	if err != nil || agg == nil {
		t.Error("Expected valid MeanAggregation from NewAggregation")
	}

	// Test invalid kind
	agg, err = NewAggregation("invalid")
	if err == nil {
		t.Error("Expected error for invalid aggregation kind")
	}
	if agg != nil {
		t.Error("Expected nil aggregation for invalid kind")
	}
}

func TestScoreAggregationKindConstants(t *testing.T) {
	// Verify the constant values are as expected
	if SumAggregation != "sum" {
		t.Errorf("Expected SumAggregation to be 'sum', got %s", SumAggregation)
	}
	if MaxAggregation != "max" {
		t.Errorf("Expected MaxAggregation to be 'max', got %s", MaxAggregation)
	}
	if MeanAggregation != "mean" {
		t.Errorf("Expected MeanAggregation to be 'mean', got %s", MeanAggregation)
	}
}
