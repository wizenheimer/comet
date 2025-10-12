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

	agg, _ := NewVectorAggregation(SumAggregation)
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

	agg, _ := NewVectorAggregation(MaxAggregation)
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

	agg, _ := NewVectorAggregation(MeanAggregation)
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
		agg, _ := NewVectorAggregation(kind)
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
		agg, _ := NewVectorAggregation(kind)
		aggregated := agg.Aggregate(results)
		if len(aggregated) != 1 {
			t.Errorf("%s: Expected 1 result", kind)
		}
		if aggregated[0].Score != 0.5 {
			t.Errorf("%s: Expected score 0.5, got %f", kind, aggregated[0].Score)
		}
	}
}

func TestNewVectorAggregation(t *testing.T) {
	// Test valid kinds
	agg, err := NewVectorAggregation(SumAggregation)
	if err != nil || agg == nil {
		t.Error("Expected valid SumAggregation")
	}
	if agg.Kind() != SumAggregation {
		t.Errorf("Expected SumAggregation, got %s", agg.Kind())
	}

	agg, err = NewVectorAggregation(MaxAggregation)
	if err != nil || agg == nil {
		t.Error("Expected valid MaxAggregation")
	}

	agg, err = NewVectorAggregation(MeanAggregation)
	if err != nil || agg == nil {
		t.Error("Expected valid MeanAggregation")
	}

	// Test invalid kind
	agg, err = NewVectorAggregation("invalid")
	if err == nil {
		t.Error("Expected error for invalid aggregation kind")
	}
	if agg != nil {
		t.Error("Expected nil aggregation for invalid kind")
	}
}

func TestNewTextAggregation(t *testing.T) {
	// Test valid kinds
	agg, err := NewTextAggregation(SumAggregation)
	if err != nil || agg == nil {
		t.Error("Expected valid SumAggregation")
	}
	if agg.Kind() != SumAggregation {
		t.Errorf("Expected SumAggregation, got %s", agg.Kind())
	}

	agg, err = NewTextAggregation(MaxAggregation)
	if err != nil || agg == nil {
		t.Error("Expected valid MaxAggregation")
	}

	agg, err = NewTextAggregation(MeanAggregation)
	if err != nil || agg == nil {
		t.Error("Expected valid MeanAggregation")
	}

	// Test invalid kind
	agg, err = NewTextAggregation("invalid")
	if err == nil {
		t.Error("Expected error for invalid aggregation kind")
	}
	if agg != nil {
		t.Error("Expected nil aggregation for invalid kind")
	}
}

func TestDefaultVectorAggregation(t *testing.T) {
	agg := DefaultVectorAggregation()
	if agg == nil {
		t.Error("Expected non-nil default aggregation")
	}
	if agg.Kind() != SumAggregation {
		t.Errorf("Expected default to be SumAggregation, got %s", agg.Kind())
	}
}

func TestDefaultTextAggregation(t *testing.T) {
	agg := DefaultTextAggregation()
	if agg == nil {
		t.Error("Expected non-nil default aggregation")
	}
	if agg.Kind() != SumAggregation {
		t.Errorf("Expected default to be SumAggregation, got %s", agg.Kind())
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

// ============================================================================
// TEXT AGGREGATION INTERFACE TESTS
// ============================================================================

func TestTextAggregationSum(t *testing.T) {
	// Create results with duplicates (note: higher scores are better for text)
	results := []TextResult{
		{Id: 1, Score: 1.5},
		{Id: 2, Score: 2.0},
		{Id: 1, Score: 1.8}, // Duplicate doc 1
		{Id: 3, Score: 3.0},
		{Id: 1, Score: 1.2}, // Another duplicate doc 1
	}

	agg, _ := NewTextAggregation(SumAggregation)
	aggregated := agg.Aggregate(results)

	// Should have 3 unique documents
	if len(aggregated) != 3 {
		t.Errorf("Expected 3 unique documents, got %d", len(aggregated))
	}

	// Verify doc 1 has summed score: 1.5 + 1.8 + 1.2 = 4.5
	found := false
	for _, result := range aggregated {
		if result.Id == 1 {
			found = true
			expected := float32(4.5)
			if result.Score != expected {
				t.Errorf("Expected doc 1 score to be %f, got %f", expected, result.Score)
			}
		}
	}
	if !found {
		t.Error("Doc 1 not found in aggregated results")
	}

	// Verify results are sorted by score (descending - higher is better)
	for i := 1; i < len(aggregated); i++ {
		if aggregated[i].Score > aggregated[i-1].Score {
			t.Error("Results are not sorted by score descending")
		}
	}
}

func TestTextAggregationMax(t *testing.T) {
	// Create results with duplicates
	results := []TextResult{
		{Id: 1, Score: 1.5},
		{Id: 2, Score: 2.0},
		{Id: 1, Score: 3.5}, // Max for doc 1
		{Id: 1, Score: 2.2},
	}

	agg, _ := NewTextAggregation(MaxAggregation)
	aggregated := agg.Aggregate(results)

	// Should have 2 unique documents
	if len(aggregated) != 2 {
		t.Errorf("Expected 2 unique documents, got %d", len(aggregated))
	}

	// Verify doc 1 has max score: 3.5
	for _, result := range aggregated {
		if result.Id == 1 {
			expected := float32(3.5)
			if result.Score != expected {
				t.Errorf("Expected doc 1 score to be %f, got %f", expected, result.Score)
			}
		}
	}
}

func TestTextAggregationMean(t *testing.T) {
	// Create results with duplicates
	results := []TextResult{
		{Id: 1, Score: 1.5},
		{Id: 2, Score: 2.0},
		{Id: 1, Score: 2.1},
		{Id: 1, Score: 2.4},
	}

	agg, _ := NewTextAggregation(MeanAggregation)
	aggregated := agg.Aggregate(results)

	// Should have 2 unique documents
	if len(aggregated) != 2 {
		t.Errorf("Expected 2 unique documents, got %d", len(aggregated))
	}

	// Verify doc 1 has average score: (1.5 + 2.1 + 2.4) / 3 = 2.0
	for _, result := range aggregated {
		if result.Id == 1 {
			expected := float32(2.0)
			if result.Score != expected {
				t.Errorf("Expected doc 1 score to be %f, got %f", expected, result.Score)
			}
		}
	}
}

func TestTextAggregationEmptyResults(t *testing.T) {
	results := []TextResult{}

	kinds := []ScoreAggregationKind{
		SumAggregation,
		MaxAggregation,
		MeanAggregation,
	}

	for _, kind := range kinds {
		agg, _ := NewTextAggregation(kind)
		aggregated := agg.Aggregate(results)
		if len(aggregated) != 0 {
			t.Errorf("%s: Expected empty results for empty input", kind)
		}
	}
}

func TestTextAggregationSingleResult(t *testing.T) {
	results := []TextResult{
		{Id: 1, Score: 2.5},
	}

	kinds := []ScoreAggregationKind{
		SumAggregation,
		MaxAggregation,
		MeanAggregation,
	}

	for _, kind := range kinds {
		agg, _ := NewTextAggregation(kind)
		aggregated := agg.Aggregate(results)
		if len(aggregated) != 1 {
			t.Errorf("%s: Expected 1 result", kind)
		}
		if aggregated[0].Score != 2.5 {
			t.Errorf("%s: Expected score 2.5, got %f", kind, aggregated[0].Score)
		}
	}
}
