package comet

import (
	"testing"
)

// TestWeightedSumFusion_EqualWeights tests weighted sum fusion with equal weights
func TestWeightedSumFusion_EqualWeights(t *testing.T) {
	fusion, err := NewFusion(WeightedSumFusion, &FusionConfig{
		VectorWeight: 1.0,
		TextWeight:   1.0,
	})
	if err != nil {
		t.Fatalf("Failed to create fusion: %v", err)
	}

	vectorResults := map[uint32]float64{
		1: 0.5,
		2: 0.3,
		3: 0.8,
	}

	textResults := map[uint32]float64{
		1: 10.0,
		2: 20.0,
		4: 15.0,
	}

	combined := fusion.Combine(vectorResults, textResults)

	// Check doc 1: should have both scores
	if combined[1] != 10.5 { // 0.5 * 1.0 + 10.0 * 1.0
		t.Errorf("Doc 1: expected 10.5, got %f", combined[1])
	}

	// Check doc 2: should have both scores
	if combined[2] != 20.3 { // 0.3 * 1.0 + 20.0 * 1.0
		t.Errorf("Doc 2: expected 20.3, got %f", combined[2])
	}

	// Check doc 3: only in vector results
	if combined[3] != 0.8 { // 0.8 * 1.0
		t.Errorf("Doc 3: expected 0.8, got %f", combined[3])
	}

	// Check doc 4: only in text results
	if combined[4] != 15.0 { // 15.0 * 1.0
		t.Errorf("Doc 4: expected 15.0, got %f", combined[4])
	}
}

// TestWeightedSumFusion_CustomWeights tests weighted sum fusion with custom weights
func TestWeightedSumFusion_CustomWeights(t *testing.T) {
	fusion, err := NewFusion(WeightedSumFusion, &FusionConfig{
		VectorWeight: 2.0,
		TextWeight:   0.5,
	})
	if err != nil {
		t.Fatalf("Failed to create fusion: %v", err)
	}

	vectorResults := map[uint32]float64{
		1: 0.5,
	}

	textResults := map[uint32]float64{
		1: 10.0,
	}

	combined := fusion.Combine(vectorResults, textResults)

	// Check doc 1: 0.5 * 2.0 + 10.0 * 0.5 = 1.0 + 5.0 = 6.0
	expected := 6.0
	if combined[1] != expected {
		t.Errorf("Doc 1: expected %f, got %f", expected, combined[1])
	}
}

// TestWeightedSumFusion_VectorOnly tests weighted sum fusion with only vector results
func TestWeightedSumFusion_VectorOnly(t *testing.T) {
	fusion := DefaultFusion()

	vectorResults := map[uint32]float64{
		1: 0.5,
		2: 0.3,
	}

	textResults := map[uint32]float64{}

	combined := fusion.Combine(vectorResults, textResults)

	if len(combined) != 2 {
		t.Errorf("Expected 2 results, got %d", len(combined))
	}

	if combined[1] != 0.5 {
		t.Errorf("Doc 1: expected 0.5, got %f", combined[1])
	}
}

// TestWeightedSumFusion_TextOnly tests weighted sum fusion with only text results
func TestWeightedSumFusion_TextOnly(t *testing.T) {
	fusion := DefaultFusion()

	vectorResults := map[uint32]float64{}

	textResults := map[uint32]float64{
		1: 10.0,
		2: 20.0,
	}

	combined := fusion.Combine(vectorResults, textResults)

	if len(combined) != 2 {
		t.Errorf("Expected 2 results, got %d", len(combined))
	}

	if combined[1] != 10.0 {
		t.Errorf("Doc 1: expected 10.0, got %f", combined[1])
	}
}

// TestWeightedSumFusion_Empty tests weighted sum fusion with empty results
func TestWeightedSumFusion_Empty(t *testing.T) {
	fusion := DefaultFusion()

	vectorResults := map[uint32]float64{}
	textResults := map[uint32]float64{}

	combined := fusion.Combine(vectorResults, textResults)

	if len(combined) != 0 {
		t.Errorf("Expected 0 results, got %d", len(combined))
	}
}

// TestReciprocalRankFusion tests reciprocal rank fusion
func TestReciprocalRankFusion(t *testing.T) {
	fusion, err := NewFusion(ReciprocalRankFusion, &FusionConfig{
		K: 60.0,
	})
	if err != nil {
		t.Fatalf("Failed to create fusion: %v", err)
	}

	// Vector results: lower score = better (distances)
	vectorResults := map[uint32]float64{
		1: 0.1, // rank 0 (best)
		2: 0.3, // rank 1
		3: 0.5, // rank 2
	}

	// Text results: higher score = better (relevance)
	textResults := map[uint32]float64{
		1: 20.0, // rank 0 (best)
		2: 15.0, // rank 1
		4: 10.0, // rank 2
	}

	combined := fusion.Combine(vectorResults, textResults)

	// Doc 1 appears in both with best rank in both
	// RRF score = 1/(60+0) + 1/(60+0) = 1/60 + 1/60 = 0.0333...
	doc1Expected := 1.0/60.0 + 1.0/60.0

	// Doc 2 appears in both with rank 1 in both
	// RRF score = 1/(60+1) + 1/(60+1) = 1/61 + 1/61
	doc2Expected := 1.0/61.0 + 1.0/61.0

	// Doc 3 only in vector with rank 2
	// RRF score = 1/(60+2) = 1/62
	doc3Expected := 1.0 / 62.0

	// Doc 4 only in text with rank 2
	// RRF score = 1/(60+2) = 1/62
	doc4Expected := 1.0 / 62.0

	tolerance := 0.0001

	if abs(combined[1]-doc1Expected) > tolerance {
		t.Errorf("Doc 1: expected %f, got %f", doc1Expected, combined[1])
	}

	if abs(combined[2]-doc2Expected) > tolerance {
		t.Errorf("Doc 2: expected %f, got %f", doc2Expected, combined[2])
	}

	if abs(combined[3]-doc3Expected) > tolerance {
		t.Errorf("Doc 3: expected %f, got %f", doc3Expected, combined[3])
	}

	if abs(combined[4]-doc4Expected) > tolerance {
		t.Errorf("Doc 4: expected %f, got %f", doc4Expected, combined[4])
	}

	// Doc 1 should have highest score (appears in both with best ranks)
	if combined[1] <= combined[2] || combined[1] <= combined[3] || combined[1] <= combined[4] {
		t.Error("Doc 1 should have highest RRF score")
	}
}

// TestMaxFusion tests max fusion strategy
func TestMaxFusion(t *testing.T) {
	fusion, err := NewFusion(MaxFusion, nil)
	if err != nil {
		t.Fatalf("Failed to create fusion: %v", err)
	}

	vectorResults := map[uint32]float64{
		1: 0.5,
		2: 0.8, // higher vector score
		3: 0.3,
	}

	textResults := map[uint32]float64{
		1: 10.0, // higher text score
		2: 5.0,
		4: 15.0,
	}

	combined := fusion.Combine(vectorResults, textResults)

	// Doc 1: max(0.5, 10.0) = 10.0
	if combined[1] != 10.0 {
		t.Errorf("Doc 1: expected 10.0, got %f", combined[1])
	}

	// Doc 2: max(0.8, 5.0) = 5.0
	if combined[2] != 5.0 {
		t.Errorf("Doc 2: expected 5.0, got %f", combined[2])
	}

	// Doc 3: only vector = 0.3
	if combined[3] != 0.3 {
		t.Errorf("Doc 3: expected 0.3, got %f", combined[3])
	}

	// Doc 4: only text = 15.0
	if combined[4] != 15.0 {
		t.Errorf("Doc 4: expected 15.0, got %f", combined[4])
	}
}

// TestMinFusion tests min fusion strategy
func TestMinFusion(t *testing.T) {
	fusion, err := NewFusion(MinFusion, nil)
	if err != nil {
		t.Fatalf("Failed to create fusion: %v", err)
	}

	vectorResults := map[uint32]float64{
		1: 0.5, // lower
		2: 0.8,
		3: 0.3,
	}

	textResults := map[uint32]float64{
		1: 10.0,
		2: 5.0, // lower
		4: 15.0,
	}

	combined := fusion.Combine(vectorResults, textResults)

	// Doc 1: min(0.5, 10.0) = 0.5
	if combined[1] != 0.5 {
		t.Errorf("Doc 1: expected 0.5, got %f", combined[1])
	}

	// Doc 2: min(0.8, 5.0) = 0.8
	if combined[2] != 0.8 {
		t.Errorf("Doc 2: expected 0.8, got %f", combined[2])
	}

	// Doc 3: only in vector, should not appear in result
	if _, exists := combined[3]; exists {
		t.Error("Doc 3 should not appear (only in one result set)")
	}

	// Doc 4: only in text, should not appear in result
	if _, exists := combined[4]; exists {
		t.Error("Doc 4 should not appear (only in one result set)")
	}

	// Only docs that appear in both should be in result
	if len(combined) != 2 {
		t.Errorf("Expected 2 results, got %d", len(combined))
	}
}

// TestMinFusion_NoOverlap tests min fusion with no overlapping documents
func TestMinFusion_NoOverlap(t *testing.T) {
	fusion, err := NewFusion(MinFusion, nil)
	if err != nil {
		t.Fatalf("Failed to create fusion: %v", err)
	}

	vectorResults := map[uint32]float64{
		1: 0.5,
		2: 0.8,
	}

	textResults := map[uint32]float64{
		3: 10.0,
		4: 15.0,
	}

	combined := fusion.Combine(vectorResults, textResults)

	// No overlap, so result should be empty
	if len(combined) != 0 {
		t.Errorf("Expected 0 results with no overlap, got %d", len(combined))
	}
}

// TestFusionKind tests fusion kind retrieval
func TestFusionKind(t *testing.T) {
	tests := []struct {
		kind     FusionKind
		expected FusionKind
	}{
		{WeightedSumFusion, WeightedSumFusion},
		{ReciprocalRankFusion, ReciprocalRankFusion},
		{MaxFusion, MaxFusion},
		{MinFusion, MinFusion},
	}

	for _, tt := range tests {
		fusion, err := NewFusion(tt.kind, nil)
		if err != nil {
			t.Fatalf("Failed to create fusion %s: %v", tt.kind, err)
		}

		if fusion.Kind() != tt.expected {
			t.Errorf("Expected kind %s, got %s", tt.expected, fusion.Kind())
		}
	}
}

// TestNewFusion_InvalidKind tests creating fusion with invalid kind
func TestNewFusion_InvalidKind(t *testing.T) {
	_, err := NewFusion("invalid", nil)
	if err == nil {
		t.Error("Expected error for invalid fusion kind")
	}
}

// TestDefaultFusion tests the default fusion strategy
func TestDefaultFusion(t *testing.T) {
	fusion := DefaultFusion()

	if fusion.Kind() != WeightedSumFusion {
		t.Errorf("Expected default fusion to be WeightedSumFusion, got %s", fusion.Kind())
	}
}

// TestDefaultFusionConfig tests the default fusion configuration
func TestDefaultFusionConfig(t *testing.T) {
	config := DefaultFusionConfig()

	if config.VectorWeight != 1.0 {
		t.Errorf("Expected VectorWeight 1.0, got %f", config.VectorWeight)
	}

	if config.TextWeight != 1.0 {
		t.Errorf("Expected TextWeight 1.0, got %f", config.TextWeight)
	}

	if config.K != 60.0 {
		t.Errorf("Expected K 60.0, got %f", config.K)
	}
}

// TestNewFusion_NilConfig tests creating fusion with nil config uses defaults
func TestNewFusion_NilConfig(t *testing.T) {
	fusion, err := NewFusion(WeightedSumFusion, nil)
	if err != nil {
		t.Fatalf("Failed to create fusion: %v", err)
	}

	// Should use default config
	vectorResults := map[uint32]float64{1: 1.0}
	textResults := map[uint32]float64{1: 2.0}

	combined := fusion.Combine(vectorResults, textResults)

	// With default weights (1.0, 1.0): 1.0 * 1.0 + 2.0 * 1.0 = 3.0
	if combined[1] != 3.0 {
		t.Errorf("Expected 3.0, got %f", combined[1])
	}
}

// TestScoreMapToRanks_Ascending tests rank conversion for distances (lower is better)
func TestScoreMapToRanks_Ascending(t *testing.T) {
	scores := map[uint32]float64{
		1: 0.1, // should be rank 0 (best)
		2: 0.5, // should be rank 2
		3: 0.3, // should be rank 1
	}

	ranks := scoreMapToRanks(scores, true)

	if ranks[1] != 0 {
		t.Errorf("Doc 1: expected rank 0, got %d", ranks[1])
	}

	if ranks[2] != 2 {
		t.Errorf("Doc 2: expected rank 2, got %d", ranks[2])
	}

	if ranks[3] != 1 {
		t.Errorf("Doc 3: expected rank 1, got %d", ranks[3])
	}
}

// TestScoreMapToRanks_Descending tests rank conversion for relevance (higher is better)
func TestScoreMapToRanks_Descending(t *testing.T) {
	scores := map[uint32]float64{
		1: 10.0, // should be rank 1
		2: 5.0,  // should be rank 2
		3: 20.0, // should be rank 0 (best)
	}

	ranks := scoreMapToRanks(scores, false)

	if ranks[1] != 1 {
		t.Errorf("Doc 1: expected rank 1, got %d", ranks[1])
	}

	if ranks[2] != 2 {
		t.Errorf("Doc 2: expected rank 2, got %d", ranks[2])
	}

	if ranks[3] != 0 {
		t.Errorf("Doc 3: expected rank 0, got %d", ranks[3])
	}
}

// TestScoreMapToRanks_Empty tests rank conversion with empty map
func TestScoreMapToRanks_Empty(t *testing.T) {
	scores := map[uint32]float64{}
	ranks := scoreMapToRanks(scores, true)

	if len(ranks) != 0 {
		t.Errorf("Expected empty ranks, got %d entries", len(ranks))
	}
}

// Helper function for floating point comparison
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// TestReciprocalRankFusion_CustomK tests RRF with custom K value
func TestReciprocalRankFusion_CustomK(t *testing.T) {
	// Test with smaller K (gives more weight to top ranks)
	fusion, err := NewFusion(ReciprocalRankFusion, &FusionConfig{
		K: 10.0,
	})
	if err != nil {
		t.Fatalf("Failed to create fusion: %v", err)
	}

	vectorResults := map[uint32]float64{
		1: 0.1, // rank 0
	}

	textResults := map[uint32]float64{}

	combined := fusion.Combine(vectorResults, textResults)

	// With K=10, rank 0: 1/(10+0) = 0.1
	expected := 1.0 / 10.0
	if abs(combined[1]-expected) > 0.0001 {
		t.Errorf("Expected %f, got %f", expected, combined[1])
	}
}

// TestWeightedSumFusion_ZeroWeights tests weighted sum with zero weights
func TestWeightedSumFusion_ZeroWeights(t *testing.T) {
	// Only vector weight, text weight is zero
	fusion, err := NewFusion(WeightedSumFusion, &FusionConfig{
		VectorWeight: 1.0,
		TextWeight:   0.0,
	})
	if err != nil {
		t.Fatalf("Failed to create fusion: %v", err)
	}

	vectorResults := map[uint32]float64{
		1: 0.5,
	}

	textResults := map[uint32]float64{
		1: 100.0, // This should be ignored
	}

	combined := fusion.Combine(vectorResults, textResults)

	// Should only have vector contribution: 0.5 * 1.0 + 100.0 * 0.0 = 0.5
	if combined[1] != 0.5 {
		t.Errorf("Expected 0.5, got %f", combined[1])
	}
}
