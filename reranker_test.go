package comet

import (
	"sort"
	"testing"
)

// ExampleReranker demonstrates a simple reranker that reverses result order
type ExampleReranker struct{}

func (r *ExampleReranker) Rerank(results []VectorResult) []VectorResult {
	// Simple example: reverse the order of results
	reranked := make([]VectorResult, len(results))
	copy(reranked, results)

	// Reverse the slice
	for i, j := 0, len(reranked)-1; i < j; i, j = i+1, j-1 {
		reranked[i], reranked[j] = reranked[j], reranked[i]
	}

	return reranked
}

// ScoreBoostReranker demonstrates a reranker that boosts scores based on custom logic
type ScoreBoostReranker struct {
	boostIDs map[uint32]float32 // Map of IDs to boost amounts
}

func NewScoreBoostReranker(boostIDs map[uint32]float32) *ScoreBoostReranker {
	return &ScoreBoostReranker{boostIDs: boostIDs}
}

func (r *ScoreBoostReranker) Rerank(results []VectorResult) []VectorResult {
	// Create a copy of results with modified scores
	reranked := make([]VectorResult, len(results))

	for i, result := range results {
		reranked[i] = result

		// If this ID should be boosted, reduce its distance score (lower is better)
		if boost, exists := r.boostIDs[result.GetId()]; exists {
			reranked[i].Score = result.Score * (1.0 - boost)
		}
	}

	// Re-sort by score (ascending - lower is better)
	sort.Slice(reranked, func(i, j int) bool {
		return reranked[i].Score < reranked[j].Score
	})

	return reranked
}

// TopKReranker demonstrates a reranker that limits results to top K
type TopKReranker struct {
	k int
}

func NewTopKReranker(k int) *TopKReranker {
	return &TopKReranker{k: k}
}

func (r *TopKReranker) Rerank(results []VectorResult) []VectorResult {
	if len(results) <= r.k {
		return results
	}
	return results[:r.k]
}

// TestRerankerWithFlatIndex tests the reranker interface with flat index
func TestRerankerWithFlatIndex(t *testing.T) {
	// Create a simple flat index
	index, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	// Add some vectors with specific IDs
	vectors := []struct {
		id  uint32
		vec []float32
	}{
		{1, []float32{1.0, 0.0, 0.0}},
		{2, []float32{0.0, 1.0, 0.0}},
		{3, []float32{0.0, 0.0, 1.0}},
		{4, []float32{0.5, 0.5, 0.0}},
	}

	for _, v := range vectors {
		node := NewVectorNodeWithID(v.id, v.vec)
		if err := index.Add(*node); err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Search without reranker
	query := []float32{1.0, 0.0, 0.0}
	resultsNoReranker, err := index.NewSearch().
		WithQuery(query).
		WithK(4).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(resultsNoReranker) == 0 {
		t.Fatal("Expected results without reranker")
	}

	// Search with reverse reranker
	reranker := &ExampleReranker{}
	resultsWithReranker, err := index.NewSearch().
		WithQuery(query).
		WithK(4).
		WithReranker(reranker).
		Execute()

	if err != nil {
		t.Fatalf("Search with reranker failed: %v", err)
	}

	if len(resultsWithReranker) != len(resultsNoReranker) {
		t.Errorf("Expected same number of results, got %d vs %d",
			len(resultsWithReranker), len(resultsNoReranker))
	}

	// Verify results are in reverse order
	for i := 0; i < len(resultsNoReranker); i++ {
		expected := resultsNoReranker[len(resultsNoReranker)-1-i]
		actual := resultsWithReranker[i]

		if expected.GetId() != actual.GetId() {
			t.Errorf("Result %d: expected ID %d, got %d",
				i, expected.GetId(), actual.GetId())
		}
	}
}

// TestScoreBoostReranker tests score-based reranking
func TestScoreBoostReranker(t *testing.T) {
	// Create a simple flat index
	index, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	// Add vectors with known IDs
	vectors := []struct {
		id  uint32
		vec []float32
	}{
		{1, []float32{1.0, 0.0, 0.0}},
		{2, []float32{0.8, 0.2, 0.0}},
		{3, []float32{0.6, 0.4, 0.0}},
		{4, []float32{0.0, 1.0, 0.0}},
	}

	for _, v := range vectors {
		node := NewVectorNodeWithID(v.id, v.vec)
		if err := index.Add(*node); err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Search with boost reranker that boosts ID 4
	boostMap := map[uint32]float32{
		4: 0.9, // 90% boost (reduces distance significantly)
	}
	reranker := NewScoreBoostReranker(boostMap)

	query := []float32{1.0, 0.0, 0.0}
	results, err := index.NewSearch().
		WithQuery(query).
		WithK(4).
		WithReranker(reranker).
		Execute()

	if err != nil {
		t.Fatalf("Search with boost reranker failed: %v", err)
	}

	// ID 4 should now be ranked higher due to boost
	// (even though it's further from the query in original space)
	if len(results) == 0 {
		t.Fatal("Expected results from boosted search")
	}

	t.Logf("Reranked results:")
	for i, result := range results {
		t.Logf("  Rank %d: ID=%d, Score=%.4f", i+1, result.GetId(), result.Score)
	}
}

// TestTopKReranker tests top-k limiting reranker
func TestTopKReranker(t *testing.T) {
	// Create a simple flat index
	index, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	// Add some vectors with specific IDs
	vectors := []struct {
		id  uint32
		vec []float32
	}{
		{1, []float32{1.0, 0.0, 0.0}},
		{2, []float32{0.0, 1.0, 0.0}},
		{3, []float32{0.0, 0.0, 1.0}},
		{4, []float32{0.5, 0.5, 0.0}},
		{5, []float32{0.3, 0.3, 0.3}},
	}

	for _, v := range vectors {
		node := NewVectorNodeWithID(v.id, v.vec)
		if err := index.Add(*node); err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Search with top-k reranker
	reranker := NewTopKReranker(2)
	query := []float32{1.0, 0.0, 0.0}
	results, err := index.NewSearch().
		WithQuery(query).
		WithK(5). // Request 5, but reranker will limit to 2
		WithReranker(reranker).
		Execute()

	if err != nil {
		t.Fatalf("Search with top-k reranker failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results after top-k reranking, got %d", len(results))
	}
}

// TestRerankerNil tests that nil reranker works correctly
func TestRerankerNil(t *testing.T) {
	// Create a simple flat index
	index, err := NewFlatIndex(3, Euclidean)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	// Add a vector with specific ID
	node := NewVectorNodeWithID(1, []float32{1.0, 0.0, 0.0})
	if err := index.Add(*node); err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	// Search with nil reranker (should work normally)
	query := []float32{1.0, 0.0, 0.0}
	results, err := index.NewSearch().
		WithQuery(query).
		WithK(1).
		WithReranker(nil). // Explicitly set to nil
		Execute()

	if err != nil {
		t.Fatalf("Search with nil reranker failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Expected results with nil reranker")
	}
}

// BenchmarkSearchWithoutReranker benchmarks search without reranker
func BenchmarkSearchWithoutReranker(b *testing.B) {
	index, _ := NewFlatIndex(128, Euclidean)

	// Add 1000 vectors
	for i := 0; i < 1000; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(i+j) / 1000.0
		}
		node := NewVectorNode(vec)
		index.Add(*node)
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index.NewSearch().WithQuery(query).WithK(10).Execute()
	}
}

// BenchmarkSearchWithReranker benchmarks search with reranker
func BenchmarkSearchWithReranker(b *testing.B) {
	index, _ := NewFlatIndex(128, Euclidean)

	// Add 1000 vectors
	for i := 0; i < 1000; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = float32(i+j) / 1000.0
		}
		node := NewVectorNode(vec)
		index.Add(*node)
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = 0.5
	}

	reranker := NewTopKReranker(5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index.NewSearch().WithQuery(query).WithK(10).WithReranker(reranker).Execute()
	}
}
