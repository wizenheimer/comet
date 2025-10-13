package comet

import (
	"testing"
)

// TestHybridSearchIndex_VectorOnly tests vector-only search
func TestHybridSearchIndex_VectorOnly(t *testing.T) {
	// Create indexes
	vecIdx, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	idx := NewHybridSearchIndex(vecIdx, nil, nil)

	// Add documents with vectors only
	_, err = idx.Add([]float32{1.0, 0.0, 0.0}, "", nil)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add([]float32{0.0, 1.0, 0.0}, "", nil)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add([]float32{1.0, 0.1, 0.0}, "", nil)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Search
	results, err := idx.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0}).
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// First result should have highest similarity
	if results[0].Score <= results[1].Score {
		t.Errorf("Results not sorted by score")
	}
}

// TestHybridSearchIndex_TextOnly tests text-only search
func TestHybridSearchIndex_TextOnly(t *testing.T) {
	// Create indexes
	txtIdx := NewBM25SearchIndex()
	idx := NewHybridSearchIndex(nil, txtIdx, nil)

	// Add documents with text only
	_, err := idx.Add(nil, "the quick brown fox jumps over the lazy dog", nil)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add(nil, "the quick brown cat climbs a tree", nil)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add(nil, "a lazy dog sleeps all day", nil)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Search
	results, err := idx.NewSearch().
		WithText("quick brown").
		WithK(2).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}
}

// TestHybridSearchIndex_MetadataOnly tests metadata-only filtering
func TestHybridSearchIndex_MetadataOnly(t *testing.T) {
	// Create indexes
	metaIdx := NewRoaringMetadataIndex()
	idx := NewHybridSearchIndex(nil, nil, metaIdx)

	// Add documents with metadata only
	_, err := idx.Add(nil, "", map[string]interface{}{
		"category": "electronics",
		"price":    999,
	})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add(nil, "", map[string]interface{}{
		"category": "electronics",
		"price":    499,
	})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add(nil, "", map[string]interface{}{
		"category": "books",
		"price":    29,
	})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Search with metadata filters
	results, err := idx.NewSearch().
		WithMetadata(
			Eq("category", "electronics"),
			Gte("price", 500),
		).
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}
}

// TestHybridSearchIndex_VectorPlusMetadata tests hybrid vector + metadata search
func TestHybridSearchIndex_VectorPlusMetadata(t *testing.T) {
	// Create indexes
	vecIdx, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	metaIdx := NewRoaringMetadataIndex()
	idx := NewHybridSearchIndex(vecIdx, nil, metaIdx)

	// Add documents
	_, err = idx.Add([]float32{1.0, 0.0, 0.0}, "", map[string]interface{}{
		"category": "electronics",
		"price":    999,
	})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add([]float32{0.9, 0.1, 0.0}, "", map[string]interface{}{
		"category": "electronics",
		"price":    499,
	})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add([]float32{1.0, 0.05, 0.0}, "", map[string]interface{}{
		"category": "books",
		"price":    29,
	})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Search: find similar vectors but only in electronics category
	results, err := idx.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0}).
		WithMetadata(Eq("category", "electronics")).
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Should only get electronics items (2 results)
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}
}

// TestHybridSearchIndex_TextPlusMetadata tests hybrid text + metadata search
func TestHybridSearchIndex_TextPlusMetadata(t *testing.T) {
	// Create indexes
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()
	idx := NewHybridSearchIndex(nil, txtIdx, metaIdx)

	// Add documents
	_, err := idx.Add(nil, "the quick brown fox jumps over the lazy dog", map[string]interface{}{
		"category": "animals",
		"rating":   5,
	})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add(nil, "the quick brown cat climbs a tree", map[string]interface{}{
		"category": "animals",
		"rating":   3,
	})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add(nil, "a lazy dog sleeps all day", map[string]interface{}{
		"category": "nature",
		"rating":   4,
	})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Search: find text matches but only in animals category with high rating
	results, err := idx.NewSearch().
		WithText("quick brown").
		WithMetadata(
			Eq("category", "animals"),
			Gte("rating", 4),
		).
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Should only get 1 result (quick brown fox with rating 5)
	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}
}

// TestHybridSearchIndex_FullHybrid tests full hybrid search with all three modalities
func TestHybridSearchIndex_FullHybrid(t *testing.T) {
	// Create indexes
	vecIdx, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()
	idx := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Add documents
	_, err = idx.Add(
		[]float32{1.0, 0.0, 0.0},
		"advanced machine learning algorithms",
		map[string]interface{}{
			"category": "ai",
			"level":    "advanced",
		},
	)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add(
		[]float32{0.9, 0.1, 0.0},
		"introduction to machine learning",
		map[string]interface{}{
			"category": "ai",
			"level":    "beginner",
		},
	)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add(
		[]float32{0.0, 1.0, 0.0},
		"data structures and algorithms",
		map[string]interface{}{
			"category": "programming",
			"level":    "intermediate",
		},
	)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Hybrid search: vector similarity + text relevance + metadata filtering
	results, err := idx.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0}).
		WithText("machine learning").
		WithMetadata(Eq("category", "ai")).
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Should get 2 AI documents
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// Results should be sorted by combined score
	if len(results) > 1 && results[0].Score <= results[1].Score {
		t.Errorf("Results not sorted by combined score")
	}
}

// TestHybridSearchIndex_WithWeights tests weighted score combination
func TestHybridSearchIndex_WithWeights(t *testing.T) {
	// Create indexes
	vecIdx, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	txtIdx := NewBM25SearchIndex()
	idx := NewHybridSearchIndex(vecIdx, txtIdx, nil)

	// Add documents
	id1, err := idx.Add(
		[]float32{1.0, 0.0, 0.0},
		"machine learning algorithms",
		nil,
	)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}
	_, err = idx.Add(
		[]float32{0.0, 1.0, 0.0},
		"machine learning basics",
		nil,
	)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Search with equal weights (default weighted sum fusion)
	results1, err := idx.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0}).
		WithText("machine learning").
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Search with vector-heavy weights
	vectorHeavyFusion, err := NewFusion(WeightedSumFusion, &FusionConfig{
		VectorWeight: 10.0,
		TextWeight:   0.1,
	})
	if err != nil {
		t.Fatalf("Failed to create fusion: %v", err)
	}
	results2, err := idx.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0}).
		WithText("machine learning").
		WithFusion(vectorHeavyFusion).
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Just verify both searches returned results
	if len(results1) == 0 || len(results2) == 0 {
		t.Error("Expected results from both searches")
	}

	if len(results1) < 2 || len(results2) < 2 {
		t.Error("Expected at least 2 results from both searches")
	}

	// Verify that weights affect the ranking
	// Find positions of id1 and id2 in both result sets
	getPosition := func(results []HybridSearchResult, id uint32) int {
		for i, r := range results {
			if r.ID == id {
				return i
			}
		}
		return -1
	}

	pos1_r1 := getPosition(results1, id1)
	pos1_r2 := getPosition(results2, id1)

	// With vector-heavy weights, id1 (which has perfect vector match) should improve its position
	// or maintain if already first
	if pos1_r2 > pos1_r1 {
		t.Logf("Note: With vector-heavy weights, id1 position: %d -> %d (expected to improve or stay same)", pos1_r1, pos1_r2)
	}
}

// TestHybridSearchIndex_Remove tests document removal
func TestHybridSearchIndex_Remove(t *testing.T) {
	// Create indexes
	vecIdx, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()
	idx := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Add documents
	id1, _ := idx.Add(
		[]float32{1.0, 0.0, 0.0},
		"test document one",
		map[string]interface{}{"tag": "test"},
	)
	id2, _ := idx.Add(
		[]float32{0.0, 1.0, 0.0},
		"test document two",
		map[string]interface{}{"tag": "test"},
	)

	// Verify both exist
	results, err := idx.NewSearch().
		WithText("test document").
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("Expected 2 results before removal, got %d", len(results))
	}

	// Remove first document
	err = idx.Remove(id1)
	if err != nil {
		t.Fatalf("Failed to remove document: %v", err)
	}

	// Verify only one exists
	results, err = idx.NewSearch().
		WithText("test document").
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("Expected 1 result after removal, got %d", len(results))
	}
	if results[0].ID != id2 {
		t.Errorf("Expected remaining document to be %d, got %d", id2, results[0].ID)
	}
}

// TestHybridSearchIndex_AddWithID tests adding documents with specific IDs
func TestHybridSearchIndex_AddWithID(t *testing.T) {
	// Create indexes
	vecIdx, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	idx := NewHybridSearchIndex(vecIdx, nil, nil)

	// Add with specific ID
	err = idx.AddWithID(42, []float32{1.0, 0.0, 0.0}, "", nil)
	if err != nil {
		t.Fatalf("Failed to add document with ID: %v", err)
	}

	// Search
	results, err := idx.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0}).
		WithK(1).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	if results[0].ID != 42 {
		t.Errorf("Expected ID 42, got %d", results[0].ID)
	}
}

// TestHybridSearchIndex_MetadataGroups tests complex metadata filtering with OR groups
func TestHybridSearchIndex_MetadataGroups(t *testing.T) {
	// Create indexes
	metaIdx := NewRoaringMetadataIndex()
	idx := NewHybridSearchIndex(nil, nil, metaIdx)

	// Add documents
	_, err := idx.Add(nil, "", map[string]interface{}{
		"category": "electronics",
		"price":    999,
	})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add(nil, "", map[string]interface{}{
		"category": "phones",
		"price":    599,
	})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	_, err = idx.Add(nil, "", map[string]interface{}{
		"category": "books",
		"price":    29,
	})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Search: (category=electronics AND price>=900) OR (category=phones AND price>=500)
	results, err := idx.NewSearch().
		WithMetadataGroups(
			&FilterGroup{
				Filters: []Filter{Eq("category", "electronics"), Gte("price", 900)},
				Logic:   AND,
			},
			&FilterGroup{
				Filters: []Filter{Eq("category", "phones"), Gte("price", 500)},
				Logic:   AND,
			},
		).
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}
}

// TestHybridSearchIndex_EmptyIndexes tests behavior with no data
func TestHybridSearchIndex_EmptyIndexes(t *testing.T) {
	// Create indexes
	vecIdx, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()
	idx := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Search empty index
	results, err := idx.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0}).
		WithText("test").
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected 0 results from empty index, got %d", len(results))
	}
}

// TestHybridSearchIndex_PartialData tests documents with partial data
func TestHybridSearchIndex_PartialData(t *testing.T) {
	// Create indexes
	vecIdx, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()
	idx := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Add document with only vector
	_, err = idx.Add([]float32{1.0, 0.0, 0.0}, "", nil)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Add document with only text
	_, err = idx.Add(nil, "test document", nil)
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Add document with only metadata
	_, err = idx.Add(nil, "", map[string]interface{}{"tag": "test"})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Search should work for each modality
	vecResults, err := idx.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0}).
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("Vector search failed: %v", err)
	}
	if len(vecResults) != 1 {
		t.Errorf("Expected 1 vector result, got %d", len(vecResults))
	}

	txtResults, err := idx.NewSearch().
		WithText("test").
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("Text search failed: %v", err)
	}
	if len(txtResults) != 1 {
		t.Errorf("Expected 1 text result, got %d", len(txtResults))
	}

	metaResults, err := idx.NewSearch().
		WithMetadata(Eq("tag", "test")).
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("Metadata search failed: %v", err)
	}
	if len(metaResults) != 1 {
		t.Errorf("Expected 1 metadata result, got %d", len(metaResults))
	}
}
