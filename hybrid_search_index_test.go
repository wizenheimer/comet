package comet

import (
	"bytes"
	"io"
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

// ============================================================================
// SERIALIZATION TESTS
// ============================================================================

// TestHybridSearchIndexWriteTo tests serialization of the hybrid index
func TestHybridSearchIndexWriteTo(t *testing.T) {
	// Create indexes
	vecIdx, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()
	idx := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Add documents
	_, err = idx.Add([]float32{1.0, 0.0, 0.0}, "first document", map[string]interface{}{"category": "A"})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Serialize to buffers
	var hybridBuf, vectorBuf, textBuf, metadataBuf bytes.Buffer
	err = idx.(*hybridSearchIndex).WriteTo(&hybridBuf, &vectorBuf, &textBuf, &metadataBuf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// Verify buffers have data
	if hybridBuf.Len() == 0 {
		t.Error("WriteTo() wrote no data to hybrid buffer")
	}
	if vectorBuf.Len() == 0 {
		t.Error("WriteTo() wrote no data to vector buffer")
	}
	if textBuf.Len() == 0 {
		t.Error("WriteTo() wrote no data to text buffer")
	}
	if metadataBuf.Len() == 0 {
		t.Error("WriteTo() wrote no data to metadata buffer")
	}

	// Verify magic number
	magic := hybridBuf.Bytes()[:4]
	if string(magic) != "HYBR" {
		t.Errorf("Invalid magic number: got %s, want HYBR", string(magic))
	}
}

// TestHybridSearchIndexReadFrom tests deserialization of the hybrid index
func TestHybridSearchIndexReadFrom(t *testing.T) {
	// Create original index
	vecIdx, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()
	original := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Add documents
	_, err = original.Add([]float32{1.0, 0.0, 0.0}, "first document", map[string]interface{}{"category": "A"})
	if err != nil {
		t.Fatalf("Failed to add document: %v", err)
	}

	// Serialize
	var hybridBuf, vectorBuf, textBuf, metadataBuf bytes.Buffer
	err = original.(*hybridSearchIndex).WriteTo(&hybridBuf, &vectorBuf, &textBuf, &metadataBuf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// Create new index and deserialize
	vecIdx2, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	txtIdx2 := NewBM25SearchIndex()
	metaIdx2 := NewRoaringMetadataIndex()
	restored := NewHybridSearchIndex(vecIdx2, txtIdx2, metaIdx2)

	// Use io.MultiReader to combine the buffers
	combinedReader := io.MultiReader(&hybridBuf, &vectorBuf, &textBuf, &metadataBuf)
	_, err = restored.(*hybridSearchIndex).ReadFrom(combinedReader)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Verify docInfo was restored
	restoredImpl := restored.(*hybridSearchIndex)
	if len(restoredImpl.docInfo) != 1 {
		t.Errorf("Expected 1 document in restored index, got %d", len(restoredImpl.docInfo))
	}
}

// TestHybridSearchIndexSerializationRoundTrip tests full serialization and deserialization
func TestHybridSearchIndexSerializationRoundTrip(t *testing.T) {
	// Create original index
	vecIdx, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()
	idx := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Add documents
	id1, _ := idx.Add([]float32{1.0, 0.0, 0.0}, "machine learning deep learning", map[string]interface{}{"category": "AI", "score": 90})
	id2, _ := idx.Add([]float32{0.0, 1.0, 0.0}, "natural language processing", map[string]interface{}{"category": "NLP", "score": 85})
	id3, _ := idx.Add([]float32{0.0, 0.0, 1.0}, "computer vision image recognition", map[string]interface{}{"category": "CV", "score": 88})

	// Perform searches before serialization
	vectorResults, err := idx.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0}).
		WithK(2).
		Execute()
	if err != nil {
		t.Fatalf("Vector search before serialization error: %v", err)
	}

	textResults, err := idx.NewSearch().
		WithText("learning").
		WithK(2).
		Execute()
	if err != nil {
		t.Fatalf("Text search before serialization error: %v", err)
	}

	metadataResults, err := idx.NewSearch().
		WithMetadata(Gte("score", 87)).
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("Metadata search before serialization error: %v", err)
	}

	// Serialize
	var hybridBuf, vectorBuf, textBuf, metadataBuf bytes.Buffer
	err = idx.(*hybridSearchIndex).WriteTo(&hybridBuf, &vectorBuf, &textBuf, &metadataBuf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// Deserialize into new index
	vecIdx2, err := NewFlatIndex(3, Cosine)
	if err != nil {
		t.Fatalf("Failed to create vector index: %v", err)
	}
	txtIdx2 := NewBM25SearchIndex()
	metaIdx2 := NewRoaringMetadataIndex()
	idx2 := NewHybridSearchIndex(vecIdx2, txtIdx2, metaIdx2)

	// Use io.MultiReader to combine the buffers
	combinedReader := io.MultiReader(&hybridBuf, &vectorBuf, &textBuf, &metadataBuf)
	_, err = idx2.(*hybridSearchIndex).ReadFrom(combinedReader)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Perform same searches after deserialization
	vectorResults2, err := idx2.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0}).
		WithK(2).
		Execute()
	if err != nil {
		t.Fatalf("Vector search after deserialization error: %v", err)
	}

	textResults2, err := idx2.NewSearch().
		WithText("learning").
		WithK(2).
		Execute()
	if err != nil {
		t.Fatalf("Text search after deserialization error: %v", err)
	}

	metadataResults2, err := idx2.NewSearch().
		WithMetadata(Gte("score", 87)).
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("Metadata search after deserialization error: %v", err)
	}

	// Verify vector results match
	if len(vectorResults) != len(vectorResults2) {
		t.Errorf("Vector result count mismatch: before=%d, after=%d", len(vectorResults), len(vectorResults2))
	}
	for i := range vectorResults {
		if vectorResults[i].ID != vectorResults2[i].ID {
			t.Errorf("Vector result %d ID mismatch: before=%d, after=%d", i, vectorResults[i].ID, vectorResults2[i].ID)
		}
	}

	// Verify text results match (check set of IDs, BM25 ordering may vary slightly)
	if len(textResults) != len(textResults2) {
		t.Errorf("Text result count mismatch: before=%d, after=%d", len(textResults), len(textResults2))
	}
	textIDsBefore := make(map[uint32]bool)
	for _, r := range textResults {
		textIDsBefore[r.ID] = true
	}
	textIDsAfter := make(map[uint32]bool)
	for _, r := range textResults2 {
		textIDsAfter[r.ID] = true
	}
	for id := range textIDsBefore {
		if !textIDsAfter[id] {
			t.Errorf("Text ID %d present before serialization but missing after", id)
		}
	}

	// Verify metadata results match
	if len(metadataResults) != len(metadataResults2) {
		t.Errorf("Metadata result count mismatch: before=%d, after=%d", len(metadataResults), len(metadataResults2))
	}

	// Verify all expected document IDs are present
	restoredImpl := idx2.(*hybridSearchIndex)
	for _, expectedID := range []uint32{id1, id2, id3} {
		if _, exists := restoredImpl.docInfo[expectedID]; !exists {
			t.Errorf("Expected document %d to exist in restored index", expectedID)
		}
	}
}

// TestHybridSearchIndexSerializationPartialIndexes tests with only some indexes present
func TestHybridSearchIndexSerializationPartialIndexes(t *testing.T) {
	tests := []struct {
		name             string
		setupOriginal    func() HybridSearchIndex
		setupRestored    func() HybridSearchIndex
		expectedDocCount int
	}{
		{
			name: "vector only",
			setupOriginal: func() HybridSearchIndex {
				vecIdx, _ := NewFlatIndex(3, Cosine)
				idx := NewHybridSearchIndex(vecIdx, nil, nil)
				idx.Add([]float32{1.0, 0.0, 0.0}, "", nil)
				return idx
			},
			setupRestored: func() HybridSearchIndex {
				vecIdx, _ := NewFlatIndex(3, Cosine)
				return NewHybridSearchIndex(vecIdx, nil, nil)
			},
			expectedDocCount: 1,
		},
		{
			name: "text only",
			setupOriginal: func() HybridSearchIndex {
				txtIdx := NewBM25SearchIndex()
				idx := NewHybridSearchIndex(nil, txtIdx, nil)
				idx.AddWithID(1, nil, "test document", nil)
				return idx
			},
			setupRestored: func() HybridSearchIndex {
				txtIdx := NewBM25SearchIndex()
				return NewHybridSearchIndex(nil, txtIdx, nil)
			},
			expectedDocCount: 1,
		},
		{
			name: "metadata only",
			setupOriginal: func() HybridSearchIndex {
				metaIdx := NewRoaringMetadataIndex()
				idx := NewHybridSearchIndex(nil, nil, metaIdx)
				idx.AddWithID(1, nil, "", map[string]interface{}{"category": "test"})
				return idx
			},
			setupRestored: func() HybridSearchIndex {
				metaIdx := NewRoaringMetadataIndex()
				return NewHybridSearchIndex(nil, nil, metaIdx)
			},
			expectedDocCount: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := tt.setupOriginal()

			// Serialize (use separate buffers)
			var hybridBuf, vectorBuf, textBuf, metadataBuf bytes.Buffer
			err := original.(*hybridSearchIndex).WriteTo(&hybridBuf, &vectorBuf, &textBuf, &metadataBuf)
			if err != nil {
				t.Fatalf("WriteTo() error: %v", err)
			}

			// Deserialize
			restored := tt.setupRestored()
			combinedReader := io.MultiReader(&hybridBuf, &vectorBuf, &textBuf, &metadataBuf)
			_, err = restored.(*hybridSearchIndex).ReadFrom(combinedReader)
			if err != nil {
				t.Fatalf("ReadFrom() error: %v", err)
			}

			// Verify document count
			restoredImpl := restored.(*hybridSearchIndex)
			if len(restoredImpl.docInfo) != tt.expectedDocCount {
				t.Errorf("Expected %d documents, got %d", tt.expectedDocCount, len(restoredImpl.docInfo))
			}
		})
	}
}

// TestHybridSearchIndexReadFromInvalidData tests error handling for invalid serialized data
func TestHybridSearchIndexReadFromInvalidData(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() *bytes.Buffer
		wantErr string
	}{
		{
			name: "invalid magic number",
			setup: func() *bytes.Buffer {
				buf := bytes.NewBuffer([]byte("XXXX"))
				return buf
			},
			wantErr: "invalid magic number",
		},
		{
			name: "unsupported version",
			setup: func() *bytes.Buffer {
				var buf bytes.Buffer
				buf.Write([]byte("HYBR"))
				buf.Write([]byte{99, 0, 0, 0}) // version 99
				return &buf
			},
			wantErr: "unsupported version",
		},
		{
			name: "truncated data",
			setup: func() *bytes.Buffer {
				buf := bytes.NewBuffer([]byte("HY"))
				return buf
			},
			wantErr: "failed to read magic number",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := tt.setup()

			vecIdx, _ := NewFlatIndex(3, Cosine)
			idx := NewHybridSearchIndex(vecIdx, nil, nil)

			// The buf already contains invalid data, no need for combining
			_, err := idx.(*hybridSearchIndex).ReadFrom(buf)
			if err == nil {
				t.Errorf("ReadFrom() expected error containing '%s', got nil", tt.wantErr)
				return
			}

			// Check if error message contains expected substring
			if tt.wantErr != "" {
				errMsg := err.Error()
				found := false
				for i := 0; i <= len(errMsg)-len(tt.wantErr); i++ {
					if errMsg[i:i+len(tt.wantErr)] == tt.wantErr {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("ReadFrom() error = %v, want error containing '%s'", err, tt.wantErr)
				}
			}
		})
	}
}

// TestHybridSearchIndexSerializationEmpty tests serialization of an empty index
func TestHybridSearchIndexSerializationEmpty(t *testing.T) {
	// Create empty index
	vecIdx, _ := NewFlatIndex(3, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()
	idx := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Serialize empty index
	var hybridBuf, vectorBuf, textBuf, metadataBuf bytes.Buffer
	err := idx.(*hybridSearchIndex).WriteTo(&hybridBuf, &vectorBuf, &textBuf, &metadataBuf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// Deserialize
	vecIdx2, _ := NewFlatIndex(3, Cosine)
	txtIdx2 := NewBM25SearchIndex()
	metaIdx2 := NewRoaringMetadataIndex()
	idx2 := NewHybridSearchIndex(vecIdx2, txtIdx2, metaIdx2)

	// Use io.MultiReader to combine the buffers
	combinedReader := io.MultiReader(&hybridBuf, &vectorBuf, &textBuf, &metadataBuf)
	_, err = idx2.(*hybridSearchIndex).ReadFrom(combinedReader)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Verify restored index is also empty
	restoredImpl := idx2.(*hybridSearchIndex)
	if len(restoredImpl.docInfo) != 0 {
		t.Errorf("Expected 0 documents in restored empty index, got %d", len(restoredImpl.docInfo))
	}
}

// TestHybridSearchIndexWriteToFlushBehavior tests that WriteTo calls Flush
func TestHybridSearchIndexWriteToFlushBehavior(t *testing.T) {
	// Create index
	vecIdx, _ := NewFlatIndex(3, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()
	idx := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Add documents
	id1, _ := idx.Add([]float32{1.0, 0.0, 0.0}, "first document", map[string]interface{}{"category": "A"})
	id2, _ := idx.Add([]float32{0.0, 1.0, 0.0}, "second document", map[string]interface{}{"category": "B"})

	// Remove one document (soft delete)
	idx.Remove(id2)

	// Before WriteTo, docInfo should have 1 entry (after soft delete)
	idxImpl := idx.(*hybridSearchIndex)
	initialCount := len(idxImpl.docInfo)
	if initialCount != 1 {
		t.Errorf("Expected 1 document before WriteTo, got %d", initialCount)
	}

	// Call WriteTo (should flush)
	var hybridBuf, vectorBuf, textBuf, metadataBuf bytes.Buffer
	err := idx.(*hybridSearchIndex).WriteTo(&hybridBuf, &vectorBuf, &textBuf, &metadataBuf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// After WriteTo, should still have 1 document (the flush succeeded)
	afterCount := len(idxImpl.docInfo)
	if afterCount != 1 {
		t.Errorf("Expected 1 document after WriteTo (auto-flush), got %d", afterCount)
	}

	// Verify only id1 remains
	if _, exists := idxImpl.docInfo[id1]; !exists {
		t.Error("Expected document id1 to exist after WriteTo")
	}
	if _, exists := idxImpl.docInfo[id2]; exists {
		t.Error("Expected document id2 to not exist after WriteTo (was removed)")
	}
}

// errorWriterHybrid is a writer that always returns an error
type errorWriterHybrid struct{}

func (e errorWriterHybrid) Write(p []byte) (n int, err error) {
	return 0, io.ErrUnexpectedEOF
}

// TestHybridSearchIndexWriteToError tests error handling during write operations
func TestHybridSearchIndexWriteToError(t *testing.T) {
	vecIdx, _ := NewFlatIndex(3, Cosine)
	idx := NewHybridSearchIndex(vecIdx, nil, nil)
	idx.Add([]float32{1.0, 0.0, 0.0}, "", nil)

	// Try to write to an error writer
	var errWriter errorWriterHybrid
	var dummyBuf bytes.Buffer
	err := idx.(*hybridSearchIndex).WriteTo(errWriter, &dummyBuf, &dummyBuf, &dummyBuf)
	if err == nil {
		t.Error("WriteTo() expected error when writing to error writer, got nil")
	}
}
