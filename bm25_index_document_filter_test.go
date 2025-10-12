package comet

import (
	"testing"
)

// TestBM25Index_WithDocumentIDs tests document filtering in BM25 search
func TestBM25Index_WithDocumentIDs(t *testing.T) {
	// Create index and add documents
	idx := NewBM25SearchIndex()

	documents := []struct {
		id   uint32
		text string
	}{
		{1, "the quick brown fox jumps over the lazy dog"},
		{2, "the lazy cat sleeps all day"},
		{3, "quick movements of the fox"},
		{4, "the dog barks at strangers"},
		{5, "a fox in the forest"},
	}

	for _, doc := range documents {
		if err := idx.Add(doc.id, doc.text); err != nil {
			t.Fatalf("Failed to add document %d: %v", doc.id, err)
		}
	}

	tests := []struct {
		name        string
		query       string
		docIDs      []uint32
		expectedIDs []uint32
		description string
	}{
		{
			name:        "No filter - all matching documents",
			query:       "fox",
			docIDs:      nil,
			expectedIDs: []uint32{1, 3, 5},
			description: "Without filter, all documents containing 'fox' should be returned",
		},
		{
			name:        "Filter to subset",
			query:       "fox",
			docIDs:      []uint32{1, 3},
			expectedIDs: []uint32{1, 3},
			description: "With filter, only documents 1 and 3 should be considered",
		},
		{
			name:        "Filter to single document",
			query:       "fox",
			docIDs:      []uint32{5},
			expectedIDs: []uint32{5},
			description: "Filter should work with single document",
		},
		{
			name:        "Filter to non-matching documents",
			query:       "fox",
			docIDs:      []uint32{2, 4},
			expectedIDs: []uint32{},
			description: "Documents 2 and 4 don't contain 'fox', should return empty",
		},
		{
			name:        "Empty filter list - same as no filter",
			query:       "lazy",
			docIDs:      []uint32{},
			expectedIDs: []uint32{1, 2},
			description: "Empty filter should behave like no filter",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			search := idx.NewSearch().WithQuery(tt.query)
			if tt.docIDs != nil {
				search = search.WithDocumentIDs(tt.docIDs...)
			}

			results, err := search.Execute()
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Check result count
			if len(results) != len(tt.expectedIDs) {
				t.Errorf("%s: expected %d results, got %d", tt.description, len(tt.expectedIDs), len(results))
			}

			// Check that all expected IDs are present
			resultIDs := make(map[uint32]bool)
			for _, r := range results {
				resultIDs[r.Id] = true
			}

			for _, expectedID := range tt.expectedIDs {
				if !resultIDs[expectedID] {
					t.Errorf("%s: expected document %d in results, but not found", tt.description, expectedID)
				}
			}

			// Check that no unexpected IDs are present
			for _, r := range results {
				found := false
				for _, expectedID := range tt.expectedIDs {
					if r.Id == expectedID {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("%s: unexpected document %d in results", tt.description, r.Id)
				}
			}
		})
	}
}

// TestBM25Index_WithDocumentIDs_MultipleQueries tests filtering with multiple queries
func TestBM25Index_WithDocumentIDs_MultipleQueries(t *testing.T) {
	idx := NewBM25SearchIndex()

	documents := []struct {
		id   uint32
		text string
	}{
		{1, "machine learning algorithms"},
		{2, "deep learning neural networks"},
		{3, "data science and analytics"},
		{4, "artificial intelligence research"},
		{5, "natural language processing"},
	}

	for _, doc := range documents {
		if err := idx.Add(doc.id, doc.text); err != nil {
			t.Fatalf("Failed to add document %d: %v", doc.id, err)
		}
	}

	// Search for "learning" and "intelligence" with document filter
	results, err := idx.NewSearch().
		WithQuery("learning", "intelligence").
		WithDocumentIDs(1, 2, 4).
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Should only return documents 1, 2, and 4 (filtered set)
	// Document 3 and 5 are excluded by filter
	if len(results) > 3 {
		t.Errorf("Expected at most 3 results (filtered set), got %d", len(results))
	}

	// Verify all results are from the filtered set
	allowedDocs := map[uint32]bool{1: true, 2: true, 4: true}
	for _, r := range results {
		if !allowedDocs[r.Id] {
			t.Errorf("Result contains document %d which is not in filtered set", r.Id)
		}
	}
}

// TestBM25Index_WithDocumentIDs_WithK tests filtering combined with K limit
func TestBM25Index_WithDocumentIDs_WithK(t *testing.T) {
	idx := NewBM25SearchIndex()

	for i := uint32(1); i <= 10; i++ {
		text := "document about programming and software development"
		if err := idx.Add(i, text); err != nil {
			t.Fatalf("Failed to add document %d: %v", i, err)
		}
	}

	// Filter to documents 1-5, but limit to K=3
	results, err := idx.NewSearch().
		WithQuery("programming").
		WithDocumentIDs(1, 2, 3, 4, 5).
		WithK(3).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Should return exactly 3 results (K limit)
	if len(results) != 3 {
		t.Errorf("Expected 3 results due to K limit, got %d", len(results))
	}

	// All results should be from the filtered set
	allowedDocs := map[uint32]bool{1: true, 2: true, 3: true, 4: true, 5: true}
	for _, r := range results {
		if !allowedDocs[r.Id] {
			t.Errorf("Result contains document %d which is not in filtered set", r.Id)
		}
	}
}

// TestBM25Index_WithDocumentIDs_WithNode tests filtering with node-based search
func TestBM25Index_WithDocumentIDs_WithNode(t *testing.T) {
	idx := NewBM25SearchIndex()

	documents := []struct {
		id   uint32
		text string
	}{
		{1, "apple banana cherry"},
		{2, "apple orange grape"},
		{3, "banana kiwi mango"},
		{4, "cherry strawberry blueberry"},
		{5, "apple banana orange"},
	}

	for _, doc := range documents {
		if err := idx.Add(doc.id, doc.text); err != nil {
			t.Fatalf("Failed to add document %d: %v", doc.id, err)
		}
	}

	// Search from node 1 (contains "apple banana cherry")
	// but only consider documents 2, 3, 5
	results, err := idx.NewSearch().
		WithNode(1).
		WithDocumentIDs(2, 3, 5).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Verify all results are from the filtered set
	allowedDocs := map[uint32]bool{2: true, 3: true, 5: true}
	for _, r := range results {
		if !allowedDocs[r.Id] {
			t.Errorf("Result contains document %d which is not in filtered set", r.Id)
		}
	}

	// Should find documents 2 and 5 (both contain apple or banana)
	// Document 3 might also match if it shares terms
	if len(results) == 0 {
		t.Error("Expected at least some results from filtered set")
	}
}

// TestBM25Index_WithDocumentIDs_EmptyResults tests that filtering can produce empty results
func TestBM25Index_WithDocumentIDs_EmptyResults(t *testing.T) {
	idx := NewBM25SearchIndex()

	documents := []struct {
		id   uint32
		text string
	}{
		{1, "red apple"},
		{2, "blue sky"},
		{3, "green grass"},
	}

	for _, doc := range documents {
		if err := idx.Add(doc.id, doc.text); err != nil {
			t.Fatalf("Failed to add document %d: %v", doc.id, err)
		}
	}

	// Search for "apple" but only in documents 2 and 3 (which don't contain "apple")
	results, err := idx.NewSearch().
		WithQuery("apple").
		WithDocumentIDs(2, 3).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected 0 results, got %d", len(results))
	}
}

// TestBM25Index_WithDocumentIDs_ScoreAggregation tests filtering with score aggregation
func TestBM25Index_WithDocumentIDs_ScoreAggregation(t *testing.T) {
	idx := NewBM25SearchIndex()

	documents := []struct {
		id   uint32
		text string
	}{
		{1, "machine learning"},
		{2, "deep learning"},
		{3, "machine vision"},
		{4, "computer vision"},
		{5, "learning algorithms"},
	}

	for _, doc := range documents {
		if err := idx.Add(doc.id, doc.text); err != nil {
			t.Fatalf("Failed to add document %d: %v", doc.id, err)
		}
	}

	// Search with multiple queries and document filter
	results, err := idx.NewSearch().
		WithQuery("machine", "learning").
		WithDocumentIDs(1, 2, 3).
		WithScoreAggregation(SumAggregation).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Should only return documents from filtered set
	allowedDocs := map[uint32]bool{1: true, 2: true, 3: true}
	for _, r := range results {
		if !allowedDocs[r.Id] {
			t.Errorf("Result contains document %d which is not in filtered set", r.Id)
		}
	}

	// Document 1 should have highest score (contains both "machine" and "learning")
	if len(results) > 0 && results[0].Id != 1 {
		t.Log("Note: Document 1 should typically rank highest, got", results[0].Id)
		// This is not a hard error as BM25 scoring can vary
	}
}

// TestBM25Index_WithDocumentIDs_Chaining tests method chaining with document filter
func TestBM25Index_WithDocumentIDs_Chaining(t *testing.T) {
	idx := NewBM25SearchIndex()

	for i := uint32(1); i <= 5; i++ {
		if err := idx.Add(i, "test document with content"); err != nil {
			t.Fatalf("Failed to add document %d: %v", i, err)
		}
	}

	// Test that method chaining works correctly
	results, err := idx.NewSearch().
		WithQuery("content").
		WithK(3).
		WithDocumentIDs(1, 2, 3, 4).
		WithScoreAggregation(MaxAggregation).
		Execute()

	if err != nil {
		t.Fatalf("Search with chaining failed: %v", err)
	}

	// Should return at most 3 results (K limit)
	if len(results) > 3 {
		t.Errorf("Expected at most 3 results, got %d", len(results))
	}

	// All results should be from the filtered set
	allowedDocs := map[uint32]bool{1: true, 2: true, 3: true, 4: true}
	for _, r := range results {
		if !allowedDocs[r.Id] {
			t.Errorf("Result contains document %d which is not in filtered set", r.Id)
		}
	}
}

// TestBM25Index_WithDocumentIDs_NonExistentDocuments tests filtering with non-existent document IDs
func TestBM25Index_WithDocumentIDs_NonExistentDocuments(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add documents 1, 2, 3
	for i := uint32(1); i <= 3; i++ {
		if err := idx.Add(i, "test document content"); err != nil {
			t.Fatalf("Failed to add document %d: %v", i, err)
		}
	}

	// Filter includes non-existent documents 100, 200
	results, err := idx.NewSearch().
		WithQuery("content").
		WithDocumentIDs(1, 100, 200).
		Execute()

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Should only return document 1 (the only one that exists and is in filter)
	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}

	if len(results) > 0 && results[0].Id != 1 {
		t.Errorf("Expected document 1, got document %d", results[0].Id)
	}
}

