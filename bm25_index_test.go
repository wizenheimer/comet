package comet

import (
	"bytes"
	"io"
	"testing"
)

// TestNewBM25SearchIndex tests the creation of a new BM25 index
func TestNewBM25SearchIndex(t *testing.T) {
	idx := NewBM25SearchIndex()

	if idx == nil {
		t.Fatal("NewBM25SearchIndex() returned nil")
	}

	if idx.numDocs.Load() != 0 {
		t.Errorf("numDocs = %d, want 0", idx.numDocs.Load())
	}

	if idx.postings == nil {
		t.Error("postings map is nil")
	}

	if idx.tf == nil {
		t.Error("tf map is nil")
	}

	if idx.docLengths == nil {
		t.Error("docLengths map is nil")
	}

	if idx.docTokens == nil {
		t.Error("docTokens map is nil")
	}
}

// TestBM25SearchIndexAdd tests adding documents to the index
func TestBM25SearchIndexAdd(t *testing.T) {
	tests := []struct {
		name    string
		docID   uint32
		text    string
		wantErr bool
	}{
		{
			name:    "add simple document",
			docID:   1,
			text:    "the quick brown fox",
			wantErr: false,
		},
		{
			name:    "add empty document",
			docID:   2,
			text:    "",
			wantErr: false,
		},
		{
			name:    "add document with special characters",
			docID:   3,
			text:    "Hello, World! How are you?",
			wantErr: false,
		},
		{
			name:    "add document with unicode",
			docID:   4,
			text:    "héllo wörld 你好世界",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx := NewBM25SearchIndex()
			err := idx.Add(tt.docID, tt.text)

			if (err != nil) != tt.wantErr {
				t.Errorf("Add() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && tt.text != "" {
				if idx.numDocs.Load() != 1 {
					t.Errorf("numDocs = %d, want 1", idx.numDocs.Load())
				}

				tokens, exists := idx.docTokens[tt.docID]
				if !exists {
					t.Error("document tokens not stored")
				}

				if len(tokens) == 0 && tt.text != "" {
					t.Error("expected non-zero tokens for non-empty text")
				}
			}
		})
	}
}

// TestBM25SearchIndexAddMultiple tests adding multiple documents
func TestBM25SearchIndexAddMultiple(t *testing.T) {
	idx := NewBM25SearchIndex()

	docs := map[uint32]string{
		1: "the quick brown fox jumps over the lazy dog",
		2: "the lazy cat sleeps under the warm sun",
		3: "quick brown rabbits run through the forest",
		4: "the forest is dark and mysterious",
		5: "dogs and cats are popular pets",
	}

	for id, text := range docs {
		if err := idx.Add(id, text); err != nil {
			t.Fatalf("Add() error = %v", err)
		}
	}

	if idx.numDocs.Load() != uint32(len(docs)) {
		t.Errorf("numDocs = %d, want %d", idx.numDocs.Load(), len(docs))
	}

	// Verify all documents are stored
	for id := range docs {
		if _, exists := idx.docTokens[id]; !exists {
			t.Errorf("document %d not found in index", id)
		}
	}
}

// TestBM25SearchIndexUpdate tests updating an existing document
func TestBM25SearchIndexUpdate(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add initial document
	if err := idx.Add(1, "original text"); err != nil {
		t.Fatalf("Add() error = %v", err)
	}

	originalTokens := idx.docTokens[1]

	// Update the document
	if err := idx.Add(1, "updated text with more content"); err != nil {
		t.Fatalf("Add() error = %v", err)
	}

	// Should still have only 1 document
	if idx.numDocs.Load() != 1 {
		t.Errorf("numDocs = %d, want 1", idx.numDocs.Load())
	}

	// Tokens should be different
	updatedTokens := idx.docTokens[1]
	if len(updatedTokens) == len(originalTokens) {
		t.Error("expected different token count after update")
	}
}

// TestBM25SearchIndexRemove tests removing documents from the index
func TestBM25SearchIndexRemove(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add documents
	docs := map[uint32]string{
		1: "document one",
		2: "document two",
		3: "document three",
	}

	for id, text := range docs {
		if err := idx.Add(id, text); err != nil {
			t.Fatalf("Add() error = %v", err)
		}
	}

	// Remove a document (soft delete)
	if err := idx.Remove(2); err != nil {
		t.Errorf("Remove() error = %v", err)
	}

	// With soft delete, document still in internal structures
	if idx.numDocs.Load() != 3 {
		t.Errorf("numDocs = %d, want 3 (before flush)", idx.numDocs.Load())
	}

	// Document should still be in docTokens (soft delete)
	if _, exists := idx.docTokens[2]; !exists {
		t.Error("soft-deleted document should still be in index before flush")
	}

	// Flush to perform hard delete
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error = %v", err)
	}

	// After flush, document should be physically removed
	if idx.numDocs.Load() != 2 {
		t.Errorf("numDocs = %d, want 2 (after flush)", idx.numDocs.Load())
	}

	if _, exists := idx.docTokens[2]; exists {
		t.Error("removed document still in index after flush")
	}

	// Remove non-existent document (should not error)
	if err := idx.Remove(999); err != nil {
		t.Errorf("Remove() error = %v", err)
	}

	// Remove all documents
	if err := idx.Remove(1); err != nil {
		t.Errorf("Remove() error = %v", err)
	}
	if err := idx.Remove(3); err != nil {
		t.Errorf("Remove() error = %v", err)
	}

	// Still 2 documents before flush
	if idx.numDocs.Load() != 2 {
		t.Errorf("numDocs = %d, want 2 (before final flush)", idx.numDocs.Load())
	}

	// Flush to remove all
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error = %v", err)
	}

	if idx.numDocs.Load() != 0 {
		t.Errorf("numDocs = %d, want 0", idx.numDocs.Load())
	}

	if idx.avgDocLen != 0 {
		t.Errorf("avgDocLen = %f, want 0", idx.avgDocLen)
	}
}

// TestBM25SoftDeleteWithSearch tests soft delete functionality with search
func TestBM25SoftDeleteWithSearch(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add documents
	docs := map[uint32]string{
		1: "the quick brown fox jumps over the lazy dog",
		2: "a fast fox runs through the forest",
		3: "lazy cats sleep all day",
		4: "the dog chases the cat",
	}

	for id, text := range docs {
		if err := idx.Add(id, text); err != nil {
			t.Fatalf("Add() error = %v", err)
		}
	}

	// Search for "fox" - should get docs 1 and 2
	results, err := idx.NewSearch().WithQuery("fox").WithK(10).Execute()
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 2 {
		t.Errorf("Before delete: got %d results, want 2", len(results))
	}

	// Soft delete doc 1
	if err := idx.Remove(1); err != nil {
		t.Errorf("Remove() error = %v", err)
	}

	// Search again for "fox" - should only get doc 2 now
	results, err = idx.NewSearch().WithQuery("fox").WithK(10).Execute()
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 1 {
		t.Errorf("After soft delete: got %d results, want 1", len(results))
	}
	if len(results) > 0 && results[0].Id != 2 {
		t.Errorf("After soft delete: got doc %d, want doc 2", results[0].Id)
	}

	// Verify doc 1 still exists in internal structure
	if _, exists := idx.docTokens[1]; !exists {
		t.Error("soft-deleted document should still be in index before flush")
	}

	// Search for "lazy" - should get docs 2 and 3 (doc 1 is soft-deleted)
	results, err = idx.NewSearch().WithQuery("lazy").WithK(10).Execute()
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 1 {
		t.Errorf("Search for 'lazy': got %d results, want 1", len(results))
	}
	if len(results) > 0 && results[0].Id != 3 {
		t.Errorf("Search for 'lazy': got doc %d, want doc 3", results[0].Id)
	}

	// Test WithNode with deleted document (should fail)
	_, err = idx.NewSearch().WithNode(1).WithK(5).Execute()
	if err == nil {
		t.Error("WithNode for deleted document should return error")
	}

	// Test WithNode with non-deleted document (should succeed)
	_, err = idx.NewSearch().WithNode(2).WithK(5).Execute()
	if err != nil {
		t.Errorf("WithNode for non-deleted document error = %v", err)
	}

	// Flush to perform hard delete
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error = %v", err)
	}

	// Verify doc 1 is physically removed after flush
	if _, exists := idx.docTokens[1]; exists {
		t.Error("deleted document should be removed from index after flush")
	}

	// Search again for "fox" - should still only get doc 2
	results, err = idx.NewSearch().WithQuery("fox").WithK(10).Execute()
	if err != nil {
		t.Fatalf("Search() after flush error = %v", err)
	}
	if len(results) != 1 {
		t.Errorf("After flush: got %d results, want 1", len(results))
	}
	if len(results) > 0 && results[0].Id != 2 {
		t.Errorf("After flush: got doc %d, want doc 2", results[0].Id)
	}

	// Verify numDocs is correct after flush
	if idx.numDocs.Load() != 3 {
		t.Errorf("numDocs = %d, want 3 after flush", idx.numDocs.Load())
	}

	// Remove already deleted document (should be no-op)
	if err := idx.Remove(1); err != nil {
		t.Errorf("Remove already deleted error = %v", err)
	}
}

// TestBM25Search tests basic search functionality
func TestBM25Search(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add test documents
	docs := map[uint32]string{
		1: "the quick brown fox jumps over the lazy dog",
		2: "the lazy cat sleeps under the warm sun",
		3: "quick brown rabbits run through the forest",
		4: "the forest is dark and mysterious",
		5: "dogs and cats are popular pets",
	}

	for id, text := range docs {
		if err := idx.Add(id, text); err != nil {
			t.Fatalf("Add() error = %v", err)
		}
	}

	tests := []struct {
		name          string
		query         string
		k             int
		minResults    int
		maxResults    int
		expectedFirst uint32 // Expected first result (if deterministic)
	}{
		{
			name:       "search for 'fox'",
			query:      "fox",
			k:          5,
			minResults: 1,
			maxResults: 1,
		},
		{
			name:       "search for 'lazy'",
			query:      "lazy",
			k:          5,
			minResults: 2,
			maxResults: 2,
		},
		{
			name:       "search for 'forest'",
			query:      "forest",
			k:          5,
			minResults: 2,
			maxResults: 2,
		},
		{
			name:       "search for multiple terms",
			query:      "quick brown",
			k:          5,
			minResults: 2,
			maxResults: 5,
		},
		{
			name:       "search for non-existent term",
			query:      "elephant",
			k:          5,
			minResults: 0,
			maxResults: 0,
		},
		{
			name:       "empty query",
			query:      "",
			k:          5,
			minResults: 0,
			maxResults: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			textResults, err := idx.NewSearch().
				WithQuery(tt.query).
				WithK(tt.k).
				Execute()
			if err != nil && tt.minResults > 0 {
				t.Fatalf("Execute() error = %v", err)
			}

			if len(textResults) < tt.minResults || len(textResults) > tt.maxResults {
				t.Errorf("Execute() returned %d results, want between %d and %d",
					len(textResults), tt.minResults, tt.maxResults)
			}
		})
	}
}

// TestBM25SearchWithScores tests search with score information
func TestBM25SearchWithScores(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add test documents
	docs := map[uint32]string{
		1: "fox fox fox",           // High term frequency
		2: "fox dog",               // Multiple terms
		3: "the lazy dog sleeps",   // Common words
		4: "quick brown fox jumps", // Relevant
		5: "cat and mouse",         // Unrelated
	}

	for id, text := range docs {
		if err := idx.Add(id, text); err != nil {
			t.Fatalf("Add() error = %v", err)
		}
	}

	results, err := idx.NewSearch().
		WithQuery("fox").
		WithK(5).
		Execute()
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Execute() returned no results")
	}

	// Verify results are sorted by score (descending)
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("Results not sorted by score: results[%d].Score (%.4f) > results[%d].Score (%.4f)",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}

	// Verify scores are positive
	for i, result := range results {
		if result.Score <= 0 {
			t.Errorf("results[%d].Score = %.4f, want > 0", i, result.Score)
		}
	}

	// Document 1 should have highest score (highest term frequency)
	if results[0].Id != 1 {
		t.Errorf("Expected document 1 to have highest score, got document %d", results[0].Id)
	}
}

// TestBM25SearchTopK tests k-limiting functionality
func TestBM25SearchTopK(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add many documents
	for i := uint32(1); i <= 10; i++ {
		text := "the quick brown fox jumps"
		if err := idx.Add(i, text); err != nil {
			t.Fatalf("Add() error = %v", err)
		}
	}

	tests := []struct {
		name       string
		k          int
		wantLength int
	}{
		{"top 3", 3, 3},
		{"top 5", 5, 5},
		{"top 10", 10, 10},
		{"top 20 (more than available)", 20, 10},
		{"all results (k=0)", 0, 10},
		{"all results (k=-1)", -1, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().
				WithQuery("quick").
				WithK(tt.k).
				Execute()
			if err != nil {
				t.Fatalf("Execute() error = %v", err)
			}

			if len(results) != tt.wantLength {
				t.Errorf("Execute() returned %d results, want %d", len(results), tt.wantLength)
			}
		})
	}
}

// TestBM25Flush tests the Flush method
func TestBM25Flush(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add documents
	docs := map[uint32]string{
		1: "test document one",
		2: "test document two",
		3: "test document three",
	}

	for id, text := range docs {
		if err := idx.Add(id, text); err != nil {
			t.Fatalf("Add() error = %v", err)
		}
	}

	// Test 1: Flush with no deletions (should be no-op)
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() with no deletions error = %v", err)
	}

	if idx.numDocs.Load() != 3 {
		t.Errorf("After flush with no deletions: numDocs = %d, want 3", idx.numDocs.Load())
	}

	if idx.deletedDocs.GetCardinality() != 0 {
		t.Errorf("deletedDocs cardinality = %d, want 0", idx.deletedDocs.GetCardinality())
	}

	// Test 2: Soft delete some documents
	if err := idx.Remove(2); err != nil {
		t.Errorf("Remove() error = %v", err)
	}

	// Verify soft delete
	if idx.numDocs.Load() != 3 {
		t.Errorf("After soft delete: numDocs = %d, want 3", idx.numDocs.Load())
	}

	if idx.deletedDocs.GetCardinality() != 1 {
		t.Errorf("deletedDocs cardinality = %d, want 1", idx.deletedDocs.GetCardinality())
	}

	// Test 3: Flush should perform hard delete
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error = %v", err)
	}

	if idx.numDocs.Load() != 2 {
		t.Errorf("After flush: numDocs = %d, want 2", idx.numDocs.Load())
	}

	if idx.deletedDocs.GetCardinality() != 0 {
		t.Errorf("After flush: deletedDocs cardinality = %d, want 0", idx.deletedDocs.GetCardinality())
	}

	if _, exists := idx.docTokens[2]; exists {
		t.Error("Deleted document still in index after flush")
	}

	// Test 4: Remaining documents should still be searchable
	results, err := idx.NewSearch().
		WithQuery("test").
		WithK(5).
		Execute()
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if len(results) != 2 {
		t.Errorf("After Flush(), Execute() returned %d results, want 2", len(results))
	}

	// Test 5: Delete all remaining documents and flush
	if err := idx.Remove(1); err != nil {
		t.Errorf("Remove() error = %v", err)
	}
	if err := idx.Remove(3); err != nil {
		t.Errorf("Remove() error = %v", err)
	}

	if err := idx.Flush(); err != nil {
		t.Errorf("Final Flush() error = %v", err)
	}

	if idx.numDocs.Load() != 0 {
		t.Errorf("After final flush: numDocs = %d, want 0", idx.numDocs.Load())
	}

	if idx.avgDocLen != 0 {
		t.Errorf("After final flush: avgDocLen = %f, want 0", idx.avgDocLen)
	}
}

// TestBM25Tokenization tests the tokenization behavior
func TestBM25Tokenization(t *testing.T) {
	tests := []struct {
		name          string
		text          string
		expectedTerms []string
	}{
		{
			name:          "simple words",
			text:          "hello world",
			expectedTerms: []string{"hello", "world"},
		},
		{
			name:          "with punctuation",
			text:          "Hello, World!",
			expectedTerms: []string{"hello", "world"},
		},
		{
			name:          "with numbers",
			text:          "test 123 document",
			expectedTerms: []string{"test", "123", "document"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens := tokenize(normalize(tt.text))

			if len(tokens) < len(tt.expectedTerms) {
				t.Errorf("tokenize() returned %d tokens, want at least %d", len(tokens), len(tt.expectedTerms))
			}

			// Check that expected terms are present
			tokenMap := make(map[string]bool)
			for _, token := range tokens {
				tokenMap[token] = true
			}

			for _, expected := range tt.expectedTerms {
				if !tokenMap[expected] {
					t.Errorf("expected token %q not found in %v", expected, tokens)
				}
			}
		})
	}
}

// TestBM25Normalization tests the normalization behavior
func TestBM25Normalization(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"Hello", "hello"},
		{"WORLD", "world"},
		{"TeSt", "test"},
		{"héllo", "héllo"}, // Unicode normalization
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := normalize(tt.input)
			if result != tt.expected {
				t.Errorf("normalize(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

// TestBM25Interface tests that BM25SearchIndex implements TextIndex
func TestBM25Interface(t *testing.T) {
	var _ TextIndex = (*BM25SearchIndex)(nil)
}

// TestBM25NewSearch tests the NewSearch method
func TestBM25NewSearch(t *testing.T) {
	idx := NewBM25SearchIndex()

	search := idx.NewSearch()
	if search == nil {
		t.Fatal("NewSearch() returned nil")
	}

	// Verify it returns a TextSearch interface
	var _ TextSearch = search
}

// TestBM25IndexWriteTo tests serialization of the BM25 index
func TestBM25IndexWriteTo(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add some documents
	idx.Add(1, "the quick brown fox jumps over the lazy dog")
	idx.Add(2, "the lazy cat sleeps in the sun")
	idx.Add(3, "quick brown foxes are clever animals")

	// Serialize to buffer
	var buf bytes.Buffer
	n, err := idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	if n <= 0 {
		t.Errorf("WriteTo() returned %d bytes, expected > 0", n)
	}

	// Verify buffer has data
	if buf.Len() == 0 {
		t.Error("WriteTo() wrote no data to buffer")
	}

	// Verify magic number
	magic := buf.Bytes()[:4]
	if string(magic) != "BM25" {
		t.Errorf("Invalid magic number: got %s, want BM25", string(magic))
	}
}

// TestBM25IndexReadFrom tests deserialization of the BM25 index
func TestBM25IndexReadFrom(t *testing.T) {
	// Create and populate original index
	original := NewBM25SearchIndex()
	original.Add(1, "the quick brown fox jumps over the lazy dog")
	original.Add(2, "the lazy cat sleeps in the sun")
	original.Add(3, "quick brown foxes are clever animals")

	// Serialize
	var buf bytes.Buffer
	_, err := original.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// Create new index and deserialize
	restored := NewBM25SearchIndex()
	n, err := restored.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	if n <= 0 {
		t.Errorf("ReadFrom() returned %d bytes, expected > 0", n)
	}

	// Verify restored index matches original
	if restored.numDocs.Load() != original.numDocs.Load() {
		t.Errorf("numDocs mismatch: got %d, want %d", restored.numDocs.Load(), original.numDocs.Load())
	}

	if restored.totalTokens != original.totalTokens {
		t.Errorf("totalTokens mismatch: got %d, want %d", restored.totalTokens, original.totalTokens)
	}

	if restored.avgDocLen != original.avgDocLen {
		t.Errorf("avgDocLen mismatch: got %f, want %f", restored.avgDocLen, original.avgDocLen)
	}

	// Verify document count
	if len(restored.docLengths) != len(original.docLengths) {
		t.Errorf("docLengths count mismatch: got %d, want %d", len(restored.docLengths), len(original.docLengths))
	}

	// Verify postings count
	if len(restored.postings) != len(original.postings) {
		t.Errorf("postings count mismatch: got %d, want %d", len(restored.postings), len(original.postings))
	}
}

// TestBM25IndexSerializationRoundTrip tests that serialization and deserialization preserve data
func TestBM25IndexSerializationRoundTrip(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add documents
	idx.Add(1, "the quick brown fox jumps over the lazy dog")
	idx.Add(2, "the lazy cat sleeps in the sun")
	idx.Add(3, "quick brown foxes are clever animals")
	idx.Add(4, "foxes and dogs are both animals")

	// Perform a search before serialization
	resultsBefore, err := idx.NewSearch().WithQuery("fox").WithK(3).Execute()
	if err != nil {
		t.Fatalf("Search before serialization error: %v", err)
	}

	// Serialize
	var buf bytes.Buffer
	_, err = idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// Deserialize into new index
	idx2 := NewBM25SearchIndex()
	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Perform same search after deserialization
	resultsAfter, err := idx2.NewSearch().WithQuery("fox").WithK(3).Execute()
	if err != nil {
		t.Fatalf("Search after deserialization error: %v", err)
	}

	// Results should match
	if len(resultsBefore) != len(resultsAfter) {
		t.Errorf("Result count mismatch: before=%d, after=%d", len(resultsBefore), len(resultsAfter))
	}

	for i := 0; i < len(resultsBefore) && i < len(resultsAfter); i++ {
		if resultsBefore[i].Id != resultsAfter[i].Id {
			t.Errorf("Result %d Id mismatch: before=%d, after=%d", i, resultsBefore[i].Id, resultsAfter[i].Id)
		}
		scoreDiff := resultsBefore[i].Score - resultsAfter[i].Score
		if scoreDiff < -0.001 || scoreDiff > 0.001 {
			t.Errorf("Result %d score mismatch: before=%.4f, after=%.4f", i, resultsBefore[i].Score, resultsAfter[i].Score)
		}
	}
}

// TestBM25IndexSerializationWithDeletions tests serialization with soft-deleted documents
func TestBM25IndexSerializationWithDeletions(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add documents
	idx.Add(1, "the quick brown fox")
	idx.Add(2, "the lazy cat")
	idx.Add(3, "quick brown foxes")
	idx.Add(4, "lazy dogs")

	// Soft delete some documents
	idx.Remove(2)
	idx.Remove(4)

	// Verify soft deletes exist before serialization
	// numDocs is not decremented until Flush(), so it should still be 4
	if idx.numDocs.Load() != 4 {
		t.Errorf("Expected 4 docs before flush (soft delete doesn't decrement numDocs), got %d", idx.numDocs.Load())
	}
	// But deletedDocs should have 2 entries
	if idx.deletedDocs.GetCardinality() != 2 {
		t.Errorf("Expected 2 soft-deleted docs, got %d", idx.deletedDocs.GetCardinality())
	}

	// Serialize (should call Flush automatically)
	var buf bytes.Buffer
	_, err := idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// After WriteTo (which calls Flush), deleted docs should be removed
	if idx.numDocs.Load() != 2 {
		t.Errorf("Expected 2 docs after WriteTo (auto-flush), got %d", idx.numDocs.Load())
	}
	// deletedDocs should be cleared
	if idx.deletedDocs.GetCardinality() != 0 {
		t.Errorf("Expected 0 soft-deleted docs after flush, got %d", idx.deletedDocs.GetCardinality())
	}

	// Deserialize
	idx2 := NewBM25SearchIndex()
	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Restored index should only have non-deleted documents
	if idx2.numDocs.Load() != 2 {
		t.Errorf("Expected 2 docs in restored index, got %d", idx2.numDocs.Load())
	}

	// Verify only docs 1 and 3 exist
	if _, exists := idx2.docLengths[1]; !exists {
		t.Error("Expected doc 1 to exist in restored index")
	}
	if _, exists := idx2.docLengths[3]; !exists {
		t.Error("Expected doc 3 to exist in restored index")
	}
	if _, exists := idx2.docLengths[2]; exists {
		t.Error("Expected doc 2 to NOT exist in restored index (deleted)")
	}
	if _, exists := idx2.docLengths[4]; exists {
		t.Error("Expected doc 4 to NOT exist in restored index (deleted)")
	}
}

// TestBM25IndexReadFromInvalidData tests error handling for invalid serialized data
func TestBM25IndexReadFromInvalidData(t *testing.T) {
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
				// Write valid magic
				buf.Write([]byte("BM25"))
				// Write invalid version
				buf.Write([]byte{99, 0, 0, 0}) // version 99
				return &buf
			},
			wantErr: "unsupported version",
		},
		{
			name: "truncated data",
			setup: func() *bytes.Buffer {
				buf := bytes.NewBuffer([]byte("BM"))
				return buf
			},
			wantErr: "failed to read magic number",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := tt.setup()

			idx := NewBM25SearchIndex()
			_, err := idx.ReadFrom(buf)
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

// TestBM25IndexSerializationEmpty tests serialization of an empty index
func TestBM25IndexSerializationEmpty(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Serialize empty index
	var buf bytes.Buffer
	n, err := idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	if n <= 0 {
		t.Errorf("WriteTo() returned %d bytes for empty index, expected > 0", n)
	}

	// Deserialize
	idx2 := NewBM25SearchIndex()
	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Verify restored index is also empty
	if idx2.numDocs.Load() != 0 {
		t.Errorf("Expected 0 docs in restored empty index, got %d", idx2.numDocs.Load())
	}

	if len(idx2.postings) != 0 {
		t.Errorf("Expected 0 postings in restored empty index, got %d", len(idx2.postings))
	}
}

// TestBM25IndexWriteToFlushBehavior tests that WriteTo calls Flush
func TestBM25IndexWriteToFlushBehavior(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add documents
	idx.Add(1, "the quick brown fox")
	idx.Add(2, "the lazy cat")
	idx.Add(3, "quick brown foxes")

	// Soft delete one document
	idx.Remove(2)

	// Before WriteTo, numDocs is NOT decremented (only on Flush)
	// but deletedDocs should have 1 entry
	if idx.numDocs.Load() != 3 {
		t.Errorf("Expected 3 docs before WriteTo (soft delete doesn't decrement), got %d", idx.numDocs.Load())
	}
	if idx.deletedDocs.GetCardinality() != 1 {
		t.Errorf("Expected 1 soft-deleted doc before WriteTo, got %d", idx.deletedDocs.GetCardinality())
	}

	// Call WriteTo (should flush)
	var buf bytes.Buffer
	_, err := idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// After WriteTo, should still have 2 docs
	if idx.numDocs.Load() != 2 {
		t.Errorf("Expected 2 docs after WriteTo (auto-flush), got %d", idx.numDocs.Load())
	}

	// Deleted bitmap should be empty (flushed)
	if idx.deletedDocs.GetCardinality() != 0 {
		t.Errorf("Expected deletedDocs to be empty after WriteTo, got cardinality %d", idx.deletedDocs.GetCardinality())
	}

	// Doc 2 should no longer exist in internal structures
	if _, exists := idx.docLengths[2]; exists {
		t.Error("Expected doc 2 to be removed after WriteTo flush")
	}
	if _, exists := idx.docTokens[2]; exists {
		t.Error("Expected doc 2 tokens to be removed after WriteTo flush")
	}
}

// errorWriterBM25 is a writer that always returns an error
type errorWriterBM25 struct{}

func (e errorWriterBM25) Write(p []byte) (n int, err error) {
	return 0, io.ErrUnexpectedEOF
}

// TestBM25IndexWriteToError tests error handling during write operations
func TestBM25IndexWriteToError(t *testing.T) {
	idx := NewBM25SearchIndex()
	idx.Add(1, "the quick brown fox")

	// Try to write to an error writer
	var errWriter errorWriterBM25
	_, err := idx.WriteTo(errWriter)
	if err == nil {
		t.Error("WriteTo() expected error when writing to error writer, got nil")
	}
}

// TestBM25IndexSerializationComplexQueries tests serialization with complex queries
func TestBM25IndexSerializationComplexQueries(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add many documents with various content
	docs := map[uint32]string{
		1:  "machine learning and artificial intelligence",
		2:  "deep learning neural networks",
		3:  "natural language processing",
		4:  "computer vision and image recognition",
		5:  "reinforcement learning algorithms",
		6:  "supervised and unsupervised learning",
		7:  "neural network architectures",
		8:  "artificial intelligence applications",
		9:  "machine learning frameworks",
		10: "deep neural networks for vision",
	}

	for docID, text := range docs {
		idx.Add(docID, text)
	}

	// Perform various searches before serialization
	queries := []string{"machine learning", "neural networks"}

	// Serialize
	var buf bytes.Buffer
	_, err := idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// Deserialize
	idx2 := NewBM25SearchIndex()
	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Verify basic statistics match
	if idx2.numDocs.Load() != idx.numDocs.Load() {
		t.Errorf("numDocs mismatch: got %d, want %d", idx2.numDocs.Load(), idx.numDocs.Load())
	}
	if idx2.totalTokens != idx.totalTokens {
		t.Errorf("totalTokens mismatch: got %d, want %d", idx2.totalTokens, idx.totalTokens)
	}

	// Perform searches after deserialization and verify results are valid
	for _, query := range queries {
		results, err := idx2.NewSearch().WithQuery(query).WithK(5).Execute()
		if err != nil {
			t.Fatalf("Search after deserialization error for query '%s': %v", query, err)
		}

		// Verify we got results
		if len(results) == 0 {
			t.Errorf("Query '%s': expected some results, got none", query)
			continue
		}

		// Verify all result IDs are valid (exist in our docs map)
		for i, r := range results {
			if _, exists := docs[r.Id]; !exists {
				t.Errorf("Query '%s': result %d has invalid ID %d", query, i, r.Id)
			}
			// Verify scores are positive (BM25 scores should be > 0 for matches)
			if r.Score <= 0 {
				t.Errorf("Query '%s': result %d has non-positive score %.4f", query, i, r.Score)
			}
		}
	}
}
