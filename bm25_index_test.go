package comet

import (
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

	// Remove a document
	if err := idx.Remove(2); err != nil {
		t.Errorf("Remove() error = %v", err)
	}

	if idx.numDocs.Load() != 2 {
		t.Errorf("numDocs = %d, want 2", idx.numDocs.Load())
	}

	if _, exists := idx.docTokens[2]; exists {
		t.Error("removed document still in index")
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

	if idx.numDocs.Load() != 0 {
		t.Errorf("numDocs = %d, want 0", idx.numDocs.Load())
	}

	if idx.avgDocLen != 0 {
		t.Errorf("avgDocLen = %f, want 0", idx.avgDocLen)
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

	// Add a document
	if err := idx.Add(1, "test document"); err != nil {
		t.Fatalf("Add() error = %v", err)
	}

	// Flush should not error
	if err := idx.Flush(); err != nil {
		t.Errorf("Flush() error = %v", err)
	}

	// Document should still be searchable after flush
	results, err := idx.NewSearch().
		WithQuery("test").
		WithK(5).
		Execute()
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if len(results) != 1 {
		t.Errorf("After Flush(), Execute() returned %d results, want 1", len(results))
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
