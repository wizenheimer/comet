// Package comet implements a BM25-based full-text search index.
//
// WHAT IS BM25?
// BM25 (Best Matching 25) is a probabilistic ranking function used to estimate
// the relevance of documents to a given search query. It is one of the most
// widely used ranking functions in information retrieval.
//
// HOW BM25 WORKS:
// For a given query Q with terms {t1, t2, ..., tn} and document D:
// 1. Tokenizes and normalizes both query and documents using UAX#29 word segmentation
// 2. For each query term, calculates:
//   - IDF (Inverse Document Frequency): log((N - df + 0.5) / (df + 0.5) + 1)
//     where N is total docs and df is docs containing the term
//   - TF component: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLen / avgDocLen)))
//     where tf is term frequency in the document
//
// 3. Final score is the sum of (IDF × TF) for all query terms
//
// KEY PARAMETERS:
//   - K1 (default 1.2): Controls term frequency saturation. Higher values mean term
//     frequency has more impact on the score. Typical range: 1.2-2.0
//   - B (default 0.75): Controls document length normalization. 0 means no normalization
//     (all docs treated equally), 1 means full normalization. Typical range: 0.75
//
// TIME COMPLEXITY:
//   - Add: O(m) where m is the number of tokens in the document
//   - Search: O(q × d) where q is query tokens and d is average docs per term
//   - Remove: O(m) where m is the number of tokens in the document
//
// MEMORY REQUIREMENTS:
// - Stores inverted index (term -> docIDs) using roaring bitmaps for compression
// - Stores term frequencies (term -> docID -> count)
// - Stores document lengths and tokens (not full text)
// - Much more memory efficient than storing full document text
//
// GUARANTEES & TRADE-OFFS:
// ✓ Pros:
//   - Excellent relevance ranking for text search
//   - Handles term frequency and document length well
//   - Fast search using inverted index
//   - Memory efficient (doesn't store original text)
//   - Thread-safe for concurrent use
//
// ✗ Cons:
//   - Requires tokenization and normalization
//   - Cannot retrieve original document text
//   - Updates require document replacement (remove + add)
//
// WHEN TO USE:
// Use BM25 index when:
// 1. You need full-text search with relevance ranking
// 2. You want fast keyword-based search
// 3. Memory efficiency is important (vs storing full text)
// 4. You have your own document store and just need search
package comet

import (
	"container/heap"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/RoaringBitmap/roaring"
	"github.com/clipperhouse/uax29/v2/words"
	"golang.org/x/text/unicode/norm"
)

// Compile-time checks to ensure BM25SearchIndex implements TextIndex
var _ TextIndex = (*BM25SearchIndex)(nil)

// BM25 parameters for ranking
const (
	// K1 controls term frequency saturation (typical range: 1.2-2.0)
	K1 = 1.2
	// B controls document length normalization (0 = no normalization, 1 = full normalization)
	B = 0.75
)

// heapPool is a sync.Pool for resultHeap to reduce allocations during search operations
var heapPool = sync.Pool{
	New: func() interface{} {
		h := &resultHeap{}
		heap.Init(h)
		return h
	},
}

// BM25SearchIndex is a full-text search index that uses BM25 scoring for relevance ranking.
// It maintains an inverted index using roaring bitmaps for efficient storage and retrieval.
// All methods are safe for concurrent use by multiple goroutines.
//
// The index stores only document IDs and tokens for efficient memory usage.
// Applications should maintain their own document store and use the returned
// document IDs to retrieve full text.
type BM25SearchIndex struct {
	mu sync.RWMutex // protects all fields except numDocs

	// inverted index: term -> docIDs
	postings map[string]*roaring.Bitmap
	// term frequencies: term -> docID -> tf
	tf map[string]map[uint32]int
	// docID -> number of tokens
	docLengths map[uint32]int
	// total number of docs (uses atomic operations for lock-free reads)
	numDocs atomic.Uint32
	// running total of all token counts for O(1) average calculation
	totalTokens int
	// average doc length
	avgDocLen float64
	// store tokens per document for removal (much lighter than full text)
	docTokens map[uint32][]string
}

// SearchResult represents a single search result with its score.
type SearchResult struct {
	DocID uint32  // Document ID
	Score float64 // BM25 relevance score
}

// NewBM25SearchIndex creates and returns a new empty BM25SearchIndex.
// The index stores only document IDs and tokens for efficient memory usage.
// Applications should maintain their own document store and use the returned
// document IDs to retrieve full text.
//
// Returns:
//   - *BM25SearchIndex: A new empty index ready to accept documents
//
// Example:
//
//	idx := NewBM25SearchIndex()
//	idx.Add(1, "the quick brown fox")
//	results := idx.NewSearch().WithQuery("fox").WithK(10).Execute()
func NewBM25SearchIndex() *BM25SearchIndex {
	return &BM25SearchIndex{
		postings:   make(map[string]*roaring.Bitmap),
		tf:         make(map[string]map[uint32]int),
		docLengths: make(map[uint32]int),
		docTokens:  make(map[uint32][]string),
	}
}

// normalize applies Unicode normalization (NFKC) and converts to lowercase.
func normalize(s string) string {
	return strings.ToLower(norm.NFKC.String(s))
}

// tokenize splits text into tokens using UAX#29 word segmentation.
func tokenize(s string) []string {
	toks := words.FromString(s)
	var tokens []string
	for toks.Next() {
		tokens = append(tokens, toks.Value())
	}
	return tokens
}

// Add indexes a document with the given docID and text.
// If a document with the same ID already exists, it will be replaced.
// This method is safe for concurrent use.
//
// Note: The index does NOT store the original text, only tokens for efficient memory usage.
//
// Parameters:
//   - id: Document ID (must be unique)
//   - text: Document text to index
//
// Returns:
//   - error: Always returns nil (exists to satisfy TextIndex interface)
//
// Time Complexity: O(m) where m is the number of tokens in the text
//
// Thread-safety: Acquires exclusive lock
//
// Example:
//
//	err := idx.Add(1, "the quick brown fox jumps over the lazy dog")
func (ix *BM25SearchIndex) Add(id uint32, text string) error {
	ix.mu.Lock()
	defer ix.mu.Unlock()

	// If updating an existing doc, remove it first (internal call, lock already held)
	if _, exists := ix.docTokens[id]; exists {
		ix.removeInternal(id)
	}

	normText := normalize(text)
	tokens := tokenize(normText)
	docLen := len(tokens)

	// Store tokens for removal support (much lighter than full text)
	ix.docTokens[id] = tokens
	ix.docLengths[id] = docLen
	ix.numDocs.Add(1)

	// Update running total for O(1) average calculation
	ix.totalTokens += docLen

	for _, t := range tokens {
		// bitmap
		if ix.postings[t] == nil {
			ix.postings[t] = roaring.New()
		}
		ix.postings[t].Add(id)
		// tf
		if ix.tf[t] == nil {
			ix.tf[t] = make(map[uint32]int)
		}
		ix.tf[t][id]++
	}

	// update average doc length (now O(1))
	ix.updateAvgDocLen()

	return nil
}

// Remove removes a document from the index.
// This method is safe for concurrent use.
//
// Parameters:
//   - id: Document ID to remove
//
// Returns:
//   - error: Always returns nil (exists to satisfy TextIndex interface)
//
// Time Complexity: O(m) where m is the number of tokens in the document
//
// Thread-safety: Acquires exclusive lock
func (ix *BM25SearchIndex) Remove(id uint32) error {
	ix.mu.Lock()
	defer ix.mu.Unlock()

	ix.removeInternal(id)
	return nil
}

// removeInternal removes a document without acquiring the lock.
// Must be called with ix.mu held.
func (ix *BM25SearchIndex) removeInternal(id uint32) bool {
	// Check if document exists and get its tokens
	tokens, exists := ix.docTokens[id]
	if !exists {
		return false
	}

	docLen := ix.docLengths[id]

	// Remove from postings and tf
	for _, t := range tokens {
		if bitmap := ix.postings[t]; bitmap != nil {
			bitmap.Remove(id)
			if bitmap.IsEmpty() {
				delete(ix.postings, t)
			}
		}
		if tfMap := ix.tf[t]; tfMap != nil {
			delete(tfMap, id)
			if len(tfMap) == 0 {
				delete(ix.tf, t)
			}
		}
	}

	delete(ix.docTokens, id)
	delete(ix.docLengths, id)
	ix.numDocs.Add(^uint32(0)) // Atomic decrement (add -1)

	// Update running total for O(1) average calculation
	ix.totalTokens -= docLen

	if ix.numDocs.Load() > 0 {
		ix.updateAvgDocLen()
	} else {
		ix.avgDocLen = 0
		ix.totalTokens = 0 // Reset to ensure consistency
	}

	return true
}

// updateAvgDocLen recalculates the average document length.
// Must be called with ix.mu held.
// Now O(1) instead of O(N) by using running total.
func (ix *BM25SearchIndex) updateAvgDocLen() {
	numDocs := ix.numDocs.Load()
	if numDocs == 0 {
		ix.avgDocLen = 0
		return
	}
	ix.avgDocLen = float64(ix.totalTokens) / float64(numDocs)
}

// NewSearch creates a new search builder for this index.
//
// Returns:
//   - TextSearch: A new search builder ready to be configured
//
// Example:
//
//	results, err := idx.NewSearch().
//		WithQuery("quick brown").
//		WithK(5).
//		Execute()
func (ix *BM25SearchIndex) NewSearch() TextSearch {
	return &bm25TextSearch{
		index:  ix,
		k:      10, // Default k value
		cutoff: -1, // Default no cutoff
	}
}

// Flush is a no-op for BM25 index since data is stored in memory.
// This method exists to satisfy the TextIndex interface.
//
// Returns:
//   - error: Always returns nil
func (ix *BM25SearchIndex) Flush() error {
	return nil
}

// resultHeap is a min-heap of SearchResults for efficient top-K retrieval.
// We keep the K results with the highest scores by maintaining a min-heap
// where the minimum score is at the root.
type resultHeap []SearchResult

func (h resultHeap) Len() int           { return len(h) }
func (h resultHeap) Less(i, j int) bool { return h[i].Score < h[j].Score } // min-heap
func (h resultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *resultHeap) Push(x interface{}) {
	*h = append(*h, x.(SearchResult))
}

func (h *resultHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
