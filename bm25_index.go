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
	"encoding/binary"
	"fmt"
	"io"
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

	// deletedDocs tracks soft-deleted document IDs using roaring bitmap
	// CRITICAL OPTIMIZATION: RoaringBitmap is much more efficient than map[uint32]bool
	// - O(log n) membership test with better memory efficiency
	// - Compressed bitmap representation
	// - Fast iteration for batch operations
	deletedDocs *roaring.Bitmap
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
		postings:    make(map[string]*roaring.Bitmap),
		tf:          make(map[string]map[uint32]int),
		docLengths:  make(map[uint32]int),
		docTokens:   make(map[uint32][]string),
		deletedDocs: roaring.New(), // Initialize empty bitmap for soft deletes
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

// Remove performs soft delete using roaring bitmap.
//
// CONCURRENCY OPTIMIZATION:
// - Uses read lock first (cheaper) to check if document exists
// - Only acquires write lock for the actual bitmap modification
// - Minimizes write lock contention
//
// SOFT DELETE MECHANISM:
// Instead of immediately removing from all data structures (expensive O(m)),
// we mark as deleted in roaring bitmap. Deleted documents are:
//   - Skipped during search
//   - Still in internal data structures
//   - Not counted as active documents
//
// Call Flush() periodically for actual cleanup and memory reclamation.
//
// Parameters:
//   - id: Document ID to remove
//
// Returns:
//   - error: Always returns nil (exists to satisfy TextIndex interface)
//
// Time Complexity: O(log n) for bitmap operation (vs O(m) for hard delete)
//
// Thread-safety: Uses read lock for validation, write lock for modification
func (ix *BM25SearchIndex) Remove(id uint32) error {
	// ════════════════════════════════════════════════════════════════════════
	// STEP 1: CHECK EXISTENCE (READ LOCK - CHEAPER)
	// ════════════════════════════════════════════════════════════════════════
	ix.mu.RLock()
	_, exists := ix.docTokens[id]
	alreadyDeleted := ix.deletedDocs.Contains(id)
	ix.mu.RUnlock()

	// Fast-fail validation outside of write lock
	if !exists {
		return nil // Document doesn't exist, nothing to do
	}
	if alreadyDeleted {
		return nil // Already deleted, nothing to do
	}

	// ════════════════════════════════════════════════════════════════════════
	// STEP 2: MARK AS DELETED (WRITE LOCK - ONLY FOR BITMAP UPDATE)
	// ════════════════════════════════════════════════════════════════════════
	ix.mu.Lock()
	ix.deletedDocs.Add(id)
	ix.mu.Unlock()

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

// Flush performs hard delete of soft-deleted documents.
//
// WHEN TO CALL:
//   - After multiple Remove() calls (batch cleanup)
//   - When deleted documents are significant (e.g., > 10% of index)
//   - During off-peak hours
//
// WHAT IT DOES:
// 1. Physically removes all soft-deleted documents from all data structures
// 2. Updates inverted index (postings), term frequencies, document lengths
// 3. Reclaims memory occupied by deleted documents
// 4. Clears the deleted documents bitmap
//
// COST: O(d × m) where d = number of deleted docs, m = avg tokens per doc
//
// Thread-safety: Acquires exclusive write lock
//
// Returns:
//   - error: Always returns nil
func (ix *BM25SearchIndex) Flush() error {
	ix.mu.Lock()
	defer ix.mu.Unlock()

	// Quick exit if nothing to flush
	deletedCount := int(ix.deletedDocs.GetCardinality())
	if deletedCount == 0 {
		return nil
	}

	// ═══════════════════════════════════════════════════════════════════════
	// PHASE 1: HARD DELETE ALL SOFT-DELETED DOCUMENTS
	// ═══════════════════════════════════════════════════════════════════════
	// Use roaring bitmap's iterator for efficient traversal
	iter := ix.deletedDocs.Iterator()
	for iter.HasNext() {
		id := iter.Next()
		ix.removeInternal(id)
	}

	// ═══════════════════════════════════════════════════════════════════════
	// PHASE 2: RESET DELETED TRACKING
	// ═══════════════════════════════════════════════════════════════════════
	ix.deletedDocs.Clear()

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

// WriteTo serializes the BM25SearchIndex to an io.Writer.
//
// IMPORTANT: This method calls Flush() before serialization to ensure all soft-deleted
// documents are permanently removed from the serialized data.
//
// The serialization format is:
// 1. Magic number (4 bytes) - "BM25" identifier for validation
// 2. Version (4 bytes) - Format version for backward compatibility
// 3. Statistics:
//   - numDocs (4 bytes)
//   - totalTokens (4 bytes)
//   - avgDocLen (8 bytes as float64)
//
// 4. Document lengths (map[uint32]int):
//   - Count (4 bytes)
//   - For each entry: docID (4 bytes) + length (4 bytes)
//
// 5. Document tokens (map[uint32][]string):
//   - Count (4 bytes)
//   - For each entry:
//   - docID (4 bytes)
//   - Token count (4 bytes)
//   - For each token: token length (4 bytes) + token bytes
//
// 6. Postings (map[string]*roaring.Bitmap):
//   - Count (4 bytes)
//   - For each entry:
//   - Term length (4 bytes) + term bytes
//   - Bitmap size (4 bytes) + bitmap bytes
//
// 7. Term frequencies (map[string]map[uint32]int):
//   - Count (4 bytes)
//   - For each term:
//   - Term length (4 bytes) + term bytes
//   - Doc count (4 bytes)
//   - For each doc: docID (4 bytes) + frequency (4 bytes)
//
// 8. Deleted docs bitmap size (4 bytes) + roaring bitmap bytes
//
// Thread-safety: Acquires read lock during serialization
//
// Returns:
//   - int64: Number of bytes written
//   - error: Returns error if write fails or flush fails
func (ix *BM25SearchIndex) WriteTo(w io.Writer) (int64, error) {
	// Flush before serializing to remove soft-deleted documents
	if err := ix.Flush(); err != nil {
		return 0, fmt.Errorf("failed to flush before serialization: %w", err)
	}

	ix.mu.RLock()
	defer ix.mu.RUnlock()

	var bytesWritten int64

	// Helper function to track writes
	write := func(data interface{}) error {
		err := binary.Write(w, binary.LittleEndian, data)
		if err == nil {
			switch data.(type) {
			case uint32, int32:
				bytesWritten += 4
			case uint64, int64, float64:
				bytesWritten += 8
			}
		}
		return err
	}

	// 1. Write magic number "BM25"
	magic := [4]byte{'B', 'M', '2', '5'}
	if _, err := w.Write(magic[:]); err != nil {
		return bytesWritten, fmt.Errorf("failed to write magic number: %w", err)
	}
	bytesWritten += 4

	// 2. Write version
	version := uint32(1)
	if err := write(version); err != nil {
		return bytesWritten, fmt.Errorf("failed to write version: %w", err)
	}

	// 3. Write statistics
	numDocs := ix.numDocs.Load()
	if err := write(numDocs); err != nil {
		return bytesWritten, fmt.Errorf("failed to write numDocs: %w", err)
	}

	if err := write(uint32(ix.totalTokens)); err != nil {
		return bytesWritten, fmt.Errorf("failed to write totalTokens: %w", err)
	}

	if err := write(ix.avgDocLen); err != nil {
		return bytesWritten, fmt.Errorf("failed to write avgDocLen: %w", err)
	}

	// 4. Write document lengths
	if err := write(uint32(len(ix.docLengths))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write docLengths count: %w", err)
	}
	for docID, length := range ix.docLengths {
		if err := write(docID); err != nil {
			return bytesWritten, fmt.Errorf("failed to write docLength docID: %w", err)
		}
		if err := write(uint32(length)); err != nil {
			return bytesWritten, fmt.Errorf("failed to write docLength value: %w", err)
		}
	}

	// 5. Write document tokens
	if err := write(uint32(len(ix.docTokens))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write docTokens count: %w", err)
	}
	for docID, tokens := range ix.docTokens {
		if err := write(docID); err != nil {
			return bytesWritten, fmt.Errorf("failed to write docTokens docID: %w", err)
		}
		if err := write(uint32(len(tokens))); err != nil {
			return bytesWritten, fmt.Errorf("failed to write token count: %w", err)
		}
		for _, token := range tokens {
			tokenBytes := []byte(token)
			if err := write(uint32(len(tokenBytes))); err != nil {
				return bytesWritten, fmt.Errorf("failed to write token length: %w", err)
			}
			if _, err := w.Write(tokenBytes); err != nil {
				return bytesWritten, fmt.Errorf("failed to write token: %w", err)
			}
			bytesWritten += int64(len(tokenBytes))
		}
	}

	// 6. Write postings (inverted index)
	if err := write(uint32(len(ix.postings))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write postings count: %w", err)
	}
	for term, bitmap := range ix.postings {
		termBytes := []byte(term)
		if err := write(uint32(len(termBytes))); err != nil {
			return bytesWritten, fmt.Errorf("failed to write term length: %w", err)
		}
		if _, err := w.Write(termBytes); err != nil {
			return bytesWritten, fmt.Errorf("failed to write term: %w", err)
		}
		bytesWritten += int64(len(termBytes))

		bitmapBytes, err := bitmap.ToBytes()
		if err != nil {
			return bytesWritten, fmt.Errorf("failed to serialize posting bitmap: %w", err)
		}
		if err := write(uint32(len(bitmapBytes))); err != nil {
			return bytesWritten, fmt.Errorf("failed to write bitmap size: %w", err)
		}
		if _, err := w.Write(bitmapBytes); err != nil {
			return bytesWritten, fmt.Errorf("failed to write bitmap data: %w", err)
		}
		bytesWritten += int64(len(bitmapBytes))
	}

	// 7. Write term frequencies
	if err := write(uint32(len(ix.tf))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write tf count: %w", err)
	}
	for term, docFreqs := range ix.tf {
		termBytes := []byte(term)
		if err := write(uint32(len(termBytes))); err != nil {
			return bytesWritten, fmt.Errorf("failed to write tf term length: %w", err)
		}
		if _, err := w.Write(termBytes); err != nil {
			return bytesWritten, fmt.Errorf("failed to write tf term: %w", err)
		}
		bytesWritten += int64(len(termBytes))

		if err := write(uint32(len(docFreqs))); err != nil {
			return bytesWritten, fmt.Errorf("failed to write tf doc count: %w", err)
		}
		for docID, freq := range docFreqs {
			if err := write(docID); err != nil {
				return bytesWritten, fmt.Errorf("failed to write tf docID: %w", err)
			}
			if err := write(uint32(freq)); err != nil {
				return bytesWritten, fmt.Errorf("failed to write tf frequency: %w", err)
			}
		}
	}

	// 8. Write deleted docs bitmap
	bitmapBytes, err := ix.deletedDocs.ToBytes()
	if err != nil {
		return bytesWritten, fmt.Errorf("failed to serialize deleted docs bitmap: %w", err)
	}
	if err := write(uint32(len(bitmapBytes))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write bitmap size: %w", err)
	}
	if _, err := w.Write(bitmapBytes); err != nil {
		return bytesWritten, fmt.Errorf("failed to write bitmap data: %w", err)
	}
	bytesWritten += int64(len(bitmapBytes))

	return bytesWritten, nil
}

// ReadFrom deserializes a BM25SearchIndex from an io.Reader.
//
// This method reconstructs a BM25SearchIndex from the serialized format created by WriteTo.
// The deserialized index is fully functional and ready to use for searches.
//
// Thread-safety: Acquires write lock during deserialization
//
// Returns:
//   - int64: Number of bytes read
//   - error: Returns error if read fails, format is invalid, or data is corrupted
//
// Example:
//
//	// Save index
//	file, _ := os.Create("bm25_index.bin")
//	idx.WriteTo(file)
//	file.Close()
//
//	// Load index
//	file, _ := os.Open("bm25_index.bin")
//	idx2 := NewBM25SearchIndex()
//	idx2.ReadFrom(file)
//	file.Close()
func (ix *BM25SearchIndex) ReadFrom(r io.Reader) (int64, error) {
	ix.mu.Lock()
	defer ix.mu.Unlock()

	var bytesRead int64

	// Helper function to track reads
	read := func(data interface{}) error {
		err := binary.Read(r, binary.LittleEndian, data)
		if err == nil {
			switch data.(type) {
			case *uint32, *int32:
				bytesRead += 4
			case *uint64, *int64, *float64:
				bytesRead += 8
			}
		}
		return err
	}

	// 1. Read and validate magic number
	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		return bytesRead, fmt.Errorf("failed to read magic number: %w", err)
	}
	bytesRead += 4
	if string(magic) != "BM25" {
		return bytesRead, fmt.Errorf("invalid magic number: expected 'BM25', got '%s'", string(magic))
	}

	// 2. Read version
	var version uint32
	if err := read(&version); err != nil {
		return bytesRead, fmt.Errorf("failed to read version: %w", err)
	}
	if version != 1 {
		return bytesRead, fmt.Errorf("unsupported version: %d", version)
	}

	// 3. Read statistics
	var numDocs uint32
	if err := read(&numDocs); err != nil {
		return bytesRead, fmt.Errorf("failed to read numDocs: %w", err)
	}

	var totalTokens uint32
	if err := read(&totalTokens); err != nil {
		return bytesRead, fmt.Errorf("failed to read totalTokens: %w", err)
	}

	var avgDocLen float64
	if err := read(&avgDocLen); err != nil {
		return bytesRead, fmt.Errorf("failed to read avgDocLen: %w", err)
	}

	// 4. Read document lengths
	var docLengthsCount uint32
	if err := read(&docLengthsCount); err != nil {
		return bytesRead, fmt.Errorf("failed to read docLengths count: %w", err)
	}
	docLengths := make(map[uint32]int, docLengthsCount)
	for i := uint32(0); i < docLengthsCount; i++ {
		var docID, length uint32
		if err := read(&docID); err != nil {
			return bytesRead, fmt.Errorf("failed to read docLength docID: %w", err)
		}
		if err := read(&length); err != nil {
			return bytesRead, fmt.Errorf("failed to read docLength value: %w", err)
		}
		docLengths[docID] = int(length)
	}

	// 5. Read document tokens
	var docTokensCount uint32
	if err := read(&docTokensCount); err != nil {
		return bytesRead, fmt.Errorf("failed to read docTokens count: %w", err)
	}
	docTokens := make(map[uint32][]string, docTokensCount)
	for i := uint32(0); i < docTokensCount; i++ {
		var docID uint32
		if err := read(&docID); err != nil {
			return bytesRead, fmt.Errorf("failed to read docTokens docID: %w", err)
		}

		var tokenCount uint32
		if err := read(&tokenCount); err != nil {
			return bytesRead, fmt.Errorf("failed to read token count: %w", err)
		}

		tokens := make([]string, tokenCount)
		for j := uint32(0); j < tokenCount; j++ {
			var tokenLen uint32
			if err := read(&tokenLen); err != nil {
				return bytesRead, fmt.Errorf("failed to read token length: %w", err)
			}

			tokenBytes := make([]byte, tokenLen)
			if _, err := io.ReadFull(r, tokenBytes); err != nil {
				return bytesRead, fmt.Errorf("failed to read token: %w", err)
			}
			bytesRead += int64(tokenLen)
			tokens[j] = string(tokenBytes)
		}
		docTokens[docID] = tokens
	}

	// 6. Read postings (inverted index)
	var postingsCount uint32
	if err := read(&postingsCount); err != nil {
		return bytesRead, fmt.Errorf("failed to read postings count: %w", err)
	}
	postings := make(map[string]*roaring.Bitmap, postingsCount)
	for i := uint32(0); i < postingsCount; i++ {
		var termLen uint32
		if err := read(&termLen); err != nil {
			return bytesRead, fmt.Errorf("failed to read term length: %w", err)
		}

		termBytes := make([]byte, termLen)
		if _, err := io.ReadFull(r, termBytes); err != nil {
			return bytesRead, fmt.Errorf("failed to read term: %w", err)
		}
		bytesRead += int64(termLen)
		term := string(termBytes)

		var bitmapSize uint32
		if err := read(&bitmapSize); err != nil {
			return bytesRead, fmt.Errorf("failed to read bitmap size: %w", err)
		}

		bitmapBytes := make([]byte, bitmapSize)
		if _, err := io.ReadFull(r, bitmapBytes); err != nil {
			return bytesRead, fmt.Errorf("failed to read bitmap data: %w", err)
		}
		bytesRead += int64(bitmapSize)

		bitmap := roaring.New()
		if err := bitmap.UnmarshalBinary(bitmapBytes); err != nil {
			return bytesRead, fmt.Errorf("failed to deserialize posting bitmap: %w", err)
		}
		postings[term] = bitmap
	}

	// 7. Read term frequencies
	var tfCount uint32
	if err := read(&tfCount); err != nil {
		return bytesRead, fmt.Errorf("failed to read tf count: %w", err)
	}
	tf := make(map[string]map[uint32]int, tfCount)
	for i := uint32(0); i < tfCount; i++ {
		var termLen uint32
		if err := read(&termLen); err != nil {
			return bytesRead, fmt.Errorf("failed to read tf term length: %w", err)
		}

		termBytes := make([]byte, termLen)
		if _, err := io.ReadFull(r, termBytes); err != nil {
			return bytesRead, fmt.Errorf("failed to read tf term: %w", err)
		}
		bytesRead += int64(termLen)
		term := string(termBytes)

		var docCount uint32
		if err := read(&docCount); err != nil {
			return bytesRead, fmt.Errorf("failed to read tf doc count: %w", err)
		}

		docFreqs := make(map[uint32]int, docCount)
		for j := uint32(0); j < docCount; j++ {
			var docID, freq uint32
			if err := read(&docID); err != nil {
				return bytesRead, fmt.Errorf("failed to read tf docID: %w", err)
			}
			if err := read(&freq); err != nil {
				return bytesRead, fmt.Errorf("failed to read tf frequency: %w", err)
			}
			docFreqs[docID] = int(freq)
		}
		tf[term] = docFreqs
	}

	// 8. Read deleted docs bitmap
	var bitmapSize uint32
	if err := read(&bitmapSize); err != nil {
		return bytesRead, fmt.Errorf("failed to read bitmap size: %w", err)
	}

	bitmapBytes := make([]byte, bitmapSize)
	if _, err := io.ReadFull(r, bitmapBytes); err != nil {
		return bytesRead, fmt.Errorf("failed to read bitmap data: %w", err)
	}
	bytesRead += int64(bitmapSize)

	deletedDocs := roaring.New()
	if err := deletedDocs.UnmarshalBinary(bitmapBytes); err != nil {
		return bytesRead, fmt.Errorf("failed to deserialize deleted docs bitmap: %w", err)
	}

	// Update index state
	ix.numDocs.Store(numDocs)
	ix.totalTokens = int(totalTokens)
	ix.avgDocLen = avgDocLen
	ix.docLengths = docLengths
	ix.docTokens = docTokens
	ix.postings = postings
	ix.tf = tf
	ix.deletedDocs = deletedDocs

	return bytesRead, nil
}
