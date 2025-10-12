package comet

import (
	"sync"

	"github.com/RoaringBitmap/roaring"
)

// DocumentFilter provides efficient document ID filtering for vector search.
// It uses roaring bitmaps for fast membership testing during search operations.
type DocumentFilter struct {
	bitmap *roaring.Bitmap
}

// documentFilterPool is a sync.Pool for DocumentFilter to reduce allocations
var documentFilterPool = sync.Pool{
	New: func() interface{} {
		return &DocumentFilter{
			bitmap: roaring.New(),
		}
	},
}

// NewDocumentFilter creates a new document filter from a list of document IDs.
// If the document IDs list is empty, returns nil (no filtering).
// The filter should be returned to the pool using ReturnDocumentFilter when done.
func NewDocumentFilter(documentIDs []uint32) *DocumentFilter {
	if len(documentIDs) == 0 {
		return nil
	}

	filter := documentFilterPool.Get().(*DocumentFilter)
	filter.bitmap.Clear() // Reset bitmap from pool

	for _, docID := range documentIDs {
		filter.bitmap.Add(docID)
	}

	return filter
}

// ReturnDocumentFilter returns a document filter to the pool for reuse.
// This should be called after the filter is no longer needed to reduce allocations.
// Do not use the filter after calling this method.
func ReturnDocumentFilter(filter *DocumentFilter) {
	if filter != nil {
		documentFilterPool.Put(filter)
	}
}

// IsEligible checks if a document ID is eligible for search.
// If filter is nil, all documents are eligible.
// Otherwise, checks if the document ID exists in the filter bitmap.
func (f *DocumentFilter) IsEligible(docID uint32) bool {
	if f == nil {
		return true
	}
	return f.bitmap.Contains(docID)
}

// ShouldSkip returns true if the document should be skipped (not eligible).
// This is a convenience method for use in loops with continue statements.
func (f *DocumentFilter) ShouldSkip(docID uint32) bool {
	return !f.IsEligible(docID)
}

// Count returns the number of eligible documents.
// Returns 0 if filter is nil (meaning all documents are eligible).
func (f *DocumentFilter) Count() uint64 {
	if f == nil {
		return 0 // All documents eligible, no specific count
	}
	return f.bitmap.GetCardinality()
}

// IsEmpty returns true if no documents are eligible.
// Returns false if filter is nil (all documents eligible).
func (f *DocumentFilter) IsEmpty() bool {
	if f == nil {
		return false
	}
	return f.bitmap.IsEmpty()
}
