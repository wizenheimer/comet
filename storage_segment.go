package comet

import (
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// segmentMetadata contains metadata about a persisted segment.
// Segments are immutable once written to disk.
type segmentMetadata struct {
	mu sync.RWMutex

	// Unique segment identifier
	id uint64

	// File paths
	hybridPath   string
	vectorPath   string
	textPath     string
	metadataPath string

	// Statistics
	createdAt time.Time
	numDocs   uint32
	sizeBytes int64

	// Lazy-loaded index (nil until first access)
	cachedIndex HybridSearchIndex
}

// newSegmentMetadata creates a new segment metadata.
func newSegmentMetadata(id uint64, hybrid, vector, text, metadata string) *segmentMetadata {
	return &segmentMetadata{
		id:           id,
		hybridPath:   hybrid,
		vectorPath:   vector,
		textPath:     text,
		metadataPath: metadata,
		createdAt:    time.Now(),
	}
}

// getIndex returns the cached index or loads it from disk if not cached.
// This implements lazy loading for memory efficiency.
//
// Parameters:
//   - vecIdx: Template vector index to use for deserialization
//   - txtIdx: Template text index to use for deserialization
//   - metaIdx: Template metadata index to use for deserialization
//
// Returns:
//   - HybridSearchIndex: The loaded index
//   - error: Error if loading fails
func (s *segmentMetadata) getIndex(vecIdx VectorIndex, txtIdx TextIndex, metaIdx MetadataIndex) (HybridSearchIndex, error) {
	s.mu.RLock()
	if s.cachedIndex != nil {
		idx := s.cachedIndex
		s.mu.RUnlock()
		return idx, nil
	}
	s.mu.RUnlock()

	// Load from disk with write lock
	s.mu.Lock()
	defer s.mu.Unlock()

	// Double-check after acquiring write lock
	if s.cachedIndex != nil {
		return s.cachedIndex, nil
	}

	// Create new hybrid index
	idx := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Open all segment files
	hybridFile, err := os.Open(s.hybridPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open hybrid file: %w", err)
	}
	defer hybridFile.Close()

	hybridGz, err := gzip.NewReader(hybridFile)
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader for hybrid: %w", err)
	}
	defer hybridGz.Close()

	var vectorGz, textGz, metadataGz io.ReadCloser

	// Open vector file if vector index exists
	if vecIdx != nil {
		vectorFile, err := os.Open(s.vectorPath)
		if err != nil {
			return nil, fmt.Errorf("failed to open vector file: %w", err)
		}
		defer vectorFile.Close()

		vectorGz, err = gzip.NewReader(vectorFile)
		if err != nil {
			return nil, fmt.Errorf("failed to create gzip reader for vector: %w", err)
		}
		defer vectorGz.Close()
	}

	// Open text file if text index exists
	if txtIdx != nil {
		textFile, err := os.Open(s.textPath)
		if err != nil {
			return nil, fmt.Errorf("failed to open text file: %w", err)
		}
		defer textFile.Close()

		textGz, err = gzip.NewReader(textFile)
		if err != nil {
			return nil, fmt.Errorf("failed to create gzip reader for text: %w", err)
		}
		defer textGz.Close()
	}

	// Open metadata file if metadata index exists
	if metaIdx != nil {
		metadataFile, err := os.Open(s.metadataPath)
		if err != nil {
			return nil, fmt.Errorf("failed to open metadata file: %w", err)
		}
		defer metadataFile.Close()

		metadataGz, err = gzip.NewReader(metadataFile)
		if err != nil {
			return nil, fmt.Errorf("failed to create gzip reader for metadata: %w", err)
		}
		defer metadataGz.Close()
	}

	// Create multi-reader for deserialization
	readers := []io.Reader{hybridGz}
	if vectorGz != nil {
		readers = append(readers, vectorGz)
	}
	if textGz != nil {
		readers = append(readers, textGz)
	}
	if metadataGz != nil {
		readers = append(readers, metadataGz)
	}

	combinedReader := io.MultiReader(readers...)

	// Deserialize the index
	if readerFrom, ok := idx.(io.ReaderFrom); ok {
		if _, err := readerFrom.ReadFrom(combinedReader); err != nil {
			return nil, fmt.Errorf("failed to deserialize segment: %w", err)
		}
	} else {
		return nil, fmt.Errorf("index does not implement io.ReaderFrom")
	}

	// Cache the loaded index
	s.cachedIndex = idx

	return idx, nil
}

// EvictCache evicts the cached index from memory.
// Call this when memory pressure is high or segment is no longer frequently accessed.
//
// Example:
//
//	segment.EvictCache() // Free memory
func (s *segmentMetadata) EvictCache() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cachedIndex = nil
}

// updateStats updates segment statistics.
func (s *segmentMetadata) updateStats(numDocs uint32, sizeBytes int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.numDocs = numDocs
	s.sizeBytes = sizeBytes
}

// segmentManager manages a collection of segments.
type segmentManager struct {
	mu       sync.RWMutex
	segments []*segmentMetadata
}

// newSegmentManager creates a new segment manager.
func newSegmentManager() *segmentManager {
	return &segmentManager{
		segments: make([]*segmentMetadata, 0),
	}
}

// add adds a segment to the manager.
func (sm *segmentManager) add(segment *segmentMetadata) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	sm.segments = append(sm.segments, segment)
}

// remove removes a segment from the manager.
func (sm *segmentManager) remove(segmentID uint64) bool {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	for i, seg := range sm.segments {
		if seg.id == segmentID {
			// Remove by swapping with last element
			sm.segments[i] = sm.segments[len(sm.segments)-1]
			sm.segments = sm.segments[:len(sm.segments)-1]
			return true
		}
	}

	return false
}

// get returns a segment by ID.
func (sm *segmentManager) get(segmentID uint64) *segmentMetadata {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	for _, seg := range sm.segments {
		if seg.id == segmentID {
			return seg
		}
	}

	return nil
}

// list returns all segments (oldest first).
func (sm *segmentManager) list() []*segmentMetadata {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	result := make([]*segmentMetadata, len(sm.segments))
	copy(result, sm.segments)
	return result
}

// Count returns the number of segments.
//
// Useful for monitoring segment accumulation and deciding when to compact.
//
// Example:
//
//	if store.segmentManager.Count() > 50 {
//	    store.TriggerCompaction()
//	}
func (sm *segmentManager) Count() int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return len(sm.segments)
}

// TotalSize returns the total size of all segments in bytes.
//
// Useful for monitoring disk usage.
//
// Example:
//
//	diskUsage := store.segmentManager.TotalSize()
//	fmt.Printf("Segments using %d bytes\n", diskUsage)
func (sm *segmentManager) TotalSize() int64 {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	var total int64
	for _, seg := range sm.segments {
		seg.mu.RLock()
		total += seg.sizeBytes
		seg.mu.RUnlock()
	}

	return total
}

// EvictAllCaches evicts all cached indexes from memory.
//
// Useful for reducing memory usage when under memory pressure.
//
// Example:
//
//	// Free up memory
//	store.segmentManager.EvictAllCaches()
func (sm *segmentManager) EvictAllCaches() {
	sm.mu.RLock()
	segments := make([]*segmentMetadata, len(sm.segments))
	copy(segments, sm.segments)
	sm.mu.RUnlock()

	for _, seg := range segments {
		seg.EvictCache()
	}
}
