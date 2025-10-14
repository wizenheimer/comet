// Package comet implements a persistent hybrid search index with LSM-tree inspired architecture.
//
// WHAT IS PERSISTENT STORAGE?
// The storage layer provides durability and scalability by persisting indexes to disk
// while maintaining fast in-memory writes through memtables. This enables:
// 1. Datasets larger than available RAM
// 2. Crash recovery and persistence
// 3. Fast writes (buffered in memory)
// 4. Efficient reads (merge from memory + disk)
//
// ARCHITECTURE:
// The storage follows an LSM-tree (Log-Structured Merge-Tree) inspired design:
//
//	┌─────────────────────────────────────────────────────────────┐
//	│                         WRITES                               │
//	└──────────────────┬──────────────────────────────────────────┘
//	                   │
//	                   ▼
//	         ┌─────────────────┐
//	         │  Active Memtable│  ← In-memory write buffer
//	         │   (Mutable)     │
//	         └────────┬────────┘
//	                  │ (fills up)
//	                  ▼
//	         ┌─────────────────┐
//	         │ Frozen Memtables│  ← Immutable, waiting flush
//	         │     (Queue)     │
//	         └────────┬────────┘
//	                  │ (background flush)
//	                  ▼
//	         ┌─────────────────┐
//	         │   Segments      │  ← Compressed on disk
//	         │   (Immutable)   │
//	         └────────┬────────┘
//	                  │ (background compaction)
//	                  ▼
//	         ┌─────────────────┐
//	         │ Merged Segments │  ← Larger, consolidated
//	         └─────────────────┘
//
// READ PATH:
// Searches read from newest to oldest:
// 1. Active memtable (most recent writes)
// 2. Frozen memtables (pending flush)
// 3. Disk segments (oldest data)
// Results are merged and deduplicated by score.
//
// MEMORY MANAGEMENT:
// - Memtables: Default 100MB per memtable
// - Flush threshold: Triggered when total memtables > 200MB
// - Segments: Lazy-loaded from disk, cached in memory
// - Cache eviction: LRU-style eviction under memory pressure
//
// WHEN TO USE:
// Use persistent storage when:
// 1. Your index needs to survive process restarts
// 2. Dataset size exceeds available RAM
// 3. You need write durability
// 4. You want automatic background compaction
package comet

import (
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// Default configuration constants
const (
	// DefaultMemtableSizeLimit is the default size limit for each memtable (100 MB)
	DefaultMemtableSizeLimit = 100 * 1024 * 1024

	// DefaultFlushThreshold is the default total memtable size before flush (200 MB)
	DefaultFlushThreshold = 200 * 1024 * 1024

	// DefaultCompactionInterval is how often to check for compaction (5 minutes)
	DefaultCompactionInterval = 5 * time.Minute

	// DefaultCompactionThreshold is the minimum number of segments before compaction
	DefaultCompactionThreshold = 5
)

// StorageConfig contains configuration for the storage layer.
type StorageConfig struct {
	// BaseDir is the directory for storing segments
	BaseDir string

	// MemtableSizeLimit is the maximum size of each memtable in bytes
	MemtableSizeLimit int64

	// FlushThreshold is the total memtable size that triggers a flush
	FlushThreshold int64

	// CompactionInterval is how often to check for compaction
	CompactionInterval time.Duration

	// CompactionThreshold is the minimum number of segments before compaction
	CompactionThreshold int

	// Index templates for creating new memtables
	VectorIndexTemplate   VectorIndex
	TextIndexTemplate     TextIndex
	MetadataIndexTemplate MetadataIndex
}

// DefaultStorageConfig returns a default storage configuration.
func DefaultStorageConfig(baseDir string) *StorageConfig {
	return &StorageConfig{
		BaseDir:             baseDir,
		MemtableSizeLimit:   DefaultMemtableSizeLimit,
		FlushThreshold:      DefaultFlushThreshold,
		CompactionInterval:  DefaultCompactionInterval,
		CompactionThreshold: DefaultCompactionThreshold,
	}
}

// PersistentHybridIndex is a persistent hybrid search index with LSM-tree architecture.
//
// It implements the HybridSearchIndex interface with full parity, providing the same
// builder-style search API while adding durability and scalability.
//
// Thread-safety: All methods are safe for concurrent use by multiple goroutines.
type PersistentHybridIndex struct {
	mu sync.RWMutex

	// Configuration
	config *StorageConfig

	// Storage components
	provider       *storageProvider
	memtableQueue  *memtableQueue
	segmentManager *segmentManager

	// Background workers
	flushChan      chan struct{}
	compactionChan chan struct{}
	closeChan      chan struct{}
	wg             sync.WaitGroup

	// State
	closed bool
}

// Compile-time check to ensure PersistentHybridIndex implements HybridSearchIndex
var _ HybridSearchIndex = (*PersistentHybridIndex)(nil)

// OpenPersistentHybridIndex opens or creates a persistent hybrid search index.
//
// If the directory already contains segments, they will be loaded for searching.
// A new active memtable is created for writes.
//
// Parameters:
//   - config: Storage configuration
//
// Returns:
//   - *PersistentHybridIndex: Opened storage instance
//   - error: Error if opening fails
//
// Example:
//
//	config := DefaultStorageConfig("./data")
//	config.VectorIndexTemplate, _ = NewFlatIndex(384, Cosine)
//	config.TextIndexTemplate = NewBM25SearchIndex()
//	config.MetadataIndexTemplate = NewRoaringMetadataIndex()
//	store, err := OpenPersistentHybridIndex(config)
func OpenPersistentHybridIndex(config *StorageConfig) (*PersistentHybridIndex, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	// Create storage provider
	provider, err := newStorageProvider(config.BaseDir)
	if err != nil {
		return nil, fmt.Errorf("failed to create storage provider: %w", err)
	}

	// Create segment manager
	segmentManager := newSegmentManager()

	// Load existing segments
	segmentIDs, err := provider.listSegments()
	if err != nil {
		provider.close()
		return nil, fmt.Errorf("failed to list segments: %w", err)
	}

	for _, id := range segmentIDs {
		hybrid, vector, text, metadata := provider.segmentPaths(id)
		segment := newSegmentMetadata(id, hybrid, vector, text, metadata)
		segmentManager.add(segment)
	}

	// Create memtable queue with fresh memtable
	memtableQueue := newMemtableQueue(
		config.VectorIndexTemplate,
		config.TextIndexTemplate,
		config.MetadataIndexTemplate,
		config.MemtableSizeLimit,
	)

	storage := &PersistentHybridIndex{
		config:         config,
		provider:       provider,
		memtableQueue:  memtableQueue,
		segmentManager: segmentManager,
		flushChan:      make(chan struct{}, 1),
		compactionChan: make(chan struct{}, 1),
		closeChan:      make(chan struct{}),
	}

	// Start background workers
	storage.wg.Add(2)
	go storage.flushWorker()
	go storage.compactionWorker()

	return storage, nil
}

// Add adds a document to the index.
// The document is written to the active memtable in memory.
//
// Parameters:
//   - vector: Document vector embedding (can be nil)
//   - text: Document text (can be empty)
//   - metadata: Document metadata (can be nil)
//
// Returns:
//   - uint32: Generated document ID
//   - error: Error if add fails
func (s *PersistentHybridIndex) Add(vector []float32, text string, metadata map[string]interface{}) (uint32, error) {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return 0, fmt.Errorf("storage is closed")
	}
	s.mu.RUnlock()

	id, err := s.memtableQueue.add(vector, text, metadata)
	if err != nil {
		return 0, err
	}

	// Maybe trigger flush
	s.maybeScheduleFlush()

	return id, nil
}

// AddWithID adds a document with a specific ID to the index.
func (s *PersistentHybridIndex) AddWithID(id uint32, vector []float32, text string, metadata map[string]interface{}) error {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return fmt.Errorf("storage is closed")
	}
	s.mu.RUnlock()

	if err := s.memtableQueue.addWithID(id, vector, text, metadata); err != nil {
		return err
	}

	// Maybe trigger flush
	s.maybeScheduleFlush()

	return nil
}

// Remove removes a document from the index.
//
// Parameters:
//   - id: Document ID to remove
//
// Returns:
//   - error: Error if removal fails
func (s *PersistentHybridIndex) Remove(id uint32) error {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return fmt.Errorf("storage is closed")
	}
	s.mu.RUnlock()

	// Remove from active memtable
	// Note: Documents in frozen memtables and segments cannot be removed
	// They will be removed during compaction
	memtables := s.memtableQueue.list()
	if len(memtables) > 0 {
		mutable := memtables[len(memtables)-1]
		return mutable.remove(id)
	}

	return nil
}

// NewSearch creates a new search builder for this index.
//
// Returns:
//   - HybridSearch: A new search builder ready to be configured
//
// Example:
//
//	results, err := store.NewSearch().
//		WithVector(queryEmbedding).
//		WithText("quick fox").
//		WithMetadata(Eq("category", "animals")).
//		WithK(10).
//		Execute()
func (s *PersistentHybridIndex) NewSearch() HybridSearch {
	return &persistentHybridSearch{
		storage:          s,
		k:                10,
		scoreAggregation: SumAggregation,
		cutoff:           -1,
		fusion:           DefaultFusion(),
	}
}

// Train trains the vector index if it requires training (e.g., IVF, PQ, IVFPQ)
//
// Parameters:
//   - vectors: Training vectors
//
// Returns:
//   - error: Error if training fails
func (s *PersistentHybridIndex) Train(vectors [][]float32) error {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return fmt.Errorf("storage is closed")
	}
	s.mu.RUnlock()

	if s.config.VectorIndexTemplate == nil {
		return fmt.Errorf("no vector index configured")
	}

	// Convert [][]float32 to []VectorNode
	nodes := make([]VectorNode, len(vectors))
	for i, vec := range vectors {
		nodes[i] = *NewVectorNodeWithID(uint32(i), vec)
	}

	return s.config.VectorIndexTemplate.Train(nodes)
}

// VectorIndex returns the underlying vector index template.
// Note: This returns the template, not a specific segment's index.
func (s *PersistentHybridIndex) VectorIndex() VectorIndex {
	return s.config.VectorIndexTemplate
}

// TextIndex returns the underlying text index template.
// Note: This returns the template, not a specific segment's index.
func (s *PersistentHybridIndex) TextIndex() TextIndex {
	return s.config.TextIndexTemplate
}

// MetadataIndex returns the underlying metadata index template.
// Note: This returns the template, not a specific segment's index.
func (s *PersistentHybridIndex) MetadataIndex() MetadataIndex {
	return s.config.MetadataIndexTemplate
}

// WriteTo is not supported for persistent storage.
// Use the storage layer's own persistence mechanism instead.
func (s *PersistentHybridIndex) WriteTo(hybridWriter, vectorWriter, textWriter, metadataWriter io.Writer) error {
	return fmt.Errorf("WriteTo not supported for persistent storage")
}

// ReadFrom is not supported for persistent storage.
// Use OpenPersistentHybridIndex to load from disk instead.
func (s *PersistentHybridIndex) ReadFrom(r io.Reader) (int64, error) {
	return 0, fmt.Errorf("ReadFrom not supported for persistent storage")
}

// persistentHybridSearch implements the HybridSearch interface for persistent storage.
type persistentHybridSearch struct {
	storage *PersistentHybridIndex

	// === Vector Search Parameters ===
	vectorQuery []float32

	// === Text Search Parameters ===
	textQueries []string

	// === Metadata Search Parameters ===
	metadataFilters []Filter
	metadataGroups  []*FilterGroup

	// === Common Search Parameters ===
	k                int
	threshold        float32
	scoreAggregation ScoreAggregationKind
	cutoff           int

	// === Vector Index Specific Parameters ===
	nProbes  int // For IVF-based indexes
	efSearch int // For HNSW index

	// === Fusion Parameters ===
	fusion Fusion // Strategy for combining vector and text scores
}

// WithVector sets the query vector for vector search
func (s *persistentHybridSearch) WithVector(query []float32) HybridSearch {
	s.vectorQuery = query
	return s
}

// WithText sets the query text(s) for text search
func (s *persistentHybridSearch) WithText(queries ...string) HybridSearch {
	s.textQueries = queries
	return s
}

// WithMetadata sets metadata filters (AND logic between filters)
func (s *persistentHybridSearch) WithMetadata(filters ...Filter) HybridSearch {
	s.metadataFilters = filters
	return s
}

// WithMetadataGroups sets complex metadata filter groups
func (s *persistentHybridSearch) WithMetadataGroups(groups ...*FilterGroup) HybridSearch {
	s.metadataGroups = groups
	return s
}

// WithK sets the number of results to return
func (s *persistentHybridSearch) WithK(k int) HybridSearch {
	s.k = k
	return s
}

// WithNProbes sets the number of probes for IVF-based vector indexes
func (s *persistentHybridSearch) WithNProbes(nProbes int) HybridSearch {
	s.nProbes = nProbes
	return s
}

// WithEfSearch sets the efSearch parameter for HNSW search
func (s *persistentHybridSearch) WithEfSearch(efSearch int) HybridSearch {
	s.efSearch = efSearch
	return s
}

// WithThreshold sets a score threshold for results
func (s *persistentHybridSearch) WithThreshold(threshold float32) HybridSearch {
	s.threshold = threshold
	return s
}

// WithScoreAggregation sets the strategy for aggregating scores
func (s *persistentHybridSearch) WithScoreAggregation(kind ScoreAggregationKind) HybridSearch {
	s.scoreAggregation = kind
	return s
}

// WithCutoff sets the autocut parameter for result cutoff
func (s *persistentHybridSearch) WithCutoff(cutoff int) HybridSearch {
	s.cutoff = cutoff
	return s
}

// WithFusion sets the fusion strategy for combining vector and text scores
func (s *persistentHybridSearch) WithFusion(fusion Fusion) HybridSearch {
	s.fusion = fusion
	return s
}

// WithFusionKind sets the fusion strategy by kind with default config
func (s *persistentHybridSearch) WithFusionKind(kind FusionKind) HybridSearch {
	fusion, err := NewFusion(kind, nil)
	if err == nil {
		s.fusion = fusion
	}
	return s
}

// Execute performs the search across all memtables and segments.
//
// The search flow is:
// 1. Search all memtables (newest first)
// 2. Search all disk segments concurrently
// 3. Merge and deduplicate results
// 4. Sort by score and return top-k
func (s *persistentHybridSearch) Execute() ([]HybridSearchResult, error) {
	s.storage.mu.RLock()
	if s.storage.closed {
		s.storage.mu.RUnlock()
		return nil, fmt.Errorf("storage is closed")
	}
	s.storage.mu.RUnlock()

	// Collect all results
	var allResults []HybridSearchResult
	resultsMu := sync.Mutex{}

	// Search memtables (newest first)
	memtables := s.storage.memtableQueue.list()
	for i := len(memtables) - 1; i >= 0; i-- {
		mt := memtables[i]

		// Create search query with all parameters
		search := mt.index.NewSearch().
			WithK(s.k).
			WithScoreAggregation(s.scoreAggregation).
			WithCutoff(s.cutoff).
			WithFusion(s.fusion)

		if s.vectorQuery != nil {
			search = search.WithVector(s.vectorQuery)
		}
		if len(s.textQueries) > 0 {
			search = search.WithText(s.textQueries...)
		}
		if len(s.metadataFilters) > 0 {
			search = search.WithMetadata(s.metadataFilters...)
		}
		if len(s.metadataGroups) > 0 {
			search = search.WithMetadataGroups(s.metadataGroups...)
		}
		if s.nProbes > 0 {
			search = search.WithNProbes(s.nProbes)
		}
		if s.efSearch > 0 {
			search = search.WithEfSearch(s.efSearch)
		}
		if s.threshold > 0 {
			search = search.WithThreshold(s.threshold)
		}

		results, err := search.Execute()
		if err != nil {
			return nil, fmt.Errorf("memtable search failed: %w", err)
		}

		resultsMu.Lock()
		allResults = append(allResults, results...)
		resultsMu.Unlock()
	}

	// Search segments concurrently
	segments := s.storage.segmentManager.list()
	if len(segments) > 0 {
		var wg sync.WaitGroup
		resultsChan := make(chan []HybridSearchResult, len(segments))

		for _, seg := range segments {
			wg.Add(1)
			go func(segment *segmentMetadata) {
				defer wg.Done()

				// Load segment index
				idx, err := segment.getIndex(
					s.storage.config.VectorIndexTemplate,
					s.storage.config.TextIndexTemplate,
					s.storage.config.MetadataIndexTemplate,
				)
				if err != nil {
					// Log error but continue with other segments
					return
				}

				// Create search query with all parameters
				search := idx.NewSearch().
					WithK(s.k).
					WithScoreAggregation(s.scoreAggregation).
					WithCutoff(s.cutoff).
					WithFusion(s.fusion)

				if s.vectorQuery != nil {
					search = search.WithVector(s.vectorQuery)
				}
				if len(s.textQueries) > 0 {
					search = search.WithText(s.textQueries...)
				}
				if len(s.metadataFilters) > 0 {
					search = search.WithMetadata(s.metadataFilters...)
				}
				if len(s.metadataGroups) > 0 {
					search = search.WithMetadataGroups(s.metadataGroups...)
				}
				if s.nProbes > 0 {
					search = search.WithNProbes(s.nProbes)
				}
				if s.efSearch > 0 {
					search = search.WithEfSearch(s.efSearch)
				}
				if s.threshold > 0 {
					search = search.WithThreshold(s.threshold)
				}

				results, err := search.Execute()
				if err != nil {
					return
				}

				resultsChan <- results
			}(seg)
		}

		// Wait for all segment searches to complete
		go func() {
			wg.Wait()
			close(resultsChan)
		}()

		// Collect segment results
		for results := range resultsChan {
			resultsMu.Lock()
			allResults = append(allResults, results...)
			resultsMu.Unlock()
		}
	}

	// Merge and deduplicate results by keeping highest score per doc
	merged := mergeResults(allResults)

	// Sort by score descending and limit to k
	sortResultsByScore(merged)
	if len(merged) > s.k {
		merged = merged[:s.k]
	}

	return merged, nil
}

// maybeScheduleFlush checks if a flush should be triggered.
func (s *PersistentHybridIndex) maybeScheduleFlush() {
	totalSize := s.memtableQueue.totalSize()

	if totalSize >= s.config.FlushThreshold {
		// Non-blocking send to flush channel
		select {
		case s.flushChan <- struct{}{}:
		default:
			// Flush already scheduled
		}
	}
}

// Flush forces a flush of all frozen memtables to disk.
// This is synchronous and blocks until flush completes.
//
// Returns:
//   - error: Error if flush fails
func (s *PersistentHybridIndex) Flush() error {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return fmt.Errorf("storage is closed")
	}
	s.mu.RUnlock()

	return s.flushMemtables()
}

// flushMemtables flushes all frozen memtables to disk segments.
func (s *PersistentHybridIndex) flushMemtables() error {
	// Get frozen memtables
	frozen := s.memtableQueue.listFrozen()
	if len(frozen) == 0 {
		return nil
	}

	for _, mt := range frozen {
		if err := s.flushMemtable(mt); err != nil {
			return fmt.Errorf("failed to flush memtable: %w", err)
		}

		// Remove from queue
		s.memtableQueue.remove(mt)
	}

	return nil
}

// flushMemtable flushes a single memtable to disk.
func (s *PersistentHybridIndex) flushMemtable(mt *memtable) error {
	// Get index from memtable
	idx, err := mt.flush()
	if err != nil {
		return fmt.Errorf("failed to get index: %w", err)
	}

	// Generate segment ID and paths
	segmentID := s.provider.nextSegmentID()
	hybridPath, vectorPath, textPath, metadataPath := s.provider.segmentPaths(segmentID)

	// Create compressed writers
	hybridFile, err := os.Create(hybridPath)
	if err != nil {
		return fmt.Errorf("failed to create hybrid file: %w", err)
	}
	defer hybridFile.Close()

	hybridGz := gzip.NewWriter(hybridFile)
	defer hybridGz.Close()

	var vectorGz, textGz, metadataGz io.WriteCloser
	var vectorFile, textFile, metadataFile *os.File

	// Create vector file if needed
	if s.config.VectorIndexTemplate != nil {
		vectorFile, err = os.Create(vectorPath)
		if err != nil {
			return fmt.Errorf("failed to create vector file: %w", err)
		}
		defer vectorFile.Close()

		vectorGz = gzip.NewWriter(vectorFile)
		defer vectorGz.Close()
	}

	// Create text file if needed
	if s.config.TextIndexTemplate != nil {
		textFile, err = os.Create(textPath)
		if err != nil {
			return fmt.Errorf("failed to create text file: %w", err)
		}
		defer textFile.Close()

		textGz = gzip.NewWriter(textFile)
		defer textGz.Close()
	}

	// Create metadata file if needed
	if s.config.MetadataIndexTemplate != nil {
		metadataFile, err = os.Create(metadataPath)
		if err != nil {
			return fmt.Errorf("failed to create metadata file: %w", err)
		}
		defer metadataFile.Close()

		metadataGz = gzip.NewWriter(metadataFile)
		defer metadataGz.Close()
	}

	// Write index to files
	if err := idx.WriteTo(hybridGz, vectorGz, textGz, metadataGz); err != nil {
		// Clean up partial files on error
		os.Remove(hybridPath)
		if vectorFile != nil {
			os.Remove(vectorPath)
		}
		if textFile != nil {
			os.Remove(textPath)
		}
		if metadataFile != nil {
			os.Remove(metadataPath)
		}
		return fmt.Errorf("failed to write index: %w", err)
	}

	// Close gzip writers to ensure all data is flushed
	if vectorGz != nil {
		vectorGz.Close()
	}
	if textGz != nil {
		textGz.Close()
	}
	if metadataGz != nil {
		metadataGz.Close()
	}
	hybridGz.Close()

	// Get file sizes
	var totalSize int64
	if info, err := os.Stat(hybridPath); err == nil {
		totalSize += info.Size()
	}
	if vectorFile != nil {
		if info, err := os.Stat(vectorPath); err == nil {
			totalSize += info.Size()
		}
	}
	if textFile != nil {
		if info, err := os.Stat(textPath); err == nil {
			totalSize += info.Size()
		}
	}
	if metadataFile != nil {
		if info, err := os.Stat(metadataPath); err == nil {
			totalSize += info.Size()
		}
	}

	// Create segment metadata
	segment := newSegmentMetadata(segmentID, hybridPath, vectorPath, textPath, metadataPath)
	segment.updateStats(mt.count(), totalSize)

	// Add to segment manager
	s.segmentManager.add(segment)

	return nil
}

// flushWorker runs in the background and handles flush requests.
func (s *PersistentHybridIndex) flushWorker() {
	defer s.wg.Done()

	for {
		select {
		case <-s.flushChan:
			if err := s.flushMemtables(); err != nil {
				// Log error but continue
				fmt.Printf("flush error: %v\n", err)
			}
		case <-s.closeChan:
			// Final flush before closing
			s.flushMemtables()
			return
		}
	}
}

// compactionWorker runs in the background and periodically compacts segments.
func (s *PersistentHybridIndex) compactionWorker() {
	defer s.wg.Done()

	ticker := time.NewTicker(s.config.CompactionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := s.maybeCompact(); err != nil {
				// Log error but continue
				fmt.Printf("compaction error: %v\n", err)
			}
		case <-s.compactionChan:
			if err := s.maybeCompact(); err != nil {
				fmt.Printf("compaction error: %v\n", err)
			}
		case <-s.closeChan:
			return
		}
	}
}

// Close closes the storage and releases all resources.
// It performs a final flush before closing.
//
// Returns:
//   - error: Error if close fails
func (s *PersistentHybridIndex) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return fmt.Errorf("storage already closed")
	}
	s.closed = true
	s.mu.Unlock()

	// Signal background workers to stop
	close(s.closeChan)

	// Wait for workers to finish
	s.wg.Wait()

	// Close provider (releases lock)
	if err := s.provider.close(); err != nil {
		return fmt.Errorf("failed to close provider: %w", err)
	}

	return nil
}
