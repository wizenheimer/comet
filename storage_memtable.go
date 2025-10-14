package comet

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// memtable is a write buffer that holds recent index updates in memory.
// Once it reaches a size threshold, it is marked immutable and flushed to disk.
//
// Thread-safety: All methods are safe for concurrent use.
type memtable struct {
	mu sync.RWMutex

	// The underlying hybrid search index
	index HybridSearchIndex

	// Memory usage tracking
	sizeUsed  atomic.Int64
	sizeLimit int64

	// State tracking
	frozen    atomic.Bool // Set to true when ready to flush
	createdAt time.Time

	// Statistics
	numDocs atomic.Uint32
}

// newMemtable creates a new memtable with the given size limit.
//
// Parameters:
//   - vecIdx: Vector index to use (can be nil)
//   - txtIdx: Text index to use (can be nil)
//   - metaIdx: Metadata index to use (can be nil)
//   - sizeLimit: Maximum memory size in bytes before rotation
//
// Returns:
//   - *memtable: New memtable instance
func newMemtable(vecIdx VectorIndex, txtIdx TextIndex, metaIdx MetadataIndex, sizeLimit int64) *memtable {
	return &memtable{
		index:     NewHybridSearchIndex(vecIdx, txtIdx, metaIdx),
		sizeLimit: sizeLimit,
		createdAt: time.Now(),
	}
}

// add adds a document to the memtable.
// Returns an error if the memtable is frozen or if the add fails.
//
// Parameters:
//   - vector: Document vector embedding (can be nil)
//   - text: Document text (can be empty)
//   - metadata: Document metadata (can be nil)
//
// Returns:
//   - uint32: Generated document ID
//   - error: Error if add fails or memtable is frozen
func (m *memtable) add(vector []float32, text string, metadata map[string]interface{}) (uint32, error) {
	if m.frozen.Load() {
		return 0, fmt.Errorf("memtable is frozen")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Add to underlying index
	id, err := m.index.Add(vector, text, metadata)
	if err != nil {
		return 0, fmt.Errorf("failed to add to index: %w", err)
	}

	// Estimate and update size
	size := m.estimateDocumentSize(vector, text, metadata)
	m.sizeUsed.Add(size)
	m.numDocs.Add(1)

	return id, nil
}

// addWithID adds a document with a specific ID to the memtable.
//
// Parameters:
//   - id: Document ID to use
//   - vector: Document vector embedding (can be nil)
//   - text: Document text (can be empty)
//   - metadata: Document metadata (can be nil)
//
// Returns:
//   - error: Error if add fails or memtable is frozen
func (m *memtable) addWithID(id uint32, vector []float32, text string, metadata map[string]interface{}) error {
	if m.frozen.Load() {
		return fmt.Errorf("memtable is frozen")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Add to underlying index
	if err := m.index.AddWithID(id, vector, text, metadata); err != nil {
		return fmt.Errorf("failed to add to index: %w", err)
	}

	// Estimate and update size
	size := m.estimateDocumentSize(vector, text, metadata)
	m.sizeUsed.Add(size)
	m.numDocs.Add(1)

	return nil
}

// remove removes a document from the memtable.
//
// Parameters:
//   - id: Document ID to remove
//
// Returns:
//   - error: Error if removal fails
func (m *memtable) remove(id uint32) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.index.Remove(id)
}

// Removed search() method - not useful as public API.
// Users should search through PersistentHybridIndex, not individual memtables.

// hasRoomFor checks if the memtable has room for a document of the given size.
//
// Parameters:
//   - vector: Document vector
//   - text: Document text
//   - metadata: Document metadata
//
// Returns:
//   - bool: True if there is room, false otherwise
func (m *memtable) hasRoomFor(vector []float32, text string, metadata map[string]interface{}) bool {
	if m.frozen.Load() {
		return false
	}

	estimatedSize := m.estimateDocumentSize(vector, text, metadata)
	currentSize := m.sizeUsed.Load()

	return currentSize+estimatedSize <= m.sizeLimit
}

// freeze marks the memtable as immutable and ready for flushing.
// After freezing, no more writes are accepted.
func (m *memtable) freeze() {
	m.frozen.Store(true)
}

// IsFrozen returns true if the memtable is frozen.
//
// Useful for monitoring memtable state.
//
// Example:
//
//	if memtable.IsFrozen() {
//	    log.Info("Memtable ready for flush")
//	}
func (m *memtable) IsFrozen() bool {
	return m.frozen.Load()
}

// size returns the current memory usage in bytes.
func (m *memtable) size() int64 {
	return m.sizeUsed.Load()
}

// count returns the number of documents in the memtable.
func (m *memtable) count() uint32 {
	return m.numDocs.Load()
}

// Age returns the age of the memtable.
//
// Useful for monitoring stale memtables.
//
// Example:
//
//	if memtable.Age() > 10*time.Minute {
//	    log.Warn("Stale memtable detected")
//	}
func (m *memtable) Age() time.Duration {
	return time.Since(m.createdAt)
}

// estimateDocumentSize estimates the memory size of a document.
// This is a heuristic and doesn't need to be exact.
//
// Memory breakdown:
//   - Vector: 4 bytes per float32 dimension
//   - Text: 1 byte per character + inverted index overhead (estimated as 2x)
//   - Metadata: Estimated based on number of fields and values
func (m *memtable) estimateDocumentSize(vector []float32, text string, metadata map[string]interface{}) int64 {
	var size int64

	// Vector size: 4 bytes per float32
	if vector != nil {
		size += int64(len(vector) * 4)
	}

	// Text size: includes tokens + inverted index overhead
	// Rough estimate: 2x the text length for tokens + posting lists
	if text != "" {
		size += int64(len(text) * 2)
	}

	// Metadata size: rough estimate
	// Each field: ~32 bytes for key + 64 bytes for value (average)
	if metadata != nil {
		size += int64(len(metadata) * 96)
	}

	// Base overhead per document (ID tracking, etc.)
	size += 64

	return size
}

// flush returns the underlying index for serialization.
// The memtable should be frozen before calling this.
func (m *memtable) flush() (HybridSearchIndex, error) {
	if !m.frozen.Load() {
		return nil, fmt.Errorf("memtable must be frozen before flush")
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.index, nil
}

// memtableQueue manages a queue of memtables for write ordering.
//
// Thread-safety: All methods are safe for concurrent use.
type memtableQueue struct {
	mu sync.RWMutex

	// Active memtable for writes (always the last in queue)
	mutable *memtable

	// Queue of memtables (includes mutable at the end)
	// Ordered from oldest (index 0) to newest (last index)
	queue []*memtable

	// Configuration
	vecIdxTemplate    VectorIndex
	txtIdxTemplate    TextIndex
	metaIdxTemplate   MetadataIndex
	memtableSizeLimit int64
}

// newMemtableQueue creates a new memtable queue.
func newMemtableQueue(vecIdx VectorIndex, txtIdx TextIndex, metaIdx MetadataIndex, sizeLimit int64) *memtableQueue {
	mutable := newMemtable(vecIdx, txtIdx, metaIdx, sizeLimit)

	return &memtableQueue{
		mutable:           mutable,
		queue:             []*memtable{mutable},
		vecIdxTemplate:    vecIdx,
		txtIdxTemplate:    txtIdx,
		metaIdxTemplate:   metaIdx,
		memtableSizeLimit: sizeLimit,
	}
}

// add adds a document to the active memtable.
// If the memtable doesn't have room, it rotates to a new one.
func (mq *memtableQueue) add(vector []float32, text string, metadata map[string]interface{}) (uint32, error) {
	mq.mu.Lock()

	// Check if we need to rotate
	if !mq.mutable.hasRoomFor(vector, text, metadata) {
		mq.rotateNoLock()
	}

	mutable := mq.mutable
	mq.mu.Unlock()

	return mutable.add(vector, text, metadata)
}

// addWithID adds a document with a specific ID to the active memtable.
func (mq *memtableQueue) addWithID(id uint32, vector []float32, text string, metadata map[string]interface{}) error {
	mq.mu.Lock()

	// Check if we need to rotate
	if !mq.mutable.hasRoomFor(vector, text, metadata) {
		mq.rotateNoLock()
	}

	mutable := mq.mutable
	mq.mu.Unlock()

	return mutable.addWithID(id, vector, text, metadata)
}

// Rotate creates a new mutable memtable and freezes the old one.
//
// Useful for forcing rotation before the size limit is reached,
// such as before maintenance windows or to ensure data is flushed.
//
// Example:
//
//	// Force rotation before maintenance
//	store.memtableQueue.Rotate()
//	store.Flush()
func (mq *memtableQueue) Rotate() {
	mq.mu.Lock()
	defer mq.mu.Unlock()
	mq.rotateNoLock()
}

// rotateNoLock performs rotation without acquiring the lock.
// Must be called with mq.mu held.
func (mq *memtableQueue) rotateNoLock() {
	// Freeze the current mutable memtable
	mq.mutable.freeze()

	// Create new mutable memtable
	mq.mutable = newMemtable(
		mq.vecIdxTemplate,
		mq.txtIdxTemplate,
		mq.metaIdxTemplate,
		mq.memtableSizeLimit,
	)

	// Add to queue
	mq.queue = append(mq.queue, mq.mutable)
}

// list returns all memtables (oldest first, including mutable).
func (mq *memtableQueue) list() []*memtable {
	mq.mu.RLock()
	defer mq.mu.RUnlock()

	result := make([]*memtable, len(mq.queue))
	copy(result, mq.queue)
	return result
}

// listFrozen returns all frozen memtables (oldest first, excluding mutable).
func (mq *memtableQueue) listFrozen() []*memtable {
	mq.mu.RLock()
	defer mq.mu.RUnlock()

	if len(mq.queue) <= 1 {
		return nil
	}

	// All except the last one (which is mutable)
	result := make([]*memtable, len(mq.queue)-1)
	copy(result, mq.queue[:len(mq.queue)-1])
	return result
}

// remove removes a memtable from the queue.
func (mq *memtableQueue) remove(m *memtable) {
	mq.mu.Lock()
	defer mq.mu.Unlock()

	for i, mt := range mq.queue {
		if mt == m {
			// Remove by swapping with last element
			mq.queue[i] = mq.queue[len(mq.queue)-1]
			mq.queue = mq.queue[:len(mq.queue)-1]
			return
		}
	}
}

// totalSize returns the total memory usage of all memtables.
func (mq *memtableQueue) totalSize() int64 {
	mq.mu.RLock()
	defer mq.mu.RUnlock()

	var total int64
	for _, mt := range mq.queue {
		total += mt.size()
	}

	return total
}

// Count returns the number of memtables in the queue.
//
// Useful for monitoring queue depth.
//
// Example:
//
//	queueDepth := store.memtableQueue.Count()
//	fmt.Printf("Memtable queue: %d\n", queueDepth)
func (mq *memtableQueue) Count() int {
	mq.mu.RLock()
	defer mq.mu.RUnlock()
	return len(mq.queue)
}
