// Package comet implements a k-Nearest Neighbors (kNN) flat index for similarity search.
//
// WHAT IS A FLAT INDEX?
// A flat index is the most naive and simple approach to similarity search. The term "flat"
// indicates that vectors are stored without any compression or transformation - they are
// stored "as-is" in their original form. This is also known as brute-force or exhaustive search.
//
// HOW kNN WORKS:
// For a given query vector Q, the algorithm:
// 1. Calculates the distance from Q to EVERY vector in the dataset
// 2. Sorts all distances
// 3. Returns the k vectors with the smallest distances
//
// TIME COMPLEXITY:
//   - Training: O(1) - No training phase required! This is one of the few ML algorithms
//     that doesn't need training.
//   - Search: O(m*n) where:
//   - n = number of vectors in the dataset
//   - m = dimensionality of each vector
//     For each of the n vectors, we need O(m) time to calculate the distance.
//
// MEMORY REQUIREMENTS:
// - 4 bytes per float32 component
// - Total per vector: 4 * d bytes (where d is the dimensionality)
// - No compression, so memory scales linearly with dataset size
//
// GUARANTEES & TRADE-OFFS:
// ✓ Pros:
//   - 100% accuracy - always finds the true nearest neighbors
//   - No training phase required
//   - Simple to implement and understand
//
// ✗ Cons:
//   - Slow for large datasets (exhaustive search)
//   - No memory compression
//   - Not scalable (O(mn) time complexity)
//
// WHEN TO USE:
// Use flat index only when:
// 1. Dataset size or embedding dimensionality is relatively small
// 2. You MUST have 100% accuracy (e.g., fingerprint matching, security applications)
// 3. Speed is not a critical concern
package comet

import (
	"encoding/binary"
	"fmt"
	"io"
	"sync"

	"github.com/RoaringBitmap/roaring"
)

// Compile-time checks to ensure FlatIndex implements VectorIndex
var _ VectorIndex = (*FlatIndex)(nil)

// FlatIndex represents a flat (brute-force) kNN index.
//
// This is the simplest form of similarity search index where vectors are stored
// with metric-appropriate preprocessing applied. For cosine distance, vectors are
// normalized to unit length. For euclidean distance, vectors are stored as-is.
//
// Thread-safety: This index is safe for concurrent use through a read-write mutex.
// Multiple readers can search simultaneously, but Add operations are exclusive.
type FlatIndex struct {
	// dim is the dimensionality of vectors stored in this index.
	// All vectors must have exactly this many dimensions.
	dim int

	// distanceKind specifies the distance function used for similarity measurement.
	distanceKind DistanceKind

	// distance is the actual distance calculator
	distance Distance

	// vectors stores all the vectors added to this index.
	// Each vector contains:
	//   - Data: the actual float32 values (dimensionality = dim)
	//   - ID: unique identifier for retrieval
	//   - For cosine metric: vectors are stored in normalized form
	//   - For euclidean metric: vectors are stored as-is
	vectors []VectorNode

	// deletedNodes tracks soft-deleted IDs using roaring bitmap
	// CRITICAL OPTIMIZATION: RoaringBitmap is much more efficient than map[uint32]bool
	// - O(log n) membership test with better memory efficiency
	// - Compressed bitmap representation
	// - Fast iteration for batch operations
	deletedNodes *roaring.Bitmap

	// mu provides thread-safe access to the index.
	// RWMutex allows multiple concurrent readers (Search) but exclusive writers (Add).
	mu sync.RWMutex
}

// NewFlatIndex creates a new flat (kNN) index with the specified dimensionality and distance metric.
//
// This is the only "training" step for a flat index, which simply initializes the data structure.
// Unlike other indexes (IVF, HNSW, PQ), there's no actual training phase or preprocessing required.
//
// Parameters:
//   - dim: The dimensionality of vectors that will be stored. All vectors must have this exact
//     dimension. For example, if you're using 384-dimensional embeddings from a sentence
//     transformer model, dim should be 384.
//   - distanceKind: The distance metric to use for similarity comparison.
//
// Returns:
//   - *FlatIndex: A new flat index ready to accept vectors via Add()
//   - error: Returns error if dim <= 0 or distance kind is invalid
//
// Time Complexity: O(1)
// Memory: O(1) initially, grows to O(n*d) where n is number of vectors added
//
// Example:
//
//	idx, err := NewFlatIndex(384, Cosine)  // For sentence embeddings
//	if err != nil { log.Fatal(err) }
func NewFlatIndex(dim int, distanceKind DistanceKind) (*FlatIndex, error) {
	// Validate dimension is positive (can't have 0 or negative dimensional vectors)
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}

	// Create distance calculator
	distance, err := NewDistance(distanceKind)
	if err != nil {
		return nil, err
	}

	// Create and return the index with an empty vector slice
	return &FlatIndex{
		dim:          dim,
		distanceKind: distanceKind,
		distance:     distance,
		vectors:      make([]VectorNode, 0),
		deletedNodes: roaring.New(), // Initialize empty bitmap for soft deletes
	}, nil
}

// Train is a no-op for flat index since vectors are stored in memory.
// This method exists to satisfy the VectorIndex interface.
//
// Returns:
//   - error: Always returns nil
func (idx *FlatIndex) Train(vectors []VectorNode) error {
	return nil
}

// Add adds a vector to the flat index.
//
// Vectors are preprocessed according to the distance metric:
//   - For cosine distance: vectors are normalized to unit length in-place
//   - For euclidean distance: vectors are stored as-is (preprocessing is a no-op)
//
// This preprocessing happens once during insertion, making all subsequent distance
// calculations more efficient by eliminating redundant norm computations and divisions.
//
// Parameters:
//   - vector: Vector to add. Must have dimensionality matching idx.dim
//
// Returns:
//   - error: Returns error if vector has wrong dimension or preprocessing fails (e.g., zero vector for cosine)
//
// Time Complexity: O(m) where m = dimensionality
//   - For L2 metric: Just appends vector (preprocessing is no-op)
//   - For cosine metric: Normalizes vector then appends
//
// Thread-safety: Acquires exclusive lock, blocking all searches during addition
func (idx *FlatIndex) Add(vector VectorNode) error {
	// Acquire exclusive write lock to prevent concurrent modifications and reads
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Validate vector dimensionality matches the index
	if len(vector.Vector()) != idx.dim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", idx.dim, len(vector.Vector()))
	}

	// Preprocess the vector according to the distance metric
	// - For cosine: normalizes to unit length (returns error if zero vector)
	// - For euclidean: no-op (always returns nil)
	if err := idx.distance.PreprocessInPlace(vector.Vector()); err != nil {
		return err
	}

	// Simply append the preprocessed vector to our flat storage
	idx.vectors = append(idx.vectors, vector)
	return nil
}

// Remove performs soft delete using roaring bitmap.
//
// CONCURRENCY OPTIMIZATION:
// - Uses read lock first (cheaper) to check if node exists
// - Only acquires write lock for the actual bitmap modification
// - Minimizes write lock contention
//
// SOFT DELETE MECHANISM:
// Instead of immediately removing from the vectors slice (expensive O(n)),
// we mark as deleted in roaring bitmap. Deleted nodes are:
//   - Skipped during search
//   - Still in vectors slice
//   - Not counted as active nodes
//
// Call Flush() periodically for actual cleanup and memory reclamation.
//
// Parameters:
//   - vector: Vector to remove (only the ID field is used for matching)
//
// Returns:
//   - error: Returns error if vector is not found or already deleted
//
// Time Complexity: O(n) for existence check + O(log n) for bitmap operation
//
// Thread-safety: Uses read lock for validation, write lock for modification
func (idx *FlatIndex) Remove(vector VectorNode) error {
	id := vector.ID()

	// ════════════════════════════════════════════════════════════════════════
	// STEP 1: CHECK EXISTENCE (READ LOCK - CHEAPER)
	// ════════════════════════════════════════════════════════════════════════
	idx.mu.RLock()
	exists := false
	for _, v := range idx.vectors {
		if v.ID() == id {
			exists = true
			break
		}
	}
	alreadyDeleted := idx.deletedNodes.Contains(id)
	idx.mu.RUnlock()

	// Fast-fail validation outside of write lock
	if !exists {
		return fmt.Errorf("vector with ID %d not found", id)
	}
	if alreadyDeleted {
		return fmt.Errorf("vector with ID %d already deleted", id)
	}

	// ════════════════════════════════════════════════════════════════════════
	// STEP 2: MARK AS DELETED (WRITE LOCK - ONLY FOR BITMAP UPDATE)
	// ════════════════════════════════════════════════════════════════════════
	idx.mu.Lock()
	idx.deletedNodes.Add(id)
	idx.mu.Unlock()

	return nil
}

// Flush performs hard delete of soft-deleted nodes.
//
// WHEN TO CALL:
//   - After multiple Remove() calls (batch cleanup)
//   - When deleted nodes are significant (e.g., > 10% of index)
//   - During off-peak hours
//
// WHAT IT DOES:
// 1. Removes all soft-deleted vectors from the vectors slice
// 2. Reclaims memory occupied by deleted vectors
// 3. Clears the deleted nodes bitmap
//
// COST: O(n) where n = number of vectors in the index
//
// Thread-safety: Acquires exclusive write lock
func (idx *FlatIndex) Flush() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Quick exit if nothing to flush
	deletedCount := int(idx.deletedNodes.GetCardinality())
	if deletedCount == 0 {
		return nil
	}

	// ═══════════════════════════════════════════════════════════════════════
	// PHASE 1: FILTER OUT DELETED VECTORS
	// ═══════════════════════════════════════════════════════════════════════
	// Pre-allocate slice with capacity for non-deleted vectors
	activeVectors := make([]VectorNode, 0, len(idx.vectors)-deletedCount)

	for _, v := range idx.vectors {
		// Keep vector only if NOT deleted
		// RoaringBitmap Contains() is very fast - O(log n)
		if !idx.deletedNodes.Contains(v.ID()) {
			activeVectors = append(activeVectors, v)
		}
	}

	// Replace vectors slice with filtered version
	idx.vectors = activeVectors

	// ═══════════════════════════════════════════════════════════════════════
	// PHASE 2: RESET DELETED TRACKING
	// ═══════════════════════════════════════════════════════════════════════
	idx.deletedNodes.Clear()

	return nil
}

// NewSearch creates a new search builder for this index.
//
// Returns:
//   - VectorSearch: A new search builder ready to be configured
func (idx *FlatIndex) NewSearch() VectorSearch {
	return &flatIndexSearch{
		index:  idx,
		k:      10, // Default k value
		cutoff: -1, // Default no cutoff
	}
}

// Dimensions returns the dimensionality of vectors stored in this index.
//
// Returns:
//   - int: The dimensionality (number of components in each vector)
func (idx *FlatIndex) Dimensions() int {
	return idx.dim
}

// DistanceKind returns the distance metric used by this index.
//
// Returns:
//   - DistanceKind: The distance metric (L2, Cosine, or Dot)
func (idx *FlatIndex) DistanceKind() DistanceKind {
	return idx.distanceKind
}

// Kind returns the type of this index.
//
// Returns:
//   - VectorIndexKind: Always returns FlatIndex
func (idx *FlatIndex) Kind() VectorIndexKind {
	return FlatIndexKind
}

// Trained returns true if the index has been trained
// This is a no-op for flat index since it doesn't need training
func (idx *FlatIndex) Trained() bool {
	return true
}

// WriteTo serializes the FlatIndex to an io.Writer.
//
// IMPORTANT: This method calls Flush() before serialization to ensure all soft-deleted
// vectors are permanently removed from the serialized data.
//
// The serialization format is:
// 1. Magic number (4 bytes) - "FLAT" identifier for validation
// 2. Version (4 bytes) - Format version for backward compatibility
// 3. Dimensionality (4 bytes)
// 4. Distance kind length (4 bytes) + distance kind string
// 5. Number of vectors (4 bytes)
// 6. For each vector:
//   - Vector ID (4 bytes)
//   - Vector dimension (4 bytes)
//   - Vector data (dim * 4 bytes as float32)
//
// 7. Deleted nodes bitmap size (4 bytes) + roaring bitmap bytes
//
// Thread-safety: Acquires read lock during serialization
//
// Returns:
//   - int64: Number of bytes written
//   - error: Returns error if write fails or flush fails
func (idx *FlatIndex) WriteTo(w io.Writer) (int64, error) {
	// Flush before serializing to remove soft-deleted vectors
	if err := idx.Flush(); err != nil {
		return 0, fmt.Errorf("failed to flush before serialization: %w", err)
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var bytesWritten int64

	// Helper function to track writes
	write := func(data interface{}) error {
		err := binary.Write(w, binary.LittleEndian, data)
		if err == nil {
			switch v := data.(type) {
			case uint32, int32, float32:
				bytesWritten += 4
			case uint64, int64, float64:
				bytesWritten += 8
			case []byte:
				bytesWritten += int64(len(v))
			case []float32:
				bytesWritten += int64(len(v) * 4)
			}
		}
		return err
	}

	// 1. Write magic number "FLAT"
	magic := [4]byte{'F', 'L', 'A', 'T'}
	if _, err := w.Write(magic[:]); err != nil {
		return bytesWritten, fmt.Errorf("failed to write magic number: %w", err)
	}
	bytesWritten += 4

	// 2. Write version
	version := uint32(1)
	if err := write(version); err != nil {
		return bytesWritten, fmt.Errorf("failed to write version: %w", err)
	}

	// 3. Write dimensionality
	if err := write(uint32(idx.dim)); err != nil {
		return bytesWritten, fmt.Errorf("failed to write dimensionality: %w", err)
	}

	// 4. Write distance kind
	distanceKindBytes := []byte(idx.distanceKind)
	if err := write(uint32(len(distanceKindBytes))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write distance kind length: %w", err)
	}
	if _, err := w.Write(distanceKindBytes); err != nil {
		return bytesWritten, fmt.Errorf("failed to write distance kind: %w", err)
	}
	bytesWritten += int64(len(distanceKindBytes))

	// 5. Write number of vectors
	if err := write(uint32(len(idx.vectors))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write vector count: %w", err)
	}

	// 6. Write each vector
	for i, node := range idx.vectors {
		// Write vector ID
		if err := write(node.ID()); err != nil {
			return bytesWritten, fmt.Errorf("failed to write vector %d ID: %w", i, err)
		}

		// Write vector dimension (for validation)
		vec := node.Vector()
		if err := write(uint32(len(vec))); err != nil {
			return bytesWritten, fmt.Errorf("failed to write vector %d dimension: %w", i, err)
		}

		// Write vector data
		for j, val := range vec {
			if err := write(val); err != nil {
				return bytesWritten, fmt.Errorf("failed to write vector %d component %d: %w", i, j, err)
			}
		}
	}

	// 7. Write deleted nodes bitmap
	bitmapBytes, err := idx.deletedNodes.ToBytes()
	if err != nil {
		return bytesWritten, fmt.Errorf("failed to serialize deleted nodes bitmap: %w", err)
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

// ReadFrom deserializes a FlatIndex from an io.Reader.
//
// This method reconstructs a FlatIndex from the serialized format created by WriteTo.
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
//	file, _ := os.Create("index.bin")
//	idx.WriteTo(file)
//	file.Close()
//
//	// Load index
//	file, _ := os.Open("index.bin")
//	idx2, _ := NewFlatIndex(384, Cosine)
//	idx2.ReadFrom(file)
//	file.Close()
func (idx *FlatIndex) ReadFrom(r io.Reader) (int64, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	var bytesRead int64

	// Helper function to track reads
	read := func(data interface{}) error {
		err := binary.Read(r, binary.LittleEndian, data)
		if err == nil {
			switch data.(type) {
			case *uint32, *int32, *float32:
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
	if string(magic) != "FLAT" {
		return bytesRead, fmt.Errorf("invalid magic number: expected 'FLAT', got '%s'", string(magic))
	}

	// 2. Read version
	var version uint32
	if err := read(&version); err != nil {
		return bytesRead, fmt.Errorf("failed to read version: %w", err)
	}
	if version != 1 {
		return bytesRead, fmt.Errorf("unsupported version: %d", version)
	}

	// 3. Read dimensionality
	var dim uint32
	if err := read(&dim); err != nil {
		return bytesRead, fmt.Errorf("failed to read dimensionality: %w", err)
	}

	// Validate dimension matches
	if int(dim) != idx.dim {
		return bytesRead, fmt.Errorf("dimension mismatch: index has dim=%d, serialized data has dim=%d", idx.dim, dim)
	}

	// 4. Read distance kind
	var distanceKindLen uint32
	if err := read(&distanceKindLen); err != nil {
		return bytesRead, fmt.Errorf("failed to read distance kind length: %w", err)
	}

	distanceKindBytes := make([]byte, distanceKindLen)
	if _, err := io.ReadFull(r, distanceKindBytes); err != nil {
		return bytesRead, fmt.Errorf("failed to read distance kind: %w", err)
	}
	bytesRead += int64(distanceKindLen)

	distanceKind := DistanceKind(distanceKindBytes)
	if distanceKind != idx.distanceKind {
		return bytesRead, fmt.Errorf("distance kind mismatch: index uses '%s', serialized data uses '%s'", idx.distanceKind, distanceKind)
	}

	// 5. Read number of vectors
	var vectorCount uint32
	if err := read(&vectorCount); err != nil {
		return bytesRead, fmt.Errorf("failed to read vector count: %w", err)
	}

	// 6. Read vectors
	vectors := make([]VectorNode, vectorCount)
	for i := uint32(0); i < vectorCount; i++ {
		// Read vector ID
		var id uint32
		if err := read(&id); err != nil {
			return bytesRead, fmt.Errorf("failed to read vector %d ID: %w", i, err)
		}

		// Read vector dimension
		var vecDim uint32
		if err := read(&vecDim); err != nil {
			return bytesRead, fmt.Errorf("failed to read vector %d dimension: %w", i, err)
		}

		// Validate dimension
		if vecDim != dim {
			return bytesRead, fmt.Errorf("vector %d has dimension %d, expected %d", i, vecDim, dim)
		}

		// Read vector data
		vec := make([]float32, vecDim)
		for j := uint32(0); j < vecDim; j++ {
			if err := read(&vec[j]); err != nil {
				return bytesRead, fmt.Errorf("failed to read vector %d component %d: %w", i, j, err)
			}
		}

		vectors[i] = *NewVectorNodeWithID(id, vec)
	}

	// 7. Read deleted nodes bitmap
	var bitmapSize uint32
	if err := read(&bitmapSize); err != nil {
		return bytesRead, fmt.Errorf("failed to read bitmap size: %w", err)
	}

	bitmapBytes := make([]byte, bitmapSize)
	if _, err := io.ReadFull(r, bitmapBytes); err != nil {
		return bytesRead, fmt.Errorf("failed to read bitmap data: %w", err)
	}
	bytesRead += int64(bitmapSize)

	deletedNodes := roaring.New()
	if err := deletedNodes.UnmarshalBinary(bitmapBytes); err != nil {
		return bytesRead, fmt.Errorf("failed to deserialize deleted nodes bitmap: %w", err)
	}

	// Update index state
	idx.vectors = vectors
	idx.deletedNodes = deletedNodes

	return bytesRead, nil
}
