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
	"fmt"
	"sync"
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
	}, nil
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

// Remove removes a vector from the flat index.
//
// This performs a linear search to find the vector with the matching ID and removes it.
// For large indexes, this operation can be slow O(n).
//
// Parameters:
//   - vector: Vector to remove (only the ID field is used for matching)
//
// Returns:
//   - error: Returns error if vector is not found
//
// Time Complexity: O(n) where n = number of vectors
//
// Thread-safety: Acquires exclusive lock
func (idx *FlatIndex) Remove(vector VectorNode) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Find the vector with matching ID
	for i, v := range idx.vectors {
		if v.ID() == vector.ID() {
			// Remove by slicing around the element
			idx.vectors = append(idx.vectors[:i], idx.vectors[i+1:]...)
			return nil
		}
	}

	return fmt.Errorf("vector with ID %d not found", vector.ID())
}

// Flush is a no-op for flat index since vectors are stored in memory.
// This method exists to satisfy the VectorIndex interface.
//
// Returns:
//   - error: Always returns nil
func (idx *FlatIndex) Flush() error {
	return nil
}

// NewSearch creates a new search builder for this index.
//
// Returns:
//   - VectorSearch: A new search builder ready to be configured
func (idx *FlatIndex) NewSearch() VectorSearch {
	return &flatIndexSearch{
		index: idx,
		k:     10, // Default k value
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
