// Package comet implements a k-Nearest Neighbors (kNN) IVF index for similarity search.
//
// WHAT IS AN IVF INDEX?
// IVF (Inverted File Index) is a partitioning-based approximate nearest neighbor search
// algorithm. It divides the vector space into Voronoi cells using k-means clustering,
// then searches only the nearest cells instead of scanning all vectors.
//
// HOW IVF WORKS:
// Training Phase:
// 1. Run k-means on training vectors to learn nlist cluster centroids
// 2. These centroids define Voronoi partitions of the vector space
//
// Indexing Phase:
// 1. For each vector, find its nearest centroid
// 2. Add the vector to that centroid's inverted list
//
// Search Phase:
// 1. Find the nprobe nearest centroids to the query vector
// 2. Search only the vectors in those nprobe inverted lists
// 3. Return the top-k nearest neighbors from candidates
//
// TIME COMPLEXITY:
//   - Training: O(iterations × nlist × n × dim) - k-means clustering
//   - Add: O(nlist × dim) - find nearest centroid
//   - Search: O(nprobe × (nlist/nprobe) × dim + nprobe × (n/nlist) × dim)
//     ≈ O(nprobe × dim + (nprobe/nlist) × n × dim)
//
// MEMORY REQUIREMENTS:
// - Vectors: 4 × n × dim bytes (stored as-is)
// - Centroids: 4 × nlist × dim bytes
// - Lists: negligible overhead (just pointers)
// - Total: ~4 × (n + nlist) × dim bytes
//
// ACCURACY VS SPEED TRADEOFF:
// - nprobe = 1: Fastest, lowest recall (~30-50%)
// - nprobe = sqrt(nlist): Good balance (~70-90% recall)
// - nprobe = nlist: Same as flat search (100% recall)
//
// CHOOSING NLIST:
// Rule of thumb: nlist = sqrt(n) or nlist = 4*sqrt(n)
// - For 1M vectors: nlist = 1,000 to 4,000
// - For 100K vectors: nlist = 316 to 1,264
// - For 10K vectors: nlist = 100 to 400
//
// WHEN TO USE IVF:
// Use IVF when:
// 1. Dataset is large (>10K vectors)
// 2. You can tolerate ~90-95% recall (not 100%)
// 3. You want 10-100x speedup over flat search
// 4. Memory usage is not a primary concern
//
// DON'T use IVF when:
// 1. Dataset is small (<10K vectors) - use flat index
// 2. You need 100% recall - use flat index
// 3. Memory is very limited - use PQ or IVFPQ
package comet

import (
	"fmt"
	"math"
	"sync"
)

// Compile-time checks to ensure IVFIndex implements VectorIndex
var _ VectorIndex = (*IVFIndex)(nil)

// Compile-time checks to ensure ivfIndexSearch implements VectorSearch
var _ VectorSearch = (*ivfIndexSearch)(nil)

// IVFIndex represents an Inverted File index for approximate nearest neighbor search.
//
// The index partitions the vector space into nlist Voronoi cells using k-means clustering.
// Each cell has an inverted list containing vectors assigned to that cell's centroid.
// During search, only the nearest nprobe cells are examined, providing a speed-accuracy tradeoff.
//
// Thread-safety: This index is safe for concurrent use through a read-write mutex.
// Multiple readers can search simultaneously, but training and adding are exclusive.
type IVFIndex struct {
	// dim is the dimensionality of vectors stored in this index
	dim int

	// distanceKind specifies the distance function used for similarity measurement
	distanceKind DistanceKind

	// distance is the actual distance calculator
	distance Distance

	// nlist is the number of Voronoi cells (clusters) in the index
	// Typical value: sqrt(n) where n is the number of vectors
	nlist int

	// centroids are the learned cluster centers that define the Voronoi partitions
	// Length: nlist, each centroid has dimensionality dim
	centroids [][]float32

	// lists are the inverted lists, one per centroid
	// lists[i] contains all vectors assigned to centroids[i]
	// This is the "inverted file" part of IVF
	lists [][]VectorNode

	// mu provides thread-safe access to the index
	// RWMutex allows multiple concurrent readers (Search) but exclusive writers (Train, Add)
	mu sync.RWMutex

	// trained indicates whether the index has been trained (centroids learned)
	// Must be true before adding or searching vectors
	trained bool
}

// NewIVFIndex creates a new IVF index with the specified parameters.
//
// The index must be trained with representative vectors before it can be used.
// Training learns the Voronoi partitions (cluster centroids) via k-means clustering.
//
// Parameters:
//   - dim: The dimensionality of vectors. All vectors must have this exact dimension.
//   - nlist: Number of Voronoi cells (clusters). Typical: sqrt(n) or 4*sqrt(n).
//   - distanceKind: The distance metric for similarity comparison.
//
// Returns:
//   - *IVFIndex: A new IVF index ready to be trained
//   - error: Returns error if parameters are invalid
//
// Choosing nlist:
//   - Too small: Poor speed improvement (searching too many vectors per cell)
//   - Too large: Poor recall (query's neighbors scattered across many cells)
//   - Rule of thumb: nlist = sqrt(expected_dataset_size)
//
// Example:
//
//	idx, err := NewIVFIndex(384, 316, Cosine)  // For ~100K vectors
//	if err != nil { log.Fatal(err) }
//
//	// Must train before use
//	err = idx.Train(trainingVectors)
func NewIVFIndex(dim int, nlist int, distanceKind DistanceKind) (*IVFIndex, error) {
	// Validate dimension is positive
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}

	// Validate nlist is positive
	if nlist <= 0 {
		return nil, fmt.Errorf("nlist must be positive")
	}

	// Create distance calculator
	distance, err := NewDistance(distanceKind)
	if err != nil {
		return nil, err
	}

	return &IVFIndex{
		dim:          dim,
		distanceKind: distanceKind,
		distance:     distance,
		nlist:        nlist,
		lists:        make([][]VectorNode, nlist),
		trained:      false,
	}, nil
}

// Train learns the Voronoi partitions (cluster centroids) using k-means clustering.
//
// Training is required before the index can accept vectors or perform searches.
// The training vectors should be representative of the full dataset distribution.
//
// Algorithm:
// 1. Run k-means clustering with k=nlist on the training vectors
// 2. Store the learned centroids
// 3. Mark the index as trained
//
// Parameters:
//   - vectors: Training vectors used to learn cluster centroids.
//     Should be at least nlist vectors, ideally 10-100x more.
//
// Returns:
//   - error: Returns error if insufficient training vectors or training fails
//
// Training Tips:
//   - Use at least max(nlist, 1000) training vectors
//   - More training vectors = better centroids = better recall
//   - Training vectors should represent your full dataset's distribution
//   - Can use a random sample of your full dataset
//
// Time Complexity: O(iterations × nlist × n × dim) where n = len(vectors)
// Typical training time: seconds to minutes depending on dataset size
//
// Example:
//
//	// Sample 100K vectors from your dataset for training
//	trainingVectors := sampleVectors(allVectors, 100000)
//	err := idx.Train(trainingVectors)
func (idx *IVFIndex) Train(vectors []VectorNode) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Validate we have enough training vectors
	if len(vectors) < idx.nlist {
		return fmt.Errorf("need at least %d training vectors for %d clusters (got %d)",
			idx.nlist, idx.nlist, len(vectors))
	}

	// Extract raw float32 slices for k-means
	rawVectors := make([][]float32, len(vectors))
	for i, v := range vectors {
		rawVectors[i] = v.Vector()
	}

	// Run k-means clustering to learn centroids
	// maxIter=20 is typically sufficient for convergence
	centroids, _ := KMeans(rawVectors, idx.nlist, idx.distance, 20)

	if centroids == nil {
		return fmt.Errorf("k-means clustering failed")
	}

	// Store the learned centroids
	idx.centroids = centroids
	idx.trained = true

	return nil
}

// Add adds a vector to the IVF index.
//
// The vector is assigned to the nearest centroid and added to that centroid's
// inverted list. This is a simple O(nlist × dim) operation.
//
// Parameters:
//   - vector: Vector to add. Must have dimensionality matching idx.dim
//
// Returns:
//   - error: Returns error if index not trained, wrong dimension, or preprocessing fails
//
// Time Complexity: O(nlist × dim) - compute distance to all centroids
//
// Thread-safety: Acquires exclusive lock, blocking all searches during addition
func (idx *IVFIndex) Add(vector VectorNode) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Verify index has been trained
	if !idx.trained {
		return fmt.Errorf("index must be trained before adding vectors")
	}

	// Validate vector dimensionality
	if len(vector.Vector()) != idx.dim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d",
			idx.dim, len(vector.Vector()))
	}

	// Preprocess the vector according to the distance metric
	// For cosine: normalizes to unit length (returns error if zero vector)
	// For euclidean: no-op (always returns nil)
	if err := idx.distance.PreprocessInPlace(vector.Vector()); err != nil {
		return err
	}

	// Find the nearest centroid (call utility directly since we already hold write lock)
	nearestCentroidIdx := FindNearestCentroidIndex(vector.Vector(), idx.centroids, idx.distance)

	// Add vector to the corresponding inverted list
	idx.lists[nearestCentroidIdx] = append(idx.lists[nearestCentroidIdx], vector)

	return nil
}

// Remove removes a vector from the IVF index.
//
// This performs a linear search through all inverted lists to find and remove
// the vector with the matching ID. For large indexes, this can be slow.
//
// Parameters:
//   - vector: Vector to remove (only the ID field is used for matching)
//
// Returns:
//   - error: Returns error if vector is not found
//
// Time Complexity: O(n) worst case - may need to scan all vectors
//
// Thread-safety: Acquires exclusive lock
func (idx *IVFIndex) Remove(vector VectorNode) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Search through all inverted lists
	for listIdx := range idx.lists {
		for i, v := range idx.lists[listIdx] {
			if v.ID() == vector.ID() {
				// Remove by slicing around the element
				idx.lists[listIdx] = append(
					idx.lists[listIdx][:i],
					idx.lists[listIdx][i+1:]...)
				return nil
			}
		}
	}

	return fmt.Errorf("vector with ID %d not found", vector.ID())
}

// Flush is a no-op for IVF index since vectors are stored in memory.
// This method exists to satisfy the VectorIndex interface.
func (idx *IVFIndex) Flush() error {
	return nil
}

// NewSearch creates a new search builder for this index.
//
// Returns:
//   - VectorSearch: A new search builder ready to be configured
func (idx *IVFIndex) NewSearch() VectorSearch {
	return &ivfIndexSearch{
		index:   idx,
		k:       10,                                 // Default k
		nprobes: int(math.Sqrt(float64(idx.nlist))), // Default nprobes = sqrt(nlist)
	}
}

// Dimensions returns the dimensionality of vectors stored in this index.
func (idx *IVFIndex) Dimensions() int {
	return idx.dim
}

// DistanceKind returns the distance metric used by this index.
func (idx *IVFIndex) DistanceKind() DistanceKind {
	return idx.distanceKind
}

// Kind returns the type of this index.
func (idx *IVFIndex) Kind() VectorIndexKind {
	return IVFIndexKind
}

// Trained returns true if the index has been trained
func (idx *IVFIndex) Trained() bool {
	return idx.trained
}
