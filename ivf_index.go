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
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"sync"

	"github.com/RoaringBitmap/roaring"
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

	// deletedNodes tracks soft-deleted IDs using roaring bitmap
	// CRITICAL OPTIMIZATION: RoaringBitmap is much more efficient than map[uint32]bool
	// - O(log n) membership test with better memory efficiency
	// - Compressed bitmap representation
	// - Fast iteration for batch operations
	deletedNodes *roaring.Bitmap

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
		deletedNodes: roaring.New(), // Initialize empty bitmap for soft deletes
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

// Remove performs soft delete using roaring bitmap.
//
// CONCURRENCY OPTIMIZATION:
// - Uses read lock first (cheaper) to check if node exists
// - Only acquires write lock for the actual bitmap modification
// - Minimizes write lock contention
//
// SOFT DELETE MECHANISM:
// Instead of immediately removing from inverted lists (expensive O(n)),
// we mark as deleted in roaring bitmap. Deleted nodes are:
//   - Skipped during search
//   - Still in inverted lists
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
func (idx *IVFIndex) Remove(vector VectorNode) error {
	id := vector.ID()

	// ════════════════════════════════════════════════════════════════════════
	// STEP 1: CHECK EXISTENCE (READ LOCK - CHEAPER)
	// ════════════════════════════════════════════════════════════════════════
	idx.mu.RLock()
	exists := false
	for _, list := range idx.lists {
		for _, v := range list {
			if v.ID() == id {
				exists = true
				break
			}
		}
		if exists {
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
// 1. Removes all soft-deleted vectors from all inverted lists
// 2. Reclaims memory occupied by deleted vectors
// 3. Clears the deleted nodes bitmap
//
// COST: O(n) where n = total number of vectors across all lists
//
// Thread-safety: Acquires exclusive write lock
func (idx *IVFIndex) Flush() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Quick exit if nothing to flush
	deletedCount := int(idx.deletedNodes.GetCardinality())
	if deletedCount == 0 {
		return nil
	}

	// ═══════════════════════════════════════════════════════════════════════
	// PHASE 1: FILTER OUT DELETED VECTORS FROM ALL INVERTED LISTS
	// ═══════════════════════════════════════════════════════════════════════
	for listIdx := range idx.lists {
		if len(idx.lists[listIdx]) == 0 {
			continue
		}

		// Filter this inverted list
		filtered := make([]VectorNode, 0, len(idx.lists[listIdx]))
		for _, v := range idx.lists[listIdx] {
			// Keep vector only if NOT deleted
			// RoaringBitmap Contains() is very fast - O(log n)
			if !idx.deletedNodes.Contains(v.ID()) {
				filtered = append(filtered, v)
			}
		}

		// Replace list with filtered version
		idx.lists[listIdx] = filtered
	}

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
func (idx *IVFIndex) NewSearch() VectorSearch {
	return &ivfIndexSearch{
		index:   idx,
		k:       10,                                 // Default k
		nprobes: int(math.Sqrt(float64(idx.nlist))), // Default nprobes = sqrt(nlist)
		cutoff:  -1,                                 // Default no cutoff
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

// WriteTo serializes the IVFIndex to an io.Writer.
//
// IMPORTANT: This method calls Flush() before serialization to ensure all soft-deleted
// vectors are permanently removed from the serialized data.
//
// The serialization format is:
// 1. Magic number (4 bytes) - "IVFX" identifier for validation
// 2. Version (4 bytes) - Format version for backward compatibility
// 3. Basic parameters:
//   - Dimensionality (4 bytes)
//   - Distance kind length (4 bytes) + distance kind string
//   - nlist (4 bytes) - number of IVF clusters
//   - trained (1 byte) - whether index is trained
//
// 4. Centroids (only if trained):
//   - For each of nlist centroids:
//   - Centroid size (4 bytes)
//   - Centroid data (dim * 4 bytes as float32)
//
// 5. Number of inverted lists (4 bytes)
// 6. For each inverted list:
//   - List size (4 bytes)
//   - For each vector:
//   - Vector ID (4 bytes)
//   - Vector data (dim * 4 bytes as float32)
//
// 7. Deleted nodes bitmap size (4 bytes) + roaring bitmap bytes
//
// Thread-safety: Acquires read lock during serialization
//
// Returns:
//   - int64: Number of bytes written
//   - error: Returns error if write fails or flush fails
func (idx *IVFIndex) WriteTo(w io.Writer) (int64, error) {
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
			switch data.(type) {
			case uint32, int32, float32:
				bytesWritten += 4
			case uint8, int8, bool:
				bytesWritten += 1
			}
		}
		return err
	}

	// 1. Write magic number "IVFX"
	magic := [4]byte{'I', 'V', 'F', 'X'}
	if _, err := w.Write(magic[:]); err != nil {
		return bytesWritten, fmt.Errorf("failed to write magic number: %w", err)
	}
	bytesWritten += 4

	// 2. Write version
	version := uint32(1)
	if err := write(version); err != nil {
		return bytesWritten, fmt.Errorf("failed to write version: %w", err)
	}

	// 3. Write basic parameters
	if err := write(uint32(idx.dim)); err != nil {
		return bytesWritten, fmt.Errorf("failed to write dimensionality: %w", err)
	}

	// Write distance kind
	distanceKindBytes := []byte(idx.distanceKind)
	if err := write(uint32(len(distanceKindBytes))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write distance kind length: %w", err)
	}
	if _, err := w.Write(distanceKindBytes); err != nil {
		return bytesWritten, fmt.Errorf("failed to write distance kind: %w", err)
	}
	bytesWritten += int64(len(distanceKindBytes))

	if err := write(uint32(idx.nlist)); err != nil {
		return bytesWritten, fmt.Errorf("failed to write nlist: %w", err)
	}

	// Write trained flag
	trainedByte := uint8(0)
	if idx.trained {
		trainedByte = 1
	}
	if err := write(trainedByte); err != nil {
		return bytesWritten, fmt.Errorf("failed to write trained flag: %w", err)
	}

	// 4. Write centroids (only if trained)
	if idx.trained {
		for i, centroid := range idx.centroids {
			// Write centroid size
			if err := write(uint32(len(centroid))); err != nil {
				return bytesWritten, fmt.Errorf("failed to write centroid %d size: %w", i, err)
			}

			// Write centroid data
			for _, val := range centroid {
				if err := write(val); err != nil {
					return bytesWritten, fmt.Errorf("failed to write centroid %d data: %w", i, err)
				}
			}
		}
	}

	// 5. Write number of inverted lists
	if err := write(uint32(len(idx.lists))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write list count: %w", err)
	}

	// 6. Write each inverted list
	for listIdx, list := range idx.lists {
		// Write list size
		if err := write(uint32(len(list))); err != nil {
			return bytesWritten, fmt.Errorf("failed to write list %d size: %w", listIdx, err)
		}

		// Write each vector in the list
		for i, node := range list {
			// Write vector ID
			if err := write(node.ID()); err != nil {
				return bytesWritten, fmt.Errorf("failed to write list %d vector %d ID: %w", listIdx, i, err)
			}

			// Write vector data
			vec := node.Vector()
			for j, val := range vec {
				if err := write(val); err != nil {
					return bytesWritten, fmt.Errorf("failed to write list %d vector %d data[%d]: %w", listIdx, i, j, err)
				}
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

// ReadFrom deserializes an IVFIndex from an io.Reader.
//
// This method reconstructs an IVFIndex from the serialized format created by WriteTo.
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
//	idx2, _ := NewIVFIndex(384, 100, Cosine)
//	idx2.ReadFrom(file)
//	file.Close()
func (idx *IVFIndex) ReadFrom(r io.Reader) (int64, error) {
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
			case *uint8, *int8, *bool:
				bytesRead += 1
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
	if string(magic) != "IVFX" {
		return bytesRead, fmt.Errorf("invalid magic number: expected 'IVFX', got '%s'", string(magic))
	}

	// 2. Read version
	var version uint32
	if err := read(&version); err != nil {
		return bytesRead, fmt.Errorf("failed to read version: %w", err)
	}
	if version != 1 {
		return bytesRead, fmt.Errorf("unsupported version: %d", version)
	}

	// 3. Read basic parameters
	var dim uint32
	if err := read(&dim); err != nil {
		return bytesRead, fmt.Errorf("failed to read dimensionality: %w", err)
	}

	// Validate dimension matches
	if int(dim) != idx.dim {
		return bytesRead, fmt.Errorf("dimension mismatch: index has dim=%d, serialized data has dim=%d", idx.dim, dim)
	}

	// Read distance kind
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

	// Read nlist
	var nlist uint32
	if err := read(&nlist); err != nil {
		return bytesRead, fmt.Errorf("failed to read nlist: %w", err)
	}

	// Validate nlist matches
	if int(nlist) != idx.nlist {
		return bytesRead, fmt.Errorf("nlist mismatch: index has nlist=%d, serialized data has nlist=%d", idx.nlist, nlist)
	}

	// Read trained flag
	var trainedByte uint8
	if err := read(&trainedByte); err != nil {
		return bytesRead, fmt.Errorf("failed to read trained flag: %w", err)
	}
	trained := trainedByte == 1

	// 4. Read centroids (only if trained)
	var centroids [][]float32
	if trained {
		centroids = make([][]float32, idx.nlist)
		for i := 0; i < idx.nlist; i++ {
			// Read centroid size
			var centroidSize uint32
			if err := read(&centroidSize); err != nil {
				return bytesRead, fmt.Errorf("failed to read centroid %d size: %w", i, err)
			}

			// Read centroid data
			centroids[i] = make([]float32, centroidSize)
			for j := uint32(0); j < centroidSize; j++ {
				if err := read(&centroids[i][j]); err != nil {
					return bytesRead, fmt.Errorf("failed to read centroid %d data: %w", i, err)
				}
			}
		}
	}

	// 5. Read number of inverted lists
	var listCount uint32
	if err := read(&listCount); err != nil {
		return bytesRead, fmt.Errorf("failed to read list count: %w", err)
	}

	// 6. Read inverted lists
	lists := make([][]VectorNode, listCount)
	for listIdx := uint32(0); listIdx < listCount; listIdx++ {
		// Read list size
		var listSize uint32
		if err := read(&listSize); err != nil {
			return bytesRead, fmt.Errorf("failed to read list %d size: %w", listIdx, err)
		}

		lists[listIdx] = make([]VectorNode, listSize)
		for i := uint32(0); i < listSize; i++ {
			// Read vector ID
			var id uint32
			if err := read(&id); err != nil {
				return bytesRead, fmt.Errorf("failed to read list %d vector %d ID: %w", listIdx, i, err)
			}

			// Read vector data
			vec := make([]float32, idx.dim)
			for j := 0; j < idx.dim; j++ {
				if err := read(&vec[j]); err != nil {
					return bytesRead, fmt.Errorf("failed to read list %d vector %d data: %w", listIdx, i, err)
				}
			}

			// Create vector node
			lists[listIdx][i] = *NewVectorNodeWithID(id, vec)
		}
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
	idx.trained = trained
	idx.centroids = centroids
	idx.lists = lists
	idx.deletedNodes = deletedNodes

	return bytesRead, nil
}
