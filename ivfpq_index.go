// Package comet implements IVFPQ (Inverted File with Product Quantization).
//
// WHAT IS IVFPQ?
// IVFPQ combines IVF (scope reduction) with PQ (compression) to create one of the
// most powerful similarity search algorithms. It's the workhorse of large-scale
// vector search systems.
//
// RESIDUAL VECTORS
// IVFPQ encodes RESIDUALS (vector - centroid) instead of original vectors.
// This dramatically improves compression quality because:
//   - Residuals are centered near 0 (low variance)
//   - Single set of codebooks works for all clusters
//   - Better approximation than encoding original vectors
//
// PERFORMANCE:
//   - Speed: 10-100x faster than flat (from IVF)
//   - Memory: 100-500x compression (from PQ on residuals)
//   - Accuracy: 85-95% recall with proper tuning
//
// TIME COMPLEXITY:
//   - Training: O(IVF_kmeans + PQ_kmeans)
//   - Add: O(nlist × dim + M × K × dsub)
//   - Search: O(nlist + nprobe × (M × K × dsub + candidates × M))
package comet

import (
	"fmt"
	"math"
	"sync"

	"github.com/RoaringBitmap/roaring"
)

// Compile-time checks to ensure IVFPQIndex implements VectorIndex
var _ VectorIndex = (*IVFPQIndex)(nil)

// CompressedVector represents a PQ-encoded residual in an inverted list.
type CompressedVector struct {
	// Node is the original VectorNode
	Node VectorNode

	// Code is the PQ-encoded residual (M bytes)
	Code []uint8
}

// IVFPQIndex represents an IVFPQ index.
//
// Memory layout:
//   - IVF centroids: nlist × dim × 4 bytes
//   - PQ codebooks: M × K × (dim/M) × 4 bytes
//   - Compressed vectors: n × M bytes
type IVFPQIndex struct {
	// dim is vector dimensionality
	dim int

	// distanceKind specifies the distance metric
	distanceKind DistanceKind

	// distance is the distance calculator
	distance Distance

	// nlist is number of IVF clusters
	nlist int

	// M is number of PQ subspaces
	M int

	// Nbits is bits per PQ code
	Nbits int

	// Ksub is centroids per subspace (K = 2^Nbits)
	Ksub int

	// dsub is subspace dimension (dim/M)
	dsub int

	// centroids stores nlist IVF cluster centers
	centroids [][]float32

	// codebooks stores M PQ codebooks (trained on residuals!)
	codebooks [][]float32

	// lists stores inverted lists with compressed vectors
	lists [][]CompressedVector

	// deletedNodes tracks soft-deleted IDs using roaring bitmap
	// CRITICAL OPTIMIZATION: RoaringBitmap is much more efficient than map[uint32]bool
	// - O(log n) membership test with better memory efficiency
	// - Compressed bitmap representation
	// - Fast iteration for batch operations
	deletedNodes *roaring.Bitmap

	// mu provides thread-safe access
	mu sync.RWMutex

	// trained indicates whether both IVF and PQ are trained
	trained bool
}

// NewIVFPQIndex creates a new IVFPQ index.
//
// Parameters:
//   - dim: Vector dimensionality (must be divisible by M)
//   - distanceKind: Distance metric
//   - nlist: Number of IVF clusters (rule of thumb: sqrt(n))
//   - m: Number of PQ subspaces (must divide dimension evenly)
//   - nbits: Bits per PQ code (most common: 8, which gives K=256)
//
// Returns:
//   - *IVFPQIndex: New untrained IVFPQ index
//   - error: Returns error if parameters invalid
func NewIVFPQIndex(dim int, distanceKind DistanceKind, nlist int, m int, nbits int) (*IVFPQIndex, error) {
	// Validate dimension
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}

	// Validate nlist
	if nlist <= 0 {
		return nil, fmt.Errorf("nlist must be positive")
	}

	// Validate M
	if m <= 0 {
		return nil, fmt.Errorf("parameter M must be positive")
	}

	// Critical: dimension must be divisible by M
	if dim%m != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by M %d", dim, m)
	}

	// Validate Nbits
	if nbits <= 0 || nbits > 16 {
		return nil, fmt.Errorf("parameter Nbits must be in [1,16]")
	}

	// Create distance calculator
	distance, err := NewDistance(distanceKind)
	if err != nil {
		return nil, err
	}

	// Calculate derived parameters
	Ksub := 1 << nbits
	dsub := dim / m

	return &IVFPQIndex{
		dim:          dim,
		distanceKind: distanceKind,
		distance:     distance,
		nlist:        nlist,
		M:            m,
		Nbits:        nbits,
		Ksub:         Ksub,
		dsub:         dsub,
		lists:        make([][]CompressedVector, nlist),
		deletedNodes: roaring.New(), // Initialize empty bitmap for soft deletes
	}, nil
}

// Train learns IVF clusters and PQ codebooks on residuals.
//
// THE TRAINING ALGORITHM:
//  1. Run k-means to create nlist IVF clusters
//  2. Assign vectors to nearest centroids
//  3. Compute residuals = vector - centroid
//  4. Train PQ codebooks on ALL residuals together
//
// The key innovation: Training PQ on residuals instead of original vectors.
// Residuals have much less variance, enabling better compression.
//
// Parameters:
//   - vectors: Training vectors (need at least nlist*10)
//
// Returns:
//   - error: Returns error if insufficient training data
func (idx *IVFPQIndex) Train(vectors []VectorNode) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Validate sufficient training data
	if len(vectors) < idx.nlist*10 {
		return fmt.Errorf("need at least %d vectors for training", idx.nlist*10)
	}

	// Validate dimensionality
	for _, v := range vectors {
		if len(v.Vector()) != idx.dim {
			return fmt.Errorf("vector dimension mismatch: expected %d, got %d",
				idx.dim, len(v.Vector()))
		}
	}

	// Extract raw vectors
	rawVectors := make([][]float32, len(vectors))
	for i, v := range vectors {
		rawVectors[i] = v.Vector()
	}

	// STEP 1: Train IVF (k-means clustering on original vectors)
	centroids, _ := KMeans(rawVectors, idx.nlist, idx.distance, 20)
	if centroids == nil {
		return fmt.Errorf("IVF k-means failed")
	}
	idx.centroids = centroids

	// STEP 2: Assign vectors to nearest centroids
	assignments := make([]int, len(rawVectors))
	for i, v := range rawVectors {
		assignments[i] = FindNearestCentroidIndex(v, idx.centroids, idx.distance)
	}

	// STEP 3: Compute residuals
	// residual = vector - centroid
	// This centers all vectors near 0, regardless of which cluster
	residuals := make([][]float32, len(rawVectors))
	for i, v := range rawVectors {
		centroid := idx.centroids[assignments[i]]
		residual := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residual[d] = v[d] - centroid[d]
		}
		residuals[i] = residual
	}

	// STEP 4: Train PQ on residuals (not original vectors!)
	// Single set of codebooks for all clusters works because
	// residuals are all centered near 0
	idx.codebooks = make([][]float32, idx.M)

	for m := 0; m < idx.M; m++ {
		// Extract subspace m from all residuals
		subVectors := make([][]float32, len(residuals))
		start := m * idx.dsub
		end := start + idx.dsub

		for i, r := range residuals {
			subVectors[i] = r[start:end]
		}

		// Run k-means on residual subspace
		centroids, _ := KMeansSubspace(subVectors, idx.Ksub, 20)
		if centroids == nil {
			return fmt.Errorf("PQ k-means failed for subspace %d", m)
		}

		// Store as flattened codebook
		idx.codebooks[m] = make([]float32, idx.Ksub*idx.dsub)
		for k := 0; k < idx.Ksub; k++ {
			copy(idx.codebooks[m][k*idx.dsub:(k+1)*idx.dsub], centroids[k])
		}
	}

	idx.trained = true
	return nil
}

// Trained returns true if the index has been trained
func (idx *IVFPQIndex) Trained() bool {
	return idx.trained
}

// Add encodes vectors as residuals and adds to inverted lists.
//
// Algorithm:
//  1. Find nearest IVF centroid
//  2. Compute residual = vector - centroid
//  3. Encode residual with PQ
//  4. Store code in inverted list
//
// Parameters:
//   - vector: Vector to add
//
// Returns:
//   - error: Returns error if not trained or dimension mismatch
func (idx *IVFPQIndex) Add(vector VectorNode) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Enforce training requirement
	if !idx.trained {
		return fmt.Errorf("index must be trained before adding")
	}

	// Validate dimensionality
	if len(vector.Vector()) != idx.dim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d",
			idx.dim, len(vector.Vector()))
	}

	// Preprocess vector
	if err := idx.distance.PreprocessInPlace(vector.Vector()); err != nil {
		return err
	}

	// Find nearest IVF centroid
	listIdx := FindNearestCentroidIndex(vector.Vector(), idx.centroids, idx.distance)

	// Compute residual = vector - centroid
	centroid := idx.centroids[listIdx]
	residual := make([]float32, idx.dim)
	for d := 0; d < idx.dim; d++ {
		residual[d] = vector.Vector()[d] - centroid[d]
	}

	// Encode residual with PQ
	code := idx.encodeResidual(residual)

	// Store compressed vector in inverted list
	idx.lists[listIdx] = append(idx.lists[listIdx], CompressedVector{
		Node: vector,
		Code: code,
	})

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
func (idx *IVFPQIndex) Remove(vector VectorNode) error {
	id := vector.ID()

	// ════════════════════════════════════════════════════════════════════════
	// STEP 1: CHECK EXISTENCE (READ LOCK - CHEAPER)
	// ════════════════════════════════════════════════════════════════════════
	idx.mu.RLock()
	exists := false
	for _, list := range idx.lists {
		for _, cv := range list {
			if cv.Node.ID() == id {
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
// 2. Reclaims memory occupied by deleted vectors and their PQ codes
// 3. Clears the deleted nodes bitmap
//
// COST: O(n) where n = total number of vectors across all lists
//
// Thread-safety: Acquires exclusive write lock
func (idx *IVFPQIndex) Flush() error {
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
		filtered := make([]CompressedVector, 0, len(idx.lists[listIdx]))
		for _, cv := range idx.lists[listIdx] {
			// Keep vector only if NOT deleted
			// RoaringBitmap Contains() is very fast - O(log n)
			if !idx.deletedNodes.Contains(cv.Node.ID()) {
				filtered = append(filtered, cv)
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

// NewSearch creates a new search builder.
func (idx *IVFPQIndex) NewSearch() VectorSearch {
	return &ivfpqIndexSearch{
		index:   idx,
		k:       10,
		nprobes: int(math.Sqrt(float64(idx.nlist))),
		cutoff:  -1, // Default no cutoff
	}
}

// Dimensions returns vector dimensionality.
func (idx *IVFPQIndex) Dimensions() int {
	return idx.dim
}

// DistanceKind returns the distance metric.
func (idx *IVFPQIndex) DistanceKind() DistanceKind {
	return idx.distanceKind
}

// Kind returns the index type.
func (idx *IVFPQIndex) Kind() VectorIndexKind {
	return IVFPQIndexKind
}

// encodeResidual encodes a residual vector with PQ.
func (idx *IVFPQIndex) encodeResidual(residual []float32) []uint8 {
	code := make([]uint8, idx.M)

	for m := 0; m < idx.M; m++ {
		// Extract subvector
		start := m * idx.dsub
		end := start + idx.dsub
		subVector := residual[start:end]

		// Find nearest centroid
		minDist := float32(math.Inf(1))
		minIdx := 0

		for ksub := 0; ksub < idx.Ksub; ksub++ {
			centroid := idx.codebooks[m][ksub*idx.dsub : (ksub+1)*idx.dsub]

			// L2 squared distance
			var dist float32
			for i := range subVector {
				diff := subVector[i] - centroid[i]
				dist += diff * diff
			}

			if dist < minDist {
				minDist = dist
				minIdx = ksub
			}
		}

		code[m] = uint8(minIdx)
	}

	return code
}
