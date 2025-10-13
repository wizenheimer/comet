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
	"encoding/binary"
	"fmt"
	"io"
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

// WriteTo serializes the IVFPQIndex to an io.Writer.
//
// IMPORTANT: This method calls Flush() before serialization to ensure all soft-deleted
// vectors are permanently removed from the serialized data.
//
// The serialization format is:
// 1. Magic number (4 bytes) - "IVPQ" identifier for validation
// 2. Version (4 bytes) - Format version for backward compatibility
// 3. Basic parameters:
//   - Dimensionality (4 bytes)
//   - Distance kind length (4 bytes) + distance kind string
//   - nlist (4 bytes) - number of IVF clusters
//   - M (4 bytes) - number of PQ subspaces
//   - Nbits (4 bytes) - bits per PQ code
//   - Ksub (4 bytes) - centroids per subspace
//   - dsub (4 bytes) - subspace dimension
//   - trained (1 byte) - whether index is trained
//
// 4. Centroids (only if trained):
//   - For each of nlist centroids:
//   - Centroid size (4 bytes)
//   - Centroid data (dim * 4 bytes as float32)
//
// 5. Codebooks (only if trained):
//   - For each of M subquantizers:
//   - Codebook size (4 bytes)
//   - Codebook data (Ksub * dsub * 4 bytes as float32)
//
// 6. Number of inverted lists (4 bytes)
// 7. For each inverted list:
//   - List size (4 bytes)
//   - For each compressed vector:
//   - Vector ID (4 bytes)
//   - PQ code (M bytes)
//
// 8. Deleted nodes bitmap size (4 bytes) + roaring bitmap bytes
//
// Thread-safety: Acquires read lock during serialization
//
// Returns:
//   - int64: Number of bytes written
//   - error: Returns error if write fails or flush fails
func (idx *IVFPQIndex) WriteTo(w io.Writer) (int64, error) {
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

	// 1. Write magic number "IVPQ"
	magic := [4]byte{'I', 'V', 'P', 'Q'}
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

	if err := write(uint32(idx.M)); err != nil {
		return bytesWritten, fmt.Errorf("failed to write M: %w", err)
	}

	if err := write(uint32(idx.Nbits)); err != nil {
		return bytesWritten, fmt.Errorf("failed to write Nbits: %w", err)
	}

	if err := write(uint32(idx.Ksub)); err != nil {
		return bytesWritten, fmt.Errorf("failed to write Ksub: %w", err)
	}

	if err := write(uint32(idx.dsub)); err != nil {
		return bytesWritten, fmt.Errorf("failed to write dsub: %w", err)
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

		// 5. Write codebooks (only if trained)
		for m := 0; m < idx.M; m++ {
			// Write codebook size
			codebookSize := uint32(len(idx.codebooks[m]))
			if err := write(codebookSize); err != nil {
				return bytesWritten, fmt.Errorf("failed to write codebook %d size: %w", m, err)
			}

			// Write codebook data
			for _, val := range idx.codebooks[m] {
				if err := write(val); err != nil {
					return bytesWritten, fmt.Errorf("failed to write codebook %d data: %w", m, err)
				}
			}
		}
	}

	// 6. Write number of inverted lists
	if err := write(uint32(len(idx.lists))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write list count: %w", err)
	}

	// 7. Write each inverted list
	for listIdx, list := range idx.lists {
		// Write list size
		if err := write(uint32(len(list))); err != nil {
			return bytesWritten, fmt.Errorf("failed to write list %d size: %w", listIdx, err)
		}

		// Write each compressed vector in the list
		for i, cv := range list {
			// Write vector ID
			if err := write(cv.Node.ID()); err != nil {
				return bytesWritten, fmt.Errorf("failed to write list %d vector %d ID: %w", listIdx, i, err)
			}

			// Write PQ code
			if _, err := w.Write(cv.Code); err != nil {
				return bytesWritten, fmt.Errorf("failed to write list %d vector %d code: %w", listIdx, i, err)
			}
			bytesWritten += int64(len(cv.Code))
		}
	}

	// 8. Write deleted nodes bitmap
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

// ReadFrom deserializes an IVFPQIndex from an io.Reader.
//
// This method reconstructs an IVFPQIndex from the serialized format created by WriteTo.
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
//	idx2, _ := NewIVFPQIndex(384, Cosine, 100, 8, 8)
//	idx2.ReadFrom(file)
//	file.Close()
func (idx *IVFPQIndex) ReadFrom(r io.Reader) (int64, error) {
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
	if string(magic) != "IVPQ" {
		return bytesRead, fmt.Errorf("invalid magic number: expected 'IVPQ', got '%s'", string(magic))
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

	// Read nlist, M, Nbits, Ksub, dsub
	var nlist, M, Nbits, Ksub, dsub uint32
	if err := read(&nlist); err != nil {
		return bytesRead, fmt.Errorf("failed to read nlist: %w", err)
	}
	if err := read(&M); err != nil {
		return bytesRead, fmt.Errorf("failed to read M: %w", err)
	}
	if err := read(&Nbits); err != nil {
		return bytesRead, fmt.Errorf("failed to read Nbits: %w", err)
	}
	if err := read(&Ksub); err != nil {
		return bytesRead, fmt.Errorf("failed to read Ksub: %w", err)
	}
	if err := read(&dsub); err != nil {
		return bytesRead, fmt.Errorf("failed to read dsub: %w", err)
	}

	// Validate parameters match
	if int(nlist) != idx.nlist {
		return bytesRead, fmt.Errorf("parameter nlist mismatch: index has nlist=%d, serialized data has nlist=%d", idx.nlist, nlist)
	}
	if int(M) != idx.M {
		return bytesRead, fmt.Errorf("parameter M mismatch: index has M=%d, serialized data has M=%d", idx.M, M)
	}
	if int(Nbits) != idx.Nbits {
		return bytesRead, fmt.Errorf("parameter Nbits mismatch: index has Nbits=%d, serialized data has Nbits=%d", idx.Nbits, Nbits)
	}
	if int(Ksub) != idx.Ksub {
		return bytesRead, fmt.Errorf("parameter Ksub mismatch: index has Ksub=%d, serialized data has Ksub=%d", idx.Ksub, Ksub)
	}
	if int(dsub) != idx.dsub {
		return bytesRead, fmt.Errorf("parameter dsub mismatch: index has dsub=%d, serialized data has dsub=%d", idx.dsub, dsub)
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

		// 5. Read codebooks (only if trained)
		codebooks := make([][]float32, idx.M)
		for m := 0; m < idx.M; m++ {
			// Read codebook size
			var codebookSize uint32
			if err := read(&codebookSize); err != nil {
				return bytesRead, fmt.Errorf("failed to read codebook %d size: %w", m, err)
			}

			// Read codebook data
			codebooks[m] = make([]float32, codebookSize)
			for j := uint32(0); j < codebookSize; j++ {
				if err := read(&codebooks[m][j]); err != nil {
					return bytesRead, fmt.Errorf("failed to read codebook %d data: %w", m, err)
				}
			}
		}
		idx.codebooks = codebooks
	}

	// 6. Read number of inverted lists
	var listCount uint32
	if err := read(&listCount); err != nil {
		return bytesRead, fmt.Errorf("failed to read list count: %w", err)
	}

	// 7. Read inverted lists
	lists := make([][]CompressedVector, listCount)
	for listIdx := uint32(0); listIdx < listCount; listIdx++ {
		// Read list size
		var listSize uint32
		if err := read(&listSize); err != nil {
			return bytesRead, fmt.Errorf("failed to read list %d size: %w", listIdx, err)
		}

		lists[listIdx] = make([]CompressedVector, listSize)
		for i := uint32(0); i < listSize; i++ {
			// Read vector ID
			var id uint32
			if err := read(&id); err != nil {
				return bytesRead, fmt.Errorf("failed to read list %d vector %d ID: %w", listIdx, i, err)
			}

			// Read PQ code
			code := make([]uint8, idx.M)
			if _, err := io.ReadFull(r, code); err != nil {
				return bytesRead, fmt.Errorf("failed to read list %d vector %d code: %w", listIdx, i, err)
			}
			bytesRead += int64(idx.M)

			// Create compressed vector (with empty original vector since IVFPQ doesn't store them)
			lists[listIdx][i] = CompressedVector{
				Node: *NewVectorNodeWithID(id, nil),
				Code: code,
			}
		}
	}

	// 8. Read deleted nodes bitmap
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
