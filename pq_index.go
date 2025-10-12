// Package comet implements Product Quantization (PQ) for similarity search.
//
// WHAT IS PRODUCT QUANTIZATION?
// PQ is a lossy compression technique that dramatically reduces memory usage for vector
// storage while enabling approximate similarity search. It achieves compression ratios
// of 10-500x by dividing vectors into subspaces and quantizing each independently.
//
// THE CORE IDEA - DIVIDE AND COMPRESS:
// Instead of storing full high-dimensional vectors:
// 1. Divide each vector into M equal-sized subvectors (subspaces)
// 2. Learn a codebook of K centroids for each subspace via k-means
// 3. Encode each subvector with the ID of its nearest centroid
// 4. Store only these compact codes instead of original vectors
//
// COMPRESSION EXAMPLE:
// Original: 768 dims × 4 bytes = 3,072 bytes
// PQ (M=8, K=256): 8 subspaces × 1 byte = 8 bytes
// Compression: 384x smaller!
//
// TIME COMPLEXITY:
//   - Training: O(M × iterations × K × n × dsub) where dsub = dim/M
//   - Add: O(M × K × dsub) per vector
//   - Search: O(M × K × dsub + n × M) - table build + lookups
//
// WHEN TO USE PQ:
// - Dataset too large for RAM
// - Can tolerate 85-95% recall
// - L2 or inner product metric
// - Want massive compression
package comet

import (
	"fmt"
	"math"
	"sync"
)

// Compile-time checks
var _ VectorIndex = (*PQIndex)(nil)

// CalculatePQParams returns recommended PQ parameters for a given dimension.
// A neat utility function to get the recommended PQ parameters for a given dimension.
// Returns:
//   - M: Number of subquantizers (subspaces) that divides dim evenly
//   - Nbits: Bits per PQ code (default: 8, giving K=256 centroids per subspace)
func CalculatePQParams(dim int) (M int, Nbits int) {
	// Find M that divides dimension evenly
	// Prefer M=8 as good balance
	m := 8
	if dim%m != 0 {
		// Find divisor close to 8
		for m = 8; m <= 32; m++ {
			if dim%m == 0 {
				break
			}
		}
		if dim%m != 0 {
			m = 4 // Fallback
		}
	}

	return m, 8 // Standard: 256 centroids per subspace
}

// PQIndex represents a Product Quantization index.
//
// Memory layout:
//   - Codes: n × M bytes (compressed vectors)
//   - Codebooks: M × K × (dim/M) × 4 bytes
//   - Typical: 10-500x smaller than original
type PQIndex struct {
	// dim is the dimensionality of original vectors
	dim int

	// distanceKind specifies the distance metric
	distanceKind DistanceKind

	// distance is the distance calculator
	distance Distance

	// M is the number of subquantizers
	M int

	// Nbits is bits per PQ code
	Nbits int

	// Ksub is centroids per subquantizer (K = 2^Nbits)
	Ksub int

	// dsub is dimension of each subspace (dim/M)
	dsub int

	// codebooks stores M independent codebooks
	// codebooks[m][k*dsub:(k+1)*dsub] is centroid k in subspace m
	codebooks [][]float32

	// codes stores compressed representations
	// codes[i] is M bytes, one per subspace
	codes [][]uint8

	// vectorNodes stores the original VectorNode metadata
	vectorNodes []VectorNode

	// mu provides thread-safe access
	mu sync.RWMutex

	// trained indicates whether codebooks have been learned
	trained bool
}

// NewPQIndex creates a new Product Quantization index.
//
// Parameters:
//   - dim: Vector dimensionality (must be divisible by M)
//   - distanceKind: Distance metric
//   - M: Number of subquantizers (subspaces). Must divide dim evenly.
//   - Nbits: Bits per PQ code, determines K=2^Nbits centroids per subspace
//
// Returns:
//   - *PQIndex: New untrained PQ index
//   - error: Returns error if parameters invalid
//
// Tip: Use CalculatePQParams(dim) to get recommended M and Nbits values.
func NewPQIndex(dim int, distanceKind DistanceKind, M int, Nbits int) (*PQIndex, error) {
	// Validate dimension
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}

	// Validate M
	if M <= 0 {
		return nil, fmt.Errorf("parameter M must be positive")
	}

	// Critical: dimension must be evenly divisible by M
	if dim%M != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by M %d", dim, M)
	}

	// Validate Nbits
	if Nbits <= 0 || Nbits > 16 {
		return nil, fmt.Errorf("parameter Nbits must be in [1,16]")
	}

	// Create distance calculator
	distance, err := NewDistance(distanceKind)
	if err != nil {
		return nil, err
	}

	// Calculate derived parameters
	Ksub := 1 << Nbits // K = 2^Nbits
	dsub := dim / M

	return &PQIndex{
		dim:          dim,
		distanceKind: distanceKind,
		distance:     distance,
		M:            M,
		Nbits:        Nbits,
		Ksub:         Ksub,
		dsub:         dsub,
		codes:        make([][]uint8, 0),
		vectorNodes:  make([]VectorNode, 0),
	}, nil
}

// Train learns codebooks for each subspace using k-means.
//
// Algorithm:
//  1. For each of M subspaces:
//     a. Extract that subspace from all training vectors
//     b. Run k-means to find K centroids
//     c. Store centroids as codebook for that subspace
//
// Parameters:
//   - vectors: Training vectors (need at least Ksub vectors)
//
// Returns:
//   - error: Returns error if insufficient training data
func (idx *PQIndex) Train(vectors []VectorNode) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Validate sufficient training vectors
	if len(vectors) < idx.Ksub {
		return fmt.Errorf("need at least %d vectors for training", idx.Ksub)
	}

	// Validate dimensionality
	for _, v := range vectors {
		if len(v.Vector()) != idx.dim {
			return fmt.Errorf("vector dimension mismatch: expected %d, got %d",
				idx.dim, len(v.Vector()))
		}
	}

	// Extract raw float32 slices for k-means
	rawVectors := make([][]float32, len(vectors))
	for i, v := range vectors {
		rawVectors[i] = v.Vector()
	}

	// Allocate codebooks
	idx.codebooks = make([][]float32, idx.M)

	// Train each subspace independently
	for m := 0; m < idx.M; m++ {
		// Extract subspace m from all vectors
		subVectors := make([][]float32, len(rawVectors))
		start := m * idx.dsub
		end := start + idx.dsub

		for i, v := range rawVectors {
			subVectors[i] = v[start:end]
		}

		// Run k-means on this subspace
		// Use our improved KMeansSubspace function
		centroids, _ := KMeansSubspace(subVectors, idx.Ksub, 20)

		if centroids == nil {
			return fmt.Errorf("k-means failed for subspace %d", m)
		}

		// Store centroids as flattened codebook
		idx.codebooks[m] = make([]float32, idx.Ksub*idx.dsub)
		for k := 0; k < idx.Ksub; k++ {
			copy(idx.codebooks[m][k*idx.dsub:(k+1)*idx.dsub], centroids[k])
		}
	}

	idx.trained = true
	return nil
}

// Add compresses and adds vectors to the index.
//
// Encoding process:
// 1. Divide vector into M subvectors
// 2. For each subvector, find nearest centroid in its codebook
// 3. Store centroid IDs as M-byte code
//
// Original vectors are discarded after encoding!
//
// Parameters:
//   - vector: Vector to compress and add
//
// Returns:
//   - error: Returns error if not trained or dimension mismatch
func (idx *PQIndex) Add(vector VectorNode) error {
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

	// Preprocess vector according to distance metric
	if err := idx.distance.PreprocessInPlace(vector.Vector()); err != nil {
		return err
	}

	// Encode vector into PQ code
	code := idx.encode(vector.Vector())

	// Store compressed code and metadata
	idx.codes = append(idx.codes, code)
	idx.vectorNodes = append(idx.vectorNodes, vector)

	return nil
}

// Remove removes a vector from the index.
func (idx *PQIndex) Remove(vector VectorNode) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Search for vector with matching ID
	for i, v := range idx.vectorNodes {
		if v.ID() == vector.ID() {
			// Remove by slicing
			idx.codes = append(idx.codes[:i], idx.codes[i+1:]...)
			idx.vectorNodes = append(idx.vectorNodes[:i], idx.vectorNodes[i+1:]...)
			return nil
		}
	}

	return fmt.Errorf("vector with ID %d not found", vector.ID())
}

// Flush is a no-op for PQ index.
func (idx *PQIndex) Flush() error {
	return nil
}

// NewSearch creates a new search builder.
func (idx *PQIndex) NewSearch() VectorSearch {
	return &pqIndexSearch{
		index:  idx,
		k:      10,
		cutoff: -1, // Default no cutoff
	}
}

// Dimensions returns the dimensionality of original vectors.
func (idx *PQIndex) Dimensions() int {
	return idx.dim
}

// DistanceKind returns the distance metric.
func (idx *PQIndex) DistanceKind() DistanceKind {
	return idx.distanceKind
}

// Kind returns the index type.
func (idx *PQIndex) Kind() VectorIndexKind {
	return PQIndexKind
}

// Trained returns true if the index has been trained
func (idx *PQIndex) Trained() bool {
	return idx.trained
}

// encode converts a vector into a compact PQ code.
//
// Time Complexity: O(M × K × dsub)
func (idx *PQIndex) encode(v []float32) []uint8 {
	code := make([]uint8, idx.M)

	// Encode each subspace independently
	for m := 0; m < idx.M; m++ {
		// Extract subvector
		start := m * idx.dsub
		end := start + idx.dsub
		subVector := v[start:end]

		// Find nearest centroid
		minDist := float32(math.Inf(1))
		minIdx := 0

		for ksub := 0; ksub < idx.Ksub; ksub++ {
			centroid := idx.codebooks[m][ksub*idx.dsub : (ksub+1)*idx.dsub]

			// Use L2 squared for efficiency
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
