package comet

import (
	"errors"
	"math"
)

// ErrUnknownDistanceKind is returned when an unknown distance kind is provided to NewDistance.
var ErrUnknownDistanceKind = errors.New("unknown distance kind")

// ErrZeroVector is returned when a zero vector is provided for a metric that doesn't support it.
var ErrZeroVector = errors.New("zero vector not allowed for this metric")

// DistanceKind represents the type of distance metric to use for vector comparisons.
// Different distance metrics are suitable for different use cases:
// - Euclidean (L2): Measures absolute spatial distance between points
// - L2Squared: Squared Euclidean distance (faster, preserves ordering)
// - Cosine: Measures angular similarity, independent of magnitude
type DistanceKind string

const (
	// Euclidean (L2) distance measures the straight-line distance between two points.
	// Use this when the magnitude of vectors matters.
	// Formula: sqrt(sum((a[i] - b[i])^2))
	Euclidean DistanceKind = "l2"

	// L2Squared (squared Euclidean) distance measures the squared distance between two points.
	// This is faster than L2 as it avoids the sqrt operation.
	// Use this when you only need to compare distances (ordering is preserved).
	// Formula: sum((a[i] - b[i])^2)
	L2Squared DistanceKind = "l2_squared"

	// Cosine distance measures the angular difference between vectors (1 - cosine similarity).
	// Use this when you care about direction but not magnitude (e.g., text embeddings).
	// Formula: 1 - (dot(a,b) / (||a|| * ||b||))
	// Range: [0, 2] where 0 = identical direction, 1 = orthogonal, 2 = opposite
	Cosine DistanceKind = "cosine"
)

// Singleton instances of distance strategies.
// These are stateless and can be safely reused across goroutines.
var (
	euclideanDistanceImpl = euclidean{}
	l2SquaredDistanceImpl = l2Squared{}
	cosineDistanceImpl    = cosine{}
)

// Distance is the interface for computing distances between vectors.
// Implementations provide different distance metrics for vector similarity search.
type Distance interface {
	// Calculate computes the distance between two vectors a and b.
	// The vectors must have the same dimensionality.
	// Returns a float32 representing the distance (lower values = more similar).
	Calculate(a, b []float32) float32

	// CalculateBatch computes distances from multiple query vectors to a single target vector.
	// This is more efficient than calling Calculate multiple times as it can optimize
	// computations (e.g., precomputing norms for cosine distance).
	//
	// Parameters:
	//   - queries: slice of query vectors (each vector is []float32)
	//   - target: single target vector to compare against
	//
	// Returns:
	//   - slice of distances where result[i] is the distance from queries[i] to target
	//
	// All vectors (queries and target) must have the same dimensionality.
	CalculateBatch(queries [][]float32, target []float32) []float32

	// PreprocessInPlace preprocesses the target vector in-place for the distance metric.
	// For cosine distance, this normalizes the vector to unit length.
	// For euclidean distance, this is a no-op.
	// Returns an error if the vector is invalid for this metric (e.g., zero vector for cosine).
	PreprocessInPlace(target []float32) error

	// Preprocess preprocesses the target vector for the distance metric, returning a new vector.
	// For cosine distance, this returns a normalized copy.
	// For euclidean distance, this returns the original vector unchanged.
	// Returns an error if the vector is invalid for this metric (e.g., zero vector for cosine).
	Preprocess(target []float32) ([]float32, error)
}

// NewDistance returns a singleton Distance implementation for the specified metric type.
// The returned instances are stateless and safe for concurrent use across goroutines.
// Returns ErrUnknownDistanceKind if the distance kind is not recognized.
//
// Example:
//
//	dist, err := NewDistance(Euclidean)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	distance := dist.Calculate([]float32{1, 2, 3}, []float32{4, 5, 6})
func NewDistance(t DistanceKind) (Distance, error) {
	switch t {
	case Euclidean:
		return euclideanDistanceImpl, nil
	case L2Squared:
		return l2SquaredDistanceImpl, nil
	case Cosine:
		return cosineDistanceImpl, nil
	default:
		return nil, ErrUnknownDistanceKind
	}
}

// euclidean implements the Distance interface using Euclidean (L2) distance.
// This measures the straight-line distance between two points in n-dimensional space.
type euclidean struct{}

// Calculate computes the Euclidean (L2) distance between two vectors.
// Formula: sqrt(sum((a[i] - b[i])^2))
// Time complexity: O(n) where n is the vector dimension
func (e euclidean) Calculate(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

func (e euclidean) CalculateBatch(queries [][]float32, target []float32) []float32 {
	results := make([]float32, len(queries))
	for i, query := range queries {
		var sum float32
		for j := range query {
			diff := query[j] - target[j]
			sum += diff * diff
		}
		results[i] = float32(math.Sqrt(float64(sum)))
	}
	return results
}

// PreprocessInPlace is a no-op for euclidean distance.
// Euclidean distance doesn't require preprocessing.
func (e euclidean) PreprocessInPlace(target []float32) error {
	// No-op: Euclidean distance doesn't need preprocessing
	return nil
}

// Preprocess is a no-op for euclidean distance, returning the vector unchanged.
// Euclidean distance doesn't require preprocessing.
func (e euclidean) Preprocess(target []float32) ([]float32, error) {
	return target, nil
}

// l2Squared implements the Distance interface using squared Euclidean (L2²) distance.
// This measures the squared straight-line distance between two points.
// This is faster than euclidean distance as it avoids the sqrt operation.
// The ordering of distances is preserved, so this is suitable for k-NN search.
type l2Squared struct{}

// Calculate computes the squared Euclidean (L2²) distance between two vectors.
// Formula: sum((a[i] - b[i])^2)
// Time complexity: O(n) where n is the vector dimension
func (l l2Squared) Calculate(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

func (l l2Squared) CalculateBatch(queries [][]float32, target []float32) []float32 {
	results := make([]float32, len(queries))
	for i, query := range queries {
		var sum float32
		for j := range query {
			diff := query[j] - target[j]
			sum += diff * diff
		}
		results[i] = sum
	}
	return results
}

// PreprocessInPlace is a no-op for L2 squared distance.
// L2 squared distance doesn't require preprocessing.
func (l l2Squared) PreprocessInPlace(target []float32) error {
	// No-op: L2 squared distance doesn't need preprocessing
	return nil
}

// Preprocess is a no-op for L2 squared distance, returning the vector unchanged.
// L2 squared distance doesn't require preprocessing.
func (l l2Squared) Preprocess(target []float32) ([]float32, error) {
	return target, nil
}

// cosine implements the Distance interface using cosine distance.
// This measures angular similarity between vectors, independent of their magnitude.
type cosine struct{}

// Calculate computes cosine distance between two vectors.
// Assumes both vectors are pre-normalized to unit length.
// For normalized vectors: cosine_distance = 1 - dot(a, b)
// Time complexity: O(n) where n is the vector dimension
func (c cosine) Calculate(a, b []float32) float32 {
	// For normalized vectors, cosine distance is simply 1 - dot product
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}

	// Clamp to [-1, 1] to handle floating point precision errors
	if dot > 1 {
		dot = 1
	} else if dot < -1 {
		dot = -1
	}

	return 1 - dot
}

func (c cosine) CalculateBatch(queries [][]float32, target []float32) []float32 {
	// For cosine distance, we expect vectors to be pre-normalized.
	// Since both queries and target are normalized, distance = 1 - dot(a, b)

	results := make([]float32, len(queries))
	for i, query := range queries {
		var dot float32
		for j := range query {
			dot += query[j] * target[j]
		}

		// Clamp to [-1, 1] to handle floating point precision errors
		if dot > 1 {
			dot = 1
		} else if dot < -1 {
			dot = -1
		}

		results[i] = 1 - dot
	}
	return results
}

// PreprocessInPlace normalizes the vector in-place to unit length for cosine distance.
// Returns ErrZeroVector if the vector has zero magnitude.
// Time complexity: O(n) where n is the vector dimension
func (c cosine) PreprocessInPlace(target []float32) error {
	// Compute norm
	var sum float32
	for _, x := range target {
		sum += x * x
	}
	norm := float32(math.Sqrt(float64(sum)))

	// Zero vectors are undefined for cosine similarity
	if norm == 0 {
		return ErrZeroVector
	}

	// Normalize in-place
	scale := 1.0 / norm
	for i := range target {
		target[i] *= scale
	}

	return nil
}

// Preprocess returns a normalized copy of the vector for cosine distance.
// Returns ErrZeroVector if the vector has zero magnitude.
// Time complexity: O(n) where n is the vector dimension
func (c cosine) Preprocess(target []float32) ([]float32, error) {
	// Compute norm
	var sum float32
	for _, x := range target {
		sum += x * x
	}
	norm := float32(math.Sqrt(float64(sum)))

	// Zero vectors are undefined for cosine similarity
	if norm == 0 {
		return nil, ErrZeroVector
	}

	// Create normalized copy
	result := make([]float32, len(target))
	scale := 1.0 / norm
	for i := range target {
		result[i] = target[i] * scale
	}

	return result, nil
}

// ============================================================================
// Public utility functions
// ============================================================================

// Norm computes the L2 norm (Euclidean length/magnitude) of a vector.
// This represents the "length" of the vector from the origin in n-dimensional space.
//
// The norm is useful for:
//   - Measuring vector magnitude
//   - Normalizing vectors to unit length
//   - Computing distances and similarities
//
// Formula: sqrt(sum(v[i]^2))
//
// Example:
//
//	v := []float32{3, 4}
//	length := Norm(v)  // Returns 5.0 (3²+4² = 25, sqrt(25) = 5)
//
// Time complexity: O(n) where n is the vector dimension
func Norm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

// Scale returns a new vector with all elements multiplied by the given scalar.
// The original vector is not modified.
//
// Use cases:
//   - Scaling vectors to a desired magnitude
//   - Implementing weighted vectors
//   - Normalizing vectors (when combined with Norm)
//
// Formula: result[i] = v[i] * scalar
//
// Example:
//
//	v := []float32{1, 2, 3}
//	doubled := Scale(v, 2.0)        // Returns [2, 4, 6]
//	inverted := Scale(v, -1.0)      // Returns [-1, -2, -3]
//	normalized := Scale(v, 1.0/Norm(v))  // Returns unit vector
//
// Time complexity: O(n) where n is the vector dimension
func Scale(v []float32, scalar float32) []float32 {
	result := make([]float32, len(v))
	for i := range v {
		result[i] = v[i] * scalar
	}
	return result
}

// Normalize returns a new vector with the same direction as v but with unit length (magnitude = 1).
// The original vector is not modified.
//
// Normalization is essential for:
//   - Cosine similarity calculations (removes magnitude, keeps direction)
//   - Comparing vectors by direction only
//   - Machine learning feature scaling
//   - Unit vector representations
//
// Special case:
//   - If the input is a zero vector (all elements are 0), returns the zero vector unchanged
//     to avoid division by zero and NaN values
//
// Formula: result = v / ||v|| where ||v|| is the L2 norm
//
// Example:
//
//	v := []float32{3, 4}
//	unit := Normalize(v)     // Returns [0.6, 0.8] (magnitude = 1)
//	zero := []float32{0, 0}
//	safe := Normalize(zero)  // Returns [0, 0] (safely handles zero vector)
//
// Time complexity: O(n) where n is the vector dimension
func Normalize(v []float32) []float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	norm := float32(math.Sqrt(float64(sum)))

	if norm == 0 {
		result := make([]float32, len(v))
		copy(result, v)
		return result
	}

	result := make([]float32, len(v))
	scale := 1.0 / norm
	for i := range v {
		result[i] = v[i] * scale
	}
	return result
}

// NormalizeInPlace normalizes the vector to unit length in-place, modifying the original vector.
// This is more memory-efficient than Normalize() as it doesn't allocate a new vector.
//
// Use this when:
//   - You don't need to keep the original vector
//   - Memory efficiency is important
//   - Processing large batches of vectors
//
// Special case:
//   - If the input is a zero vector (all elements are 0), the vector remains unchanged
//     to avoid division by zero and NaN values
//
// Formula: v[i] = v[i] / ||v|| for all i
//
// Example:
//
//	v := []float32{3, 4}
//	NormalizeInPlace(v)      // v is now [0.6, 0.8] (magnitude = 1)
//
//	zero := []float32{0, 0}
//	NormalizeInPlace(zero)   // zero remains [0, 0] (safely handles zero vector)
//
// Time complexity: O(n) where n is the vector dimension
func NormalizeInPlace(v []float32) {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	norm := float32(math.Sqrt(float64(sum)))

	if norm == 0 {
		return
	}

	scale := 1.0 / norm
	for i := range v {
		v[i] *= scale
	}
}
