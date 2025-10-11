package comet

import (
	"errors"
	"math"
)

// ErrUnknownDistanceKind is returned when an unknown distance kind is provided to NewDistance.
var ErrUnknownDistanceKind = errors.New("unknown distance kind")

// DistanceKind represents the type of distance metric to use for vector comparisons.
// Different distance metrics are suitable for different use cases:
// - Euclidean (L2): Measures absolute spatial distance between points
// - Cosine: Measures angular similarity, independent of magnitude
// - DotProduct: Measures negative inner product, useful for Maximum Inner Product Search (MIPS)
type DistanceKind string

const (
	// Euclidean (L2) distance measures the straight-line distance between two points.
	// Use this when the magnitude of vectors matters.
	// Formula: sqrt(sum((a[i] - b[i])^2))
	Euclidean DistanceKind = "l2"

	// Cosine distance measures the angular difference between vectors (1 - cosine similarity).
	// Use this when you care about direction but not magnitude (e.g., text embeddings).
	// Formula: 1 - (dot(a,b) / (||a|| * ||b||))
	// Range: [0, 2] where 0 = identical direction, 1 = orthogonal, 2 = opposite
	Cosine DistanceKind = "cosine"

	// DotProduct computes negative inner product, useful for Maximum Inner Product Search.
	// Use this when vectors are already normalized or when you want to find maximum similarity.
	// Formula: -dot(a, b)
	DotProduct DistanceKind = "dot"
)

// Singleton instances of distance strategies.
// These are stateless and can be safely reused across goroutines.
var (
	euclideanDistance  = euclidean{}
	cosineDist         = cosine{}
	dotProductDistance = dot{}
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
		return euclideanDistance, nil
	case Cosine:
		return cosineDist, nil
	case DotProduct:
		return dotProductDistance, nil
	default:
		return nil, ErrUnknownDistanceKind
	}
}

// euclidean implements the Distance interface using Euclidean (L2) distance.
// This measures the straight-line distance between two points in n-dimensional space.
type euclidean struct{}

func (e euclidean) Calculate(a, b []float32) float32 {
	return l2Distance(a, b)
}

func (e euclidean) CalculateBatch(queries [][]float32, target []float32) []float32 {
	results := make([]float32, len(queries))
	for i, query := range queries {
		results[i] = l2Distance(query, target)
	}
	return results
}

// cosine implements the Distance interface using cosine distance.
// This measures angular similarity between vectors, independent of their magnitude.
type cosine struct{}

func (c cosine) Calculate(a, b []float32) float32 {
	return cosineDistance(a, b)
}

func (c cosine) CalculateBatch(queries [][]float32, target []float32) []float32 {
	// Optimize by precomputing the target's norm once
	normTarget := Norm(target)

	results := make([]float32, len(queries))
	for i, query := range queries {
		normQuery := Norm(query)
		results[i] = cosineDistanceWithNorms(query, target, normQuery, normTarget)
	}
	return results
}

// dot implements the Distance interface using negative inner product.
// This is useful for Maximum Inner Product Search (MIPS).
type dot struct{}

func (d dot) Calculate(a, b []float32) float32 {
	return innerProduct(a, b)
}

func (d dot) CalculateBatch(queries [][]float32, target []float32) []float32 {
	results := make([]float32, len(queries))
	for i, query := range queries {
		results[i] = innerProduct(query, target)
	}
	return results
}

// ============================================================================
// Unexported helper functions
// ============================================================================

// l2Distance computes the Euclidean (L2) distance between two vectors.
// This is the most common distance metric, measuring straight-line distance.
//
// Formula: sqrt(sum((a[i] - b[i])^2))
//
// Time complexity: O(n) where n is the vector dimension
func l2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// l2DistanceSquared computes the squared Euclidean distance.
// This is faster than l2Distance as it avoids the sqrt operation.
// Use this when you only need to compare distances (ordering is preserved).
//
// Formula: sum((a[i] - b[i])^2)
//
// Time complexity: O(n) where n is the vector dimension
func l2DistanceSquared(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// dotProduct computes the dot product (inner product) of two vectors.
// This measures how much two vectors align with each other.
//
// Formula: sum(a[i] * b[i])
//
// Returns:
//   - Positive value: vectors point in similar directions
//   - Zero: vectors are orthogonal (perpendicular)
//   - Negative value: vectors point in opposite directions
//
// Time complexity: O(n) where n is the vector dimension
func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

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

// normSquared computes the squared L2 norm of a vector.
// This is faster than Norm() as it avoids the sqrt operation.
// Use this when you only need to compare magnitudes.
//
// Formula: sum(v[i]^2)
//
// Time complexity: O(n) where n is the vector dimension
func normSquared(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return sum
}

// cosineDistance computes the cosine distance between two vectors.
// Cosine distance = 1 - cosine similarity, measuring angular difference.
//
// Formula: 1 - (dot(a,b) / (||a|| * ||b||))
//
// Range: [0, 2]
//   - 0: vectors point in the same direction (identical)
//   - 1: vectors are orthogonal (perpendicular)
//   - 2: vectors point in opposite directions
//
// Special cases:
//   - If either vector has zero magnitude, returns 1.0
//   - Clamps similarity to [-1, 1] to handle floating point errors
//
// Time complexity: O(n) where n is the vector dimension
func cosineDistance(a, b []float32) float32 {
	dot := dotProduct(a, b)
	normA := Norm(a)
	normB := Norm(b)

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dot / (normA * normB)
	// Clamp to [-1, 1] to handle floating point precision errors
	if similarity > 1 {
		similarity = 1
	} else if similarity < -1 {
		similarity = -1
	}
	return 1 - similarity
}

// cosineDistanceWithNorms computes cosine distance using precomputed norms.
// This is more efficient when you need to compute distances from one vector
// to many others, as you can precompute the target vector's norm once.
//
// Parameters:
//   - a, b: the two vectors to compare
//   - normA, normB: precomputed norms (magnitudes) of vectors a and b
//
// This function is ~6x faster than cosineDistance when norms are precomputed.
//
// Time complexity: O(n) for dot product only (norms already computed)
func cosineDistanceWithNorms(a, b []float32, normA, normB float32) float32 {
	dot := dotProduct(a, b)

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dot / (normA * normB)
	// Clamp to [-1, 1] to handle floating point precision errors
	if similarity > 1 {
		similarity = 1
	} else if similarity < -1 {
		similarity = -1
	}
	return 1 - similarity
}

// innerProduct computes the negative inner product of two vectors.
// This is used in Maximum Inner Product Search (MIPS), where we want to
// find vectors with the highest inner product (maximum similarity).
// By negating, we convert a maximization problem into a minimization problem.
//
// Formula: -sum(a[i] * b[i])
//
// Use cases:
//   - Recommendation systems
//   - Information retrieval
//   - When working with already normalized vectors
//
// Time complexity: O(n) where n is the vector dimension
func innerProduct(a, b []float32) float32 {
	return -dotProduct(a, b)
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
	norm := Norm(v)
	if norm == 0 {
		return v
	}
	return Scale(v, 1.0/norm)
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
	norm := Norm(v)
	if norm == 0 {
		return
	}
	scale := 1.0 / norm
	for i := range v {
		v[i] *= scale
	}
}
