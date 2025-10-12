package comet

import (
	"math"
	"testing"
)

// epsilon for floating point comparisons
const epsilon = 1e-6

// Helper function to compare float32 values with tolerance
func almostEqual(a, b, tolerance float32) bool {
	return math.Abs(float64(a-b)) < float64(tolerance)
}

// ============================================================================
// NewDistance Tests
// ============================================================================

func TestNewDistance(t *testing.T) {
	tests := []struct {
		name         string
		distanceKind DistanceKind
		wantErr      bool
		errType      error
	}{
		{
			name:         "valid Euclidean",
			distanceKind: Euclidean,
			wantErr:      false,
		},
		{
			name:         "valid L2Squared",
			distanceKind: L2Squared,
			wantErr:      false,
		},
		{
			name:         "valid Cosine",
			distanceKind: Cosine,
			wantErr:      false,
		},
		{
			name:         "invalid distance kind",
			distanceKind: "invalid",
			wantErr:      true,
			errType:      ErrUnknownDistanceKind,
		},
		{
			name:         "empty distance kind",
			distanceKind: "",
			wantErr:      true,
			errType:      ErrUnknownDistanceKind,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dist, err := NewDistance(tt.distanceKind)

			if tt.wantErr {
				if err == nil {
					t.Errorf("NewDistance() expected error but got none")
					return
				}
				if tt.errType != nil && err != tt.errType {
					t.Errorf("NewDistance() error = %v, want %v", err, tt.errType)
				}
				return
			}

			if err != nil {
				t.Errorf("NewDistance() unexpected error: %v", err)
				return
			}

			if dist == nil {
				t.Fatal("NewDistance() returned nil distance")
			}
		})
	}
}

// ============================================================================
// Euclidean Distance Tests
// ============================================================================

func TestEuclideanCalculate(t *testing.T) {
	dist, err := NewDistance(Euclidean)
	if err != nil {
		t.Fatalf("NewDistance() error: %v", err)
	}

	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float32
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			expected: 0.0,
		},
		{
			name:     "simple 2D distance",
			a:        []float32{0, 0},
			b:        []float32{3, 4},
			expected: 5.0,
		},
		{
			name:     "simple 3D distance",
			a:        []float32{1, 2, 2},
			b:        []float32{1, 2, 3},
			expected: 1.0,
		},
		{
			name:     "negative values",
			a:        []float32{-1, -2},
			b:        []float32{1, 2},
			expected: 4.472136, // sqrt(20)
		},
		{
			name:     "zero vectors",
			a:        []float32{0, 0, 0},
			b:        []float32{0, 0, 0},
			expected: 0.0,
		},
		{
			name:     "single dimension",
			a:        []float32{5},
			b:        []float32{2},
			expected: 3.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := dist.Calculate(tt.a, tt.b)
			if !almostEqual(result, tt.expected, epsilon) {
				t.Errorf("Calculate() = %f, want %f", result, tt.expected)
			}
		})
	}
}

func TestEuclideanCalculateBatch(t *testing.T) {
	dist, err := NewDistance(Euclidean)
	if err != nil {
		t.Fatalf("NewDistance() error: %v", err)
	}

	queries := [][]float32{
		{0, 0},
		{3, 4},
		{1, 1},
	}
	target := []float32{0, 0}

	results := dist.CalculateBatch(queries, target)

	if len(results) != len(queries) {
		t.Errorf("CalculateBatch() returned %d results, want %d", len(results), len(queries))
	}

	expectedDistances := []float32{0.0, 5.0, 1.414214}
	for i, expected := range expectedDistances {
		if !almostEqual(results[i], expected, epsilon) {
			t.Errorf("CalculateBatch()[%d] = %f, want %f", i, results[i], expected)
		}
	}
}

func TestEuclideanPreprocess(t *testing.T) {
	dist, err := NewDistance(Euclidean)
	if err != nil {
		t.Fatalf("NewDistance() error: %v", err)
	}

	vector := []float32{1, 2, 3}

	// Test Preprocess (should return same vector)
	result, err := dist.Preprocess(vector)
	if err != nil {
		t.Errorf("Preprocess() error: %v", err)
	}
	// For Euclidean, it should return the same slice (not a copy)
	if len(result) != len(vector) {
		t.Errorf("Preprocess() returned different length vector")
	}
	for i := range result {
		if result[i] != vector[i] {
			t.Errorf("Preprocess()[%d] = %f, want %f", i, result[i], vector[i])
		}
	}

	// Test PreprocessInPlace (should be no-op)
	original := []float32{1, 2, 3}
	err = dist.PreprocessInPlace(original)
	if err != nil {
		t.Errorf("PreprocessInPlace() error: %v", err)
	}
	for i, val := range original {
		if val != vector[i] {
			t.Errorf("PreprocessInPlace() modified vector at index %d: got %f, want %f", i, val, vector[i])
		}
	}
}

// ============================================================================
// L2Squared Distance Tests
// ============================================================================

func TestL2SquaredCalculate(t *testing.T) {
	dist, err := NewDistance(L2Squared)
	if err != nil {
		t.Fatalf("NewDistance() error: %v", err)
	}

	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float32
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			expected: 0.0,
		},
		{
			name:     "simple 2D distance",
			a:        []float32{0, 0},
			b:        []float32{3, 4},
			expected: 25.0, // 3^2 + 4^2
		},
		{
			name:     "simple 3D distance",
			a:        []float32{1, 2, 2},
			b:        []float32{1, 2, 3},
			expected: 1.0, // 1^2
		},
		{
			name:     "negative values",
			a:        []float32{-1, -2},
			b:        []float32{1, 2},
			expected: 20.0, // 2^2 + 4^2
		},
		{
			name:     "zero vectors",
			a:        []float32{0, 0, 0},
			b:        []float32{0, 0, 0},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := dist.Calculate(tt.a, tt.b)
			if !almostEqual(result, tt.expected, epsilon) {
				t.Errorf("Calculate() = %f, want %f", result, tt.expected)
			}
		})
	}
}

func TestL2SquaredCalculateBatch(t *testing.T) {
	dist, err := NewDistance(L2Squared)
	if err != nil {
		t.Fatalf("NewDistance() error: %v", err)
	}

	queries := [][]float32{
		{0, 0},
		{3, 4},
		{1, 1},
	}
	target := []float32{0, 0}

	results := dist.CalculateBatch(queries, target)

	if len(results) != len(queries) {
		t.Errorf("CalculateBatch() returned %d results, want %d", len(results), len(queries))
	}

	expectedDistances := []float32{0.0, 25.0, 2.0}
	for i, expected := range expectedDistances {
		if !almostEqual(results[i], expected, epsilon) {
			t.Errorf("CalculateBatch()[%d] = %f, want %f", i, results[i], expected)
		}
	}
}

func TestL2SquaredPreprocess(t *testing.T) {
	dist, err := NewDistance(L2Squared)
	if err != nil {
		t.Fatalf("NewDistance() error: %v", err)
	}

	vector := []float32{1, 2, 3}

	// Test Preprocess (should return same vector)
	result, err := dist.Preprocess(vector)
	if err != nil {
		t.Errorf("Preprocess() error: %v", err)
	}
	// For L2Squared, it should return the same slice (not a copy)
	if len(result) != len(vector) {
		t.Errorf("Preprocess() returned different length vector")
	}
	for i := range result {
		if result[i] != vector[i] {
			t.Errorf("Preprocess()[%d] = %f, want %f", i, result[i], vector[i])
		}
	}

	// Test PreprocessInPlace (should be no-op)
	original := []float32{1, 2, 3}
	err = dist.PreprocessInPlace(original)
	if err != nil {
		t.Errorf("PreprocessInPlace() error: %v", err)
	}
	for i, val := range original {
		if val != vector[i] {
			t.Errorf("PreprocessInPlace() modified vector at index %d: got %f, want %f", i, val, vector[i])
		}
	}
}

// ============================================================================
// Cosine Distance Tests
// ============================================================================

func TestCosineCalculate(t *testing.T) {
	dist, err := NewDistance(Cosine)
	if err != nil {
		t.Fatalf("NewDistance() error: %v", err)
	}

	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float32
	}{
		{
			name:     "identical normalized vectors",
			a:        []float32{0.6, 0.8},
			b:        []float32{0.6, 0.8},
			expected: 0.0,
		},
		{
			name:     "orthogonal normalized vectors",
			a:        []float32{1, 0},
			b:        []float32{0, 1},
			expected: 1.0,
		},
		{
			name:     "opposite normalized vectors",
			a:        []float32{1, 0},
			b:        []float32{-1, 0},
			expected: 2.0,
		},
		{
			name:     "45 degree angle",
			a:        []float32{0.707107, 0.707107},
			b:        []float32{1, 0},
			expected: 0.292893, // 1 - 0.707107
		},
		{
			name:     "same direction normalized vectors",
			a:        []float32{0.5, 0.5, 0.5, 0.5},
			b:        []float32{0.5, 0.5, 0.5, 0.5},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := dist.Calculate(tt.a, tt.b)
			if !almostEqual(result, tt.expected, epsilon) {
				t.Errorf("Calculate() = %f, want %f", result, tt.expected)
			}
		})
	}
}

func TestCosineCalculateBatch(t *testing.T) {
	dist, err := NewDistance(Cosine)
	if err != nil {
		t.Fatalf("NewDistance() error: %v", err)
	}

	// Use normalized vectors
	queries := [][]float32{
		{1, 0},
		{0, 1},
		{0.707107, 0.707107},
	}
	target := []float32{1, 0}

	results := dist.CalculateBatch(queries, target)

	if len(results) != len(queries) {
		t.Errorf("CalculateBatch() returned %d results, want %d", len(results), len(queries))
	}

	expectedDistances := []float32{0.0, 1.0, 0.292893}
	for i, expected := range expectedDistances {
		if !almostEqual(results[i], expected, epsilon) {
			t.Errorf("CalculateBatch()[%d] = %f, want %f", i, results[i], expected)
		}
	}
}

func TestCosinePreprocess(t *testing.T) {
	dist, err := NewDistance(Cosine)
	if err != nil {
		t.Fatalf("NewDistance() error: %v", err)
	}

	t.Run("valid vector", func(t *testing.T) {
		vector := []float32{3, 4}
		result, err := dist.Preprocess(vector)
		if err != nil {
			t.Errorf("Preprocess() error: %v", err)
		}

		// Should be normalized to [0.6, 0.8]
		if !almostEqual(result[0], 0.6, epsilon) || !almostEqual(result[1], 0.8, epsilon) {
			t.Errorf("Preprocess() = [%f, %f], want [0.6, 0.8]", result[0], result[1])
		}

		// Verify original is unchanged
		if vector[0] != 3.0 || vector[1] != 4.0 {
			t.Errorf("Preprocess() modified original vector")
		}

		// Verify result is unit length
		norm := Norm(result)
		if !almostEqual(norm, 1.0, epsilon) {
			t.Errorf("Preprocess() result norm = %f, want 1.0", norm)
		}
	})

	t.Run("zero vector", func(t *testing.T) {
		vector := []float32{0, 0, 0}
		result, err := dist.Preprocess(vector)
		if err != ErrZeroVector {
			t.Errorf("Preprocess() with zero vector error = %v, want %v", err, ErrZeroVector)
		}
		if result != nil {
			t.Errorf("Preprocess() with zero vector should return nil result")
		}
	})
}

func TestCosinePreprocessInPlace(t *testing.T) {
	dist, err := NewDistance(Cosine)
	if err != nil {
		t.Fatalf("NewDistance() error: %v", err)
	}

	t.Run("valid vector", func(t *testing.T) {
		vector := []float32{3, 4}
		err := dist.PreprocessInPlace(vector)
		if err != nil {
			t.Errorf("PreprocessInPlace() error: %v", err)
		}

		// Should be normalized to [0.6, 0.8]
		if !almostEqual(vector[0], 0.6, epsilon) || !almostEqual(vector[1], 0.8, epsilon) {
			t.Errorf("PreprocessInPlace() = [%f, %f], want [0.6, 0.8]", vector[0], vector[1])
		}

		// Verify it's unit length
		norm := Norm(vector)
		if !almostEqual(norm, 1.0, epsilon) {
			t.Errorf("PreprocessInPlace() result norm = %f, want 1.0", norm)
		}
	})

	t.Run("zero vector", func(t *testing.T) {
		vector := []float32{0, 0, 0}
		err := dist.PreprocessInPlace(vector)
		if err != ErrZeroVector {
			t.Errorf("PreprocessInPlace() with zero vector error = %v, want %v", err, ErrZeroVector)
		}
	})
}

// ============================================================================
// Distance Metric Comparison Tests
// ============================================================================

func TestDistanceOrderingPreservation(t *testing.T) {
	// L2Squared should preserve ordering compared to Euclidean
	euclidean, _ := NewDistance(Euclidean)
	l2Squared, _ := NewDistance(L2Squared)

	vectors := [][]float32{
		{1, 1},
		{2, 2},
		{3, 3},
		{4, 4},
	}
	query := []float32{0, 0}

	euclideanDists := make([]float32, len(vectors))
	l2SquaredDists := make([]float32, len(vectors))

	for i, vec := range vectors {
		euclideanDists[i] = euclidean.Calculate(query, vec)
		l2SquaredDists[i] = l2Squared.Calculate(query, vec)
	}

	// Verify ordering is preserved
	for i := 1; i < len(vectors); i++ {
		euclideanOrdering := euclideanDists[i] > euclideanDists[i-1]
		l2SquaredOrdering := l2SquaredDists[i] > l2SquaredDists[i-1]

		if euclideanOrdering != l2SquaredOrdering {
			t.Errorf("L2Squared doesn't preserve ordering at index %d", i)
		}
	}
}

// ============================================================================
// Norm Tests
// ============================================================================

func TestNorm(t *testing.T) {
	tests := []struct {
		name     string
		vector   []float32
		expected float32
	}{
		{
			name:     "simple 2D vector",
			vector:   []float32{3, 4},
			expected: 5.0,
		},
		{
			name:     "unit vector",
			vector:   []float32{1, 0, 0},
			expected: 1.0,
		},
		{
			name:     "zero vector",
			vector:   []float32{0, 0, 0},
			expected: 0.0,
		},
		{
			name:     "negative values",
			vector:   []float32{-3, -4},
			expected: 5.0,
		},
		{
			name:     "single dimension",
			vector:   []float32{7},
			expected: 7.0,
		},
		{
			name:     "high dimensional",
			vector:   []float32{1, 1, 1, 1},
			expected: 2.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Norm(tt.vector)
			if !almostEqual(result, tt.expected, epsilon) {
				t.Errorf("Norm() = %f, want %f", result, tt.expected)
			}
		})
	}
}

// ============================================================================
// Scale Tests
// ============================================================================

func TestScale(t *testing.T) {
	tests := []struct {
		name     string
		vector   []float32
		scalar   float32
		expected []float32
	}{
		{
			name:     "scale by 2",
			vector:   []float32{1, 2, 3},
			scalar:   2.0,
			expected: []float32{2, 4, 6},
		},
		{
			name:     "scale by 0",
			vector:   []float32{1, 2, 3},
			scalar:   0.0,
			expected: []float32{0, 0, 0},
		},
		{
			name:     "scale by -1",
			vector:   []float32{1, 2, 3},
			scalar:   -1.0,
			expected: []float32{-1, -2, -3},
		},
		{
			name:     "scale by 0.5",
			vector:   []float32{2, 4, 6},
			scalar:   0.5,
			expected: []float32{1, 2, 3},
		},
		{
			name:     "scale zero vector",
			vector:   []float32{0, 0, 0},
			scalar:   5.0,
			expected: []float32{0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Scale(tt.vector, tt.scalar)

			if len(result) != len(tt.expected) {
				t.Errorf("Scale() length = %d, want %d", len(result), len(tt.expected))
			}

			for i := range result {
				if !almostEqual(result[i], tt.expected[i], epsilon) {
					t.Errorf("Scale()[%d] = %f, want %f", i, result[i], tt.expected[i])
				}
			}

			// Verify original is unchanged
			original := []float32{1, 2, 3}
			if tt.vector[0] == original[0] {
				for i := range tt.vector {
					if tt.vector[i] != original[i] {
						t.Errorf("Scale() modified original vector")
						break
					}
				}
			}
		})
	}
}

// ============================================================================
// Normalize Tests
// ============================================================================

func TestNormalize(t *testing.T) {
	tests := []struct {
		name     string
		vector   []float32
		expected []float32
	}{
		{
			name:     "simple 2D vector",
			vector:   []float32{3, 4},
			expected: []float32{0.6, 0.8},
		},
		{
			name:     "already unit vector",
			vector:   []float32{1, 0, 0},
			expected: []float32{1, 0, 0},
		},
		{
			name:     "zero vector",
			vector:   []float32{0, 0, 0},
			expected: []float32{0, 0, 0},
		},
		{
			name:     "negative values",
			vector:   []float32{-3, -4},
			expected: []float32{-0.6, -0.8},
		},
		{
			name:     "uniform vector",
			vector:   []float32{1, 1, 1, 1},
			expected: []float32{0.5, 0.5, 0.5, 0.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Normalize(tt.vector)

			if len(result) != len(tt.expected) {
				t.Errorf("Normalize() length = %d, want %d", len(result), len(tt.expected))
			}

			for i := range result {
				if !almostEqual(result[i], tt.expected[i], epsilon) {
					t.Errorf("Normalize()[%d] = %f, want %f", i, result[i], tt.expected[i])
				}
			}

			// Verify result is unit length (except for zero vector)
			if Norm(tt.vector) != 0 {
				norm := Norm(result)
				if !almostEqual(norm, 1.0, epsilon) {
					t.Errorf("Normalize() result norm = %f, want 1.0", norm)
				}
			}

			// Verify original is unchanged
			originalNorm := Norm(tt.vector)
			if !almostEqual(Norm(tt.vector), originalNorm, epsilon) {
				t.Errorf("Normalize() modified original vector")
			}
		})
	}
}

// ============================================================================
// NormalizeInPlace Tests
// ============================================================================

func TestNormalizeInPlace(t *testing.T) {
	tests := []struct {
		name     string
		vector   []float32
		expected []float32
	}{
		{
			name:     "simple 2D vector",
			vector:   []float32{3, 4},
			expected: []float32{0.6, 0.8},
		},
		{
			name:     "already unit vector",
			vector:   []float32{1, 0, 0},
			expected: []float32{1, 0, 0},
		},
		{
			name:     "zero vector",
			vector:   []float32{0, 0, 0},
			expected: []float32{0, 0, 0},
		},
		{
			name:     "negative values",
			vector:   []float32{-3, -4},
			expected: []float32{-0.6, -0.8},
		},
		{
			name:     "uniform vector",
			vector:   []float32{2, 2, 2, 2},
			expected: []float32{0.5, 0.5, 0.5, 0.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy to check if it's modified
			vector := make([]float32, len(tt.vector))
			copy(vector, tt.vector)

			NormalizeInPlace(vector)

			for i := range vector {
				if !almostEqual(vector[i], tt.expected[i], epsilon) {
					t.Errorf("NormalizeInPlace()[%d] = %f, want %f", i, vector[i], tt.expected[i])
				}
			}

			// Verify result is unit length (except for zero vector)
			originalNorm := Norm(tt.vector)
			if originalNorm != 0 {
				norm := Norm(vector)
				if !almostEqual(norm, 1.0, epsilon) {
					t.Errorf("NormalizeInPlace() result norm = %f, want 1.0", norm)
				}
			}
		})
	}
}

// ============================================================================
// Edge Cases and Integration Tests
// ============================================================================

func TestHighDimensionalVectors(t *testing.T) {
	dim := 768
	a := make([]float32, dim)
	b := make([]float32, dim)

	for i := 0; i < dim; i++ {
		a[i] = float32(i % 10)
		b[i] = float32((i + 1) % 10)
	}

	// Test with all distance metrics
	distances := []DistanceKind{Euclidean, L2Squared, Cosine}

	for _, distKind := range distances {
		t.Run(string(distKind), func(t *testing.T) {
			dist, err := NewDistance(distKind)
			if err != nil {
				t.Fatalf("NewDistance() error: %v", err)
			}

			// For cosine, normalize first
			if distKind == Cosine {
				NormalizeInPlace(a)
				NormalizeInPlace(b)
			}

			result := dist.Calculate(a, b)
			if math.IsNaN(float64(result)) || math.IsInf(float64(result), 0) {
				t.Errorf("Calculate() returned invalid value: %f", result)
			}
		})
	}
}

func TestSingletonInstances(t *testing.T) {
	// Verify that multiple calls to NewDistance return the same instance
	dist1, _ := NewDistance(Euclidean)
	dist2, _ := NewDistance(Euclidean)

	// Should be the same singleton instance
	if dist1 != dist2 {
		t.Errorf("NewDistance() should return singleton instances")
	}
}

func TestNormalizeVsNormalizeInPlace(t *testing.T) {
	original := []float32{3, 4}

	// Test Normalize (non-mutating)
	normalized := Normalize(original)
	if original[0] != 3.0 || original[1] != 4.0 {
		t.Errorf("Normalize() modified original vector")
	}

	// Test NormalizeInPlace (mutating)
	copy := make([]float32, len(original))
	copy[0], copy[1] = 3.0, 4.0
	NormalizeInPlace(copy)

	// Results should be the same
	for i := range normalized {
		if !almostEqual(normalized[i], copy[i], epsilon) {
			t.Errorf("Normalize() and NormalizeInPlace() produce different results at index %d", i)
		}
	}
}

func TestErrorConstants(t *testing.T) {
	if ErrUnknownDistanceKind == nil {
		t.Error("ErrUnknownDistanceKind should not be nil")
	}

	if ErrZeroVector == nil {
		t.Error("ErrZeroVector should not be nil")
	}

	if ErrUnknownDistanceKind.Error() == "" {
		t.Error("ErrUnknownDistanceKind should have error message")
	}

	if ErrZeroVector.Error() == "" {
		t.Error("ErrZeroVector should have error message")
	}
}

func TestDistanceKindConstants(t *testing.T) {
	if Euclidean != "l2" {
		t.Errorf("Euclidean = %q, want %q", Euclidean, "l2")
	}

	if L2Squared != "l2_squared" {
		t.Errorf("L2Squared = %q, want %q", L2Squared, "l2_squared")
	}

	if Cosine != "cosine" {
		t.Errorf("Cosine = %q, want %q", Cosine, "cosine")
	}
}

func TestCalculateBatchConsistency(t *testing.T) {
	// Verify CalculateBatch produces same results as multiple Calculate calls
	distances := []DistanceKind{Euclidean, L2Squared, Cosine}

	queries := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	target := []float32{0, 0, 0}

	for _, distKind := range distances {
		t.Run(string(distKind), func(t *testing.T) {
			dist, err := NewDistance(distKind)
			if err != nil {
				t.Fatalf("NewDistance() error: %v", err)
			}

			// For cosine, normalize
			if distKind == Cosine {
				for i := range queries {
					NormalizeInPlace(queries[i])
				}
				NormalizeInPlace(target)
			}

			// Calculate using batch
			batchResults := dist.CalculateBatch(queries, target)

			// Calculate individually
			for i, query := range queries {
				individual := dist.Calculate(query, target)
				if !almostEqual(batchResults[i], individual, epsilon) {
					t.Errorf("CalculateBatch()[%d] = %f, individual Calculate() = %f",
						i, batchResults[i], individual)
				}
			}
		})
	}
}

func TestEmptyVectors(t *testing.T) {
	dist, _ := NewDistance(L2Squared)

	// Test with empty vectors
	a := []float32{}
	b := []float32{}

	result := dist.Calculate(a, b)
	if result != 0.0 {
		t.Errorf("Calculate() with empty vectors = %f, want 0.0", result)
	}

	// Test batch with empty queries
	emptyQueries := [][]float32{}
	emptyTarget := []float32{1, 2, 3}
	batchResults := dist.CalculateBatch(emptyQueries, emptyTarget)

	if len(batchResults) != 0 {
		t.Errorf("CalculateBatch() with empty queries should return empty results")
	}
}
