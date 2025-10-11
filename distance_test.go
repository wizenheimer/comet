package comet

import (
	"math"
	"testing"
)

const epsilon = 1e-6

func almostEqual(a, b float32) bool {
	return math.Abs(float64(a-b)) < epsilon
}

func TestNewDistance(t *testing.T) {
	tests := []struct {
		name         string
		distanceKind DistanceKind
		expectError  bool
		expectedErr  error
	}{
		{
			name:         "euclidean distance",
			distanceKind: Euclidean,
			expectError:  false,
		},
		{
			name:         "cosine distance",
			distanceKind: Cosine,
			expectError:  false,
		},
		{
			name:         "dot product distance",
			distanceKind: DotProduct,
			expectError:  false,
		},
		{
			name:         "unknown distance",
			distanceKind: "unknown",
			expectError:  true,
			expectedErr:  ErrUnknownDistanceKind,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dist, err := NewDistance(tt.distanceKind)
			if tt.expectError {
				if err == nil {
					t.Errorf("NewDistance(%s) expected error, got nil", tt.distanceKind)
				}
				if tt.expectedErr != nil && err != tt.expectedErr {
					t.Errorf("NewDistance(%s) expected error %v, got %v", tt.distanceKind, tt.expectedErr, err)
				}
			} else {
				if err != nil {
					t.Errorf("NewDistance(%s) unexpected error: %v", tt.distanceKind, err)
				}
				if dist == nil {
					t.Errorf("NewDistance(%s) returned nil distance", tt.distanceKind)
				}
			}
		})
	}
}

func TestSingletonInstances(t *testing.T) {
	// Test that NewDistance returns the same singleton instance for the same distance kind
	dist1, _ := NewDistance(Euclidean)
	dist2, _ := NewDistance(Euclidean)

	// In Go, comparing interfaces to the same struct instance will be equal
	if dist1 != dist2 {
		t.Error("NewDistance should return the same singleton instance for Euclidean")
	}

	distCosine1, _ := NewDistance(Cosine)
	distCosine2, _ := NewDistance(Cosine)

	if distCosine1 != distCosine2 {
		t.Error("NewDistance should return the same singleton instance for Cosine")
	}

	distDot1, _ := NewDistance(DotProduct)
	distDot2, _ := NewDistance(DotProduct)

	if distDot1 != distDot2 {
		t.Error("NewDistance should return the same singleton instance for DotProduct")
	}
}

func TestEuclideanDistance(t *testing.T) {
	dist, err := NewDistance(Euclidean)
	if err != nil {
		t.Fatalf("Failed to create Euclidean distance: %v", err)
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
			expected: 0,
		},
		{
			name:     "simple distance",
			a:        []float32{0, 0, 0},
			b:        []float32{3, 4, 0},
			expected: 5,
		},
		{
			name:     "negative values",
			a:        []float32{-1, -2, -3},
			b:        []float32{1, 2, 3},
			expected: float32(math.Sqrt(56)),
		},
		{
			name:     "single dimension",
			a:        []float32{5},
			b:        []float32{2},
			expected: 3,
		},
		{
			name:     "high dimensional",
			a:        []float32{1, 2, 3, 4, 5},
			b:        []float32{5, 4, 3, 2, 1},
			expected: float32(math.Sqrt(40)),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := dist.Calculate(tt.a, tt.b)
			if !almostEqual(result, tt.expected) {
				t.Errorf("Euclidean.Calculate(%v, %v) = %v, want %v", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestCosineDistance(t *testing.T) {
	dist, err := NewDistance(Cosine)
	if err != nil {
		t.Fatalf("Failed to create Cosine distance: %v", err)
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
			expected: 0,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 1,
		},
		{
			name:     "opposite vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{-1, -2, -3},
			expected: 2,
		},
		{
			name:     "zero vector a",
			a:        []float32{0, 0, 0},
			b:        []float32{1, 2, 3},
			expected: 1,
		},
		{
			name:     "zero vector b",
			a:        []float32{1, 2, 3},
			b:        []float32{0, 0, 0},
			expected: 1,
		},
		{
			name:     "45 degree angle",
			a:        []float32{1, 0},
			b:        []float32{1, 1},
			expected: 1 - float32(math.Sqrt(2)/2),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := dist.Calculate(tt.a, tt.b)
			if !almostEqual(result, tt.expected) {
				t.Errorf("Cosine.Calculate(%v, %v) = %v, want %v", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestDotProductDistance(t *testing.T) {
	dist, err := NewDistance(DotProduct)
	if err != nil {
		t.Fatalf("Failed to create DotProduct distance: %v", err)
	}

	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float32
	}{
		{
			name:     "positive dot product",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			expected: -14,
		},
		{
			name:     "negative dot product",
			a:        []float32{1, 2, 3},
			b:        []float32{-1, -2, -3},
			expected: 14,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 0,
		},
		{
			name:     "zero vector",
			a:        []float32{0, 0, 0},
			b:        []float32{1, 2, 3},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := dist.Calculate(tt.a, tt.b)
			if !almostEqual(result, tt.expected) {
				t.Errorf("DotProduct.Calculate(%v, %v) = %v, want %v", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestCalculateBatchEuclidean(t *testing.T) {
	dist, err := NewDistance(Euclidean)
	if err != nil {
		t.Fatalf("Failed to create Euclidean distance: %v", err)
	}

	target := []float32{0, 0, 0}
	queries := [][]float32{
		{3, 4, 0},   // distance 5
		{1, 0, 0},   // distance 1
		{0, 0, 0},   // distance 0
		{-3, -4, 0}, // distance 5
	}

	expected := []float32{5, 1, 0, 5}

	results := dist.CalculateBatch(queries, target)

	if len(results) != len(expected) {
		t.Fatalf("CalculateBatch returned %d results, want %d", len(results), len(expected))
	}

	for i := range results {
		if !almostEqual(results[i], expected[i]) {
			t.Errorf("CalculateBatch result[%d] = %v, want %v", i, results[i], expected[i])
		}
	}
}

func TestCalculateBatchCosine(t *testing.T) {
	dist, err := NewDistance(Cosine)
	if err != nil {
		t.Fatalf("Failed to create Cosine distance: %v", err)
	}

	target := []float32{1, 0, 0}
	queries := [][]float32{
		{1, 0, 0},  // same direction: distance 0
		{0, 1, 0},  // orthogonal: distance 1
		{-1, 0, 0}, // opposite: distance 2
		{1, 1, 0},  // 45 degrees: distance 1 - sqrt(2)/2
	}

	expected := []float32{
		0,
		1,
		2,
		1 - float32(math.Sqrt(2)/2),
	}

	results := dist.CalculateBatch(queries, target)

	if len(results) != len(expected) {
		t.Fatalf("CalculateBatch returned %d results, want %d", len(results), len(expected))
	}

	for i := range results {
		if !almostEqual(results[i], expected[i]) {
			t.Errorf("CalculateBatch result[%d] = %v, want %v", i, results[i], expected[i])
		}
	}
}

func TestCalculateBatchDotProduct(t *testing.T) {
	dist, err := NewDistance(DotProduct)
	if err != nil {
		t.Fatalf("Failed to create DotProduct distance: %v", err)
	}

	target := []float32{1, 2, 3}
	queries := [][]float32{
		{1, 2, 3},    // dot product 14, result -14
		{-1, -2, -3}, // dot product -14, result 14
		{1, 0, 0},    // dot product 1, result -1
		{0, 0, 0},    // dot product 0, result 0
	}

	expected := []float32{-14, 14, -1, 0}

	results := dist.CalculateBatch(queries, target)

	if len(results) != len(expected) {
		t.Fatalf("CalculateBatch returned %d results, want %d", len(results), len(expected))
	}

	for i := range results {
		if !almostEqual(results[i], expected[i]) {
			t.Errorf("CalculateBatch result[%d] = %v, want %v", i, results[i], expected[i])
		}
	}
}

func TestCalculateBatchConsistency(t *testing.T) {
	// Test that CalculateBatch produces the same results as multiple Calculate calls
	distanceTypes := []DistanceKind{Euclidean, Cosine, DotProduct}

	target := []float32{1, 2, 3, 4}
	queries := [][]float32{
		{5, 6, 7, 8},
		{-1, -2, -3, -4},
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{4, 3, 2, 1},
	}

	for _, distType := range distanceTypes {
		t.Run(string(distType), func(t *testing.T) {
			dist, err := NewDistance(distType)
			if err != nil {
				t.Fatalf("Failed to create %s distance: %v", distType, err)
			}

			// Get batch results
			batchResults := dist.CalculateBatch(queries, target)

			// Compare with individual Calculate calls
			for i, query := range queries {
				individualResult := dist.Calculate(query, target)
				if !almostEqual(batchResults[i], individualResult) {
					t.Errorf("%s: CalculateBatch[%d] = %v, but Calculate = %v",
						distType, i, batchResults[i], individualResult)
				}
			}
		})
	}
}

func TestCalculateBatchEmpty(t *testing.T) {
	dist, _ := NewDistance(Euclidean)
	target := []float32{1, 2, 3}
	queries := [][]float32{}

	results := dist.CalculateBatch(queries, target)

	if len(results) != 0 {
		t.Errorf("CalculateBatch with empty queries returned %d results, want 0", len(results))
	}
}

// Test unexported helper functions
func TestL2DistanceSquared(t *testing.T) {
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
			expected: 0,
		},
		{
			name:     "simple distance squared",
			a:        []float32{0, 0, 0},
			b:        []float32{3, 4, 0},
			expected: 25,
		},
		{
			name:     "negative values",
			a:        []float32{-1, -2, -3},
			b:        []float32{1, 2, 3},
			expected: 56,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := l2DistanceSquared(tt.a, tt.b)
			if !almostEqual(result, tt.expected) {
				t.Errorf("l2DistanceSquared(%v, %v) = %v, want %v", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestDotProduct(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float32
	}{
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 0,
		},
		{
			name:     "parallel vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			expected: 14,
		},
		{
			name:     "negative dot product",
			a:        []float32{1, 2, 3},
			b:        []float32{-1, -2, -3},
			expected: -14,
		},
		{
			name:     "zero vector",
			a:        []float32{0, 0, 0},
			b:        []float32{1, 2, 3},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := dotProduct(tt.a, tt.b)
			if !almostEqual(result, tt.expected) {
				t.Errorf("dotProduct(%v, %v) = %v, want %v", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestNorm(t *testing.T) {
	tests := []struct {
		name     string
		v        []float32
		expected float32
	}{
		{
			name:     "unit vector",
			v:        []float32{1, 0, 0},
			expected: 1,
		},
		{
			name:     "3-4-5 triangle",
			v:        []float32{3, 4, 0},
			expected: 5,
		},
		{
			name:     "zero vector",
			v:        []float32{0, 0, 0},
			expected: 0,
		},
		{
			name:     "negative values",
			v:        []float32{-3, 4},
			expected: 5,
		},
		{
			name:     "single element",
			v:        []float32{5},
			expected: 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Norm(tt.v)
			if !almostEqual(result, tt.expected) {
				t.Errorf("Norm(%v) = %v, want %v", tt.v, result, tt.expected)
			}
		})
	}
}

func TestNormSquared(t *testing.T) {
	tests := []struct {
		name     string
		v        []float32
		expected float32
	}{
		{
			name:     "unit vector",
			v:        []float32{1, 0, 0},
			expected: 1,
		},
		{
			name:     "3-4-5 triangle",
			v:        []float32{3, 4, 0},
			expected: 25,
		},
		{
			name:     "zero vector",
			v:        []float32{0, 0, 0},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := normSquared(tt.v)
			if !almostEqual(result, tt.expected) {
				t.Errorf("normSquared(%v) = %v, want %v", tt.v, result, tt.expected)
			}
		})
	}
}

func TestCosineDistanceWithNorms(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		normA    float32
		normB    float32
		expected float32
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			normA:    float32(math.Sqrt(14)),
			normB:    float32(math.Sqrt(14)),
			expected: 0,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			normA:    1,
			normB:    1,
			expected: 1,
		},
		{
			name:     "zero norm a",
			a:        []float32{0, 0, 0},
			b:        []float32{1, 2, 3},
			normA:    0,
			normB:    float32(math.Sqrt(14)),
			expected: 1,
		},
		{
			name:     "zero norm b",
			a:        []float32{1, 2, 3},
			b:        []float32{0, 0, 0},
			normA:    float32(math.Sqrt(14)),
			normB:    0,
			expected: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cosineDistanceWithNorms(tt.a, tt.b, tt.normA, tt.normB)
			if !almostEqual(result, tt.expected) {
				t.Errorf("cosineDistanceWithNorms(%v, %v, %v, %v) = %v, want %v",
					tt.a, tt.b, tt.normA, tt.normB, result, tt.expected)
			}
		})
	}
}

func TestCosineDistanceConsistency(t *testing.T) {
	// Test that cosineDistance and cosineDistanceWithNorms produce the same results
	tests := []struct {
		name string
		a    []float32
		b    []float32
	}{
		{
			name: "random vectors 1",
			a:    []float32{1, 2, 3, 4},
			b:    []float32{5, 6, 7, 8},
		},
		{
			name: "random vectors 2",
			a:    []float32{-1, 2, -3, 4},
			b:    []float32{5, -6, 7, -8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result1 := cosineDistance(tt.a, tt.b)
			normA := Norm(tt.a)
			normB := Norm(tt.b)
			result2 := cosineDistanceWithNorms(tt.a, tt.b, normA, normB)
			if !almostEqual(result1, result2) {
				t.Errorf("cosineDistance(%v, %v) = %v, but cosineDistanceWithNorms = %v",
					tt.a, tt.b, result1, result2)
			}
		})
	}
}

// Benchmark tests
func BenchmarkEuclideanDistance(b *testing.B) {
	dist, _ := NewDistance(Euclidean)
	v1 := make([]float32, 128)
	v2 := make([]float32, 128)
	for i := range v1 {
		v1[i] = float32(i)
		v2[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dist.Calculate(v1, v2)
	}
}

func BenchmarkCosineDistance(b *testing.B) {
	dist, _ := NewDistance(Cosine)
	v1 := make([]float32, 128)
	v2 := make([]float32, 128)
	for i := range v1 {
		v1[i] = float32(i)
		v2[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dist.Calculate(v1, v2)
	}
}

func BenchmarkDotProductDistance(b *testing.B) {
	dist, _ := NewDistance(DotProduct)
	v1 := make([]float32, 128)
	v2 := make([]float32, 128)
	for i := range v1 {
		v1[i] = float32(i)
		v2[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dist.Calculate(v1, v2)
	}
}

func BenchmarkL2Distance(b *testing.B) {
	v1 := make([]float32, 128)
	v2 := make([]float32, 128)
	for i := range v1 {
		v1[i] = float32(i)
		v2[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l2Distance(v1, v2)
	}
}

func BenchmarkL2DistanceSquared(b *testing.B) {
	v1 := make([]float32, 128)
	v2 := make([]float32, 128)
	for i := range v1 {
		v1[i] = float32(i)
		v2[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l2DistanceSquared(v1, v2)
	}
}

func BenchmarkDotProduct(b *testing.B) {
	v1 := make([]float32, 128)
	v2 := make([]float32, 128)
	for i := range v1 {
		v1[i] = float32(i)
		v2[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dotProduct(v1, v2)
	}
}

func BenchmarkNorm(b *testing.B) {
	v := make([]float32, 128)
	for i := range v {
		v[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Norm(v)
	}
}

func BenchmarkCosineDistanceWithNorms(b *testing.B) {
	v1 := make([]float32, 128)
	v2 := make([]float32, 128)
	for i := range v1 {
		v1[i] = float32(i)
		v2[i] = float32(i + 1)
	}
	normA := Norm(v1)
	normB := Norm(v2)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cosineDistanceWithNorms(v1, v2, normA, normB)
	}
}

func BenchmarkCalculateBatchEuclidean(b *testing.B) {
	dist, _ := NewDistance(Euclidean)
	target := make([]float32, 128)
	queries := make([][]float32, 100)

	for i := range target {
		target[i] = float32(i)
	}
	for i := range queries {
		queries[i] = make([]float32, 128)
		for j := range queries[i] {
			queries[i][j] = float32(i + j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dist.CalculateBatch(queries, target)
	}
}

func BenchmarkCalculateBatchCosine(b *testing.B) {
	dist, _ := NewDistance(Cosine)
	target := make([]float32, 128)
	queries := make([][]float32, 100)

	for i := range target {
		target[i] = float32(i)
	}
	for i := range queries {
		queries[i] = make([]float32, 128)
		for j := range queries[i] {
			queries[i][j] = float32(i + j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dist.CalculateBatch(queries, target)
	}
}

func BenchmarkCalculateBatchDotProduct(b *testing.B) {
	dist, _ := NewDistance(DotProduct)
	target := make([]float32, 128)
	queries := make([][]float32, 100)

	for i := range target {
		target[i] = float32(i)
	}
	for i := range queries {
		queries[i] = make([]float32, 128)
		for j := range queries[i] {
			queries[i][j] = float32(i + j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dist.CalculateBatch(queries, target)
	}
}
