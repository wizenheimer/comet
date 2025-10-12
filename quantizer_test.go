package comet

import (
	"math"
	"testing"
)

// ============================================================================
// FACTORY FUNCTION TESTS
// ============================================================================

func TestNewQuantizer(t *testing.T) {
	tests := []struct {
		name         string
		qType        QuantizerType
		expectError  bool
		expectedType QuantizerType
	}{
		{
			name:         "create full precision quantizer",
			qType:        FullPrecision,
			expectError:  false,
			expectedType: FullPrecision,
		},
		{
			name:         "create half precision quantizer",
			qType:        HalfPrecision,
			expectError:  false,
			expectedType: HalfPrecision,
		},
		{
			name:         "create int8 quantizer",
			qType:        Int8Precision,
			expectError:  false,
			expectedType: Int8Precision,
		},
		{
			name:        "invalid quantizer type",
			qType:       QuantizerType("invalid"),
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q, err := NewQuantizer(tt.qType)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if q.Type() != tt.expectedType {
				t.Errorf("expected type %s, got %s", tt.expectedType, q.Type())
			}
		})
	}
}

// ============================================================================
// FULL PRECISION QUANTIZER TESTS
// ============================================================================

func TestFullPrecisionQuantizer_Basic(t *testing.T) {
	q := &FullPrecisionQuantizer{}

	// Should always be trained
	if !q.IsTrained() {
		t.Error("FullPrecisionQuantizer should always be trained")
	}

	// Type check
	if q.Type() != FullPrecision {
		t.Errorf("expected type %s, got %s", FullPrecision, q.Type())
	}

	// Train should be no-op (shouldn't panic)
	q.Train([][]float32{{1.0, 2.0, 3.0}})
}

func TestFullPrecisionQuantizer_QuantizeDequantize(t *testing.T) {
	q := &FullPrecisionQuantizer{}

	testVectors := [][]float32{
		{1.0, 2.0, 3.0},
		{-1.5, 0.0, 2.5},
		{0.0, 0.0, 0.0},
		{100.5, -200.3, 50.1},
	}

	for _, vec := range testVectors {
		// Quantize
		quantized, err := q.Quantize(vec)
		if err != nil {
			t.Errorf("Quantize error: %v", err)
			continue
		}

		// Check type
		_, ok := quantized.([]float32)
		if !ok {
			t.Errorf("expected []float32, got %T", quantized)
			continue
		}

		// Dequantize
		dequantized, err := q.Dequantize(quantized)
		if err != nil {
			t.Errorf("Dequantize error: %v", err)
			continue
		}

		// Should be exact match for full precision
		if len(dequantized) != len(vec) {
			t.Errorf("length mismatch: expected %d, got %d", len(vec), len(dequantized))
			continue
		}

		for i := range vec {
			if dequantized[i] != vec[i] {
				t.Errorf("value mismatch at index %d: expected %f, got %f", i, vec[i], dequantized[i])
			}
		}
	}
}

func TestFullPrecisionQuantizer_Isolation(t *testing.T) {
	q := &FullPrecisionQuantizer{}

	original := []float32{1.0, 2.0, 3.0}

	quantized, err := q.Quantize(original)
	if err != nil {
		t.Fatalf("Quantize error: %v", err)
	}

	quantizedVec := quantized.([]float32)

	// Modify the quantized vector
	quantizedVec[0] = 999.0

	// Original should be unchanged
	if original[0] != 1.0 {
		t.Error("modifying quantized vector affected original")
	}

	// Dequantize
	dequantized, err := q.Dequantize(quantized)
	if err != nil {
		t.Fatalf("Dequantize error: %v", err)
	}

	// Modify dequantized
	dequantized[1] = 888.0

	// Quantized should be unchanged
	if quantizedVec[1] != 2.0 {
		t.Error("modifying dequantized vector affected quantized")
	}
}

func TestFullPrecisionQuantizer_InvalidType(t *testing.T) {
	q := &FullPrecisionQuantizer{}

	// Try to dequantize wrong type
	_, err := q.Dequantize([]int8{1, 2, 3})
	if err == nil {
		t.Error("expected error when dequantizing wrong type")
	}
}

// ============================================================================
// HALF PRECISION QUANTIZER TESTS
// ============================================================================

func TestHalfPrecisionQuantizer_Basic(t *testing.T) {
	q := &HalfPrecisionQuantizer{}

	// Should always be trained
	if !q.IsTrained() {
		t.Error("HalfPrecisionQuantizer should always be trained")
	}

	// Type check
	if q.Type() != HalfPrecision {
		t.Errorf("expected type %s, got %s", HalfPrecision, q.Type())
	}

	// Train should be no-op (shouldn't panic)
	q.Train([][]float32{{1.0, 2.0, 3.0}})
}

func TestHalfPrecisionQuantizer_QuantizeDequantize(t *testing.T) {
	q := &HalfPrecisionQuantizer{}

	testCases := []struct {
		name      string
		vector    []float32
		tolerance float32
	}{
		{
			name:      "simple positive values",
			vector:    []float32{1.0, 2.0, 3.0},
			tolerance: 0.001,
		},
		{
			name:      "mixed signs",
			vector:    []float32{-1.5, 0.0, 2.5},
			tolerance: 0.001,
		},
		{
			name:      "zeros",
			vector:    []float32{0.0, 0.0, 0.0},
			tolerance: 0.0,
		},
		{
			name:      "small values",
			vector:    []float32{0.1, 0.2, 0.3},
			tolerance: 0.001,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Quantize
			quantized, err := q.Quantize(tc.vector)
			if err != nil {
				t.Fatalf("Quantize error: %v", err)
			}

			// Check type
			quantizedVec, ok := quantized.([]uint16)
			if !ok {
				t.Fatalf("expected []uint16, got %T", quantized)
			}

			if len(quantizedVec) != len(tc.vector) {
				t.Fatalf("length mismatch: expected %d, got %d", len(tc.vector), len(quantizedVec))
			}

			// Dequantize
			dequantized, err := q.Dequantize(quantized)
			if err != nil {
				t.Fatalf("Dequantize error: %v", err)
			}

			// Check approximate equality (float16 has precision loss)
			for i := range tc.vector {
				diff := math.Abs(float64(dequantized[i] - tc.vector[i]))
				if diff > float64(tc.tolerance) {
					t.Errorf("value at index %d: expected ~%f, got %f (diff: %f)",
						i, tc.vector[i], dequantized[i], diff)
				}
			}
		})
	}
}

func TestHalfPrecisionQuantizer_InvalidType(t *testing.T) {
	q := &HalfPrecisionQuantizer{}

	// Try to dequantize wrong type
	_, err := q.Dequantize([]float32{1.0, 2.0, 3.0})
	if err == nil {
		t.Error("expected error when dequantizing wrong type")
	}

	_, err = q.Dequantize([]int8{1, 2, 3})
	if err == nil {
		t.Error("expected error when dequantizing wrong type")
	}
}

// ============================================================================
// INT8 QUANTIZER TESTS
// ============================================================================

func TestInt8Quantizer_Basic(t *testing.T) {
	q := &Int8Quantizer{}

	// Should not be trained initially
	if q.IsTrained() {
		t.Error("Int8Quantizer should not be trained initially")
	}

	// Type check
	if q.Type() != Int8Precision {
		t.Errorf("expected type %s, got %s", Int8Precision, q.Type())
	}
}

func TestInt8Quantizer_Training(t *testing.T) {
	q := &Int8Quantizer{}

	// Training data
	vectors := [][]float32{
		{1.0, 2.0, 3.0},
		{-1.0, -2.0, -3.0},
		{0.5, 1.5, 2.5},
	}

	// Train
	q.Train(vectors)

	// Should be trained now
	if !q.IsTrained() {
		t.Error("quantizer should be trained after calling Train()")
	}

	// Check that absMax was set correctly (should be 3.0)
	expectedMax := float32(3.0)
	if q.GetAbsMax() != expectedMax {
		t.Errorf("expected absMax %f, got %f", expectedMax, q.GetAbsMax())
	}
}

func TestInt8Quantizer_QuantizeBeforeTraining(t *testing.T) {
	q := &Int8Quantizer{}

	vector := []float32{1.0, 2.0, 3.0}

	_, err := q.Quantize(vector)
	if err == nil {
		t.Error("expected error when quantizing before training")
	}
}

func TestInt8Quantizer_DequantizeBeforeTraining(t *testing.T) {
	q := &Int8Quantizer{}

	quantized := []int8{10, 20, 30}

	_, err := q.Dequantize(quantized)
	if err == nil {
		t.Error("expected error when dequantizing before training")
	}
}

func TestInt8Quantizer_QuantizeDequantize(t *testing.T) {
	q := &Int8Quantizer{}

	// Train with sample data
	trainingData := [][]float32{
		{-10.0, -5.0, 0.0, 5.0, 10.0},
		{-8.0, -4.0, 0.0, 4.0, 8.0},
		{-6.0, -3.0, 0.0, 3.0, 6.0},
	}
	q.Train(trainingData)

	testCases := []struct {
		name      string
		vector    []float32
		tolerance float32
	}{
		{
			name:      "values within range",
			vector:    []float32{5.0, -5.0, 0.0},
			tolerance: 0.1,
		},
		{
			name:      "max values",
			vector:    []float32{10.0, -10.0, 0.0},
			tolerance: 0.1,
		},
		{
			name:      "small values",
			vector:    []float32{0.5, -0.5, 0.0},
			tolerance: 0.2, // Higher tolerance for small values
		},
		{
			name:      "zeros",
			vector:    []float32{0.0, 0.0, 0.0},
			tolerance: 0.01,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Quantize
			quantized, err := q.Quantize(tc.vector)
			if err != nil {
				t.Fatalf("Quantize error: %v", err)
			}

			// Check type
			quantizedVec, ok := quantized.([]int8)
			if !ok {
				t.Fatalf("expected []int8, got %T", quantized)
			}

			if len(quantizedVec) != len(tc.vector) {
				t.Fatalf("length mismatch: expected %d, got %d", len(tc.vector), len(quantizedVec))
			}

			// Verify quantized values are in valid range [-127, 127]
			// Note: int8 range is [-128, 127] but we use [-127, 127] for symmetric quantization
			for i, val := range quantizedVec {
				if val < -127 {
					t.Errorf("quantized value at index %d out of range: %d", i, val)
				}
			}

			// Dequantize
			dequantized, err := q.Dequantize(quantized)
			if err != nil {
				t.Fatalf("Dequantize error: %v", err)
			}

			// Check approximate equality
			for i := range tc.vector {
				diff := math.Abs(float64(dequantized[i] - tc.vector[i]))
				if diff > float64(tc.tolerance) {
					t.Errorf("value at index %d: expected ~%f, got %f (diff: %f, quantized: %d)",
						i, tc.vector[i], dequantized[i], diff, quantizedVec[i])
				}
			}
		})
	}
}

func TestInt8Quantizer_Symmetry(t *testing.T) {
	q := &Int8Quantizer{}

	// Train with symmetric data
	trainingData := [][]float32{
		{-5.0, 0.0, 5.0},
	}
	q.Train(trainingData)

	// Test that positive and negative values quantize symmetrically
	positiveVec := []float32{5.0, 2.5, 1.25}
	negativeVec := []float32{-5.0, -2.5, -1.25}

	posQuantized, _ := q.Quantize(positiveVec)
	negQuantized, _ := q.Quantize(negativeVec)

	posVec := posQuantized.([]int8)
	negVec := negQuantized.([]int8)

	for i := range posVec {
		if posVec[i] != -negVec[i] {
			t.Errorf("symmetry broken at index %d: pos=%d, neg=%d", i, posVec[i], negVec[i])
		}
	}
}

func TestInt8Quantizer_InvalidType(t *testing.T) {
	q := &Int8Quantizer{}
	q.Train([][]float32{{1.0, 2.0, 3.0}})

	// Try to dequantize wrong types
	_, err := q.Dequantize([]float32{1.0, 2.0, 3.0})
	if err == nil {
		t.Error("expected error when dequantizing wrong type")
	}

	_, err = q.Dequantize([]uint16{1, 2, 3})
	if err == nil {
		t.Error("expected error when dequantizing wrong type")
	}
}

func TestInt8Quantizer_GetSetAbsMax(t *testing.T) {
	q := &Int8Quantizer{}

	// Set absMax manually (e.g., for deserialization)
	expectedMax := float32(7.5)
	q.SetAbsMax(expectedMax)

	// Should be trained now
	if !q.IsTrained() {
		t.Error("quantizer should be trained after SetAbsMax()")
	}

	// Check value
	if q.GetAbsMax() != expectedMax {
		t.Errorf("expected absMax %f, got %f", expectedMax, q.GetAbsMax())
	}

	// Should be able to quantize now
	vec := []float32{3.75, -3.75, 0.0}
	quantized, err := q.Quantize(vec)
	if err != nil {
		t.Errorf("unexpected error after SetAbsMax: %v", err)
	}

	if quantized == nil {
		t.Error("expected non-nil quantized vector")
	}
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

func TestQuantizers_EmptyVector(t *testing.T) {
	quantizers := []Quantizer{
		&FullPrecisionQuantizer{},
		&HalfPrecisionQuantizer{},
	}

	// Int8 needs training
	int8Q := &Int8Quantizer{}
	int8Q.Train([][]float32{{1.0}})
	quantizers = append(quantizers, int8Q)

	emptyVec := []float32{}

	for _, q := range quantizers {
		t.Run(string(q.Type()), func(t *testing.T) {
			quantized, err := q.Quantize(emptyVec)
			if err != nil {
				t.Errorf("error quantizing empty vector: %v", err)
				return
			}

			dequantized, err := q.Dequantize(quantized)
			if err != nil {
				t.Errorf("error dequantizing empty vector: %v", err)
				return
			}

			if len(dequantized) != 0 {
				t.Errorf("expected empty vector, got length %d", len(dequantized))
			}
		})
	}
}

func TestQuantizers_SingleElement(t *testing.T) {
	quantizers := []Quantizer{
		&FullPrecisionQuantizer{},
		&HalfPrecisionQuantizer{},
	}

	// Int8 needs training
	int8Q := &Int8Quantizer{}
	int8Q.Train([][]float32{{5.0}})
	quantizers = append(quantizers, int8Q)

	testVec := []float32{3.14}

	for _, q := range quantizers {
		t.Run(string(q.Type()), func(t *testing.T) {
			quantized, err := q.Quantize(testVec)
			if err != nil {
				t.Errorf("error quantizing: %v", err)
				return
			}

			dequantized, err := q.Dequantize(quantized)
			if err != nil {
				t.Errorf("error dequantizing: %v", err)
				return
			}

			if len(dequantized) != 1 {
				t.Errorf("expected length 1, got %d", len(dequantized))
			}
		})
	}
}

func TestInt8Quantizer_TrainWithEmptyVectors(t *testing.T) {
	q := &Int8Quantizer{}

	// Train with empty data
	q.Train([][]float32{})

	// Should not be trained
	if q.IsTrained() {
		t.Error("quantizer should not be trained with empty data")
	}
}

func TestInt8Quantizer_TrainWithZeros(t *testing.T) {
	q := &Int8Quantizer{}

	// Train with all zeros
	q.Train([][]float32{
		{0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0},
	})

	// absMax should be 0, so not trained
	if q.IsTrained() {
		t.Error("quantizer should not be trained when all values are zero")
	}
}

// ============================================================================
// BENCHMARK TESTS
// ============================================================================

func BenchmarkFullPrecisionQuantizer_Quantize(b *testing.B) {
	q := &FullPrecisionQuantizer{}
	vec := make([]float32, 512)
	for i := range vec {
		vec[i] = float32(i) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.Quantize(vec)
	}
}

func BenchmarkHalfPrecisionQuantizer_Quantize(b *testing.B) {
	q := &HalfPrecisionQuantizer{}
	vec := make([]float32, 512)
	for i := range vec {
		vec[i] = float32(i) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.Quantize(vec)
	}
}

func BenchmarkInt8Quantizer_Quantize(b *testing.B) {
	q := &Int8Quantizer{}
	trainingData := [][]float32{{-100.0, 0.0, 100.0}}
	q.Train(trainingData)

	vec := make([]float32, 512)
	for i := range vec {
		vec[i] = float32(i) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.Quantize(vec)
	}
}

func BenchmarkFullPrecisionQuantizer_Dequantize(b *testing.B) {
	q := &FullPrecisionQuantizer{}
	vec := make([]float32, 512)
	for i := range vec {
		vec[i] = float32(i) * 0.1
	}
	quantized, _ := q.Quantize(vec)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.Dequantize(quantized)
	}
}

func BenchmarkHalfPrecisionQuantizer_Dequantize(b *testing.B) {
	q := &HalfPrecisionQuantizer{}
	vec := make([]float32, 512)
	for i := range vec {
		vec[i] = float32(i) * 0.1
	}
	quantized, _ := q.Quantize(vec)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.Dequantize(quantized)
	}
}

func BenchmarkInt8Quantizer_Dequantize(b *testing.B) {
	q := &Int8Quantizer{}
	trainingData := [][]float32{{-100.0, 0.0, 100.0}}
	q.Train(trainingData)

	vec := make([]float32, 512)
	for i := range vec {
		vec[i] = float32(i) * 0.1
	}
	quantized, _ := q.Quantize(vec)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.Dequantize(quantized)
	}
}
