package comet

import (
	"fmt"
	"math"

	"github.com/x448/float16"
)

// ============================================================================
// QUANTIZER INTERFACE
// ============================================================================

// QuantizerType represents the type of quantization
type QuantizerType string

const (
	FullPrecision QuantizerType = "float32"
	HalfPrecision QuantizerType = "float16"
	Int8Precision QuantizerType = "int8"
)

// Quantizer defines the interface for vector quantization operations.
// Implementations handle conversion between float32 vectors and their
// compressed representations, supporting different precision levels.
type Quantizer interface {
	// Train prepares the quantizer using sample vectors.
	// Required for Int8Quantizer, no-op for others.
	Train(vectors [][]float32)

	// IsTrained returns true if the quantizer is ready to use.
	// Always true for FullPrecision and HalfPrecision.
	// For Int8, true only after Train() has been called.
	IsTrained() bool

	// Quantize converts a float32 vector to the quantizer's storage format.
	// Returns:
	//   - []float32 for FullPrecisionQuantizer
	//   - []uint16 for HalfPrecisionQuantizer (float16 bits)
	//   - []int8 for Int8Quantizer
	Quantize(vector []float32) (any, error)

	// Dequantize converts a stored vector back to float32.
	// The input type must match the quantizer's storage format.
	Dequantize(stored any) ([]float32, error)

	// Type returns the quantizer type
	Type() QuantizerType
}

// ============================================================================
// FACTORY FUNCTION
// ============================================================================

// NewQuantizer creates a quantizer of the specified type.
func NewQuantizer(qType QuantizerType) (Quantizer, error) {
	switch qType {
	case FullPrecision:
		return &FullPrecisionQuantizer{}, nil
	case HalfPrecision:
		return &HalfPrecisionQuantizer{}, nil
	case Int8Precision:
		return &Int8Quantizer{}, nil
	default:
		return nil, fmt.Errorf("unsupported quantizer type: %s", qType)
	}
}

// ============================================================================
// FULL PRECISION QUANTIZER (Float32)
// ============================================================================

// FullPrecisionQuantizer stores vectors in full 32-bit floating point.
//
// Memory: 4 bytes per dimension
// Accuracy: Full IEEE 754 single precision
// Training: Not required
//
// This is essentially a no-op quantizer that provides the interface
// for consistency while maintaining full precision.
type FullPrecisionQuantizer struct{}

func (q *FullPrecisionQuantizer) Train(vectors [][]float32) {
	// No training needed for full precision
}

func (q *FullPrecisionQuantizer) IsTrained() bool {
	return true // Always ready
}

func (q *FullPrecisionQuantizer) Quantize(vector []float32) (any, error) {
	// Return a copy to prevent external modifications
	result := make([]float32, len(vector))
	copy(result, vector)
	return result, nil
}

func (q *FullPrecisionQuantizer) Dequantize(stored any) ([]float32, error) {
	vec, ok := stored.([]float32)
	if !ok {
		return nil, fmt.Errorf("expected []float32, got %T", stored)
	}

	// Return a copy
	result := make([]float32, len(vec))
	copy(result, vec)
	return result, nil
}

func (q *FullPrecisionQuantizer) Type() QuantizerType {
	return FullPrecision
}

// ============================================================================
// HALF PRECISION QUANTIZER (Float16)
// ============================================================================

// HalfPrecisionQuantizer compresses vectors to 16-bit floating point.
//
// Memory: 2 bytes per dimension (50% savings vs float32)
// Accuracy: IEEE 754 half precision (1 sign, 5 exp, 10 mantissa bits)
// Training: Not required
//
// Trade-off: Significant memory savings with minimal accuracy loss for
// most use cases. Values are stored as uint16 bit representations.
type HalfPrecisionQuantizer struct{}

func (q *HalfPrecisionQuantizer) Train(vectors [][]float32) {
	// No training needed for half precision
}

func (q *HalfPrecisionQuantizer) IsTrained() bool {
	return true // Always ready
}

func (q *HalfPrecisionQuantizer) Quantize(vector []float32) (any, error) {
	// Convert float32 -> float16 (stored as uint16 bit representation)
	f16Vec := make([]uint16, len(vector))
	for i, v := range vector {
		f16Vec[i] = float16.Fromfloat32(v).Bits()
	}
	return f16Vec, nil
}

func (q *HalfPrecisionQuantizer) Dequantize(stored any) ([]float32, error) {
	vec, ok := stored.([]uint16)
	if !ok {
		return nil, fmt.Errorf("expected []uint16, got %T", stored)
	}

	// Convert float16 -> float32
	f32Vec := make([]float32, len(vec))
	for i, bits := range vec {
		f32Vec[i] = float16.Frombits(bits).Float32()
	}
	return f32Vec, nil
}

func (q *HalfPrecisionQuantizer) Type() QuantizerType {
	return HalfPrecision
}

// ============================================================================
// INT8 QUANTIZER (Symmetric Scalar Quantization)
// ============================================================================

// Int8Quantizer uses symmetric scalar quantization to compress vectors.
//
// Memory: 1 byte per dimension (75% savings vs float32)
// Accuracy: Maps [-AbsMax, AbsMax] to [-127, 127]
// Training: REQUIRED - must call Train() before use
//
// How it works:
// 1. Train: Finds the maximum absolute value across sample vectors
// 2. Quantize: Maps [-AbsMax, AbsMax] to [-127, 127] using linear scaling
// 3. Dequantize: Reverses the scaling to approximate original values
//
// Trade-off: Maximum memory savings with some accuracy loss. Best for
// large-scale deployments where memory is critical.
type Int8Quantizer struct {
	absMax float32 // Maximum absolute value from training
}

func (q *Int8Quantizer) Train(vectors [][]float32) {
	var max float32
	for _, vec := range vectors {
		for _, val := range vec {
			absVal := float32(math.Abs(float64(val)))
			if absVal > max {
				max = absVal
			}
		}
	}
	q.absMax = max
}

func (q *Int8Quantizer) IsTrained() bool {
	return q.absMax > 0
}

func (q *Int8Quantizer) Quantize(vector []float32) (any, error) {
	if !q.IsTrained() {
		return nil, fmt.Errorf("quantizer must be trained before use")
	}

	quantized := make([]int8, len(vector))
	for i, val := range vector {
		// Map [-AbsMax, AbsMax] -> [-127, 127]
		scaled := (val / q.absMax) * 127.0
		quantized[i] = int8(math.Round(float64(scaled)))
	}
	return quantized, nil
}

func (q *Int8Quantizer) Dequantize(stored any) ([]float32, error) {
	vec, ok := stored.([]int8)
	if !ok {
		return nil, fmt.Errorf("expected []int8, got %T", stored)
	}

	if !q.IsTrained() {
		return nil, fmt.Errorf("quantizer must be trained before dequantization")
	}

	dequantized := make([]float32, len(vec))
	for i, val := range vec {
		// Reverse the quantization formula:
		// quantized = round((original / AbsMax) * 127)
		// original ≈ (quantized / 127) * AbsMax
		dequantized[i] = (float32(val) / 127.0) * q.absMax
	}
	return dequantized, nil
}

func (q *Int8Quantizer) Type() QuantizerType {
	return Int8Precision
}

// GetAbsMax returns the trained maximum absolute value (for serialization)
func (q *Int8Quantizer) GetAbsMax() float32 {
	return q.absMax
}

// SetAbsMax sets the maximum absolute value (for deserialization)
func (q *Int8Quantizer) SetAbsMax(absMax float32) {
	q.absMax = absMax
}
