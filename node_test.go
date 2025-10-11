package comet

import (
	"math"
	"sync"
	"testing"
)

func TestNewVectorNode(t *testing.T) {
	tests := []struct {
		name   string
		vector []float32
	}{
		{
			name:   "simple vector",
			vector: []float32{1, 2, 3},
		},
		{
			name:   "empty vector",
			vector: []float32{},
		},
		{
			name:   "single element",
			vector: []float32{42},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := NewVectorNode(tt.vector)
			if node == nil {
				t.Fatal("NewVectorNode returned nil")
			}
			if node.ID() == 0 {
				t.Error("Node ID should not be 0")
			}
			if len(node.Vector()) != len(tt.vector) {
				t.Errorf("Vector length mismatch: got %d, want %d", len(node.Vector()), len(tt.vector))
			}
		})
	}
}

func TestNewVectorNodeConcurrency(t *testing.T) {
	const numGoroutines = 100
	nodes := make([]*Node, numGoroutines)
	var wg sync.WaitGroup

	wg.Add(numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			nodes[idx] = NewVectorNode([]float32{1, 2, 3})
		}(i)
	}
	wg.Wait()

	// Check all IDs are unique
	ids := make(map[uint32]bool)
	for _, node := range nodes {
		if ids[node.ID()] {
			t.Errorf("Duplicate ID found: %d", node.ID())
		}
		ids[node.ID()] = true
	}
}

func TestNodeIDAndVector(t *testing.T) {
	vector := []float32{1.5, 2.5, 3.5}
	node := NewVectorNode(vector)

	if node.ID() == 0 {
		t.Error("ID should not be 0")
	}

	nodeVector := node.Vector()
	if len(nodeVector) != len(vector) {
		t.Errorf("Vector length mismatch: got %d, want %d", len(nodeVector), len(vector))
	}

	for i := range vector {
		if nodeVector[i] != vector[i] {
			t.Errorf("Vector[%d] = %f, want %f", i, nodeVector[i], vector[i])
		}
	}
}

func TestNodeComparableToNode(t *testing.T) {
	tests := []struct {
		name     string
		node1    []float32
		node2    []float32
		expected bool
	}{
		{
			name:     "same dimension",
			node1:    []float32{1, 2, 3},
			node2:    []float32{4, 5, 6},
			expected: true,
		},
		{
			name:     "different dimension",
			node1:    []float32{1, 2, 3},
			node2:    []float32{4, 5},
			expected: false,
		},
		{
			name:     "both empty",
			node1:    []float32{},
			node2:    []float32{},
			expected: true,
		},
		{
			name:     "one empty",
			node1:    []float32{1},
			node2:    []float32{},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n1 := NewVectorNode(tt.node1)
			n2 := NewVectorNode(tt.node2)
			result := n1.ComparableToNode(n2)
			if result != tt.expected {
				t.Errorf("ComparableToNode() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestNodeComparableToVector(t *testing.T) {
	tests := []struct {
		name     string
		node     []float32
		vector   []float32
		expected bool
	}{
		{
			name:     "same dimension",
			node:     []float32{1, 2, 3},
			vector:   []float32{4, 5, 6},
			expected: true,
		},
		{
			name:     "different dimension",
			node:     []float32{1, 2, 3},
			vector:   []float32{4, 5},
			expected: false,
		},
		{
			name:     "both empty",
			node:     []float32{},
			vector:   []float32{},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := NewVectorNode(tt.node)
			result := n.ComparableToVector(tt.vector)
			if result != tt.expected {
				t.Errorf("ComparableToVector() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestNodeCopy(t *testing.T) {
	original := NewVectorNode([]float32{1, 2, 3})
	copied := original.Copy()

	// Check IDs are the same
	if copied.ID() != original.ID() {
		t.Errorf("Copy ID = %d, want %d", copied.ID(), original.ID())
	}

	// Check vectors are equal
	origVec := original.Vector()
	copyVec := copied.Vector()
	for i := range origVec {
		if copyVec[i] != origVec[i] {
			t.Errorf("Copy vector[%d] = %f, want %f", i, copyVec[i], origVec[i])
		}
	}

	// Check deep copy - modify original shouldn't affect copy
	origVec[0] = 999
	if copyVec[0] == 999 {
		t.Error("Copy should be independent of original")
	}
}

func TestNodeAdd(t *testing.T) {
	tests := []struct {
		name     string
		initial  []float32
		add      []float32
		expected []float32
	}{
		{
			name:     "simple addition",
			initial:  []float32{1, 2, 3},
			add:      []float32{4, 5, 6},
			expected: []float32{5, 7, 9},
		},
		{
			name:     "negative values",
			initial:  []float32{1, 2, 3},
			add:      []float32{-1, -2, -3},
			expected: []float32{0, 0, 0},
		},
		{
			name:     "zero vector",
			initial:  []float32{1, 2, 3},
			add:      []float32{0, 0, 0},
			expected: []float32{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := NewVectorNode(tt.initial)
			node.Add(tt.add)
			result := node.Vector()

			for i := range tt.expected {
				if !floatEqual(result[i], tt.expected[i]) {
					t.Errorf("Add result[%d] = %f, want %f", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestNodeSub(t *testing.T) {
	tests := []struct {
		name     string
		initial  []float32
		sub      []float32
		expected []float32
	}{
		{
			name:     "simple subtraction",
			initial:  []float32{5, 7, 9},
			sub:      []float32{1, 2, 3},
			expected: []float32{4, 5, 6},
		},
		{
			name:     "negative values",
			initial:  []float32{1, 2, 3},
			sub:      []float32{-1, -2, -3},
			expected: []float32{2, 4, 6},
		},
		{
			name:     "zero vector",
			initial:  []float32{1, 2, 3},
			sub:      []float32{0, 0, 0},
			expected: []float32{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := NewVectorNode(tt.initial)
			node.Sub(tt.sub)
			result := node.Vector()

			for i := range tt.expected {
				if !floatEqual(result[i], tt.expected[i]) {
					t.Errorf("Sub result[%d] = %f, want %f", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestNodeScale(t *testing.T) {
	tests := []struct {
		name     string
		initial  []float32
		scalar   float32
		expected []float32
	}{
		{
			name:     "scale by 2",
			initial:  []float32{1, 2, 3},
			scalar:   2,
			expected: []float32{2, 4, 6},
		},
		{
			name:     "scale by 0.5",
			initial:  []float32{2, 4, 6},
			scalar:   0.5,
			expected: []float32{1, 2, 3},
		},
		{
			name:     "scale by -1",
			initial:  []float32{1, 2, 3},
			scalar:   -1,
			expected: []float32{-1, -2, -3},
		},
		{
			name:     "scale by 0",
			initial:  []float32{1, 2, 3},
			scalar:   0,
			expected: []float32{0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := NewVectorNode(tt.initial)
			node.Scale(tt.scalar)
			result := node.Vector()

			for i := range tt.expected {
				if !floatEqual(result[i], tt.expected[i]) {
					t.Errorf("Scale result[%d] = %f, want %f", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestNodeL2Norm(t *testing.T) {
	tests := []struct {
		name     string
		vector   []float32
		expected float32
	}{
		{
			name:     "unit vector",
			vector:   []float32{1, 0, 0},
			expected: 1.0,
		},
		{
			name:     "3-4-5 triangle",
			vector:   []float32{3, 4, 0},
			expected: 5.0,
		},
		{
			name:     "zero vector",
			vector:   []float32{0, 0, 0},
			expected: 0.0,
		},
		{
			name:     "negative values",
			vector:   []float32{-3, -4, 0},
			expected: 5.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := NewVectorNode(tt.vector)
			result := node.L2Norm()
			if !floatEqual(result, tt.expected) {
				t.Errorf("L2Norm() = %f, want %f", result, tt.expected)
			}
		})
	}
}

func TestNodeL2NormSquared(t *testing.T) {
	tests := []struct {
		name     string
		vector   []float32
		expected float32
	}{
		{
			name:     "unit vector",
			vector:   []float32{1, 0, 0},
			expected: 1.0,
		},
		{
			name:     "3-4-5 triangle",
			vector:   []float32{3, 4, 0},
			expected: 25.0,
		},
		{
			name:     "zero vector",
			vector:   []float32{0, 0, 0},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := NewVectorNode(tt.vector)
			result := node.L2NormSquared()
			if !floatEqual(result, tt.expected) {
				t.Errorf("L2NormSquared() = %f, want %f", result, tt.expected)
			}
		})
	}
}

func TestNodeNormalize(t *testing.T) {
	tests := []struct {
		name     string
		vector   []float32
		expected []float32
	}{
		{
			name:     "unit vector",
			vector:   []float32{1, 0, 0},
			expected: []float32{1, 0, 0},
		},
		{
			name:     "3-4 vector",
			vector:   []float32{3, 4, 0},
			expected: []float32{0.6, 0.8, 0},
		},
		{
			name:     "zero vector - should return unchanged",
			vector:   []float32{0, 0, 0},
			expected: []float32{0, 0, 0},
		},
		{
			name:     "negative values",
			vector:   []float32{-3, -4, 0},
			expected: []float32{-0.6, -0.8, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := NewVectorNode(tt.vector)
			originalID := node.ID()
			normalized := node.Normalize()

			// Check ID is preserved
			if normalized.ID() != originalID {
				t.Errorf("Normalize changed ID: got %d, want %d", normalized.ID(), originalID)
			}

			// Check original is unchanged
			origVec := node.Vector()
			for i := range tt.vector {
				if origVec[i] != tt.vector[i] {
					t.Error("Normalize should not modify original node")
				}
			}

			// Check normalized values
			result := normalized.Vector()
			for i := range tt.expected {
				if !floatEqual(result[i], tt.expected[i]) {
					t.Errorf("Normalize result[%d] = %f, want %f", i, result[i], tt.expected[i])
				}
			}

			// Check norm of normalized vector is 1 (unless it was zero)
			if tt.expected[0] != 0 || tt.expected[1] != 0 || tt.expected[2] != 0 {
				normResult := normalized.L2Norm()
				if !floatEqual(normResult, 1.0) {
					t.Errorf("Normalized vector norm = %f, want 1.0", normResult)
				}
			}
		})
	}
}

func TestNodeNormalizeInPlace(t *testing.T) {
	tests := []struct {
		name     string
		vector   []float32
		expected []float32
	}{
		{
			name:     "unit vector",
			vector:   []float32{1, 0, 0},
			expected: []float32{1, 0, 0},
		},
		{
			name:     "3-4 vector",
			vector:   []float32{3, 4, 0},
			expected: []float32{0.6, 0.8, 0},
		},
		{
			name:     "zero vector - should return unchanged",
			vector:   []float32{0, 0, 0},
			expected: []float32{0, 0, 0},
		},
		{
			name:     "negative values",
			vector:   []float32{-3, -4, 0},
			expected: []float32{-0.6, -0.8, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := NewVectorNode(tt.vector)
			originalID := node.ID()
			node.NormalizeInPlace()

			// Check ID is preserved
			if node.ID() != originalID {
				t.Errorf("NormalizeInPlace changed ID: got %d, want %d", node.ID(), originalID)
			}

			// Check normalized values
			result := node.Vector()
			for i := range tt.expected {
				if !floatEqual(result[i], tt.expected[i]) {
					t.Errorf("NormalizeInPlace result[%d] = %f, want %f", i, result[i], tt.expected[i])
				}
			}

			// Check norm is 1 (unless it was zero)
			if tt.expected[0] != 0 || tt.expected[1] != 0 || tt.expected[2] != 0 {
				normResult := node.L2Norm()
				if !floatEqual(normResult, 1.0) {
					t.Errorf("Normalized vector norm = %f, want 1.0", normResult)
				}
			}
		})
	}
}

func TestNodeNormalizeZeroVectorNoNaN(t *testing.T) {
	// This test specifically checks that normalizing a zero vector doesn't produce NaN
	zeroNode := NewVectorNode([]float32{0, 0, 0})

	// Test Normalize
	normalized := zeroNode.Normalize()
	for i, v := range normalized.Vector() {
		if math.IsNaN(float64(v)) {
			t.Errorf("Normalize produced NaN at index %d", i)
		}
	}

	// Test NormalizeInPlace
	zeroNode2 := NewVectorNode([]float32{0, 0, 0})
	zeroNode2.NormalizeInPlace()
	for i, v := range zeroNode2.Vector() {
		if math.IsNaN(float64(v)) {
			t.Errorf("NormalizeInPlace produced NaN at index %d", i)
		}
	}
}

// Helper function to compare floats with tolerance
func floatEqual(a, b float32) bool {
	const epsilon = 1e-6
	return math.Abs(float64(a-b)) < epsilon
}

// Benchmarks

func BenchmarkNewVectorNode(b *testing.B) {
	vector := []float32{1, 2, 3, 4, 5}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewVectorNode(vector)
	}
}

func BenchmarkNodeCopy(b *testing.B) {
	node := NewVectorNode([]float32{1, 2, 3, 4, 5})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node.Copy()
	}
}

func BenchmarkNodeAdd(b *testing.B) {
	node := NewVectorNode([]float32{1, 2, 3, 4, 5})
	add := []float32{1, 1, 1, 1, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node.Add(add)
	}
}

func BenchmarkNodeScale(b *testing.B) {
	node := NewVectorNode([]float32{1, 2, 3, 4, 5})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node.Scale(2.0)
	}
}

func BenchmarkNodeL2Norm(b *testing.B) {
	node := NewVectorNode([]float32{1, 2, 3, 4, 5})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node.L2Norm()
	}
}

func BenchmarkNodeNormalize(b *testing.B) {
	node := NewVectorNode([]float32{1, 2, 3, 4, 5})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node.Normalize()
	}
}

func BenchmarkNodeNormalizeInPlace(b *testing.B) {
	vector := []float32{1, 2, 3, 4, 5}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		node := NewVectorNode(append([]float32{}, vector...))
		b.StartTimer()
		node.NormalizeInPlace()
	}
}
