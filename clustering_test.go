package comet

import (
	"math"
	"testing"
)

func TestKMeans(t *testing.T) {
	// Create simple 2D test data with clear clusters
	vectors := [][]float32{
		// Cluster 1: around (0, 0)
		{0, 0},
		{0.1, 0.1},
		{-0.1, 0.1},
		{0.1, -0.1},
		// Cluster 2: around (5, 5)
		{5, 5},
		{5.1, 5.1},
		{4.9, 5.1},
		{5.1, 4.9},
		// Cluster 3: around (10, 0)
		{10, 0},
		{10.1, 0.1},
		{9.9, 0.1},
		{10.1, -0.1},
	}

	dist, _ := NewDistance(Euclidean)
	centroids, assignments := KMeans(vectors, 3, dist, 20)

	// Check that we got 3 centroids
	if len(centroids) != 3 {
		t.Errorf("Expected 3 centroids, got %d", len(centroids))
	}

	// Check that all vectors have assignments
	if len(assignments) != len(vectors) {
		t.Errorf("Expected %d assignments, got %d", len(vectors), len(assignments))
	}

	// Check that assignments are valid (0, 1, or 2)
	for i, a := range assignments {
		if a < 0 || a >= 3 {
			t.Errorf("Invalid assignment %d for vector %d", a, i)
		}
	}

	// Verify that vectors in the same cluster have the same assignment
	// Vectors 0-3 should be in the same cluster
	cluster1 := assignments[0]
	for i := 1; i < 4; i++ {
		if assignments[i] != cluster1 {
			t.Errorf("Vectors 0-%d should be in the same cluster", i)
		}
	}

	// Vectors 4-7 should be in the same cluster
	cluster2 := assignments[4]
	for i := 5; i < 8; i++ {
		if assignments[i] != cluster2 {
			t.Errorf("Vectors 4-%d should be in the same cluster", i)
		}
	}

	// Vectors 8-11 should be in the same cluster
	cluster3 := assignments[8]
	for i := 9; i < 12; i++ {
		if assignments[i] != cluster3 {
			t.Errorf("Vectors 8-%d should be in the same cluster", i)
		}
	}

	// Check that the three clusters are different
	if cluster1 == cluster2 || cluster1 == cluster3 || cluster2 == cluster3 {
		t.Error("The three clusters should have different assignments")
	}
}

func TestKMeansWithCosineDistance(t *testing.T) {
	// Create vectors that are similar in direction but different magnitudes
	vectors := [][]float32{
		// Cluster 1: direction (1, 0)
		{1, 0},
		{2, 0.1},
		{3, -0.1},
		// Cluster 2: direction (0, 1)
		{0, 1},
		{0.1, 2},
		{-0.1, 3},
	}

	dist, _ := NewDistance(Cosine)
	centroids, assignments := KMeans(vectors, 2, dist, 20)

	// Check that we got 2 centroids
	if len(centroids) != 2 {
		t.Errorf("Expected 2 centroids, got %d", len(centroids))
	}

	// Vectors 0-2 should be in the same cluster (similar direction)
	cluster1 := assignments[0]
	for i := 1; i < 3; i++ {
		if assignments[i] != cluster1 {
			t.Errorf("Vectors 0-%d should be in the same cluster (similar direction)", i)
		}
	}

	// Vectors 3-5 should be in the same cluster
	cluster2 := assignments[3]
	for i := 4; i < 6; i++ {
		if assignments[i] != cluster2 {
			t.Errorf("Vectors 3-%d should be in the same cluster", i)
		}
	}

	// The two clusters should be different
	if cluster1 == cluster2 {
		t.Error("The two clusters should have different assignments")
	}
}

func TestKMeansSingleCluster(t *testing.T) {
	vectors := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}

	dist, _ := NewDistance(Euclidean)
	centroids, assignments := KMeans(vectors, 1, dist, 20)

	// Check that we got 1 centroid
	if len(centroids) != 1 {
		t.Errorf("Expected 1 centroid, got %d", len(centroids))
	}

	// All vectors should be assigned to cluster 0
	for i, a := range assignments {
		if a != 0 {
			t.Errorf("Vector %d should be assigned to cluster 0, got %d", i, a)
		}
	}

	// Centroid should be approximately the mean of all vectors
	expectedCentroid := []float32{4, 5, 6}
	for d := 0; d < 3; d++ {
		if !almostEqual(centroids[0][d], expectedCentroid[d]) {
			t.Errorf("Centroid[%d] = %f, want %f", d, centroids[0][d], expectedCentroid[d])
		}
	}
}

func TestKMeansIdenticalVectors(t *testing.T) {
	// All vectors are identical
	vectors := [][]float32{
		{1, 2, 3},
		{1, 2, 3},
		{1, 2, 3},
		{1, 2, 3},
	}

	dist, _ := NewDistance(Euclidean)
	centroids, _ := KMeans(vectors, 2, dist, 20)

	// Check that we got 2 centroids
	if len(centroids) != 2 {
		t.Errorf("Expected 2 centroids, got %d", len(centroids))
	}

	// All centroids should converge to the same location since all vectors are identical
	expected := []float32{1, 2, 3}
	for i := range centroids {
		for d := 0; d < 3; d++ {
			if !almostEqual(centroids[i][d], expected[d]) {
				t.Errorf("Centroid %d dim %d = %f, want %f", i, d, centroids[i][d], expected[d])
			}
		}
	}
}

func TestKMeansEmptyInput(t *testing.T) {
	vectors := [][]float32{}
	dist, _ := NewDistance(Euclidean)
	centroids, assignments := KMeans(vectors, 3, dist, 20)

	if centroids != nil {
		t.Error("Expected nil centroids for empty input")
	}
	if assignments != nil {
		t.Error("Expected nil assignments for empty input")
	}
}

func TestKMeansInvalidK(t *testing.T) {
	vectors := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
	}

	dist, _ := NewDistance(Euclidean)

	// Test k = 0
	centroids, assignments := KMeans(vectors, 0, dist, 20)
	if centroids != nil || assignments != nil {
		t.Error("Expected nil for k=0")
	}

	// Test k > len(vectors)
	centroids, assignments = KMeans(vectors, 5, dist, 20)
	if centroids != nil || assignments != nil {
		t.Error("Expected nil for k > len(vectors)")
	}
}

func TestKMeansConvergence(t *testing.T) {
	// Create data where k-means should converge quickly
	vectors := [][]float32{
		{0, 0},
		{0, 1},
		{1, 0},
		{10, 10},
		{10, 11},
		{11, 10},
	}

	dist, _ := NewDistance(Euclidean)

	// Run with limited iterations
	centroids1, assignments1 := KMeans(vectors, 2, dist, 5)

	// Run with more iterations
	centroids2, assignments2 := KMeans(vectors, 2, dist, 100)

	// Assignments should be the same (converged within 5 iterations)
	for i := range assignments1 {
		if assignments1[i] != assignments2[i] {
			t.Errorf("Assignments differ at index %d after more iterations", i)
		}
	}

	// Centroids should be approximately the same
	for i := range centroids1 {
		for d := range centroids1[i] {
			if !almostEqual(centroids1[i][d], centroids2[i][d]) {
				t.Errorf("Centroids differ after more iterations")
				return
			}
		}
	}
}

func TestKMeansSubspace(t *testing.T) {
	// Create simple subspace vectors with clear clusters
	vectors := [][]float32{
		// Cluster 1: around (0, 0)
		{0, 0},
		{0.1, 0.1},
		{-0.1, 0.1},
		// Cluster 2: around (5, 5)
		{5, 5},
		{5.1, 5.1},
		{4.9, 5.1},
	}

	centroids, assignments := KMeansSubspace(vectors, 2, 20)

	// Check that we got 2 centroids
	if len(centroids) != 2 {
		t.Errorf("Expected 2 centroids, got %d", len(centroids))
	}

	// Check that all vectors have assignments
	if len(assignments) != len(vectors) {
		t.Errorf("Expected %d assignments, got %d", len(vectors), len(assignments))
	}

	// Vectors 0-2 should be in the same cluster
	cluster1 := assignments[0]
	for i := 1; i < 3; i++ {
		if assignments[i] != cluster1 {
			t.Errorf("Vectors 0-%d should be in the same cluster", i)
		}
	}

	// Vectors 3-5 should be in the same cluster
	cluster2 := assignments[3]
	for i := 4; i < 6; i++ {
		if assignments[i] != cluster2 {
			t.Errorf("Vectors 3-%d should be in the same cluster", i)
		}
	}

	// The two clusters should be different
	if cluster1 == cluster2 {
		t.Error("The two clusters should have different assignments")
	}

	// Check centroid positions are reasonable
	// One centroid should be near (0, 0) and another near (5, 5)
	foundCluster1 := false
	foundCluster2 := false

	for _, c := range centroids {
		// Check if centroid is near (0, 0) - within 0.5 in both dimensions
		if math.Abs(float64(c[0])) < 0.5 && math.Abs(float64(c[1])) < 0.5 {
			foundCluster1 = true
		}
		// Check if centroid is near (5, 5) - within 0.5 in both dimensions
		if math.Abs(float64(c[0]-5)) < 0.5 && math.Abs(float64(c[1]-5)) < 0.5 {
			foundCluster2 = true
		}
	}

	if !foundCluster1 || !foundCluster2 {
		t.Errorf("Centroids are not in expected positions. Got: %v and %v", centroids[0], centroids[1])
	}
}

func TestKMeansSubspaceHighDimensional(t *testing.T) {
	// Test with higher dimensional subspace vectors
	dim := 32
	vectors := make([][]float32, 100)

	// Create 100 vectors: 50 near origin, 50 near (10, 10, ..., 10)
	for i := 0; i < 50; i++ {
		vectors[i] = make([]float32, dim)
		for d := 0; d < dim; d++ {
			vectors[i][d] = float32(i%3) * 0.1 // Small random variation
		}
	}

	for i := 50; i < 100; i++ {
		vectors[i] = make([]float32, dim)
		for d := 0; d < dim; d++ {
			vectors[i][d] = 10.0 + float32(i%3)*0.1
		}
	}

	centroids, assignments := KMeansSubspace(vectors, 2, 50)

	// Check that we got 2 centroids
	if len(centroids) != 2 {
		t.Errorf("Expected 2 centroids, got %d", len(centroids))
	}

	// Check that vectors 0-49 are mostly in one cluster
	cluster1Count := 0
	firstCluster := assignments[0]
	for i := 0; i < 50; i++ {
		if assignments[i] == firstCluster {
			cluster1Count++
		}
	}

	// At least 80% should be in the same cluster
	if cluster1Count < 40 {
		t.Errorf("Expected most of first 50 vectors in same cluster, got %d", cluster1Count)
	}

	// Check that vectors 50-99 are mostly in the other cluster
	cluster2Count := 0
	secondCluster := assignments[50]
	for i := 50; i < 100; i++ {
		if assignments[i] == secondCluster {
			cluster2Count++
		}
	}

	if cluster2Count < 40 {
		t.Errorf("Expected most of last 50 vectors in same cluster, got %d", cluster2Count)
	}

	// The two dominant clusters should be different
	if firstCluster == secondCluster {
		t.Error("Expected two different clusters")
	}
}

func TestKMeansSubspaceEmptyInput(t *testing.T) {
	vectors := [][]float32{}
	centroids, assignments := KMeansSubspace(vectors, 3, 20)

	if centroids != nil {
		t.Error("Expected nil centroids for empty input")
	}
	if assignments != nil {
		t.Error("Expected nil assignments for empty input")
	}
}

func TestKMeansSubspaceInvalidK(t *testing.T) {
	vectors := [][]float32{
		{1, 2},
		{3, 4},
	}

	// Test k = 0
	centroids, assignments := KMeansSubspace(vectors, 0, 20)
	if centroids != nil || assignments != nil {
		t.Error("Expected nil for k=0")
	}

	// Test k > len(vectors)
	centroids, assignments = KMeansSubspace(vectors, 5, 20)
	if centroids != nil || assignments != nil {
		t.Error("Expected nil for k > len(vectors)")
	}
}

func TestKMeansSubspaceSingleCluster(t *testing.T) {
	vectors := [][]float32{
		{1, 2},
		{3, 4},
		{5, 6},
	}

	centroids, assignments := KMeansSubspace(vectors, 1, 20)

	// Check that we got 1 centroid
	if len(centroids) != 1 {
		t.Errorf("Expected 1 centroid, got %d", len(centroids))
	}

	// All vectors should be assigned to cluster 0
	for i, a := range assignments {
		if a != 0 {
			t.Errorf("Vector %d should be assigned to cluster 0, got %d", i, a)
		}
	}

	// Centroid should be approximately the mean
	expectedCentroid := []float32{3, 4}
	for d := 0; d < 2; d++ {
		if !almostEqual(centroids[0][d], expectedCentroid[d]) {
			t.Errorf("Centroid[%d] = %f, want %f", d, centroids[0][d], expectedCentroid[d])
		}
	}
}

// Benchmark tests
func BenchmarkKMeans(b *testing.B) {
	// Create realistic sized data: 1000 vectors, 128 dimensions, 10 clusters
	vectors := make([][]float32, 1000)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		for d := range vectors[i] {
			vectors[i][d] = float32(i%10) * 10.0
		}
	}

	dist, _ := NewDistance(Euclidean)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		KMeans(vectors, 10, dist, 20)
	}
}

func BenchmarkKMeansSubspace(b *testing.B) {
	// Create realistic sized subspace data: 1000 vectors, 96 dimensions (768/8), 256 clusters
	vectors := make([][]float32, 1000)
	for i := range vectors {
		vectors[i] = make([]float32, 96)
		for d := range vectors[i] {
			vectors[i][d] = float32(i%256) * 0.1
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		KMeansSubspace(vectors, 256, 20)
	}
}

func BenchmarkKMeansLargeK(b *testing.B) {
	// Test with large k value (realistic for IVF)
	vectors := make([][]float32, 10000)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		for d := range vectors[i] {
			vectors[i][d] = float32(i%100) * 0.5
		}
	}

	dist, _ := NewDistance(Euclidean)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		KMeans(vectors, 100, dist, 20)
	}
}
