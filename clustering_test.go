package comet

import (
	"math"
	"testing"
)

// TestKMeansBasic tests basic k-means clustering functionality
func TestKMeansBasic(t *testing.T) {
	// Create simple 2D vectors that naturally form 2 clusters
	vectors := [][]float32{
		{0.0, 0.0},
		{1.0, 1.0},
		{0.5, 0.5},
		{10.0, 10.0},
		{11.0, 11.0},
		{10.5, 10.5},
	}

	dist, _ := NewDistance(L2Squared)
	centroids, assignments := KMeans(vectors, 2, dist, DefaultMaxIter)

	// Verify we got 2 centroids
	if len(centroids) != 2 {
		t.Errorf("KMeans() returned %d centroids, want 2", len(centroids))
	}

	// Verify all vectors are assigned
	if len(assignments) != len(vectors) {
		t.Errorf("KMeans() returned %d assignments, want %d", len(assignments), len(vectors))
	}

	// Verify assignments are valid cluster IDs
	for i, assignment := range assignments {
		if assignment < 0 || assignment >= 2 {
			t.Errorf("assignments[%d] = %d, want value in range [0,1]", i, assignment)
		}
	}

	// Verify the first 3 vectors are in the same cluster
	if assignments[0] != assignments[1] || assignments[1] != assignments[2] {
		t.Errorf("First 3 vectors should be in same cluster, got assignments: %v, %v, %v",
			assignments[0], assignments[1], assignments[2])
	}

	// Verify the last 3 vectors are in the same cluster
	if assignments[3] != assignments[4] || assignments[4] != assignments[5] {
		t.Errorf("Last 3 vectors should be in same cluster, got assignments: %v, %v, %v",
			assignments[3], assignments[4], assignments[5])
	}

	// Verify the two groups are in different clusters
	if assignments[0] == assignments[3] {
		t.Errorf("First and last groups should be in different clusters")
	}
}

// TestKMeansEmptyVectors tests k-means with empty input
func TestKMeansEmptyVectors(t *testing.T) {
	dist, _ := NewDistance(L2Squared)
	centroids, assignments := KMeans([][]float32{}, 2, dist, DefaultMaxIter)

	if centroids != nil {
		t.Errorf("KMeans() with empty vectors returned non-nil centroids: %v", centroids)
	}

	if assignments != nil {
		t.Errorf("KMeans() with empty vectors returned non-nil assignments: %v", assignments)
	}
}

// TestKMeansInvalidK tests k-means with invalid k values
func TestKMeansInvalidK(t *testing.T) {
	vectors := [][]float32{
		{1.0, 2.0},
		{3.0, 4.0},
	}
	dist, _ := NewDistance(L2Squared)

	tests := []struct {
		name string
		k    int
	}{
		{"zero k", 0},
		{"negative k", -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			centroids, assignments := KMeans(vectors, tt.k, dist, DefaultMaxIter)

			if centroids != nil {
				t.Errorf("KMeans() with k=%d returned non-nil centroids", tt.k)
			}

			if assignments != nil {
				t.Errorf("KMeans() with k=%d returned non-nil assignments", tt.k)
			}
		})
	}
}

// TestKMeansKGreaterThanN tests k-means when k > number of vectors
func TestKMeansKGreaterThanN(t *testing.T) {
	vectors := [][]float32{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
	}
	dist, _ := NewDistance(L2Squared)

	// Request more clusters than vectors
	centroids, assignments := KMeans(vectors, 10, dist, DefaultMaxIter)

	// Should auto-adjust k to number of vectors
	if len(centroids) != len(vectors) {
		t.Errorf("KMeans() with k>n returned %d centroids, want %d", len(centroids), len(vectors))
	}

	if len(assignments) != len(vectors) {
		t.Errorf("KMeans() returned %d assignments, want %d", len(assignments), len(vectors))
	}

	// Each vector should be its own cluster
	uniqueClusters := make(map[int]bool)
	for _, assignment := range assignments {
		uniqueClusters[assignment] = true
	}

	if len(uniqueClusters) != len(vectors) {
		t.Errorf("Expected %d unique clusters, got %d", len(vectors), len(uniqueClusters))
	}
}

// TestKMeansInvalidMaxIter tests k-means with invalid maxIter
func TestKMeansInvalidMaxIter(t *testing.T) {
	vectors := [][]float32{
		{0.0, 0.0},
		{1.0, 1.0},
		{10.0, 10.0},
		{11.0, 11.0},
	}
	dist, _ := NewDistance(L2Squared)

	tests := []struct {
		name    string
		maxIter int
	}{
		{"zero maxIter", 0},
		{"negative maxIter", -5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			centroids, assignments := KMeans(vectors, 2, dist, tt.maxIter)

			// Should use DefaultMaxIter and still work
			if centroids == nil {
				t.Errorf("KMeans() with maxIter=%d returned nil centroids", tt.maxIter)
			}

			if assignments == nil {
				t.Errorf("KMeans() with maxIter=%d returned nil assignments", tt.maxIter)
			}

			if len(centroids) != 2 {
				t.Errorf("KMeans() returned %d centroids, want 2", len(centroids))
			}
		})
	}
}

// TestKMeansSingleVector tests k-means with a single vector
func TestKMeansSingleVector(t *testing.T) {
	vectors := [][]float32{
		{1.0, 2.0, 3.0},
	}
	dist, _ := NewDistance(L2Squared)

	centroids, assignments := KMeans(vectors, 1, dist, DefaultMaxIter)

	if len(centroids) != 1 {
		t.Errorf("KMeans() returned %d centroids, want 1", len(centroids))
	}

	if len(assignments) != 1 {
		t.Errorf("KMeans() returned %d assignments, want 1", len(assignments))
	}

	if assignments[0] != 0 {
		t.Errorf("assignments[0] = %d, want 0", assignments[0])
	}

	// Centroid should be equal to the single vector
	for i := range centroids[0] {
		if centroids[0][i] != vectors[0][i] {
			t.Errorf("centroid[%d] = %f, want %f", i, centroids[0][i], vectors[0][i])
		}
	}
}

// TestKMeansConvergence tests that k-means converges properly
func TestKMeansConvergence(t *testing.T) {
	// Create 3 distinct clusters
	vectors := [][]float32{
		// Cluster 1 around (0, 0)
		{0.0, 0.0},
		{0.1, 0.1},
		{-0.1, -0.1},
		{0.2, -0.1},
		// Cluster 2 around (5, 5)
		{5.0, 5.0},
		{5.1, 5.1},
		{4.9, 4.9},
		{5.2, 5.1},
		// Cluster 3 around (10, 0)
		{10.0, 0.0},
		{10.1, 0.1},
		{9.9, -0.1},
		{10.2, 0.0},
	}

	dist, _ := NewDistance(L2Squared)
	centroids, assignments := KMeans(vectors, 3, dist, 100)

	if len(centroids) != 3 {
		t.Errorf("KMeans() returned %d centroids, want 3", len(centroids))
	}

	// Verify each group is in its own cluster
	// First 4 vectors should be in same cluster
	cluster0 := assignments[0]
	for i := 1; i < 4; i++ {
		if assignments[i] != cluster0 {
			t.Errorf("Vector %d should be in cluster %d, got %d", i, cluster0, assignments[i])
		}
	}

	// Next 4 vectors should be in same cluster
	cluster1 := assignments[4]
	for i := 5; i < 8; i++ {
		if assignments[i] != cluster1 {
			t.Errorf("Vector %d should be in cluster %d, got %d", i, cluster1, assignments[i])
		}
	}

	// Last 4 vectors should be in same cluster
	cluster2 := assignments[8]
	for i := 9; i < 12; i++ {
		if assignments[i] != cluster2 {
			t.Errorf("Vector %d should be in cluster %d, got %d", i, cluster2, assignments[i])
		}
	}

	// All clusters should be different
	if cluster0 == cluster1 || cluster1 == cluster2 || cluster0 == cluster2 {
		t.Errorf("All clusters should be different, got: %d, %d, %d", cluster0, cluster1, cluster2)
	}
}

// TestKMeansCentroidAccuracy tests that centroids are computed correctly
func TestKMeansCentroidAccuracy(t *testing.T) {
	// Create 2 clusters with known means
	vectors := [][]float32{
		{0.0, 0.0},
		{2.0, 2.0}, // Mean of first cluster: (1, 1)
		{10.0, 10.0},
		{12.0, 12.0}, // Mean of second cluster: (11, 11)
	}

	dist, _ := NewDistance(L2Squared)
	centroids, assignments := KMeans(vectors, 2, dist, DefaultMaxIter)

	if len(centroids) != 2 {
		t.Fatalf("KMeans() returned %d centroids, want 2", len(centroids))
	}

	// Determine which cluster is which based on first vector's assignment
	cluster0Idx := assignments[0]
	cluster1Idx := 1 - cluster0Idx

	// Check cluster 0 centroid (should be around (1, 1))
	expectedCentroid0 := []float32{1.0, 1.0}
	for i := range expectedCentroid0 {
		diff := math.Abs(float64(centroids[cluster0Idx][i] - expectedCentroid0[i]))
		if diff > 0.01 {
			t.Errorf("centroid[%d][%d] = %f, want ~%f (diff: %f)",
				cluster0Idx, i, centroids[cluster0Idx][i], expectedCentroid0[i], diff)
		}
	}

	// Check cluster 1 centroid (should be around (11, 11))
	expectedCentroid1 := []float32{11.0, 11.0}
	for i := range expectedCentroid1 {
		diff := math.Abs(float64(centroids[cluster1Idx][i] - expectedCentroid1[i]))
		if diff > 0.01 {
			t.Errorf("centroid[%d][%d] = %f, want ~%f (diff: %f)",
				cluster1Idx, i, centroids[cluster1Idx][i], expectedCentroid1[i], diff)
		}
	}
}

// TestKMeansWithDifferentDistances tests k-means with different distance metrics
func TestKMeansWithDifferentDistances(t *testing.T) {
	vectors := [][]float32{
		{1.0, 0.0},
		{0.9, 0.1},
		{-1.0, 0.0},
		{-0.9, -0.1},
	}

	tests := []struct {
		name         string
		distanceKind DistanceKind
	}{
		{"L2Squared", L2Squared},
		{"Euclidean", Euclidean},
		{"Cosine", Cosine},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dist, err := NewDistance(tt.distanceKind)
			if err != nil {
				t.Fatalf("NewDistance() error: %v", err)
			}

			centroids, assignments := KMeans(vectors, 2, dist, DefaultMaxIter)

			if len(centroids) != 2 {
				t.Errorf("KMeans() returned %d centroids, want 2", len(centroids))
			}

			if len(assignments) != len(vectors) {
				t.Errorf("KMeans() returned %d assignments, want %d", len(assignments), len(vectors))
			}

			// Verify all assignments are valid
			for i, assignment := range assignments {
				if assignment < 0 || assignment >= 2 {
					t.Errorf("assignments[%d] = %d, want value in range [0,1]", i, assignment)
				}
			}
		})
	}
}

// TestKMeansHighDimensional tests k-means with high-dimensional vectors
func TestKMeansHighDimensional(t *testing.T) {
	dim := 128
	numVectors := 20
	k := 4

	// Create vectors
	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float32, dim)
		// Create distinct patterns for different clusters
		clusterID := i % k
		for j := 0; j < dim; j++ {
			vectors[i][j] = float32(clusterID*10 + j%3)
		}
	}

	dist, _ := NewDistance(L2Squared)
	centroids, assignments := KMeans(vectors, k, dist, DefaultMaxIter)

	if len(centroids) != k {
		t.Errorf("KMeans() returned %d centroids, want %d", len(centroids), k)
	}

	if len(assignments) != numVectors {
		t.Errorf("KMeans() returned %d assignments, want %d", len(assignments), numVectors)
	}

	// Verify centroid dimensions
	for i, centroid := range centroids {
		if len(centroid) != dim {
			t.Errorf("centroid[%d] has dimension %d, want %d", i, len(centroid), dim)
		}
	}
}

// TestKMeansSubspace tests the KMeansSubspace function
func TestKMeansSubspace(t *testing.T) {
	// Create simple subspace vectors
	vectors := [][]float32{
		{0.0, 0.0},
		{0.5, 0.5},
		{1.0, 1.0},
		{10.0, 10.0},
		{10.5, 10.5},
		{11.0, 11.0},
	}

	centroids, assignments := KMeansSubspace(vectors, 2, DefaultMaxIter)

	// Verify we got 2 centroids
	if len(centroids) != 2 {
		t.Errorf("KMeansSubspace() returned %d centroids, want 2", len(centroids))
	}

	// Verify all vectors are assigned
	if len(assignments) != len(vectors) {
		t.Errorf("KMeansSubspace() returned %d assignments, want %d", len(assignments), len(vectors))
	}

	// Verify assignments are valid
	for i, assignment := range assignments {
		if assignment < 0 || assignment >= 2 {
			t.Errorf("assignments[%d] = %d, want value in range [0,1]", i, assignment)
		}
	}

	// Verify the first 3 vectors are in the same cluster
	if assignments[0] != assignments[1] || assignments[1] != assignments[2] {
		t.Errorf("First 3 vectors should be in same cluster")
	}

	// Verify the last 3 vectors are in the same cluster
	if assignments[3] != assignments[4] || assignments[4] != assignments[5] {
		t.Errorf("Last 3 vectors should be in same cluster")
	}

	// Verify the two groups are in different clusters
	if assignments[0] == assignments[3] {
		t.Errorf("First and last groups should be in different clusters")
	}
}

// TestKMeansSubspaceEmptyVectors tests KMeansSubspace with empty input
func TestKMeansSubspaceEmptyVectors(t *testing.T) {
	centroids, assignments := KMeansSubspace([][]float32{}, 2, DefaultMaxIter)

	if centroids != nil {
		t.Errorf("KMeansSubspace() with empty vectors returned non-nil centroids")
	}

	if assignments != nil {
		t.Errorf("KMeansSubspace() with empty vectors returned non-nil assignments")
	}
}

// TestKMeansSubspaceTypicalCodebookSize tests with typical codebook size (256)
func TestKMeansSubspaceTypicalCodebookSize(t *testing.T) {
	// Simulate typical Product Quantization scenario
	subspaceDim := 96 // 768/8
	numVectors := 1000
	k := 256 // 2^8 bits

	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float32, subspaceDim)
		for j := 0; j < subspaceDim; j++ {
			vectors[i][j] = float32((i*j)%100) * 0.1
		}
	}

	centroids, assignments := KMeansSubspace(vectors, k, 10)

	if len(centroids) != k {
		t.Errorf("KMeansSubspace() returned %d centroids, want %d", len(centroids), k)
	}

	if len(assignments) != numVectors {
		t.Errorf("KMeansSubspace() returned %d assignments, want %d", len(assignments), numVectors)
	}

	// Verify all centroids have correct dimension
	for i, centroid := range centroids {
		if len(centroid) != subspaceDim {
			t.Errorf("centroid[%d] has dimension %d, want %d", i, len(centroid), subspaceDim)
		}
	}

	// Verify assignments are in valid range
	for i, assignment := range assignments {
		if assignment < 0 || assignment >= k {
			t.Errorf("assignments[%d] = %d, want value in range [0,%d)", i, assignment, k)
		}
	}
}

// TestKMeansMaxIterLimit tests that k-means respects maxIter limit
func TestKMeansMaxIterLimit(t *testing.T) {
	// Create vectors that would take many iterations to converge
	vectors := [][]float32{
		{0.0, 0.0},
		{1.0, 0.0},
		{2.0, 0.0},
		{3.0, 0.0},
		{4.0, 0.0},
		{5.0, 0.0},
		{6.0, 0.0},
		{7.0, 0.0},
		{8.0, 0.0},
		{9.0, 0.0},
	}

	dist, _ := NewDistance(L2Squared)

	// With only 1 iteration, may not fully converge but should still return valid results
	centroids, assignments := KMeans(vectors, 3, dist, 1)

	if len(centroids) != 3 {
		t.Errorf("KMeans() returned %d centroids, want 3", len(centroids))
	}

	if len(assignments) != len(vectors) {
		t.Errorf("KMeans() returned %d assignments, want %d", len(assignments), len(vectors))
	}

	// Verify all assignments are valid even with limited iterations
	for i, assignment := range assignments {
		if assignment < 0 || assignment >= 3 {
			t.Errorf("assignments[%d] = %d, want value in range [0,2]", i, assignment)
		}
	}
}

// TestKMeansIdenticalVectors tests k-means with all identical vectors
func TestKMeansIdenticalVectors(t *testing.T) {
	vectors := [][]float32{
		{5.0, 5.0},
		{5.0, 5.0},
		{5.0, 5.0},
		{5.0, 5.0},
	}

	dist, _ := NewDistance(L2Squared)
	centroids, assignments := KMeans(vectors, 2, dist, DefaultMaxIter)

	if len(centroids) != 2 {
		t.Errorf("KMeans() returned %d centroids, want 2", len(centroids))
	}

	if len(assignments) != len(vectors) {
		t.Errorf("KMeans() returned %d assignments, want %d", len(assignments), len(vectors))
	}

	// All centroids should be close to (5, 5)
	for i, centroid := range centroids {
		for j, val := range centroid {
			diff := math.Abs(float64(val - 5.0))
			if diff > 0.01 {
				t.Errorf("centroid[%d][%d] = %f, want ~5.0 (diff: %f)", i, j, val, diff)
			}
		}
	}
}

// TestKMeansDimensions tests k-means with various dimensions
func TestKMeansDimensions(t *testing.T) {
	dimensions := []int{1, 2, 3, 10, 64, 128}

	for _, dim := range dimensions {
		t.Run(string(rune(dim)), func(t *testing.T) {
			// Create simple test vectors
			vectors := make([][]float32, 6)
			for i := 0; i < 6; i++ {
				vectors[i] = make([]float32, dim)
				// Create two distinct patterns
				val := float32(0.0)
				if i >= 3 {
					val = 10.0
				}
				for j := 0; j < dim; j++ {
					vectors[i][j] = val + float32(j)*0.1
				}
			}

			dist, _ := NewDistance(L2Squared)
			centroids, assignments := KMeans(vectors, 2, dist, DefaultMaxIter)

			if len(centroids) != 2 {
				t.Errorf("KMeans() returned %d centroids, want 2", len(centroids))
			}

			// Verify centroid dimensions
			for i, centroid := range centroids {
				if len(centroid) != dim {
					t.Errorf("centroid[%d] has dimension %d, want %d", i, len(centroid), dim)
				}
			}

			// Verify clustering worked
			if assignments[0] == assignments[3] {
				t.Errorf("First and fourth vectors should be in different clusters")
			}
		})
	}
}

// TestKMeansAssignmentConsistency tests that assignments match nearest centroids
func TestKMeansAssignmentConsistency(t *testing.T) {
	vectors := [][]float32{
		{0.0, 0.0},
		{1.0, 1.0},
		{10.0, 10.0},
		{11.0, 11.0},
	}

	dist, _ := NewDistance(L2Squared)
	centroids, assignments := KMeans(vectors, 2, dist, DefaultMaxIter)

	// Verify each vector is assigned to its nearest centroid
	for i, vector := range vectors {
		nearestDist := float32(math.Inf(1))
		nearestCluster := -1

		for j, centroid := range centroids {
			d := dist.Calculate(vector, centroid)
			if d < nearestDist {
				nearestDist = d
				nearestCluster = j
			}
		}

		if assignments[i] != nearestCluster {
			t.Errorf("Vector %d assigned to cluster %d, but nearest is cluster %d",
				i, assignments[i], nearestCluster)
		}
	}
}

// TestDefaultMaxIter tests that DefaultMaxIter is reasonable
func TestDefaultMaxIter(t *testing.T) {
	if DefaultMaxIter <= 0 {
		t.Errorf("DefaultMaxIter = %d, want positive value", DefaultMaxIter)
	}

	if DefaultMaxIter < 10 {
		t.Errorf("DefaultMaxIter = %d, might be too small for convergence", DefaultMaxIter)
	}

	if DefaultMaxIter > 1000 {
		t.Errorf("DefaultMaxIter = %d, might be too large (slow)", DefaultMaxIter)
	}
}

// TestUnassignedCluster tests the UnassignedCluster constant
func TestUnassignedCluster(t *testing.T) {
	if UnassignedCluster != -1 {
		t.Errorf("UnassignedCluster = %d, want -1", UnassignedCluster)
	}
}
