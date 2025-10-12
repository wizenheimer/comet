package comet

import (
	"math"
)

const (
	// UnassignedCluster indicates a vector hasn't been assigned to any cluster yet
	UnassignedCluster = -1
)

var (
	// DefaultMaxIter is the default maximum number of iterations for k-means clustering.
	DefaultMaxIter = 20
)

// KMeans performs k-means clustering to learn cluster centroids.
//
// # K-MEANS CLUSTERING ALGORITHM
//
// K-means partitions data into k clusters by iteratively refining cluster assignments
// and centroid positions. Each cluster is represented by its centroid (center point).
//
// Algorithm Steps:
//  1. INITIALIZATION: Select k initial centroid positions (uniform spacing)
//  2. ASSIGNMENT: Assign each vector to its nearest centroid
//  3. UPDATE: Recompute centroids as the mean of assigned vectors
//  4. REPEAT: Steps 2-3 until convergence or max iterations
//
// CONVERGENCE:
// The algorithm converges when assignments stop changing, meaning all vectors are
// assigned to their nearest centroid and centroids are at the center of their clusters.
// Typical convergence: 5-20 iterations for most datasets.
//
// VORONOI PARTITIONS:
// K-means naturally creates Voronoi partitions where each cluster forms a Voronoi cell.
// Vectors in each cluster are guaranteed to be closer to their centroid than to any
// other centroid.
//
// TIME COMPLEXITY:
// O(iterations × k × n × dim) where:
//   - iterations: number of iterations until convergence (typically 10-20)
//   - k: number of clusters
//   - n: number of training vectors
//   - dim: dimensionality
//
// For 100K training vectors, 768 dims, k=316, 10 iterations:
//   - About 243 billion floating point operations
//   - Takes seconds to minutes depending on hardware
//
// Parameters:
//   - vectors: Training vectors to cluster (each is []float32)
//   - k: Number of clusters to create
//   - distance: Distance metric to use for computing distances
//   - maxIter: Maximum number of k-means iterations (typical: 20-100)
//
// Returns:
//   - [][]float32: k learned centroids that define the clusters
//   - []int: cluster assignments for each input vector (vector i -> cluster assignments[i])
func KMeans(vectors [][]float32, k int, distance Distance, maxIter int) (centroids [][]float32, vectorToClusterMapping []int) {
	return kmeansInternal(vectors, k, distance, maxIter)
}

// KMeansSubspace performs k-means clustering on subspace vectors for codebook learning.
//
// # K-MEANS FOR SUBSPACE CLUSTERING
//
// This function learns k centroids that best represent the distribution of vectors
// in a lower-dimensional subspace. Commonly used for Product Quantization (PQ)
// codebook learning and other subspace quantization methods.
//
// SUBSPACE CLUSTERING ADVANTAGES:
//   - Faster: fewer dimensions to process
//   - More effective: less curse of dimensionality
//   - Parallelizable: independent across subspaces
//
// ALGORITHM:
//  1. INITIALIZATION: Choose k initial centroids using uniform sampling
//  2. ASSIGNMENT: Assign each subvector to nearest centroid (using squared L2 distance)
//  3. UPDATE: Recompute centroids as mean of assigned subvectors
//  4. REPEAT: Steps 2-3 until convergence or max iterations
//
// CONVERGENCE:
// Converges when no assignments change. Typically takes 3-10 iterations for most
// subspaces due to lower dimensionality providing more stability.
//
// DISTANCE METRIC:
// Uses squared L2 distance (Euclidean) - standard for:
//   - Product Quantization
//   - Vector quantization
//   - Codebook learning
//
// TIME COMPLEXITY:
// O(iterations × k × n × dsub) where:
//   - iterations: number of k-means iterations (typically 5-20)
//   - k: number of centroids (often 256 = 2^8)
//   - n: number of training subvectors
//   - dsub: subspace dimension (e.g., 768/8 = 96)
//
// For 100K training vectors, dsub=96, k=256, 10 iterations:
//   - About 2.5 billion floating point operations per subspace
//   - Takes milliseconds to seconds per subspace
//
// Parameters:
//   - vectors: Subspace vectors to cluster (all from same subspace)
//   - k: Number of centroids to learn (typically 256 = 2^8 bits)
//   - maxIter: Maximum k-means iterations (typical: 20-50)
//
// Returns:
//   - [][]float32: k learned centroids for this subspace (the codebook)
//   - []int: cluster assignments for each input subvector
func KMeansSubspace(vectors [][]float32, k int, maxIter int) (centroids [][]float32, vectorToClusterMapping []int) {
	dist, _ := NewDistance(L2Squared)
	return kmeansInternal(vectors, k, dist, maxIter)
}

// kmeansInternal is the shared implementation for both KMeans and KMeansSubspace.
// This eliminates code duplication and ensures consistent behavior.
func kmeansInternal(vectors [][]float32, k int, distance Distance, maxIter int) (centroids [][]float32, vectorToClusterMapping []int) {
	// ═══════════════════════════════════════════════════════════════════════════
	// INPUT VALIDATION
	// ═══════════════════════════════════════════════════════════════════════════
	if len(vectors) == 0 {
		return nil, nil
	}

	if k <= 0 {
		return nil, nil
	}

	// Auto-adjust k if it's larger than the number of vectors
	if k > len(vectors) {
		k = len(vectors)
	}

	// Set default maxIter if invalid
	if maxIter <= 0 {
		maxIter = DefaultMaxIter
	}

	dimensions := len(vectors[0])

	// ═══════════════════════════════════════════════════════════════════════════
	// STEP 1: INITIALIZE CENTROIDS
	// ═══════════════════════════════════════════════════════════════════════════
	// Use uniform spacing: pick every (n/k)-th vector as initial centroid
	centroids = make([][]float32, k)
	samplingStep := len(vectors) / k
	if samplingStep == 0 {
		samplingStep = 1
	}

	for clusterIdx := 0; clusterIdx < k; clusterIdx++ {
		vectorIdx := clusterIdx * samplingStep
		if vectorIdx >= len(vectors) {
			vectorIdx = len(vectors) - 1
		}

		// Copy the vector data (don't modify original)
		centroids[clusterIdx] = make([]float32, dimensions)
		copy(centroids[clusterIdx], vectors[vectorIdx])
	}

	// Initialize mapping: vectorToClusterMapping[i] = which cluster vector i belongs to
	// Start with UnassignedCluster to indicate unassigned
	vectorToClusterMapping = make([]int, len(vectors))
	for i := range vectorToClusterMapping {
		vectorToClusterMapping[i] = UnassignedCluster
	}

	// ═══════════════════════════════════════════════════════════════════════════
	// STEP 2-4: ITERATE UNTIL CONVERGENCE
	// ═══════════════════════════════════════════════════════════════════════════
	for iteration := 0; iteration < maxIter; iteration++ {
		// ───────────────────────────────────────────────────────────────────────
		// ASSIGNMENT STEP: Assign each vector to its nearest centroid
		// ───────────────────────────────────────────────────────────────────────
		assignmentsChanged := false

		for vectorIdx, vector := range vectors {
			// Find nearest centroid for this vector
			nearestDistance := float32(math.Inf(1))
			nearestCluster := 0

			for clusterIdx, centroid := range centroids {
				dist := distance.Calculate(vector, centroid)
				if dist < nearestDistance {
					nearestDistance = dist
					nearestCluster = clusterIdx
				}
			}

			// Check if assignment changed from previous iteration
			if vectorToClusterMapping[vectorIdx] != nearestCluster {
				assignmentsChanged = true
				vectorToClusterMapping[vectorIdx] = nearestCluster
			}
		}

		// ───────────────────────────────────────────────────────────────────────
		// CONVERGENCE CHECK: If no assignments changed, we're done!
		// ───────────────────────────────────────────────────────────────────────
		if !assignmentsChanged {
			break // Converged - clustering is stable
		}

		// ───────────────────────────────────────────────────────────────────────
		// UPDATE STEP: Recompute centroids as mean of assigned vectors
		// OPTIMIZED: Single pass through vectors instead of k passes
		// ───────────────────────────────────────────────────────────────────────

		// Initialize accumulators
		clusterSums := make([][]float32, k)
		clusterSizes := make([]int, k)
		for i := range clusterSums {
			clusterSums[i] = make([]float32, dimensions)
		}

		// Single pass: accumulate sums for each cluster - O(n × dim)
		for vectorIdx, assignedCluster := range vectorToClusterMapping {
			if assignedCluster != UnassignedCluster {
				for dimIdx := range vectors[vectorIdx] {
					clusterSums[assignedCluster][dimIdx] += vectors[vectorIdx][dimIdx]
				}
				clusterSizes[assignedCluster]++
			}
		}

		// Compute means: centroid = sum / count - O(k × dim)
		for clusterIdx := range centroids {
			if clusterSizes[clusterIdx] > 0 {
				for dimIdx := range centroids[clusterIdx] {
					centroids[clusterIdx][dimIdx] = clusterSums[clusterIdx][dimIdx] / float32(clusterSizes[clusterIdx])
				}
			}
			// Note: If clusterSize==0 (empty cluster), we keep the old centroid position
			// This is rare but can happen. The centroid will potentially attract vectors
			// in the next iteration if it's positioned between other clusters.
		}
	}

	return centroids, vectorToClusterMapping
}
