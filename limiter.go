package comet

// sanitizeK ensures k is within valid bounds [1, maxResults].
//
// If k is <= 0 or exceeds maxResults, it returns maxResults.
// This provides a consistent way to handle k values across all search implementations.
//
// Usage:
//
//	k := sanitizeK(requestedK, len(results))
//	return results[:k]
func sanitizeK(k, maxResults int) int {
	if k <= 0 || k > maxResults {
		return maxResults
	}
	return k
}

// limitResults applies k-limiting to a result slice.
//
// This is a convenience wrapper around sanitizeK that both sanitizes k
// and returns the sliced results in one call.
//
// Usage:
//
//	return limitResults(results, requestedK)
func limitResults(results []VectorResult, k int) []VectorResult {
	k = sanitizeK(k, len(results))
	return results[:k]
}

// autocutResults applies autocut algorithm to determine optimal result cutoff.
//
// It extracts scores from VectorResult slice and uses the Autocut algorithm
// to find the optimal cutoff point based on the score distribution.
//
// Parameters:
//   - results: slice of VectorResult to analyze
//   - cutoff: number of extrema to find before cutting (-1 disables autocut)
//
// Returns the sliced results up to the autocut point. If cutoff is -1, returns
// results unchanged (no-op).
//
// Usage:
//
//	return autocutResults(results, 1)  // Apply autocut with 1 extremum
//	return autocutResults(results, -1) // No-op, returns all results
func autocutResults(results []VectorResult, cutoff int) []VectorResult {
	// No-op if cutoff is -1 or no results
	if cutoff == -1 || len(results) == 0 {
		return results
	}

	// Extract scores from results
	scores := make([]float32, len(results))
	for i, result := range results {
		scores[i] = result.Score
	}

	// Apply autocut algorithm
	cutIndex := Autocut(scores, cutoff)

	return results[:cutIndex]
}

// Autocut determines optimal cutoff point in a score distribution.
//
// It analyzes the normalized difference between actual scores and ideal linear
// distribution to find local maxima (extrema). Returns the index before the
// Nth extremum where N is the cutOff parameter.
//
// Parameters:
//   - yValues: slice of score values (typically distances or similarity scores)
//   - cutOff: number of extrema to encounter before cutting
//
// Returns the index at which to cut the results.
func Autocut(yValues []float32, cutOff int) int {
	if len(yValues) <= 1 {
		return len(yValues)
	}

	diff := make([]float32, len(yValues))
	step := 1. / (float32(len(yValues)) - 1.)

	for i := range yValues {
		xValue := 0. + float32(i)*step
		yValueNorm := (yValues[i] - yValues[0]) / (yValues[len(yValues)-1] - yValues[0])
		diff[i] = yValueNorm - xValue
	}

	extremaCount := 0
	for i := range diff {
		if i == 0 {
			continue // we want the index _before_ the extrema
		}

		if i == len(diff)-1 && len(diff) > 1 { // for last element there is no "next" point
			if diff[i] > diff[i-1] && diff[i] > diff[i-2] {
				extremaCount += 1
				if extremaCount >= cutOff {
					return i
				}
			}
		} else {
			if diff[i] > diff[i-1] && diff[i] > diff[i+1] {
				extremaCount += 1
				if extremaCount >= cutOff {
					return i
				}
			}
		}
	}
	return len(yValues)
}
