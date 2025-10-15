package comet

import "sort"

// mergeResults merges search results from multiple sources.
// Keeps the highest score for each document ID (deduplication).
//
// Parameters:
//   - results: Search results from multiple indexes
//
// Returns:
//   - []HybridSearchResult: Merged and deduplicated results
func mergeResults(results []HybridSearchResult) []HybridSearchResult {
	if len(results) == 0 {
		return nil
	}

	// Use map to deduplicate and keep highest score per doc
	scoreMap := make(map[uint32]float64)

	for _, result := range results {
		existingScore, exists := scoreMap[result.ID]
		if !exists || result.Score > existingScore {
			scoreMap[result.ID] = result.Score
		}
	}

	// Convert map to slice
	merged := make([]HybridSearchResult, 0, len(scoreMap))
	for id, score := range scoreMap {
		merged = append(merged, HybridSearchResult{
			ID:    id,
			Score: score,
		})
	}

	return merged
}

// sortResultsByScore sorts results by score in descending order.
// Modifies the input slice in place.
func sortResultsByScore(results []HybridSearchResult) {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
}
