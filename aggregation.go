package comet

import (
	"fmt"
	"sort"
)

// ScoreAggregationKind defines the type of score aggregation strategy.
type ScoreAggregationKind string

const (
	// SumAggregation sums all scores for the same node
	SumAggregation ScoreAggregationKind = "sum"

	// MaxAggregation takes the maximum score for the same node
	MaxAggregation ScoreAggregationKind = "max"

	// MeanAggregation averages all scores for the same node
	MeanAggregation ScoreAggregationKind = "mean"
)

// ============================================================================
// VECTOR AGGREGATION
// ============================================================================

// VectorAggregation defines how to aggregate scores when the same vector node
// appears in results from multiple queries or node searches.
//
// When performing multi-query or multi-node searches, the same vector may be
// returned multiple times with different scores. The aggregation strategy
// determines how to combine these scores and deduplicate results by node ID.
//
// Note: Vector results use distance metrics where LOWER score = BETTER match.
type VectorAggregation interface {
	// Kind returns the kind of aggregation strategy
	Kind() ScoreAggregationKind

	// Aggregate takes a slice of VectorResults (potentially with duplicate node IDs),
	// deduplicates by node ID, aggregates scores for each unique node,
	// and returns the deduplicated results sorted by aggregated score (ascending).
	Aggregate(results []VectorResult) []VectorResult
}

// Singleton instances for vector aggregation
var (
	vectorSumAgg  *vectorSumAggregation
	vectorMaxAgg  *vectorMaxAggregation
	vectorMeanAgg *vectorMeanAggregation
)

// Singleton instances for text aggregation
var (
	textSumAgg  *textSumAggregation
	textMaxAgg  *textMaxAggregation
	textMeanAgg *textMeanAggregation
)

func init() {
	// Initialize vector aggregation singletons
	vectorSumAgg = &vectorSumAggregation{}
	vectorMaxAgg = &vectorMaxAggregation{}
	vectorMeanAgg = &vectorMeanAggregation{}

	// Initialize text aggregation singletons
	textSumAgg = &textSumAggregation{}
	textMaxAgg = &textMaxAggregation{}
	textMeanAgg = &textMeanAggregation{}
}

// NewVectorAggregation returns the singleton vector aggregation instance for the given kind.
// Returns error if the kind is not recognized.
func NewVectorAggregation(kind ScoreAggregationKind) (VectorAggregation, error) {
	switch kind {
	case SumAggregation:
		return vectorSumAgg, nil
	case MaxAggregation:
		return vectorMaxAgg, nil
	case MeanAggregation:
		return vectorMeanAgg, nil
	default:
		return nil, fmt.Errorf("unknown aggregation kind: %s", kind)
	}
}

// DefaultVectorAggregation returns the default vector aggregation strategy (Sum).
func DefaultVectorAggregation() VectorAggregation {
	return vectorSumAgg
}

// ============================================================================
// VECTOR SUM AGGREGATION
// ============================================================================

// vectorSumAggregation sums all scores for the same vector node.
//
// Use case: When you want to emphasize nodes that appear frequently across
// multiple queries or have consistently high relevance.
//
// Example: If node 42 appears in 3 queries with scores [0.1, 0.2, 0.15],
// the final score will be 0.45.
type vectorSumAggregation struct{}

func (s *vectorSumAggregation) Kind() ScoreAggregationKind {
	return SumAggregation
}

func (s *vectorSumAggregation) Aggregate(results []VectorResult) []VectorResult {
	if len(results) == 0 {
		return results
	}

	// Map to collect scores for each node ID
	nodeScores := make(map[uint32][]float32)
	nodeMap := make(map[uint32]VectorNode)

	// Collect all scores per node ID
	for _, result := range results {
		nodeID := result.Node.ID()
		nodeScores[nodeID] = append(nodeScores[nodeID], result.Score)
		nodeMap[nodeID] = result.Node
	}

	// Aggregate scores by summing
	aggregated := make([]VectorResult, 0, len(nodeScores))
	for nodeID, scores := range nodeScores {
		sum := float32(0)
		for _, score := range scores {
			sum += score
		}
		aggregated = append(aggregated, VectorResult{
			Node:  nodeMap[nodeID],
			Score: sum,
		})
	}

	// Sort by score (ascending - smaller is better)
	sort.Slice(aggregated, func(i, j int) bool {
		return aggregated[i].Score < aggregated[j].Score
	})

	return aggregated
}

// ============================================================================
// VECTOR MAX AGGREGATION
// ============================================================================

// vectorMaxAggregation takes the maximum (worst) score for the same vector node.
//
// Use case: Conservative approach where you want to use the worst-case distance.
// Useful when you want only nodes that are close to ALL queries.
//
// Example: If node 42 appears in 3 queries with scores [0.1, 0.2, 0.15],
// the final score will be 0.2 (highest distance = furthest = worst).
//
// Note: In distance metrics, higher score = worse match (further away).
type vectorMaxAggregation struct{}

func (m *vectorMaxAggregation) Kind() ScoreAggregationKind {
	return MaxAggregation
}

func (m *vectorMaxAggregation) Aggregate(results []VectorResult) []VectorResult {
	if len(results) == 0 {
		return results
	}

	// Map to collect scores for each node ID
	nodeScores := make(map[uint32][]float32)
	nodeMap := make(map[uint32]VectorNode)

	// Collect all scores per node ID
	for _, result := range results {
		nodeID := result.Node.ID()
		nodeScores[nodeID] = append(nodeScores[nodeID], result.Score)
		nodeMap[nodeID] = result.Node
	}

	// Aggregate scores by taking maximum
	aggregated := make([]VectorResult, 0, len(nodeScores))
	for nodeID, scores := range nodeScores {
		max := scores[0]
		for _, score := range scores[1:] {
			if score > max {
				max = score
			}
		}
		aggregated = append(aggregated, VectorResult{
			Node:  nodeMap[nodeID],
			Score: max,
		})
	}

	// Sort by score (ascending - smaller is better)
	sort.Slice(aggregated, func(i, j int) bool {
		return aggregated[i].Score < aggregated[j].Score
	})

	return aggregated
}

// ============================================================================
// VECTOR MEAN AGGREGATION
// ============================================================================

// vectorMeanAggregation averages all scores for the same vector node.
//
// Use case: Balanced approach that considers all query results equally.
// Good for general-purpose multi-query search.
//
// Example: If node 42 appears in 3 queries with scores [0.1, 0.2, 0.15],
// the final score will be 0.15 (average).
type vectorMeanAggregation struct{}

func (a *vectorMeanAggregation) Kind() ScoreAggregationKind {
	return MeanAggregation
}

func (a *vectorMeanAggregation) Aggregate(results []VectorResult) []VectorResult {
	if len(results) == 0 {
		return results
	}

	// Map to collect scores for each node ID
	nodeScores := make(map[uint32][]float32)
	nodeMap := make(map[uint32]VectorNode)

	// Collect all scores per node ID
	for _, result := range results {
		nodeID := result.Node.ID()
		nodeScores[nodeID] = append(nodeScores[nodeID], result.Score)
		nodeMap[nodeID] = result.Node
	}

	// Aggregate scores by averaging
	aggregated := make([]VectorResult, 0, len(nodeScores))
	for nodeID, scores := range nodeScores {
		sum := float32(0)
		for _, score := range scores {
			sum += score
		}
		avg := sum / float32(len(scores))
		aggregated = append(aggregated, VectorResult{
			Node:  nodeMap[nodeID],
			Score: avg,
		})
	}

	// Sort by score (ascending - smaller is better)
	sort.Slice(aggregated, func(i, j int) bool {
		return aggregated[i].Score < aggregated[j].Score
	})

	return aggregated
}

// ============================================================================
// TEXT AGGREGATION
// ============================================================================

// TextAggregation defines how to aggregate scores when the same text document
// appears in results from multiple queries.
//
// When performing multi-query text searches, the same document may be
// returned multiple times with different scores. The aggregation strategy
// determines how to combine these scores and deduplicate results by document ID.
//
// Note: Text results use relevance scores where HIGHER score = BETTER match (e.g., BM25).
type TextAggregation interface {
	// Kind returns the kind of aggregation strategy
	Kind() ScoreAggregationKind

	// Aggregate takes a slice of TextResults (potentially with duplicate document IDs),
	// deduplicates by document ID, aggregates scores for each unique document,
	// and returns the deduplicated results sorted by aggregated score (descending).
	Aggregate(results []TextResult) []TextResult
}

// NewTextAggregation returns the singleton text aggregation instance for the given kind.
// Returns error if the kind is not recognized.
func NewTextAggregation(kind ScoreAggregationKind) (TextAggregation, error) {
	switch kind {
	case SumAggregation:
		return textSumAgg, nil
	case MaxAggregation:
		return textMaxAgg, nil
	case MeanAggregation:
		return textMeanAgg, nil
	default:
		return nil, fmt.Errorf("unknown aggregation kind: %s", kind)
	}
}

// DefaultTextAggregation returns the default text aggregation strategy (Sum).
func DefaultTextAggregation() TextAggregation {
	return textSumAgg
}

// ============================================================================
// TEXT SUM AGGREGATION
// ============================================================================

// textSumAggregation sums all scores for the same document.
//
// Use case: When you want to emphasize documents that appear frequently across
// multiple queries or have consistently high relevance.
//
// Example: If document 42 appears in 3 queries with scores [1.5, 2.0, 1.8],
// the final score will be 5.3.
type textSumAggregation struct{}

func (s *textSumAggregation) Kind() ScoreAggregationKind {
	return SumAggregation
}

func (s *textSumAggregation) Aggregate(results []TextResult) []TextResult {
	if len(results) == 0 {
		return results
	}

	// Map to collect scores for each document ID
	docScores := make(map[uint32]float32)

	for _, result := range results {
		docScores[result.Id] += result.Score
	}

	// Convert back to slice
	aggregated := make([]TextResult, 0, len(docScores))
	for id, score := range docScores {
		aggregated = append(aggregated, TextResult{
			Id:    id,
			Score: score,
		})
	}

	// Sort by score descending (higher score = better for BM25)
	sort.Slice(aggregated, func(i, j int) bool {
		return aggregated[i].Score > aggregated[j].Score
	})

	return aggregated
}

// ============================================================================
// TEXT MAX AGGREGATION
// ============================================================================

// textMaxAggregation takes the maximum (best) score for the same document.
//
// Use case: When you want to use the best-case score across queries.
// Useful when you want documents that match at least one query very well.
//
// Example: If document 42 appears in 3 queries with scores [1.5, 2.0, 1.8],
// the final score will be 2.0 (highest relevance score = best).
//
// Note: In relevance metrics like BM25, higher score = better match.
type textMaxAggregation struct{}

func (m *textMaxAggregation) Kind() ScoreAggregationKind {
	return MaxAggregation
}

func (m *textMaxAggregation) Aggregate(results []TextResult) []TextResult {
	if len(results) == 0 {
		return results
	}

	// Map to collect maximum score for each document ID
	docScores := make(map[uint32]float32)

	for _, result := range results {
		if existing, exists := docScores[result.Id]; !exists || result.Score > existing {
			docScores[result.Id] = result.Score
		}
	}

	// Convert back to slice
	aggregated := make([]TextResult, 0, len(docScores))
	for id, score := range docScores {
		aggregated = append(aggregated, TextResult{
			Id:    id,
			Score: score,
		})
	}

	// Sort by score descending (higher score = better for BM25)
	sort.Slice(aggregated, func(i, j int) bool {
		return aggregated[i].Score > aggregated[j].Score
	})

	return aggregated
}

// ============================================================================
// TEXT MEAN AGGREGATION
// ============================================================================

// textMeanAggregation averages all scores for the same document.
//
// Use case: Balanced approach that considers all query results equally.
// Good for general-purpose multi-query text search.
//
// Example: If document 42 appears in 3 queries with scores [1.5, 2.0, 1.8],
// the final score will be 1.77 (average).
type textMeanAggregation struct{}

func (a *textMeanAggregation) Kind() ScoreAggregationKind {
	return MeanAggregation
}

func (a *textMeanAggregation) Aggregate(results []TextResult) []TextResult {
	if len(results) == 0 {
		return results
	}

	// Map to collect scores and counts for each document ID
	type scoreInfo struct {
		sum   float32
		count int
	}
	docScores := make(map[uint32]*scoreInfo)

	for _, result := range results {
		if _, exists := docScores[result.Id]; !exists {
			docScores[result.Id] = &scoreInfo{}
		}
		docScores[result.Id].sum += result.Score
		docScores[result.Id].count++
	}

	// Convert back to slice with averaged scores
	aggregated := make([]TextResult, 0, len(docScores))
	for id, info := range docScores {
		aggregated = append(aggregated, TextResult{
			Id:    id,
			Score: info.sum / float32(info.count),
		})
	}

	// Sort by score descending (higher score = better for BM25)
	sort.Slice(aggregated, func(i, j int) bool {
		return aggregated[i].Score > aggregated[j].Score
	})

	return aggregated
}
