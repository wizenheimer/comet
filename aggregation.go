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

	// MaxAggregation takes the maximum (worst) score for the same node
	MaxAggregation ScoreAggregationKind = "max"

	// MeanAggregation averages all scores for the same node
	MeanAggregation ScoreAggregationKind = "mean"
)

// ScoreAggregation defines how to aggregate scores when the same node
// appears in results from multiple queries or node searches.
//
// When performing multi-query or multi-node searches, the same vector may be
// returned multiple times with different scores. The aggregation strategy
// determines how to combine these scores and deduplicate results by node ID.
type ScoreAggregation interface {
	// Kind returns the kind of aggregation strategy
	Kind() ScoreAggregationKind

	// Aggregate takes a slice of VectorResults (potentially with duplicate node IDs),
	// deduplicates by node ID, aggregates scores for each unique node,
	// and returns the deduplicated results sorted by aggregated score.
	Aggregate(results []VectorResult) []VectorResult
}

// Singleton instances
var (
	sumAgg  *sumAggregation
	maxAgg  *maxAggregation
	meanAgg *meanAggregation
)

func init() {
	sumAgg = &sumAggregation{}
	maxAgg = &maxAggregation{}
	meanAgg = &meanAggregation{}
}

// NewAggregation returns the singleton aggregation instance for the given kind.
// Returns error if the kind is not recognized.
func NewAggregation(kind ScoreAggregationKind) (ScoreAggregation, error) {
	switch kind {
	case SumAggregation:
		return sumAgg, nil
	case MaxAggregation:
		return maxAgg, nil
	case MeanAggregation:
		return meanAgg, nil
	default:
		return nil, fmt.Errorf("unknown aggregation kind: %s", kind)
	}
}

// GetScoreAggregation is an alias for NewAggregation for backward compatibility.
func GetScoreAggregation(kind ScoreAggregationKind) (ScoreAggregation, error) {
	return NewAggregation(kind)
}

// DefaultScoreAggregation returns the default aggregation strategy (Sum).
func DefaultScoreAggregation() ScoreAggregation {
	return sumAgg
}

// ============================================================================
// SUM AGGREGATION
// ============================================================================

// sumAggregation sums all scores for the same node.
//
// Use case: When you want to emphasize nodes that appear frequently across
// multiple queries or have consistently high relevance.
//
// Example: If node 42 appears in 3 queries with scores [0.1, 0.2, 0.15],
// the final score will be 0.45.
type sumAggregation struct{}

func (s *sumAggregation) Kind() ScoreAggregationKind {
	return SumAggregation
}

func (s *sumAggregation) Aggregate(results []VectorResult) []VectorResult {
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
// MAX AGGREGATION
// ============================================================================

// maxAggregation takes the maximum (worst) score for the same node.
//
// Use case: Conservative approach where you want to use the worst-case distance.
// Useful when you want only nodes that are close to ALL queries.
//
// Example: If node 42 appears in 3 queries with scores [0.1, 0.2, 0.15],
// the final score will be 0.2 (highest distance = furthest = worst).
//
// Note: In distance metrics, higher score = worse match (further away).
type maxAggregation struct{}

func (m *maxAggregation) Kind() ScoreAggregationKind {
	return MaxAggregation
}

func (m *maxAggregation) Aggregate(results []VectorResult) []VectorResult {
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
// MEAN AGGREGATION
// ============================================================================

// meanAggregation averages all scores for the same node.
//
// Use case: Balanced approach that considers all query results equally.
// Good for general-purpose multi-query search.
//
// Example: If node 42 appears in 3 queries with scores [0.1, 0.2, 0.15],
// the final score will be 0.15 (average).
type meanAggregation struct{}

func (a *meanAggregation) Kind() ScoreAggregationKind {
	return MeanAggregation
}

func (a *meanAggregation) Aggregate(results []VectorResult) []VectorResult {
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
