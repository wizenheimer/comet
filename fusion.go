package comet

import "fmt"

// FusionKind defines the type of score fusion strategy for combining results
// from multiple search modalities (vector, text, metadata).
type FusionKind string

const (
	// WeightedSumFusion combines scores using weighted sum
	// finalScore = vectorScore * vectorWeight + textScore * textWeight
	WeightedSumFusion FusionKind = "weighted_sum"

	// ReciprocalRankFusion (RRF) combines results based on their ranks
	// More robust to differences in score scales between modalities
	ReciprocalRankFusion FusionKind = "reciprocal_rank"

	// MaxFusion takes the maximum score across modalities
	MaxFusion FusionKind = "max"

	// MinFusion takes the minimum score across modalities
	MinFusion FusionKind = "min"
)

// Fusion defines how to combine scores from different search modalities
// (vector search, text search) into a single ranking.
//
// Different fusion strategies handle score normalization and combination differently:
// - WeightedSum: Direct weighted combination of scores
// - ReciprocalRank: Rank-based fusion (more robust to score scale differences)
// - Max/Min: Simple maximum or minimum across modalities
type Fusion interface {
	// Kind returns the kind of fusion strategy
	Kind() FusionKind

	// Combine takes separate result sets from vector and text search and
	// combines them into a single ranked list
	//
	// Parameters:
	//   - vectorResults: Results from vector search (map of docID -> score)
	//   - textResults: Results from text search (map of docID -> score)
	//
	// Returns:
	//   - Combined results as map of docID -> score
	Combine(vectorResults map[uint32]float64, textResults map[uint32]float64) map[uint32]float64
}

// FusionConfig holds configuration for fusion strategies
type FusionConfig struct {
	// VectorWeight is the weight for vector search scores (used by WeightedSumFusion)
	VectorWeight float64

	// TextWeight is the weight for text search scores (used by WeightedSumFusion)
	TextWeight float64

	// K is the constant used in Reciprocal Rank Fusion (default: 60)
	// Lower k gives more weight to top-ranked items
	K float64
}

// DefaultFusionConfig returns the default fusion configuration
func DefaultFusionConfig() *FusionConfig {
	return &FusionConfig{
		VectorWeight: 1.0,
		TextWeight:   1.0,
		K:            60.0,
	}
}

// Singleton instances for fusion strategies
var (
	defaultWeightedSumFusion *weightedSumFusion
	maxFusionInstance        *maxFusion
	minFusionInstance        *minFusion
)

func init() {
	defaultWeightedSumFusion = &weightedSumFusion{
		config: DefaultFusionConfig(),
	}
	maxFusionInstance = &maxFusion{}
	minFusionInstance = &minFusion{}
}

// NewFusion creates a new fusion strategy with the given configuration
func NewFusion(kind FusionKind, config *FusionConfig) (Fusion, error) {
	if config == nil {
		config = DefaultFusionConfig()
	}

	switch kind {
	case WeightedSumFusion:
		return &weightedSumFusion{config: config}, nil
	case ReciprocalRankFusion:
		return &reciprocalRankFusion{config: config}, nil
	case MaxFusion:
		return maxFusionInstance, nil
	case MinFusion:
		return minFusionInstance, nil
	default:
		return nil, fmt.Errorf("unknown fusion kind: %s", kind)
	}
}

// DefaultFusion returns the default fusion strategy (WeightedSum)
func DefaultFusion() Fusion {
	return defaultWeightedSumFusion
}

// ============================================================================
// WEIGHTED SUM FUSION
// ============================================================================

// weightedSumFusion combines scores using weighted sum
//
// Use case: When you want direct control over the relative importance of
// vector vs text search. Simple and interpretable.
//
// Formula: finalScore = vectorScore * vectorWeight + textScore * textWeight
//
// Example: If vectorWeight=2.0 and textWeight=1.0, vector similarity is
// weighted twice as much as text relevance.
type weightedSumFusion struct {
	config *FusionConfig
}

func (f *weightedSumFusion) Kind() FusionKind {
	return WeightedSumFusion
}

func (f *weightedSumFusion) Combine(vectorResults map[uint32]float64, textResults map[uint32]float64) map[uint32]float64 {
	combined := make(map[uint32]float64)

	// Add weighted vector scores
	for docID, score := range vectorResults {
		combined[docID] = score * f.config.VectorWeight
	}

	// Add weighted text scores
	for docID, score := range textResults {
		if existing, exists := combined[docID]; exists {
			combined[docID] = existing + score*f.config.TextWeight
		} else {
			combined[docID] = score * f.config.TextWeight
		}
	}

	return combined
}

// ============================================================================
// RECIPROCAL RANK FUSION (RRF)
// ============================================================================

// reciprocalRankFusion uses rank-based score combination
//
// Use case: When vector and text search have different score scales that are
// hard to normalize. RRF is more robust as it only uses ranks, not raw scores.
//
// Formula: score = sum(1 / (k + rank_i)) for each result list
//
// This is the fusion method used by many hybrid search systems because it
// doesn't require score normalization and handles multi-modal search well.
//
// Reference: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
type reciprocalRankFusion struct {
	config *FusionConfig
}

func (f *reciprocalRankFusion) Kind() FusionKind {
	return ReciprocalRankFusion
}

func (f *reciprocalRankFusion) Combine(vectorResults map[uint32]float64, textResults map[uint32]float64) map[uint32]float64 {
	combined := make(map[uint32]float64)
	k := f.config.K

	// Convert vector results to ranks (sorted by score)
	vectorRanks := scoreMapToRanks(vectorResults, true) // ascending for distances

	// Convert text results to ranks (sorted by score)
	textRanks := scoreMapToRanks(textResults, false) // descending for relevance

	// Apply RRF formula for vector results
	for docID, rank := range vectorRanks {
		combined[docID] = 1.0 / (k + float64(rank))
	}

	// Add RRF scores from text results
	for docID, rank := range textRanks {
		rrfScore := 1.0 / (k + float64(rank))
		if existing, exists := combined[docID]; exists {
			combined[docID] = existing + rrfScore
		} else {
			combined[docID] = rrfScore
		}
	}

	return combined
}

// scoreMapToRanks converts a score map to ranks
// ascending=true means lower scores get better (lower) ranks (for distances)
// ascending=false means higher scores get better (lower) ranks (for relevance)
func scoreMapToRanks(scores map[uint32]float64, ascending bool) map[uint32]int {
	if len(scores) == 0 {
		return make(map[uint32]int)
	}

	// Create sorted list of (docID, score) pairs
	type docScore struct {
		docID uint32
		score float64
	}

	sorted := make([]docScore, 0, len(scores))
	for docID, score := range scores {
		sorted = append(sorted, docScore{docID: docID, score: score})
	}

	// Sort by score
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			shouldSwap := false
			if ascending {
				shouldSwap = sorted[i].score > sorted[j].score
			} else {
				shouldSwap = sorted[i].score < sorted[j].score
			}
			if shouldSwap {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	// Assign ranks (0-indexed)
	ranks := make(map[uint32]int)
	for i, ds := range sorted {
		ranks[ds.docID] = i
	}

	return ranks
}

// ============================================================================
// MAX FUSION
// ============================================================================

// maxFusion takes the maximum score across modalities
//
// Use case: When you want documents that excel in at least one modality
type maxFusion struct{}

func (f *maxFusion) Kind() FusionKind {
	return MaxFusion
}

func (f *maxFusion) Combine(vectorResults map[uint32]float64, textResults map[uint32]float64) map[uint32]float64 {
	combined := make(map[uint32]float64)

	for docID, score := range vectorResults {
		combined[docID] = score
	}

	for docID, score := range textResults {
		if existing, exists := combined[docID]; exists {
			if score > existing {
				combined[docID] = score
			}
		} else {
			combined[docID] = score
		}
	}

	return combined
}

// ============================================================================
// MIN FUSION
// ============================================================================

// minFusion takes the minimum score across modalities
//
// Use case: When you want documents that perform well in all modalities
type minFusion struct{}

func (f *minFusion) Kind() FusionKind {
	return MinFusion
}

func (f *minFusion) Combine(vectorResults map[uint32]float64, textResults map[uint32]float64) map[uint32]float64 {
	combined := make(map[uint32]float64)

	// Only include documents that appear in both result sets
	for docID, vectorScore := range vectorResults {
		if textScore, exists := textResults[docID]; exists {
			if vectorScore < textScore {
				combined[docID] = vectorScore
			} else {
				combined[docID] = textScore
			}
		}
	}

	return combined
}
