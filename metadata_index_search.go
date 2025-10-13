package comet

import (
	"fmt"

	"github.com/RoaringBitmap/roaring"
)

// MetadataResult represents a search result for a metadata node
type MetadataResult struct {
	Node MetadataNode
}

func (r MetadataResult) GetId() uint32 {
	return r.Node.ID()
}

// Compile-time checks to ensure metadataFilterSearch implements MetadataSearch
var _ MetadataSearch = (*metadataFilterSearch)(nil)

// metadataFilterSearch implements the MetadataSearch interface.
//
// This search builder supports two query styles:
// 1. Simple filters: Multiple filters combined with AND logic
// 2. Filter groups: Complex boolean expressions with OR logic between groups
type metadataFilterSearch struct {
	index        *RoaringMetadataIndex
	filters      []Filter
	filterGroups []*FilterGroup
}

// FilterGroup represents a group of filters with a logical operator
type FilterGroup struct {
	Filters []Filter
	Logic   LogicOperator
}

// LogicOperator defines how filters are combined
type LogicOperator string

const (
	AND LogicOperator = "AND"
	OR  LogicOperator = "OR"
)

// WithFilters sets the filters to apply.
// Multiple filters are combined with AND logic.
//
// Example:
//
//	results, err := idx.NewSearch().
//		WithFilters(
//			Eq("category", "electronics"),
//			Gte("price", 500),
//		).
//		Execute()
func (s *metadataFilterSearch) WithFilters(filters ...Filter) MetadataSearch {
	s.filters = filters
	return s
}

// WithFilterGroups sets complex filter groups.
// Filter groups are combined with OR logic.
//
// Example:
//
//	results, err := idx.NewSearch().
//		WithFilterGroups(
//			&FilterGroup{
//				Filters: []Filter{Eq("category", "electronics"), Gte("price", 1000)},
//				Logic:   AND,
//			},
//			&FilterGroup{
//				Filters: []Filter{Eq("category", "phones"), Gte("price", 500)},
//				Logic:   AND,
//			},
//		).
//		Execute()
func (s *metadataFilterSearch) WithFilterGroups(groups ...*FilterGroup) MetadataSearch {
	s.filterGroups = groups
	return s
}

// Execute performs the actual search and returns matching document IDs.
//
// The search behavior depends on what was configured:
//   - If filters are set: Uses simple AND logic between all filters
//   - If filter groups are set: Uses OR logic between groups (filters within groups use AND)
//   - If neither: Returns all documents
//
// Returns:
//   - []MetadataResult: List of matching documents
//   - error: Returns error if search configuration is invalid
func (s *metadataFilterSearch) Execute() ([]MetadataResult, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	var resultBitmap *roaring.Bitmap

	// If filter groups are specified, use complex query logic
	if len(s.filterGroups) > 0 {
		var err error
		resultBitmap, err = s.executeFilterGroups()
		if err != nil {
			return nil, err
		}
	} else if len(s.filters) > 0 {
		// Otherwise use simple AND logic for filters
		var err error
		resultBitmap, err = s.executeSimpleFilters()
		if err != nil {
			return nil, err
		}
	} else {
		// No filters: return all documents
		resultBitmap = s.index.allDocs.Clone()
	}

	// Convert bitmap to results
	docIDs := resultBitmap.ToArray()
	results := make([]MetadataResult, len(docIDs))
	for i, docID := range docIDs {
		results[i] = MetadataResult{
			Node: MetadataNode{id: docID},
		}
	}

	return results, nil
}

// executeSimpleFilters executes filters with simple AND logic.
// Must be called with s.index.mu held (at least read lock).
func (s *metadataFilterSearch) executeSimpleFilters() (*roaring.Bitmap, error) {
	var result *roaring.Bitmap

	for _, filter := range s.filters {
		bitmap, err := s.evaluateFilter(filter)
		if err != nil {
			return nil, err
		}

		// Intersect with previous results (AND operation)
		if result == nil {
			result = bitmap
		} else {
			result.And(bitmap)
		}

		// Early exit if no documents match
		if result.IsEmpty() {
			return result, nil
		}
	}

	if result == nil {
		result = roaring.New()
	}

	return result, nil
}

// executeFilterGroups executes filter groups with OR logic between groups.
// Must be called with s.index.mu held (at least read lock).
func (s *metadataFilterSearch) executeFilterGroups() (*roaring.Bitmap, error) {
	var finalResult *roaring.Bitmap

	for groupIdx, group := range s.filterGroups {
		groupResult, err := s.executeGroup(group)
		if err != nil {
			return nil, fmt.Errorf("error executing group %d: %w", groupIdx, err)
		}

		if finalResult == nil {
			finalResult = groupResult
		} else {
			// Combine groups with OR logic
			finalResult.Or(groupResult)
		}
	}

	if finalResult == nil {
		finalResult = roaring.New()
	}

	return finalResult, nil
}

// executeGroup executes a single filter group.
// Must be called with s.index.mu held (at least read lock).
func (s *metadataFilterSearch) executeGroup(group *FilterGroup) (*roaring.Bitmap, error) {
	if len(group.Filters) == 0 {
		return s.index.allDocs.Clone(), nil
	}

	var result *roaring.Bitmap

	for _, filter := range group.Filters {
		bitmap, err := s.evaluateFilter(filter)
		if err != nil {
			return nil, err
		}

		// Combine filters within the group
		if result == nil {
			result = bitmap
		} else {
			if group.Logic == AND {
				result.And(bitmap)
			} else {
				result.Or(bitmap)
			}
		}

		// Early exit for AND if no matches
		if group.Logic == AND && result.IsEmpty() {
			return result, nil
		}
	}

	return result, nil
}

// evaluateFilter evaluates a single filter and returns the matching document bitmap.
// Must be called with s.index.mu held (at least read lock).
func (s *metadataFilterSearch) evaluateFilter(filter Filter) (*roaring.Bitmap, error) {
	// Handle special operators
	switch filter.Operator {
	case OpExists:
		return s.index.getExistenceBitmap(filter.Field), nil
	case OpNotExists:
		bitmap := s.index.allDocs.Clone()
		bitmap.AndNot(s.index.getExistenceBitmap(filter.Field))
		return bitmap, nil
	}

	// Check if it's a numeric field
	if bsiIndex, exists := s.index.numeric[filter.Field]; exists {
		return s.index.queryNumeric(bsiIndex, filter)
	}

	// Categorical field
	return s.index.queryCategorical(filter)
}

// MetadataFilterQueryBuilder provides a type-safe fluent API for building complex metadata filter queries
type MetadataFilterQueryBuilder struct {
	groups []*FilterGroup
}

// NewMetadataFilterQuery creates a new query builder
//
// Example:
//
//	results, err := NewMetadataFilterQuery().
//		Where(Eq("category", "electronics"), Gte("price", 500)).
//		Or(Eq("category", "phones"), Gte("rating", 4.5)).
//		Execute(idx)
func NewMetadataFilterQuery() *MetadataFilterQueryBuilder {
	return &MetadataFilterQueryBuilder{
		groups: make([]*FilterGroup, 0),
	}
}

// Where starts a new filter group with AND logic
func (qb *MetadataFilterQueryBuilder) Where(filters ...Filter) *MetadataFilterQueryBuilder {
	if len(filters) > 0 {
		qb.groups = append(qb.groups, &FilterGroup{
			Filters: filters,
			Logic:   AND,
		})
	}
	return qb
}

// Or starts a new filter group (filters within are combined with AND, but this group is OR'd with previous groups)
func (qb *MetadataFilterQueryBuilder) Or(filters ...Filter) *MetadataFilterQueryBuilder {
	if len(filters) > 0 {
		qb.groups = append(qb.groups, &FilterGroup{
			Filters: filters,
			Logic:   AND,
		})
	}
	return qb
}

// And adds filters with AND logic to the last group
func (qb *MetadataFilterQueryBuilder) And(filters ...Filter) *MetadataFilterQueryBuilder {
	if len(qb.groups) > 0 && len(filters) > 0 {
		lastGroup := qb.groups[len(qb.groups)-1]
		lastGroup.Filters = append(lastGroup.Filters, filters...)
		lastGroup.Logic = AND
	} else if len(filters) > 0 {
		qb.Where(filters...)
	}
	return qb
}

// Build returns the constructed filter groups
func (qb *MetadataFilterQueryBuilder) Build() []*FilterGroup {
	return qb.groups
}

// Execute runs the query against a metadata index
//
// Example:
//
//	results, err := NewMetadataFilterQuery().
//		Where(Eq("brand", "Apple")).
//		Or(Eq("category", "phone"), Eq("brand", "Samsung")).
//		Execute(idx)
func (qb *MetadataFilterQueryBuilder) Execute(idx MetadataIndex) ([]MetadataResult, error) {
	if roaringIdx, ok := idx.(*RoaringMetadataIndex); ok {
		return roaringIdx.NewSearch().WithFilterGroups(qb.groups...).Execute()
	}
	return nil, fmt.Errorf("unsupported index type")
}
