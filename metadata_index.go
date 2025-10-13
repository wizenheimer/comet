// Package comet implements a metadata filtering index for vector search.
//
// WHAT IS A METADATA INDEX?
// A metadata index enables fast filtering of documents based on structured metadata
// attributes before performing expensive vector similarity searches. This dramatically
// improves search performance by reducing the candidate set.
//
// HOW IT WORKS:
// The index uses two specialized data structures:
// 1. Roaring Bitmaps: For categorical fields (strings, booleans)
//   - Memory-efficient compressed bitmaps
//   - Fast set operations (AND, OR, NOT)
//
// 2. Bit-Sliced Index (BSI): For numeric fields (integers, floats)
//   - Enables range queries without scanning all documents
//   - Fast comparison operations (>, <, =, BETWEEN)
//
// QUERY TYPES:
// - Equality: field = value
// - Inequality: field != value
// - Comparisons: field > value, field >= value, field < value, field <= value
// - Range: field BETWEEN min AND max
// - Set membership: field IN (val1, val2, val3)
// - Set exclusion: field NOT IN (val1, val2)
// - Existence: field EXISTS, field NOT EXISTS
//
// TIME COMPLEXITY:
//   - Add: O(m) where m is the number of metadata fields
//   - Query: O(f × log(n)) where f is filter count, n is documents
//   - Remove: O(m) where m is the number of metadata fields
//
// MEMORY REQUIREMENTS:
// - Roaring bitmaps: Highly compressed, typically 1-10% of uncompressed size
// - BSI: ~64 bits per numeric value (compressed with roaring)
// - Much more efficient than traditional B-tree indexes for high-cardinality data
//
// GUARANTEES & TRADE-OFFS:
// ✓ Pros:
//   - Extremely fast filtering (microseconds for millions of documents)
//   - Memory efficient with compression
//   - Supports complex boolean queries
//   - Thread-safe for concurrent use
//
// ✗ Cons:
//   - Only supports exact matches and ranges (no fuzzy matching)
//   - Floats converted to integers (precision loss)
//   - Updates require full document replacement
//
// WHEN TO USE:
// Use metadata index when:
// 1. Pre-filtering documents before vector search
// 2. Need to filter by structured attributes (price, date, category, etc.)
// 3. Working with large datasets (100K+ documents)
// 4. Need sub-millisecond filter performance
package comet

import (
	"fmt"
	"sync"

	"github.com/RoaringBitmap/roaring"
	bsi "github.com/RoaringBitmap/roaring/BitSliceIndexing"
)

// Compile-time check to ensure RoaringMetadataIndex implements MetadataIndex
var _ MetadataIndex = (*RoaringMetadataIndex)(nil)

// RoaringMetadataIndex provides fast filtering using Roaring Bitmaps and BSI.
//
// This index maintains two types of indexes:
// 1. Categorical index: Maps "field:value" to document IDs using roaring bitmaps
// 2. Numeric index: Maps field names to BSI structures for range queries
//
// Thread-safety: This index is safe for concurrent use through a read-write mutex.
type RoaringMetadataIndex struct {
	mu sync.RWMutex

	// Categorical fields: map of "field:value" -> bitmap of doc IDs
	categorical map[string]*roaring.Bitmap

	// Numeric fields: map of field name -> BSI
	numeric map[string]*bsi.BSI

	// Track all document IDs that exist
	allDocs *roaring.Bitmap
}

// NewRoaringMetadataIndex creates a new roaring bitmap-based metadata index.
//
// Returns:
//   - *RoaringMetadataIndex: A new empty index ready to accept documents
//
// Example:
//
//	idx := NewRoaringMetadataIndex()
//	node := NewMetadataNode(1, map[string]interface{}{
//		"category": "electronics",
//		"price": 999,
//	})
//	idx.Add(node)
func NewRoaringMetadataIndex() *RoaringMetadataIndex {
	return &RoaringMetadataIndex{
		categorical: make(map[string]*roaring.Bitmap),
		numeric:     make(map[string]*bsi.BSI),
		allDocs:     roaring.New(),
	}
}

// Add adds a document with its metadata to the index.
//
// The metadata is automatically classified into categorical or numeric fields:
//   - Numeric: int, int64, float64 (stored in BSI)
//   - Categorical: string, bool (stored in roaring bitmaps)
//
// Parameters:
//   - node: MetadataNode containing document ID and metadata fields
//
// Returns:
//   - error: Returns error if metadata contains unsupported types
//
// Time Complexity: O(m) where m is the number of metadata fields
//
// Thread-safety: Acquires exclusive lock
func (idx *RoaringMetadataIndex) Add(node MetadataNode) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	docID := node.ID()
	metadata := node.Metadata()

	idx.allDocs.Add(docID)

	for key, value := range metadata {
		switch v := value.(type) {
		case int:
			idx.addNumeric(key, docID, int64(v))
		case int64:
			idx.addNumeric(key, docID, v)
		case float64:
			// Convert float to int by multiplying by 100 (for 2 decimal precision)
			idx.addNumeric(key, docID, int64(v*100))
		case string:
			idx.addCategorical(key, v, docID)
		case bool:
			idx.addCategorical(key, fmt.Sprintf("%v", v), docID)
		default:
			return fmt.Errorf("unsupported type for key %s: %T", key, value)
		}
	}

	return nil
}

// addCategorical adds a categorical field value to the index.
// Must be called with idx.mu held.
func (idx *RoaringMetadataIndex) addCategorical(field, value string, docID uint32) {
	key := fmt.Sprintf("%s:%s", field, value)
	if idx.categorical[key] == nil {
		idx.categorical[key] = roaring.New()
	}
	idx.categorical[key].Add(docID)
}

// addNumeric adds a numeric field value to the index.
// Must be called with idx.mu held.
func (idx *RoaringMetadataIndex) addNumeric(field string, docID uint32, value int64) {
	if idx.numeric[field] == nil {
		// Initialize BSI with a reasonable range
		idx.numeric[field] = bsi.NewBSI(bsi.Min64BitSigned, bsi.Max64BitSigned)
	}
	idx.numeric[field].SetValue(uint64(docID), value)
}

// Remove removes a document from all indexes.
//
// Parameters:
//   - node: MetadataNode to remove (only the ID field is used for matching)
//
// Returns:
//   - error: Always returns nil (exists to satisfy MetadataIndex interface)
//
// Time Complexity: O(m) where m is the number of metadata fields
//
// Thread-safety: Acquires exclusive lock
func (idx *RoaringMetadataIndex) Remove(node MetadataNode) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	docID := node.ID()

	idx.allDocs.Remove(docID)

	// Remove from categorical indexes
	for _, bitmap := range idx.categorical {
		bitmap.Remove(docID)
	}

	// Remove from numeric indexes
	for _, bsi := range idx.numeric {
		bsi.ClearValues(roaring.BitmapOf(docID))
	}

	return nil
}

// NewSearch creates a new search builder for this index.
//
// Returns:
//   - MetadataSearch: A new search builder ready to be configured
//
// Example:
//
//	results, err := idx.NewSearch().
//		WithFilters(
//			Eq("category", "electronics"),
//			Gte("price", 500),
//		).
//		Execute()
func (idx *RoaringMetadataIndex) NewSearch() MetadataSearch {
	return &metadataFilterSearch{
		index: idx,
	}
}

// Flush is a no-op for roaring metadata index since data is stored in memory.
// This method exists to satisfy the MetadataIndex interface.
//
// Returns:
//   - error: Always returns nil
func (idx *RoaringMetadataIndex) Flush() error {
	return nil
}

// getExistenceBitmap returns a bitmap of all documents that have a value for the field.
// Must be called with idx.mu held (at least read lock).
func (idx *RoaringMetadataIndex) getExistenceBitmap(field string) *roaring.Bitmap {
	// Check numeric fields
	if bsiIndex, exists := idx.numeric[field]; exists {
		return bsiIndex.GetExistenceBitmap().Clone()
	}

	// Check categorical fields - OR all bitmaps for this field
	result := roaring.New()
	prefix := field + ":"
	for key, bitmap := range idx.categorical {
		if len(key) > len(prefix) && key[:len(prefix)] == prefix {
			result.Or(bitmap)
		}
	}

	return result
}

// queryCategorical handles categorical field queries.
// Must be called with idx.mu held (at least read lock).
func (idx *RoaringMetadataIndex) queryCategorical(filter Filter) (*roaring.Bitmap, error) {
	switch filter.Operator {
	case OpEqual, "": // Default to equality
		key := fmt.Sprintf("%s:%v", filter.Field, filter.Value)
		if bitmap, exists := idx.categorical[key]; exists {
			return bitmap.Clone(), nil
		}
		return roaring.New(), nil

	case OpNotEqual: // Not equal
		key := fmt.Sprintf("%s:%v", filter.Field, filter.Value)
		result := idx.allDocs.Clone()
		if bitmap, exists := idx.categorical[key]; exists {
			result.AndNot(bitmap)
		}
		return result, nil

	case OpIn: // Value is in a list
		result := roaring.New()

		// Handle different slice types
		switch vals := filter.Value.(type) {
		case []string:
			for _, val := range vals {
				key := fmt.Sprintf("%s:%s", filter.Field, val)
				if bitmap, exists := idx.categorical[key]; exists {
					result.Or(bitmap)
				}
			}
		case []interface{}:
			for _, val := range vals {
				key := fmt.Sprintf("%s:%v", filter.Field, val)
				if bitmap, exists := idx.categorical[key]; exists {
					result.Or(bitmap)
				}
			}
		default:
			return nil, fmt.Errorf("'in' operator requires []string or []interface{} value")
		}

		return result, nil

	case OpNotIn: // Value is not in a list
		result := idx.allDocs.Clone()

		// Handle different slice types
		switch vals := filter.Value.(type) {
		case []string:
			for _, val := range vals {
				key := fmt.Sprintf("%s:%s", filter.Field, val)
				if bitmap, exists := idx.categorical[key]; exists {
					result.AndNot(bitmap)
				}
			}
		case []interface{}:
			for _, val := range vals {
				key := fmt.Sprintf("%s:%v", filter.Field, val)
				if bitmap, exists := idx.categorical[key]; exists {
					result.AndNot(bitmap)
				}
			}
		default:
			return nil, fmt.Errorf("'not_in' operator requires []string or []interface{} value")
		}

		return result, nil

	default:
		return nil, fmt.Errorf("unsupported operator for categorical field: %s", filter.Operator)
	}
}

// queryNumeric handles numeric field queries using BSI.
// Must be called with idx.mu held (at least read lock).
func (idx *RoaringMetadataIndex) queryNumeric(bsiIndex *bsi.BSI, filter Filter) (*roaring.Bitmap, error) {
	switch filter.Operator {
	case OpEqual, "": // Equality
		value, err := toInt64(filter.Value)
		if err != nil {
			return nil, err
		}
		return bsiIndex.CompareValue(0, bsi.EQ, value, 0, nil), nil

	case OpNotEqual: // Not equal
		value, err := toInt64(filter.Value)
		if err != nil {
			return nil, err
		}
		eq := bsiIndex.CompareValue(0, bsi.EQ, value, 0, nil)
		result := bsiIndex.GetExistenceBitmap().Clone()
		result.AndNot(eq)
		return result, nil

	case OpGreaterThan: // Greater than
		value, err := toInt64(filter.Value)
		if err != nil {
			return nil, err
		}
		return bsiIndex.CompareValue(0, bsi.GT, value, 0, nil), nil

	case OpGreaterThanOrEqual: // Greater than or equal
		value, err := toInt64(filter.Value)
		if err != nil {
			return nil, err
		}
		return bsiIndex.CompareValue(0, bsi.GE, value, 0, nil), nil

	case OpLessThan: // Less than
		value, err := toInt64(filter.Value)
		if err != nil {
			return nil, err
		}
		return bsiIndex.CompareValue(0, bsi.LT, value, 0, nil), nil

	case OpLessThanOrEqual: // Less than or equal
		value, err := toInt64(filter.Value)
		if err != nil {
			return nil, err
		}
		return bsiIndex.CompareValue(0, bsi.LE, value, 0, nil), nil

	case OpRange: // Range query [value, value2]
		minVal, err := toInt64(filter.Value)
		if err != nil {
			return nil, err
		}
		maxVal, err := toInt64(filter.Value2)
		if err != nil {
			return nil, err
		}
		return bsiIndex.CompareValue(0, bsi.RANGE, minVal, maxVal, nil), nil

	default:
		return nil, fmt.Errorf("unsupported operator for numeric field: %s", filter.Operator)
	}
}

// toInt64 converts various numeric types to int64
func toInt64(value interface{}) (int64, error) {
	switch v := value.(type) {
	case int:
		return int64(v), nil
	case int64:
		return v, nil
	case float64:
		// Convert float to int by multiplying by 100 (for 2 decimal precision)
		return int64(v * 100), nil
	default:
		return 0, fmt.Errorf("cannot convert %T to int64", value)
	}
}

// Operator represents a filter operation type
type Operator string

// Operator constants
const (
	// Equality operators
	OpEqual    Operator = "eq" // Equal to
	OpNotEqual Operator = "ne" // Not equal to

	// Comparison operators (numeric)
	OpGreaterThan        Operator = "gt"  // Greater than
	OpGreaterThanOrEqual Operator = "gte" // Greater than or equal to
	OpLessThan           Operator = "lt"  // Less than
	OpLessThanOrEqual    Operator = "lte" // Less than or equal to

	// Set operators
	OpIn    Operator = "in"     // In a set of values
	OpNotIn Operator = "not_in" // Not in a set of values

	// Range operators
	OpRange Operator = "range" // Within a range [Value, Value2]

	// Existence operators
	OpExists    Operator = "exists"     // Field exists (has any value)
	OpNotExists Operator = "not_exists" // Field doesn't exist
)

// Filter represents a query filter
type Filter struct {
	Field    string
	Operator Operator
	Value    interface{}
	Value2   interface{} // Used for range queries
}

// Type-safe filter constructors

// Eq creates an equality filter
func Eq(field string, value interface{}) Filter {
	return Filter{Field: field, Operator: OpEqual, Value: value}
}

// Ne creates a not-equal filter
func Ne(field string, value interface{}) Filter {
	return Filter{Field: field, Operator: OpNotEqual, Value: value}
}

// Gt creates a greater-than filter
func Gt(field string, value interface{}) Filter {
	return Filter{Field: field, Operator: OpGreaterThan, Value: value}
}

// Gte creates a greater-than-or-equal filter
func Gte(field string, value interface{}) Filter {
	return Filter{Field: field, Operator: OpGreaterThanOrEqual, Value: value}
}

// Lt creates a less-than filter
func Lt(field string, value interface{}) Filter {
	return Filter{Field: field, Operator: OpLessThan, Value: value}
}

// Lte creates a less-than-or-equal filter
func Lte(field string, value interface{}) Filter {
	return Filter{Field: field, Operator: OpLessThanOrEqual, Value: value}
}

// In creates an in-set filter
func In(field string, values ...interface{}) Filter {
	return Filter{Field: field, Operator: OpIn, Value: values}
}

// NotIn creates a not-in-set filter
func NotIn(field string, values ...interface{}) Filter {
	return Filter{Field: field, Operator: OpNotIn, Value: values}
}

// Range creates a range filter [min, max]
func Range(field string, min, max interface{}) Filter {
	return Filter{Field: field, Operator: OpRange, Value: min, Value2: max}
}

// Between is an alias for Range
func Between(field string, min, max interface{}) Filter {
	return Range(field, min, max)
}

// Exists checks if a field exists (has any value)
func Exists(field string) Filter {
	return Filter{Field: field, Operator: OpExists}
}

// NotExists checks if a field doesn't exist
func NotExists(field string) Filter {
	return Filter{Field: field, Operator: OpNotExists}
}

// IsNull checks if a field is null/doesn't exist
func IsNull(field string) Filter {
	return NotExists(field)
}

// IsNotNull checks if a field is not null/exists
func IsNotNull(field string) Filter {
	return Exists(field)
}

// Not creates a negation filter by inverting the operator
func Not(filter Filter) Filter {
	// Invert the operator
	switch filter.Operator {
	case OpEqual:
		filter.Operator = OpNotEqual
	case OpNotEqual:
		filter.Operator = OpEqual
	case OpGreaterThan:
		filter.Operator = OpLessThanOrEqual
	case OpGreaterThanOrEqual:
		filter.Operator = OpLessThan
	case OpLessThan:
		filter.Operator = OpGreaterThanOrEqual
	case OpLessThanOrEqual:
		filter.Operator = OpGreaterThan
	case OpIn:
		filter.Operator = OpNotIn
	case OpNotIn:
		filter.Operator = OpIn
	case OpExists:
		filter.Operator = OpNotExists
	case OpNotExists:
		filter.Operator = OpExists
	}
	return filter
}

// AnyOf creates an OR group for multiple values on the same field (alias for In)
func AnyOf(field string, values ...interface{}) Filter {
	return In(field, values...)
}

// NoneOf creates a NOT IN filter (alias for NotIn)
func NoneOf(field string, values ...interface{}) Filter {
	return NotIn(field, values...)
}
