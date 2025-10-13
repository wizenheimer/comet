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
	"encoding/binary"
	"fmt"
	"io"
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

// WriteTo serializes the RoaringMetadataIndex to an io.Writer.
//
// IMPORTANT: This method calls Flush() before serialization (though Flush is a no-op
// for metadata index since it's memory-based).
//
// The serialization format is:
// 1. Magic number (4 bytes) - "MTIX" identifier for validation
// 2. Version (4 bytes) - Format version for backward compatibility
// 3. All docs bitmap size (4 bytes) + bitmap bytes
// 4. Number of categorical entries (4 bytes)
// 5. For each categorical entry:
//   - Key length (4 bytes) + key string
//   - Bitmap size (4 bytes) + bitmap bytes
//
// 6. Number of numeric entries (4 bytes)
// 7. For each numeric entry:
//   - Field name length (4 bytes) + field name string
//   - BSI size (4 bytes) + BSI bytes
//
// Thread-safety: Acquires read lock during serialization
//
// Returns:
//   - int64: Number of bytes written
//   - error: Returns error if write fails or flush fails
func (idx *RoaringMetadataIndex) WriteTo(w io.Writer) (int64, error) {
	// Flush before serializing (no-op for metadata index)
	if err := idx.Flush(); err != nil {
		return 0, fmt.Errorf("failed to flush before serialization: %w", err)
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var bytesWritten int64

	// Helper function to track writes
	write := func(data interface{}) error {
		err := binary.Write(w, binary.LittleEndian, data)
		if err == nil {
			switch data.(type) {
			case uint32, int32, float32:
				bytesWritten += 4
			case uint8, int8, bool:
				bytesWritten += 1
			}
		}
		return err
	}

	// 1. Write magic number "MTIX"
	magic := [4]byte{'M', 'T', 'I', 'X'}
	if _, err := w.Write(magic[:]); err != nil {
		return bytesWritten, fmt.Errorf("failed to write magic number: %w", err)
	}
	bytesWritten += 4

	// 2. Write version
	version := uint32(1)
	if err := write(version); err != nil {
		return bytesWritten, fmt.Errorf("failed to write version: %w", err)
	}

	// 3. Write allDocs bitmap
	allDocsBytes, err := idx.allDocs.ToBytes()
	if err != nil {
		return bytesWritten, fmt.Errorf("failed to serialize allDocs bitmap: %w", err)
	}
	if err := write(uint32(len(allDocsBytes))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write allDocs size: %w", err)
	}
	if _, err := w.Write(allDocsBytes); err != nil {
		return bytesWritten, fmt.Errorf("failed to write allDocs data: %w", err)
	}
	bytesWritten += int64(len(allDocsBytes))

	// 4. Write categorical entries
	if err := write(uint32(len(idx.categorical))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write categorical count: %w", err)
	}

	for key, bitmap := range idx.categorical {
		// Write key
		keyBytes := []byte(key)
		if err := write(uint32(len(keyBytes))); err != nil {
			return bytesWritten, fmt.Errorf("failed to write key length: %w", err)
		}
		if _, err := w.Write(keyBytes); err != nil {
			return bytesWritten, fmt.Errorf("failed to write key: %w", err)
		}
		bytesWritten += int64(len(keyBytes))

		// Write bitmap
		bitmapBytes, err := bitmap.ToBytes()
		if err != nil {
			return bytesWritten, fmt.Errorf("failed to serialize bitmap for key %s: %w", key, err)
		}
		if err := write(uint32(len(bitmapBytes))); err != nil {
			return bytesWritten, fmt.Errorf("failed to write bitmap size for key %s: %w", key, err)
		}
		if _, err := w.Write(bitmapBytes); err != nil {
			return bytesWritten, fmt.Errorf("failed to write bitmap data for key %s: %w", key, err)
		}
		bytesWritten += int64(len(bitmapBytes))
	}

	// 6. Write numeric entries
	if err := write(uint32(len(idx.numeric))); err != nil {
		return bytesWritten, fmt.Errorf("failed to write numeric count: %w", err)
	}

	for field, bsiIndex := range idx.numeric {
		// Write field name
		fieldBytes := []byte(field)
		if err := write(uint32(len(fieldBytes))); err != nil {
			return bytesWritten, fmt.Errorf("failed to write field name length: %w", err)
		}
		if _, err := w.Write(fieldBytes); err != nil {
			return bytesWritten, fmt.Errorf("failed to write field name: %w", err)
		}
		bytesWritten += int64(len(fieldBytes))

		// Write BSI
		bsiBytes, err := bsiIndex.MarshalBinary()
		if err != nil {
			return bytesWritten, fmt.Errorf("failed to serialize BSI for field %s: %w", field, err)
		}
		// Write number of byte slices
		if err := write(uint32(len(bsiBytes))); err != nil {
			return bytesWritten, fmt.Errorf("failed to write BSI slice count for field %s: %w", field, err)
		}
		// Write each byte slice
		for j, slice := range bsiBytes {
			if err := write(uint32(len(slice))); err != nil {
				return bytesWritten, fmt.Errorf("failed to write BSI slice %d size for field %s: %w", j, field, err)
			}
			if _, err := w.Write(slice); err != nil {
				return bytesWritten, fmt.Errorf("failed to write BSI slice %d data for field %s: %w", j, field, err)
			}
			bytesWritten += int64(len(slice))
		}
	}

	return bytesWritten, nil
}

// ReadFrom deserializes a RoaringMetadataIndex from an io.Reader.
//
// This method reconstructs a RoaringMetadataIndex from the serialized format created by WriteTo.
// The deserialized index is fully functional and ready to use for searches.
//
// Thread-safety: Acquires write lock during deserialization
//
// Returns:
//   - int64: Number of bytes read
//   - error: Returns error if read fails, format is invalid, or data is corrupted
//
// Example:
//
//	// Save index
//	file, _ := os.Create("index.bin")
//	idx.WriteTo(file)
//	file.Close()
//
//	// Load index
//	file, _ := os.Open("index.bin")
//	idx2 := NewRoaringMetadataIndex()
//	idx2.ReadFrom(file)
//	file.Close()
func (idx *RoaringMetadataIndex) ReadFrom(r io.Reader) (int64, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	var bytesRead int64

	// Helper function to track reads
	read := func(data interface{}) error {
		err := binary.Read(r, binary.LittleEndian, data)
		if err == nil {
			switch data.(type) {
			case *uint32, *int32, *float32:
				bytesRead += 4
			case *uint8, *int8, *bool:
				bytesRead += 1
			}
		}
		return err
	}

	// 1. Read and validate magic number
	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		return bytesRead, fmt.Errorf("failed to read magic number: %w", err)
	}
	bytesRead += 4
	if string(magic) != "MTIX" {
		return bytesRead, fmt.Errorf("invalid magic number: expected 'MTIX', got '%s'", string(magic))
	}

	// 2. Read version
	var version uint32
	if err := read(&version); err != nil {
		return bytesRead, fmt.Errorf("failed to read version: %w", err)
	}
	if version != 1 {
		return bytesRead, fmt.Errorf("unsupported version: %d", version)
	}

	// 3. Read allDocs bitmap
	var allDocsSize uint32
	if err := read(&allDocsSize); err != nil {
		return bytesRead, fmt.Errorf("failed to read allDocs size: %w", err)
	}

	allDocsBytes := make([]byte, allDocsSize)
	if _, err := io.ReadFull(r, allDocsBytes); err != nil {
		return bytesRead, fmt.Errorf("failed to read allDocs data: %w", err)
	}
	bytesRead += int64(allDocsSize)

	allDocs := roaring.New()
	if err := allDocs.UnmarshalBinary(allDocsBytes); err != nil {
		return bytesRead, fmt.Errorf("failed to deserialize allDocs bitmap: %w", err)
	}

	// 4. Read categorical entries
	var categoricalCount uint32
	if err := read(&categoricalCount); err != nil {
		return bytesRead, fmt.Errorf("failed to read categorical count: %w", err)
	}

	categorical := make(map[string]*roaring.Bitmap, categoricalCount)
	for i := uint32(0); i < categoricalCount; i++ {
		// Read key
		var keyLen uint32
		if err := read(&keyLen); err != nil {
			return bytesRead, fmt.Errorf("failed to read key length for entry %d: %w", i, err)
		}

		keyBytes := make([]byte, keyLen)
		if _, err := io.ReadFull(r, keyBytes); err != nil {
			return bytesRead, fmt.Errorf("failed to read key for entry %d: %w", i, err)
		}
		bytesRead += int64(keyLen)
		key := string(keyBytes)

		// Read bitmap
		var bitmapSize uint32
		if err := read(&bitmapSize); err != nil {
			return bytesRead, fmt.Errorf("failed to read bitmap size for key %s: %w", key, err)
		}

		bitmapBytes := make([]byte, bitmapSize)
		if _, err := io.ReadFull(r, bitmapBytes); err != nil {
			return bytesRead, fmt.Errorf("failed to read bitmap data for key %s: %w", key, err)
		}
		bytesRead += int64(bitmapSize)

		bitmap := roaring.New()
		if err := bitmap.UnmarshalBinary(bitmapBytes); err != nil {
			return bytesRead, fmt.Errorf("failed to deserialize bitmap for key %s: %w", key, err)
		}

		categorical[key] = bitmap
	}

	// 6. Read numeric entries
	var numericCount uint32
	if err := read(&numericCount); err != nil {
		return bytesRead, fmt.Errorf("failed to read numeric count: %w", err)
	}

	numeric := make(map[string]*bsi.BSI, numericCount)
	for i := uint32(0); i < numericCount; i++ {
		// Read field name
		var fieldLen uint32
		if err := read(&fieldLen); err != nil {
			return bytesRead, fmt.Errorf("failed to read field name length for entry %d: %w", i, err)
		}

		fieldBytes := make([]byte, fieldLen)
		if _, err := io.ReadFull(r, fieldBytes); err != nil {
			return bytesRead, fmt.Errorf("failed to read field name for entry %d: %w", i, err)
		}
		bytesRead += int64(fieldLen)
		field := string(fieldBytes)

		// Read BSI
		var bsiSliceCount uint32
		if err := read(&bsiSliceCount); err != nil {
			return bytesRead, fmt.Errorf("failed to read BSI slice count for field %s: %w", field, err)
		}

		bsiBytes := make([][]byte, bsiSliceCount)
		for j := uint32(0); j < bsiSliceCount; j++ {
			var sliceSize uint32
			if err := read(&sliceSize); err != nil {
				return bytesRead, fmt.Errorf("failed to read BSI slice %d size for field %s: %w", j, field, err)
			}

			slice := make([]byte, sliceSize)
			if _, err := io.ReadFull(r, slice); err != nil {
				return bytesRead, fmt.Errorf("failed to read BSI slice %d data for field %s: %w", j, field, err)
			}
			bytesRead += int64(sliceSize)
			bsiBytes[j] = slice
		}

		bsiIndex := bsi.NewBSI(bsi.Min64BitSigned, bsi.Max64BitSigned)
		if err := bsiIndex.UnmarshalBinary(bsiBytes); err != nil {
			return bytesRead, fmt.Errorf("failed to deserialize BSI for field %s: %w", field, err)
		}

		numeric[field] = bsiIndex
	}

	// Update index state
	idx.allDocs = allDocs
	idx.categorical = categorical
	idx.numeric = numeric

	return bytesRead, nil
}
