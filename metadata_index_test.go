package comet

import (
	"bytes"
	"fmt"
	"io"
	"sort"
	"testing"
)

// TestNewRoaringMetadataIndex tests the creation of a new metadata index
func TestNewRoaringMetadataIndex(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	if idx == nil {
		t.Fatal("NewRoaringMetadataIndex() returned nil")
	}

	if idx.categorical == nil {
		t.Error("categorical map not initialized")
	}

	if idx.numeric == nil {
		t.Error("numeric map not initialized")
	}

	if idx.allDocs == nil {
		t.Error("allDocs bitmap not initialized")
	}
}

// TestMetadataIndexAdd tests adding nodes to the index
func TestMetadataIndexAdd(t *testing.T) {
	tests := []struct {
		name     string
		metadata map[string]interface{}
		wantErr  bool
		errMsg   string
	}{
		{
			name: "valid metadata with all types",
			metadata: map[string]interface{}{
				"category": "electronics",
				"price":    100,
				"rating":   4.5,
				"in_stock": true,
			},
			wantErr: false,
		},
		{
			name: "metadata with int64",
			metadata: map[string]interface{}{
				"user_id": int64(123456789),
				"count":   int64(42),
			},
			wantErr: false,
		},
		{
			name: "metadata with float64",
			metadata: map[string]interface{}{
				"score":  9.99,
				"rating": 4.75,
			},
			wantErr: false,
		},
		{
			name: "metadata with boolean",
			metadata: map[string]interface{}{
				"active":   true,
				"verified": false,
			},
			wantErr: false,
		},
		{
			name: "metadata with string",
			metadata: map[string]interface{}{
				"name":     "Product A",
				"category": "books",
			},
			wantErr: false,
		},
		{
			name: "unsupported type",
			metadata: map[string]interface{}{
				"invalid": []int{1, 2, 3},
			},
			wantErr: true,
			errMsg:  "unsupported type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx := NewRoaringMetadataIndex()
			node := NewMetadataNodeWithID(1, tt.metadata)

			err := idx.Add(*node)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Add() expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Add() unexpected error: %v", err)
				return
			}

			// Verify document was added to allDocs
			if !idx.allDocs.Contains(1) {
				t.Error("Document not added to allDocs bitmap")
			}
		})
	}
}

// TestMetadataIndexAddMultiple tests adding multiple nodes
func TestMetadataIndexAddMultiple(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"category": "electronics", "price": 100}),
		NewMetadataNodeWithID(2, map[string]interface{}{"category": "electronics", "price": 200}),
		NewMetadataNodeWithID(3, map[string]interface{}{"category": "books", "price": 15}),
		NewMetadataNodeWithID(4, map[string]interface{}{"category": "books", "price": 25}),
		NewMetadataNodeWithID(5, map[string]interface{}{"category": "clothing", "price": 50}),
	}

	for _, node := range nodes {
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Failed to add node %d: %v", node.ID(), err)
		}
	}

	// Verify all documents are in allDocs
	if idx.allDocs.GetCardinality() != 5 {
		t.Errorf("Expected 5 documents in allDocs, got %d", idx.allDocs.GetCardinality())
	}

	// Verify categorical index
	electronicsBitmap := idx.categorical["category:electronics"]
	if electronicsBitmap == nil {
		t.Fatal("electronics category bitmap not found")
	}
	if electronicsBitmap.GetCardinality() != 2 {
		t.Errorf("Expected 2 electronics, got %d", electronicsBitmap.GetCardinality())
	}

	// Verify numeric index
	priceBSI := idx.numeric["price"]
	if priceBSI == nil {
		t.Fatal("price numeric index not found")
	}
	existingDocs := priceBSI.GetExistenceBitmap()
	if existingDocs.GetCardinality() != 5 {
		t.Errorf("Expected 5 documents with price, got %d", existingDocs.GetCardinality())
	}
}

// TestMetadataIndexRemove tests removing nodes from the index
func TestMetadataIndexRemove(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add documents
	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"category": "electronics", "price": 100}),
		NewMetadataNodeWithID(2, map[string]interface{}{"category": "electronics", "price": 200}),
		NewMetadataNodeWithID(3, map[string]interface{}{"category": "books", "price": 15}),
	}

	for _, node := range nodes {
		idx.Add(*node)
	}

	// Verify initial state
	if idx.allDocs.GetCardinality() != 3 {
		t.Fatalf("Expected 3 documents initially, got %d", idx.allDocs.GetCardinality())
	}

	// Remove document 1
	err := idx.Remove(*nodes[0])
	if err != nil {
		t.Fatalf("Remove() error: %v", err)
	}

	// Verify document removed from allDocs
	if idx.allDocs.Contains(1) {
		t.Error("Document 1 still in allDocs after removal")
	}

	if idx.allDocs.GetCardinality() != 2 {
		t.Errorf("Expected 2 documents after removal, got %d", idx.allDocs.GetCardinality())
	}

	// Verify removed from categorical index
	electronicsBitmap := idx.categorical["category:electronics"]
	if electronicsBitmap.Contains(1) {
		t.Error("Document 1 still in electronics category after removal")
	}

	// Verify removed from numeric index
	priceBSI := idx.numeric["price"]
	existingDocs := priceBSI.GetExistenceBitmap()
	if existingDocs.Contains(1) {
		t.Error("Document 1 still in price index after removal")
	}
}

// TestMetadataIndexRemoveNonexistent tests removing a non-existent node
func TestMetadataIndexRemoveNonexistent(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	node := NewMetadataNodeWithID(999, map[string]interface{}{"category": "test"})

	// Should not error when removing non-existent document
	err := idx.Remove(*node)
	if err != nil {
		t.Errorf("Remove() unexpected error for non-existent document: %v", err)
	}
}

// TestMetadataIndexFlush tests the flush operation
func TestMetadataIndexFlush(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add some data
	node := NewMetadataNodeWithID(1, map[string]interface{}{"category": "test"})
	idx.Add(*node)

	// Flush should succeed (no-op)
	err := idx.Flush()
	if err != nil {
		t.Errorf("Flush() unexpected error: %v", err)
	}

	// Data should still be present
	if !idx.allDocs.Contains(1) {
		t.Error("Data lost after Flush()")
	}
}

// TestMetadataIndexCategoricalStorage tests categorical field storage
func TestMetadataIndexCategoricalStorage(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add nodes with different categorical values
	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"color": "red"}),
		NewMetadataNodeWithID(2, map[string]interface{}{"color": "blue"}),
		NewMetadataNodeWithID(3, map[string]interface{}{"color": "red"}),
		NewMetadataNodeWithID(4, map[string]interface{}{"color": "green"}),
	}

	for _, node := range nodes {
		idx.Add(*node)
	}

	// Verify separate bitmaps for each value
	redBitmap := idx.categorical["color:red"]
	if redBitmap == nil || redBitmap.GetCardinality() != 2 {
		t.Errorf("Expected 2 red items, got %v", redBitmap)
	}

	blueBitmap := idx.categorical["color:blue"]
	if blueBitmap == nil || blueBitmap.GetCardinality() != 1 {
		t.Errorf("Expected 1 blue item, got %v", blueBitmap)
	}

	greenBitmap := idx.categorical["color:green"]
	if greenBitmap == nil || greenBitmap.GetCardinality() != 1 {
		t.Errorf("Expected 1 green item, got %v", greenBitmap)
	}
}

// TestMetadataIndexNumericStorage tests numeric field storage
func TestMetadataIndexNumericStorage(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add nodes with numeric values
	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"score": 100}),
		NewMetadataNodeWithID(2, map[string]interface{}{"score": 200}),
		NewMetadataNodeWithID(3, map[string]interface{}{"score": 150}),
	}

	for _, node := range nodes {
		idx.Add(*node)
	}

	// Verify BSI created
	scoreBSI := idx.numeric["score"]
	if scoreBSI == nil {
		t.Fatal("score BSI not created")
	}

	// Verify all documents are in the existence bitmap
	existingDocs := scoreBSI.GetExistenceBitmap()
	if existingDocs.GetCardinality() != 3 {
		t.Errorf("Expected 3 documents with score, got %d", existingDocs.GetCardinality())
	}
}

// TestMetadataIndexBooleanStorage tests boolean field storage
func TestMetadataIndexBooleanStorage(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add nodes with boolean values
	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"active": true}),
		NewMetadataNodeWithID(2, map[string]interface{}{"active": false}),
		NewMetadataNodeWithID(3, map[string]interface{}{"active": true}),
	}

	for _, node := range nodes {
		idx.Add(*node)
	}

	// Booleans stored as categorical strings
	trueBitmap := idx.categorical["active:true"]
	if trueBitmap == nil || trueBitmap.GetCardinality() != 2 {
		t.Errorf("Expected 2 active:true items, got %v", trueBitmap)
	}

	falseBitmap := idx.categorical["active:false"]
	if falseBitmap == nil || falseBitmap.GetCardinality() != 1 {
		t.Errorf("Expected 1 active:false item, got %v", falseBitmap)
	}
}

// TestMetadataIndexFloatPrecision tests float to int conversion
func TestMetadataIndexFloatPrecision(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add node with float value
	node := NewMetadataNodeWithID(1, map[string]interface{}{"rating": 4.55})
	idx.Add(*node)

	// Float should be converted to int (455 = 4.55 * 100)
	ratingBSI := idx.numeric["rating"]
	if ratingBSI == nil {
		t.Fatal("rating BSI not created")
	}

	// We can't directly check the stored value, but we can verify it exists
	existingDocs := ratingBSI.GetExistenceBitmap()
	if !existingDocs.Contains(1) {
		t.Error("Document 1 not in rating index")
	}
}

// TestMetadataIndexMixedFields tests nodes with mixed field types
func TestMetadataIndexMixedFields(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add nodes with various field combinations
	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{
			"name":     "Product A",
			"price":    100,
			"rating":   4.5,
			"in_stock": true,
		}),
		NewMetadataNodeWithID(2, map[string]interface{}{
			"name":  "Product B",
			"price": 200,
		}),
		NewMetadataNodeWithID(3, map[string]interface{}{
			"name":     "Product C",
			"in_stock": false,
		}),
	}

	for _, node := range nodes {
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Failed to add node %d: %v", node.ID(), err)
		}
	}

	// Verify categorical fields
	if len(idx.categorical) == 0 {
		t.Error("No categorical fields stored")
	}

	// Verify numeric fields
	if len(idx.numeric) == 0 {
		t.Error("No numeric fields stored")
	}

	// All documents should be tracked
	if idx.allDocs.GetCardinality() != 3 {
		t.Errorf("Expected 3 documents, got %d", idx.allDocs.GetCardinality())
	}
}

// TestMetadataIndexConcurrentAdd tests concurrent additions
func TestMetadataIndexConcurrentAdd(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	const numGoroutines = 10
	const nodesPerGoroutine = 100

	errCh := make(chan error, numGoroutines)
	doneCh := make(chan bool, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(base uint32) {
			for i := uint32(0); i < nodesPerGoroutine; i++ {
				id := base*nodesPerGoroutine + i
				node := NewMetadataNodeWithID(id, map[string]interface{}{
					"category": fmt.Sprintf("cat%d", id%5),
					"value":    int(id),
				})
				if err := idx.Add(*node); err != nil {
					errCh <- err
					return
				}
			}
			doneCh <- true
		}(uint32(g))
	}

	// Wait for all goroutines
	for i := 0; i < numGoroutines; i++ {
		select {
		case err := <-errCh:
			t.Fatalf("Concurrent add error: %v", err)
		case <-doneCh:
			// Success
		}
	}

	// Verify all documents were added
	expected := uint64(numGoroutines * nodesPerGoroutine)
	if idx.allDocs.GetCardinality() != expected {
		t.Errorf("Expected %d documents, got %d", expected, idx.allDocs.GetCardinality())
	}
}

// Helper function to extract and sort IDs from results
func extractIDs(results []MetadataResult) []uint32 {
	ids := make([]uint32, len(results))
	for i, r := range results {
		ids[i] = r.GetId()
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

// Benchmark tests
func BenchmarkMetadataIndexAdd(b *testing.B) {
	idx := NewRoaringMetadataIndex()

	metadata := map[string]interface{}{
		"category": "electronics",
		"brand":    "Apple",
		"price":    1000,
		"rating":   4.5,
		"in_stock": true,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node := NewMetadataNodeWithID(uint32(i), metadata)
		_ = idx.Add(*node)
	}
}

func BenchmarkMetadataIndexAddWithManyFields(b *testing.B) {
	idx := NewRoaringMetadataIndex()

	metadata := map[string]interface{}{
		"field1":  "value1",
		"field2":  "value2",
		"field3":  100,
		"field4":  200,
		"field5":  4.5,
		"field6":  3.8,
		"field7":  true,
		"field8":  false,
		"field9":  "value9",
		"field10": 500,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node := NewMetadataNodeWithID(uint32(i), metadata)
		_ = idx.Add(*node)
	}
}

func BenchmarkMetadataIndexRemove(b *testing.B) {
	idx := NewRoaringMetadataIndex()

	// Pre-populate index
	for i := 0; i < b.N; i++ {
		node := NewMetadataNodeWithID(uint32(i), map[string]interface{}{
			"category": "test",
			"value":    i,
		})
		idx.Add(*node)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node := NewMetadataNodeWithID(uint32(i), nil)
		idx.Remove(*node)
	}
}

func BenchmarkMetadataIndexConcurrentAdd(b *testing.B) {
	idx := NewRoaringMetadataIndex()

	metadata := map[string]interface{}{
		"category": "electronics",
		"price":    1000,
	}

	b.RunParallel(func(pb *testing.PB) {
		i := uint32(0)
		for pb.Next() {
			node := NewMetadataNodeWithID(i, metadata)
			idx.Add(*node)
			i++
		}
	})
}

// TestMetadataIndexWriteTo tests serialization of the metadata index
func TestMetadataIndexWriteTo(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add some documents with various metadata types
	idx.Add(*NewMetadataNodeWithID(1, map[string]interface{}{
		"category": "electronics",
		"price":    999,
		"rating":   4.5,
		"active":   true,
	}))
	idx.Add(*NewMetadataNodeWithID(2, map[string]interface{}{
		"category": "books",
		"price":    20,
		"rating":   5.0,
		"active":   false,
	}))
	idx.Add(*NewMetadataNodeWithID(3, map[string]interface{}{
		"category": "electronics",
		"price":    1500,
		"rating":   4.0,
		"active":   true,
	}))

	// Serialize to buffer
	var buf bytes.Buffer
	n, err := idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	if n <= 0 {
		t.Errorf("WriteTo() returned %d bytes, expected > 0", n)
	}

	// Verify buffer has data
	if buf.Len() == 0 {
		t.Error("WriteTo() wrote no data to buffer")
	}

	// Verify magic number
	magic := buf.Bytes()[:4]
	if string(magic) != "MTIX" {
		t.Errorf("Invalid magic number: got %s, want MTIX", string(magic))
	}
}

// TestMetadataIndexReadFrom tests deserialization of the metadata index
func TestMetadataIndexReadFrom(t *testing.T) {
	// Create and populate original index
	original := NewRoaringMetadataIndex()

	original.Add(*NewMetadataNodeWithID(1, map[string]interface{}{
		"category": "electronics",
		"price":    999,
		"rating":   4.5,
		"active":   true,
	}))
	original.Add(*NewMetadataNodeWithID(2, map[string]interface{}{
		"category": "books",
		"price":    20,
		"rating":   5.0,
		"active":   false,
	}))
	original.Add(*NewMetadataNodeWithID(3, map[string]interface{}{
		"category": "electronics",
		"price":    1500,
		"rating":   4.0,
		"active":   true,
	}))

	// Serialize
	var buf bytes.Buffer
	_, err := original.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// Create new index and deserialize
	restored := NewRoaringMetadataIndex()

	n, err := restored.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	if n <= 0 {
		t.Errorf("ReadFrom() returned %d bytes, expected > 0", n)
	}

	// Verify allDocs match
	if original.allDocs.GetCardinality() != restored.allDocs.GetCardinality() {
		t.Errorf("allDocs cardinality mismatch: got %d, want %d",
			restored.allDocs.GetCardinality(), original.allDocs.GetCardinality())
	}

	// Verify categorical entries match
	if len(original.categorical) != len(restored.categorical) {
		t.Errorf("categorical count mismatch: got %d, want %d",
			len(restored.categorical), len(original.categorical))
	}

	// Verify numeric entries match
	if len(original.numeric) != len(restored.numeric) {
		t.Errorf("numeric count mismatch: got %d, want %d",
			len(restored.numeric), len(original.numeric))
	}
}

// TestMetadataIndexSerializationRoundTrip tests that serialization and deserialization preserve data
func TestMetadataIndexSerializationRoundTrip(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add documents with various metadata
	for i := uint32(1); i <= 10; i++ {
		idx.Add(*NewMetadataNodeWithID(i, map[string]interface{}{
			"category": fmt.Sprintf("cat%d", i%3),
			"price":    int(i * 100),
			"rating":   float64(i) / 2.0,
			"active":   i%2 == 0,
		}))
	}

	// Perform a search before serialization
	resultsBefore, err := idx.NewSearch().
		WithFilters(Eq("category", "cat1"), Gte("price", 300)).
		Execute()
	if err != nil {
		t.Fatalf("Search before serialization error: %v", err)
	}

	// Serialize
	var buf bytes.Buffer
	_, err = idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// Deserialize into new index
	idx2 := NewRoaringMetadataIndex()

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Perform same search after deserialization
	resultsAfter, err := idx2.NewSearch().
		WithFilters(Eq("category", "cat1"), Gte("price", 300)).
		Execute()
	if err != nil {
		t.Fatalf("Search after deserialization error: %v", err)
	}

	// Results should be identical
	if len(resultsBefore) != len(resultsAfter) {
		t.Errorf("Result count mismatch: before=%d, after=%d", len(resultsBefore), len(resultsAfter))
	}

	// Sort results for comparison
	sortResults := func(results []MetadataResult) {
		sort.Slice(results, func(i, j int) bool {
			return results[i].Node.ID() < results[j].Node.ID()
		})
	}

	sortResults(resultsBefore)
	sortResults(resultsAfter)

	for i := range resultsBefore {
		if resultsBefore[i].Node.ID() != resultsAfter[i].Node.ID() {
			t.Errorf("Result %d ID mismatch: before=%d, after=%d", i, resultsBefore[i].Node.ID(), resultsAfter[i].Node.ID())
		}
	}
}

// TestMetadataIndexSerializationEmpty tests serialization of an empty index
func TestMetadataIndexSerializationEmpty(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Serialize empty index
	var buf bytes.Buffer
	n, err := idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	if n <= 0 {
		t.Errorf("WriteTo() returned %d bytes for empty index, expected > 0", n)
	}

	// Deserialize
	idx2 := NewRoaringMetadataIndex()

	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Verify restored index is also empty
	if idx2.allDocs.GetCardinality() != 0 {
		t.Errorf("Expected 0 documents in restored index, got %d", idx2.allDocs.GetCardinality())
	}

	if len(idx2.categorical) != 0 {
		t.Errorf("Expected 0 categorical entries, got %d", len(idx2.categorical))
	}

	if len(idx2.numeric) != 0 {
		t.Errorf("Expected 0 numeric entries, got %d", len(idx2.numeric))
	}
}

// TestMetadataIndexReadFromInvalidData tests error handling for invalid serialized data
func TestMetadataIndexReadFromInvalidData(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() *bytes.Buffer
		wantErr string
	}{
		{
			name: "invalid magic number",
			setup: func() *bytes.Buffer {
				buf := bytes.NewBuffer([]byte("XXXX"))
				return buf
			},
			wantErr: "invalid magic number",
		},
		{
			name: "unsupported version",
			setup: func() *bytes.Buffer {
				var buf bytes.Buffer
				// Write valid magic
				buf.Write([]byte("MTIX"))
				// Write invalid version
				buf.Write([]byte{99, 0, 0, 0}) // version 99
				return &buf
			},
			wantErr: "unsupported version",
		},
		{
			name: "truncated data",
			setup: func() *bytes.Buffer {
				buf := bytes.NewBuffer([]byte("MT"))
				return buf
			},
			wantErr: "failed to read magic number",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := tt.setup()

			idx := NewRoaringMetadataIndex()

			_, err := idx.ReadFrom(buf)
			if err == nil {
				t.Errorf("ReadFrom() expected error containing '%s', got nil", tt.wantErr)
				return
			}

			// Check if error message contains expected substring
			if tt.wantErr != "" {
				errMsg := err.Error()
				found := false
				for i := 0; i <= len(errMsg)-len(tt.wantErr); i++ {
					if errMsg[i:i+len(tt.wantErr)] == tt.wantErr {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("ReadFrom() error = %v, want error containing '%s'", err, tt.wantErr)
				}
			}
		})
	}
}

// TestMetadataIndexSerializationComplexQueries tests serialization with complex queries
func TestMetadataIndexSerializationComplexQueries(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add documents
	for i := uint32(1); i <= 100; i++ {
		idx.Add(*NewMetadataNodeWithID(i, map[string]interface{}{
			"category": fmt.Sprintf("cat%d", i%5),
			"price":    int(i * 10),
			"rating":   float64(i%10) / 2.0,
			"active":   i%3 == 0,
			"brand":    fmt.Sprintf("brand%d", i%7),
		}))
	}

	// Serialize
	var buf bytes.Buffer
	_, err := idx.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error: %v", err)
	}

	// Deserialize
	idx2 := NewRoaringMetadataIndex()
	_, err = idx2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error: %v", err)
	}

	// Test various complex queries
	testCases := []struct {
		name    string
		filters []Filter
	}{
		{
			name:    "range query",
			filters: []Filter{Range("price", 100, 500)},
		},
		{
			name:    "in query",
			filters: []Filter{In("category", "cat1", "cat2", "cat3")},
		},
		{
			name:    "not in query",
			filters: []Filter{NotIn("brand", "brand1", "brand2")},
		},
		{
			name: "multiple filters",
			filters: []Filter{
				Eq("active", "true"),
				Gte("price", 300),
				Lt("price", 700),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Search on original
			resultsBefore, err := idx.NewSearch().WithFilters(tc.filters...).Execute()
			if err != nil {
				t.Fatalf("Search on original error: %v", err)
			}

			// Search on restored
			resultsAfter, err := idx2.NewSearch().WithFilters(tc.filters...).Execute()
			if err != nil {
				t.Fatalf("Search on restored error: %v", err)
			}

			// Compare results
			if len(resultsBefore) != len(resultsAfter) {
				t.Errorf("Result count mismatch: before=%d, after=%d", len(resultsBefore), len(resultsAfter))
			}

			// Sort and compare IDs
			sortResults := func(results []MetadataResult) {
				sort.Slice(results, func(i, j int) bool {
					return results[i].Node.ID() < results[j].Node.ID()
				})
			}

			sortResults(resultsBefore)
			sortResults(resultsAfter)

			for i := range resultsBefore {
				if resultsBefore[i].Node.ID() != resultsAfter[i].Node.ID() {
					t.Errorf("Result %d ID mismatch: before=%d, after=%d", i, resultsBefore[i].Node.ID(), resultsAfter[i].Node.ID())
				}
			}
		})
	}
}

// errorWriterMetadata is a writer that always returns an error
type errorWriterMetadata struct{}

func (e errorWriterMetadata) Write(p []byte) (n int, err error) {
	return 0, io.ErrUnexpectedEOF
}

// TestMetadataIndexWriteToError tests error handling during write operations
func TestMetadataIndexWriteToError(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add some data
	idx.Add(*NewMetadataNodeWithID(1, map[string]interface{}{
		"category": "test",
		"price":    100,
	}))

	// Try to write to an error writer
	var errWriter errorWriterMetadata
	_, err := idx.WriteTo(errWriter)
	if err == nil {
		t.Error("WriteTo() expected error when writing to error writer, got nil")
	}
}
