package comet

import (
	"testing"
)

// TestMemtable_BasicOperations tests basic memtable operations.
func TestMemtable_BasicOperations(t *testing.T) {
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	mt := newMemtable(vecIdx, txtIdx, metaIdx, 10000)

	// Add document
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	id, err := mt.add(vec, "hello world", map[string]interface{}{"key": "value"})
	if err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	if id == 0 {
		t.Error("expected non-zero document ID")
	}

	// Verify count
	if mt.count() != 1 {
		t.Errorf("expected count 1, got %d", mt.count())
	}

	// Add another document
	vec2 := []float32{0.0, 1.0, 0.0, 0.0}
	id2, err := mt.add(vec2, "goodbye world", nil)
	if err != nil {
		t.Fatalf("failed to add second document: %v", err)
	}

	if id2 == id {
		t.Error("expected different document IDs")
	}

	if mt.count() != 2 {
		t.Errorf("expected count 2, got %d", mt.count())
	}
}

// TestMemtable_SizeTracking tests size tracking.
func TestMemtable_SizeTracking(t *testing.T) {
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	mt := newMemtable(vecIdx, txtIdx, metaIdx, 10000)

	initialSize := mt.size()
	if initialSize != 0 {
		t.Errorf("expected initial size 0, got %d", initialSize)
	}

	// Add document
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	_, err := mt.add(vec, "hello world", map[string]interface{}{"key": "value"})
	if err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	afterSize := mt.size()
	if afterSize == 0 {
		t.Error("expected size > 0 after adding document")
	}

	// Size should increase with more documents
	vec2 := []float32{0.0, 1.0, 0.0, 0.0}
	_, err = mt.add(vec2, "another document with more text", nil)
	if err != nil {
		t.Fatalf("failed to add second document: %v", err)
	}

	finalSize := mt.size()
	if finalSize <= afterSize {
		t.Errorf("expected size to increase, got %d -> %d", afterSize, finalSize)
	}
}

// TestMemtable_HasRoomFor tests room checking.
func TestMemtable_HasRoomFor(t *testing.T) {
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	mt := newMemtable(vecIdx, txtIdx, metaIdx, 1000) // Small limit

	vec := []float32{1.0, 0.0, 0.0, 0.0}

	// Small document should fit
	hasRoom := mt.hasRoomFor(vec, "small", nil)
	if !hasRoom {
		t.Error("expected room for small document")
	}

	// Very large document should not fit
	largeText := make([]byte, 20000)
	for i := range largeText {
		largeText[i] = 'a'
	}

	hasRoom = mt.hasRoomFor(vec, string(largeText), nil)
	if hasRoom {
		t.Error("expected no room for large document")
	}
}

// TestMemtable_Freeze tests freezing.
func TestMemtable_Freeze(t *testing.T) {
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	mt := newMemtable(vecIdx, txtIdx, metaIdx, 10000)

	if mt.IsFrozen() {
		t.Error("memtable should not be frozen initially")
	}

	// Add document
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	_, err := mt.add(vec, "hello", nil)
	if err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	// Freeze
	mt.freeze()

	if !mt.IsFrozen() {
		t.Error("memtable should be frozen")
	}

	// Try to add after freeze (should fail)
	_, err = mt.add(vec, "world", nil)
	if err == nil {
		t.Error("expected error when adding to frozen memtable")
	}

	// HasRoomFor should return false when frozen
	hasRoom := mt.hasRoomFor(vec, "test", nil)
	if hasRoom {
		t.Error("frozen memtable should have no room")
	}
}

// TestMemtable_Remove tests document removal.
func TestMemtable_Remove(t *testing.T) {
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	mt := newMemtable(vecIdx, txtIdx, metaIdx, 10000)

	// Add document
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	id, err := mt.add(vec, "hello world", nil)
	if err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	// Remove document
	if err := mt.remove(id); err != nil {
		t.Fatalf("failed to remove document: %v", err)
	}

	// Note: Count doesn't decrease on remove (soft delete)
	// This is consistent with the underlying index behavior
}

// TestMemtable_Flush tests flushing.
func TestMemtable_Flush(t *testing.T) {
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	mt := newMemtable(vecIdx, txtIdx, metaIdx, 10000)

	// Try to flush unfrozen memtable (should fail)
	_, err := mt.flush()
	if err == nil {
		t.Error("expected error when flushing unfrozen memtable")
	}

	// Freeze and flush
	mt.freeze()
	idx, err := mt.flush()
	if err != nil {
		t.Fatalf("failed to flush: %v", err)
	}

	if idx == nil {
		t.Error("flushed index should not be nil")
	}
}

// TestMemtableQueue_BasicOperations tests queue operations.
func TestMemtableQueue_BasicOperations(t *testing.T) {
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	mq := newMemtableQueue(vecIdx, txtIdx, metaIdx, 1000)

	// Initially has one mutable memtable
	if mq.Count() != 1 {
		t.Errorf("expected count 1, got %d", mq.Count())
	}

	// Add document
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	id, err := mq.add(vec, "test", nil)
	if err != nil {
		t.Fatalf("failed to add: %v", err)
	}

	if id == 0 {
		t.Error("expected non-zero ID")
	}
}

// TestMemtableQueue_Rotation tests rotation.
func TestMemtableQueue_Rotation(t *testing.T) {
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	mq := newMemtableQueue(vecIdx, txtIdx, metaIdx, 1000)

	initialCount := mq.Count()
	if initialCount != 1 {
		t.Errorf("expected 1 memtable initially, got %d", initialCount)
	}

	// Rotate
	mq.Rotate()

	afterCount := mq.Count()
	if afterCount != 2 {
		t.Errorf("expected 2 memtables after rotation, got %d", afterCount)
	}

	// Check frozen list
	frozen := mq.listFrozen()
	if len(frozen) != 1 {
		t.Errorf("expected 1 frozen memtable, got %d", len(frozen))
	}

	if !frozen[0].IsFrozen() {
		t.Error("expected first memtable to be frozen")
	}
}

// TestMemtableQueue_AutoRotation tests automatic rotation on full.
func TestMemtableQueue_AutoRotation(t *testing.T) {
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	mq := newMemtableQueue(vecIdx, txtIdx, metaIdx, 100) // Small limit

	// Add documents until rotation happens
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	for i := 0; i < 10; i++ {
		_, err := mq.add(vec, "test document", nil)
		if err != nil {
			t.Fatalf("failed to add: %v", err)
		}
	}

	// Should have rotated
	if mq.Count() < 2 {
		t.Error("expected auto-rotation to have occurred")
	}
}

// TestMemtableQueue_TotalSize tests total size calculation.
func TestMemtableQueue_TotalSize(t *testing.T) {
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	mq := newMemtableQueue(vecIdx, txtIdx, metaIdx, 10000)

	initialSize := mq.totalSize()
	if initialSize != 0 {
		t.Errorf("expected initial size 0, got %d", initialSize)
	}

	// Add documents
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	for i := 0; i < 5; i++ {
		_, err := mq.add(vec, "test document", nil)
		if err != nil {
			t.Fatalf("failed to add: %v", err)
		}
	}

	totalSize := mq.totalSize()
	if totalSize == 0 {
		t.Error("expected total size > 0")
	}
}

// TestMemtableQueue_Remove tests removing memtables from queue.
func TestMemtableQueue_Remove(t *testing.T) {
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	mq := newMemtableQueue(vecIdx, txtIdx, metaIdx, 1000)

	// Rotate to create multiple memtables
	mq.Rotate()
	mq.Rotate()

	initialCount := mq.Count()
	if initialCount < 3 {
		t.Fatalf("expected at least 3 memtables, got %d", initialCount)
	}

	// Get first memtable
	all := mq.list()
	first := all[0]

	// Remove it
	mq.remove(first)

	if mq.Count() != initialCount-1 {
		t.Errorf("expected count %d after remove, got %d", initialCount-1, mq.Count())
	}
}
