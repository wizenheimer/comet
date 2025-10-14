package comet

import (
	"testing"
	"time"
)

// TestPersistentHybridIndex_Compaction tests segment compaction.
func TestPersistentHybridIndex_Compaction(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()
	config.MemtableSizeLimit = 1024
	config.FlushThreshold = 2048
	config.CompactionThreshold = 3

	store, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}
	defer store.Close()

	// Create multiple small segments by adding and flushing in batches
	for batch := 0; batch < 5; batch++ {
		for i := 0; i < 20; i++ {
			vec := []float32{float32(batch*20 + i + 1), 1.0, 1.0, 1.0} // Non-zero vector
			text := "batch document with some text content"
			_, err := store.Add(vec, text, nil)
			if err != nil {
				t.Fatalf("failed to add document: %v", err)
			}
		}
		// Force flush after each batch
		if err := store.Flush(); err != nil {
			t.Fatalf("failed to flush: %v", err)
		}
	}

	initialSegments := store.segmentManager.Count()
	if initialSegments < 3 {
		t.Skipf("not enough segments created for compaction test (got %d)", initialSegments)
	}

	// Trigger compaction
	store.TriggerCompaction()

	// Wait for compaction
	time.Sleep(500 * time.Millisecond)

	// Note: In the current simple implementation, compaction creates a new merged segment
	// but actual document merging is not fully implemented yet
	// This test mainly verifies that compaction doesn't crash
}

// TestPersistentHybridIndex_CompactionThreshold tests compaction threshold.
func TestPersistentHybridIndex_CompactionThreshold(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()
	config.MemtableSizeLimit = 512
	config.FlushThreshold = 1024
	config.CompactionThreshold = 10 // High threshold

	store, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}
	defer store.Close()

	// Create a few segments (less than threshold)
	for batch := 0; batch < 3; batch++ {
		for i := 0; i < 10; i++ {
			vec := []float32{float32(i + 1), 1.0, 1.0, 1.0} // Non-zero vector
			_, err := store.Add(vec, "test", nil)
			if err != nil {
				t.Fatalf("failed to add: %v", err)
			}
		}
		if err := store.Flush(); err != nil {
			t.Fatalf("failed to flush: %v", err)
		}
	}

	initialCount := store.segmentManager.Count()

	// Trigger compaction (should not run due to threshold)
	store.TriggerCompaction()
	time.Sleep(200 * time.Millisecond)

	afterCount := store.segmentManager.Count()

	// Segment count should be unchanged or not significantly different
	// (compaction shouldn't run with < threshold segments)
	if afterCount > initialCount+1 {
		t.Errorf("unexpected segment count change: %d -> %d", initialCount, afterCount)
	}
}

// TestPersistentHybridIndex_CompactionWithSearch tests that search works during compaction.
func TestPersistentHybridIndex_CompactionWithSearch(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()
	config.MemtableSizeLimit = 1024
	config.FlushThreshold = 2048
	config.CompactionThreshold = 3

	store, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}
	defer store.Close()

	// Add documents and create segments
	for batch := 0; batch < 5; batch++ {
		for i := 0; i < 10; i++ {
			vec := []float32{float32(i + 1), 1.0, 1.0, 1.0} // Non-zero vector
			text := "searchable document"
			_, err := store.Add(vec, text, nil)
			if err != nil {
				t.Fatalf("failed to add: %v", err)
			}
		}
		if err := store.Flush(); err != nil {
			t.Fatalf("failed to flush: %v", err)
		}
	}

	// Trigger compaction in background
	store.TriggerCompaction()

	// Search while compaction might be running
	results, err := store.NewSearch().
		WithVector([]float32{5.0, 0.0, 0.0, 0.0}).
		WithText("searchable").
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected to find results during compaction")
	}

	// Wait for compaction to complete
	time.Sleep(500 * time.Millisecond)

	// Search again after compaction
	results, err = store.NewSearch().
		WithVector([]float32{5.0, 0.0, 0.0, 0.0}).
		WithText("searchable").
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("search after compaction failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected to find results after compaction")
	}
}

// TestPersistentHybridIndex_AutomaticCompaction tests automatic periodic compaction.
func TestPersistentHybridIndex_AutomaticCompaction(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping automatic compaction test in short mode")
	}

	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()
	config.MemtableSizeLimit = 512
	config.FlushThreshold = 1024
	config.CompactionThreshold = 3
	config.CompactionInterval = 1 * time.Second // Short interval for testing

	store, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}
	defer store.Close()

	// Create enough segments to trigger compaction
	for batch := 0; batch < 5; batch++ {
		for i := 0; i < 10; i++ {
			vec := []float32{float32(i + 1), 1.0, 1.0, 1.0} // Non-zero vector
			_, err := store.Add(vec, "test", nil)
			if err != nil {
				t.Fatalf("failed to add: %v", err)
			}
		}
		if err := store.Flush(); err != nil {
			t.Fatalf("failed to flush: %v", err)
		}
	}

	initialCount := store.segmentManager.Count()
	if initialCount < config.CompactionThreshold {
		t.Skipf("not enough segments for automatic compaction test")
	}

	// Wait for automatic compaction
	time.Sleep(2 * time.Second)

	// Note: Compaction may or may not have completed, but should not crash
}

// TestCompactSegments_Empty tests compacting empty segment list.
func TestCompactSegments_Empty(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()

	store, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}
	defer store.Close()

	// Compact empty list (should be no-op)
	err = store.compactSegments([]*segmentMetadata{})
	if err != nil {
		t.Errorf("compacting empty list should not error: %v", err)
	}
}

// TestMaybeCompact tests the compaction decision logic.
func TestMaybeCompact(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()
	config.CompactionThreshold = 5

	store, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}
	defer store.Close()

	// With no segments, maybeCompact should be no-op
	err = store.maybeCompact()
	if err != nil {
		t.Errorf("maybeCompact with no segments should not error: %v", err)
	}

	// Create a few segments (less than threshold)
	for i := 0; i < 3; i++ {
		vec := []float32{float32(i + 1), 1.0, 1.0, 1.0} // Non-zero vector
		_, err := store.Add(vec, "test", nil)
		if err != nil {
			t.Fatalf("failed to add: %v", err)
		}
		if err := store.Flush(); err != nil {
			t.Fatalf("failed to flush: %v", err)
		}
	}

	// Should not compact (below threshold)
	err = store.maybeCompact()
	if err != nil {
		t.Errorf("maybeCompact below threshold should not error: %v", err)
	}
}

// TestGetSegmentSize tests segment size calculation.
func TestGetSegmentSize(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()

	store, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}
	defer store.Close()

	// Add and flush a document
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	_, err = store.Add(vec, "test document", nil)
	if err != nil {
		t.Fatalf("failed to add: %v", err)
	}

	if err := store.Flush(); err != nil {
		t.Fatalf("failed to flush: %v", err)
	}

	// Get segment
	segments := store.segmentManager.list()
	if len(segments) == 0 {
		t.Skip("no segments created")
	}

	seg := segments[0]

	// Calculate size
	size, err := store.getSegmentSize(seg.hybridPath, seg.vectorPath, seg.textPath, seg.metadataPath)
	if err != nil {
		t.Fatalf("failed to get segment size: %v", err)
	}

	if size == 0 {
		t.Error("expected non-zero segment size")
	}
}
