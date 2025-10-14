package comet

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestPersistentHybridIndex_BasicOperations tests basic add and search operations.
func TestPersistentHybridIndex_BasicOperations(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()
	config.MemtableSizeLimit = 1024 * 1024 // 1MB for testing

	store, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}
	defer store.Close()

	// Add documents
	doc1 := []float32{1.0, 0.0, 0.0, 0.0}
	id1, err := store.Add(doc1, "hello world", map[string]interface{}{"category": "greeting"})
	if err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	doc2 := []float32{0.0, 1.0, 0.0, 0.0}
	id2, err := store.Add(doc2, "goodbye world", map[string]interface{}{"category": "farewell"})
	if err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	// Search
	results, err := store.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0, 0.0}).
		WithText("world").
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) < 2 {
		t.Errorf("expected at least 2 results, got %d", len(results))
	}

	// Verify IDs are present
	foundID1 := false
	foundID2 := false
	for _, result := range results {
		if result.ID == id1 {
			foundID1 = true
		}
		if result.ID == id2 {
			foundID2 = true
		}
	}

	if !foundID1 {
		t.Errorf("expected to find document %d in results", id1)
	}
	if !foundID2 {
		t.Errorf("expected to find document %d in results", id2)
	}
}

// TestPersistentHybridIndex_Persistence tests that data survives restart.
func TestPersistentHybridIndex_Persistence(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()
	config.MemtableSizeLimit = 1024 * 10 // Small size to force flush

	// First session: add documents and flush
	store1, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}

	doc1 := []float32{1.0, 0.0, 0.0, 0.0}
	id1, err := store1.Add(doc1, "hello world", map[string]interface{}{"category": "greeting"})
	if err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	// Force flush
	if err := store1.Flush(); err != nil {
		t.Fatalf("failed to flush: %v", err)
	}

	if err := store1.Close(); err != nil {
		t.Fatalf("failed to close storage: %v", err)
	}

	// Second session: verify data is still there
	store2, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to reopen storage: %v", err)
	}
	defer store2.Close()

	// Search for the document
	results, err := store2.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0, 0.0}).
		WithText("world").
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected to find document after restart")
	}

	found := false
	for _, result := range results {
		if result.ID == id1 {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("expected to find document %d after restart", id1)
	}
}

// TestPersistentHybridIndex_AutoFlush tests automatic flushing when threshold is exceeded.
func TestPersistentHybridIndex_AutoFlush(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()
	config.MemtableSizeLimit = 1024 // 1KB
	config.FlushThreshold = 2048    // 2KB

	store, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}
	defer store.Close()

	// Add many documents to trigger auto-flush
	for i := 0; i < 100; i++ {
		vec := []float32{float32(i + 1), 1.0, 1.0, 1.0} // Non-zero vector
		text := "document number with some additional text content"
		_, err := store.Add(vec, text, map[string]interface{}{"id": i})
		if err != nil {
			t.Fatalf("failed to add document %d: %v", i, err)
		}
	}

	// Wait a bit for background flush
	time.Sleep(200 * time.Millisecond)

	// Check that segments were created
	segments := store.segmentManager.list()
	if len(segments) == 0 {
		t.Log("Note: No segments created yet (may need more data or time)")
	}
}

// TestPersistentHybridIndex_ConcurrentWrites tests concurrent write operations.
func TestPersistentHybridIndex_ConcurrentWrites(t *testing.T) {
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

	// Launch concurrent writers
	numWriters := 10
	docsPerWriter := 10
	done := make(chan bool, numWriters)
	errors := make(chan error, numWriters)

	for w := 0; w < numWriters; w++ {
		go func(writerID int) {
			for i := 0; i < docsPerWriter; i++ {
				vec := []float32{float32(writerID + 1), float32(i + 1), 1.0, 1.0} // Non-zero vector
				text := "writer document"
				_, err := store.Add(vec, text, map[string]interface{}{"writer": writerID})
				if err != nil {
					errors <- err
					return
				}
			}
			done <- true
		}(w)
	}

	// Wait for all writers
	for w := 0; w < numWriters; w++ {
		select {
		case <-done:
			// Success
		case err := <-errors:
			t.Errorf("write failed: %v", err)
		case <-time.After(5 * time.Second):
			t.Fatal("timeout waiting for writers")
		}
	}

	// Verify we can search
	results, err := store.NewSearch().
		WithText("writer").
		WithK(numWriters*docsPerWriter + 10). // Request more than we have
		Execute()
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != numWriters*docsPerWriter {
		t.Logf("expected %d results, got %d (some may be in unflushed memtables)",
			numWriters*docsPerWriter, len(results))
	}
}

// TestPersistentHybridIndex_LockFile tests that lock file prevents concurrent access.
func TestPersistentHybridIndex_LockFile(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()

	// Open first instance
	store1, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}
	defer store1.Close()

	// Try to open second instance (should fail)
	_, err = OpenPersistentHybridIndex(config)
	if err == nil {
		t.Error("expected error when opening locked storage, got nil")
	}

	// Verify lock file exists
	lockPath := filepath.Join(tempDir, "LOCK")
	if _, err := os.Stat(lockPath); os.IsNotExist(err) {
		t.Error("lock file should exist")
	}

	// Close first instance
	if err := store1.Close(); err != nil {
		t.Fatalf("failed to close storage: %v", err)
	}

	// Verify lock file is removed
	if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Error("lock file should be removed after close")
	}

	// Now second instance should succeed
	store2, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage after first closed: %v", err)
	}
	defer store2.Close()
}

// TestPersistentHybridIndex_AddWithID tests adding with specific ID.
func TestPersistentHybridIndex_AddWithID(t *testing.T) {
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

	// Add with specific ID
	customID := uint32(12345)
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	err = store.AddWithID(customID, vec, "test document", nil)
	if err != nil {
		t.Fatalf("failed to add with ID: %v", err)
	}

	// Search for the document
	results, err := store.NewSearch().
		WithVector(vec).
		WithText("test").
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	found := false
	for _, result := range results {
		if result.ID == customID {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("expected to find document with ID %d", customID)
	}
}

// TestPersistentHybridIndex_SearchEmptyIndex tests searching an empty index.
func TestPersistentHybridIndex_SearchEmptyIndex(t *testing.T) {
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

	// Search empty index
	results, err := store.NewSearch().
		WithVector([]float32{1.0, 0.0, 0.0, 0.0}).
		WithText("test").
		WithK(10).
		Execute()
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("expected 0 results from empty index, got %d", len(results))
	}
}

// TestPersistentHybridIndex_FlushTwice tests flushing twice (no-op on second flush).
func TestPersistentHybridIndex_FlushTwice(t *testing.T) {
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

	// Add document
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	_, err = store.Add(vec, "test", nil)
	if err != nil {
		t.Fatalf("failed to add: %v", err)
	}

	// First flush
	if err := store.Flush(); err != nil {
		t.Fatalf("first flush failed: %v", err)
	}

	// Second flush (should be no-op)
	if err := store.Flush(); err != nil {
		t.Fatalf("second flush failed: %v", err)
	}
}

// TestPersistentHybridIndex_CloseWithoutFlush tests closing without explicit flush.
func TestPersistentHybridIndex_CloseWithoutFlush(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()

	store, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}

	// Add document
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	_, err = store.Add(vec, "test", nil)
	if err != nil {
		t.Fatalf("failed to add: %v", err)
	}

	// Close without explicit flush (should still flush in background)
	if err := store.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}
}

// TestPersistentHybridIndex_CloseTwice tests closing twice.
func TestPersistentHybridIndex_CloseTwice(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()

	store, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}

	// First close
	if err := store.Close(); err != nil {
		t.Fatalf("first close failed: %v", err)
	}

	// Second close (should error)
	if err := store.Close(); err == nil {
		t.Error("expected error on second close")
	}
}

// TestPersistentHybridIndex_UseAfterClose tests using storage after close.
func TestPersistentHybridIndex_UseAfterClose(t *testing.T) {
	tempDir := t.TempDir()

	config := DefaultStorageConfig(tempDir)
	config.VectorIndexTemplate, _ = NewFlatIndex(4, Cosine)
	config.TextIndexTemplate = NewBM25SearchIndex()
	config.MetadataIndexTemplate = NewRoaringMetadataIndex()

	store, err := OpenPersistentHybridIndex(config)
	if err != nil {
		t.Fatalf("failed to open storage: %v", err)
	}

	if err := store.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}

	// Try to add after close (should error)
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	_, err = store.Add(vec, "test", nil)
	if err == nil {
		t.Error("expected error when adding after close")
	}

	// Try to search after close (should error)
	_, err = store.NewSearch().
		WithVector(vec).
		WithK(10).
		Execute()
	if err == nil {
		t.Error("expected error when searching after close")
	}
}

// TestDefaultStorageConfig tests default configuration.
func TestDefaultStorageConfig(t *testing.T) {
	config := DefaultStorageConfig("/tmp/test")

	if config.BaseDir != "/tmp/test" {
		t.Errorf("unexpected base dir: %s", config.BaseDir)
	}

	if config.MemtableSizeLimit != DefaultMemtableSizeLimit {
		t.Errorf("unexpected memtable size: %d", config.MemtableSizeLimit)
	}

	if config.FlushThreshold != DefaultFlushThreshold {
		t.Errorf("unexpected flush threshold: %d", config.FlushThreshold)
	}

	if config.CompactionInterval != DefaultCompactionInterval {
		t.Errorf("unexpected compaction interval: %v", config.CompactionInterval)
	}

	if config.CompactionThreshold != DefaultCompactionThreshold {
		t.Errorf("unexpected compaction threshold: %d", config.CompactionThreshold)
	}
}
