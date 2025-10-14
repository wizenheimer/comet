package comet

import (
	"os"
	"path/filepath"
	"testing"
)

// TestStorageProvider_LockFile tests lock file acquisition and release.
func TestStorageProvider_LockFile(t *testing.T) {
	tempDir := t.TempDir()

	// Create first provider
	provider1, err := newStorageProvider(tempDir)
	if err != nil {
		t.Fatalf("failed to create first provider: %v", err)
	}

	// Verify lock file exists
	lockPath := filepath.Join(tempDir, "LOCK")
	if _, err := os.Stat(lockPath); os.IsNotExist(err) {
		t.Error("lock file should exist")
	}

	// Try to create second provider (should fail)
	_, err = newStorageProvider(tempDir)
	if err == nil {
		t.Error("expected error when acquiring locked directory")
	}

	// Close first provider
	if err := provider1.close(); err != nil {
		t.Fatalf("failed to close provider: %v", err)
	}

	// Verify lock file is removed
	if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Error("lock file should be removed after close")
	}

	// Now second provider should succeed
	provider2, err := newStorageProvider(tempDir)
	if err != nil {
		t.Fatalf("failed to create provider after first closed: %v", err)
	}
	defer provider2.close()
}

// TestStorageProvider_SegmentIDGeneration tests segment ID generation.
func TestStorageProvider_SegmentIDGeneration(t *testing.T) {
	tempDir := t.TempDir()

	provider, err := newStorageProvider(tempDir)
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}
	defer provider.close()

	// Generate sequential IDs
	id1 := provider.nextSegmentID()
	id2 := provider.nextSegmentID()
	id3 := provider.nextSegmentID()

	if id1 >= id2 || id2 >= id3 {
		t.Errorf("segment IDs should be sequential: %d, %d, %d", id1, id2, id3)
	}
}

// TestStorageProvider_SegmentPaths tests segment path generation.
func TestStorageProvider_SegmentPaths(t *testing.T) {
	tempDir := t.TempDir()

	provider, err := newStorageProvider(tempDir)
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}
	defer provider.close()

	segmentID := uint64(42)
	hybrid, vector, _, _ := provider.segmentPaths(segmentID)

	// Verify paths are in base directory
	if filepath.Dir(hybrid) != tempDir {
		t.Errorf("hybrid path not in base dir: %s", hybrid)
	}

	// Verify filenames contain segment ID
	if !filepath.IsAbs(hybrid) {
		t.Error("path should be absolute")
	}

	expectedHybrid := filepath.Join(tempDir, "hybrid_000042.bin.gz")
	if hybrid != expectedHybrid {
		t.Errorf("expected hybrid path %s, got %s", expectedHybrid, hybrid)
	}

	expectedVector := filepath.Join(tempDir, "vector_000042.bin.gz")
	if vector != expectedVector {
		t.Errorf("expected vector path %s, got %s", expectedVector, vector)
	}
}

// TestStorageProvider_ListSegments tests listing segments.
func TestStorageProvider_ListSegments(t *testing.T) {
	tempDir := t.TempDir()

	provider, err := newStorageProvider(tempDir)
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}
	defer provider.close()

	// Initially no segments
	segments, err := provider.listSegments()
	if err != nil {
		t.Fatalf("failed to list segments: %v", err)
	}
	if len(segments) != 0 {
		t.Errorf("expected 0 segments, got %d", len(segments))
	}

	// Create some segment files
	for i := 1; i <= 3; i++ {
		hybrid, _, _, _ := provider.segmentPaths(uint64(i))
		if err := os.WriteFile(hybrid, []byte("test"), 0644); err != nil {
			t.Fatalf("failed to create segment file: %v", err)
		}
	}

	// List segments again
	segments, err = provider.listSegments()
	if err != nil {
		t.Fatalf("failed to list segments: %v", err)
	}
	if len(segments) != 3 {
		t.Errorf("expected 3 segments, got %d", len(segments))
	}

	// Verify segments are sorted
	for i := 1; i < len(segments); i++ {
		if segments[i-1] >= segments[i] {
			t.Errorf("segments not sorted: %v", segments)
		}
	}
}

// TestStorageProvider_DeleteSegment tests segment deletion.
func TestStorageProvider_DeleteSegment(t *testing.T) {
	tempDir := t.TempDir()

	provider, err := newStorageProvider(tempDir)
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}
	defer provider.close()

	// Create segment files
	segmentID := uint64(1)
	hybrid, vector, text, metadata := provider.segmentPaths(segmentID)

	files := []string{hybrid, vector, text, metadata}
	for _, file := range files {
		if err := os.WriteFile(file, []byte("test"), 0644); err != nil {
			t.Fatalf("failed to create file: %v", err)
		}
	}

	// Verify files exist
	for _, file := range files {
		if _, err := os.Stat(file); os.IsNotExist(err) {
			t.Errorf("file should exist: %s", file)
		}
	}

	// Delete segment
	if err := provider.deleteSegment(segmentID); err != nil {
		t.Fatalf("failed to delete segment: %v", err)
	}

	// Verify files are deleted
	for _, file := range files {
		if _, err := os.Stat(file); !os.IsNotExist(err) {
			t.Errorf("file should be deleted: %s", file)
		}
	}
}

// TestStorageProvider_InitSegmentCounter tests segment counter initialization.
func TestStorageProvider_InitSegmentCounter(t *testing.T) {
	tempDir := t.TempDir()

	// Create some segment files manually
	for i := 1; i <= 5; i++ {
		filename := filepath.Join(tempDir, "hybrid_00000"+string(rune('0'+i))+".bin.gz")
		if err := os.WriteFile(filename, []byte("test"), 0644); err != nil {
			t.Fatalf("failed to create file: %v", err)
		}
	}

	// Create provider (should init counter from existing files)
	provider, err := newStorageProvider(tempDir)
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}
	defer provider.close()

	// Next ID should be > 5
	nextID := provider.nextSegmentID()
	if nextID <= 5 {
		t.Errorf("expected next ID > 5, got %d", nextID)
	}
}
