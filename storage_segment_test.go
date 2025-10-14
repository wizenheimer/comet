package comet

import (
	"compress/gzip"
	"os"
	"testing"
)

// TestSegmentMetadata_GetIndex tests lazy loading of segment index.
func TestSegmentMetadata_GetIndex(t *testing.T) {
	tempDir := t.TempDir()

	// Create test index
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	idx := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Add some test data
	vec := []float32{1.0, 0.0, 0.0, 0.0}
	_, err := idx.Add(vec, "test document", map[string]interface{}{"key": "value"})
	if err != nil {
		t.Fatalf("failed to add to index: %v", err)
	}

	// Write to files
	hybridPath := tempDir + "/hybrid_test.bin.gz"
	vectorPath := tempDir + "/vector_test.bin.gz"
	textPath := tempDir + "/text_test.bin.gz"
	metadataPath := tempDir + "/metadata_test.bin.gz"

	hybridFile, _ := os.Create(hybridPath)
	vectorFile, _ := os.Create(vectorPath)
	textFile, _ := os.Create(textPath)
	metadataFile, _ := os.Create(metadataPath)

	hybridGz := gzip.NewWriter(hybridFile)
	vectorGz := gzip.NewWriter(vectorFile)
	textGz := gzip.NewWriter(textFile)
	metadataGz := gzip.NewWriter(metadataFile)

	if err := idx.WriteTo(hybridGz, vectorGz, textGz, metadataGz); err != nil {
		t.Fatalf("failed to write index: %v", err)
	}

	hybridGz.Close()
	vectorGz.Close()
	textGz.Close()
	metadataGz.Close()
	hybridFile.Close()
	vectorFile.Close()
	textFile.Close()
	metadataFile.Close()

	// Create segment metadata
	segment := newSegmentMetadata(1, hybridPath, vectorPath, textPath, metadataPath)

	// Verify index is not loaded initially
	if segment.cachedIndex != nil {
		t.Error("index should not be cached initially")
	}

	// Get index (should load from disk)
	vecIdx2, _ := NewFlatIndex(4, Cosine)
	txtIdx2 := NewBM25SearchIndex()
	metaIdx2 := NewRoaringMetadataIndex()

	loadedIdx, err := segment.getIndex(vecIdx2, txtIdx2, metaIdx2)
	if err != nil {
		t.Fatalf("failed to get index: %v", err)
	}

	if loadedIdx == nil {
		t.Error("loaded index should not be nil")
	}

	// Verify index is now cached
	if segment.cachedIndex == nil {
		t.Error("index should be cached after load")
	}

	// Get index again (should return cached)
	loadedIdx2, err := segment.getIndex(vecIdx2, txtIdx2, metaIdx2)
	if err != nil {
		t.Fatalf("failed to get cached index: %v", err)
	}

	if loadedIdx2 != loadedIdx {
		t.Error("should return same cached instance")
	}
}

// TestSegmentMetadata_EvictCache tests cache eviction.
func TestSegmentMetadata_EvictCache(t *testing.T) {
	segment := newSegmentMetadata(1, "h", "v", "t", "m")

	// Mock cached index
	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()
	segment.cachedIndex = NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Evict cache
	segment.EvictCache()

	if segment.cachedIndex != nil {
		t.Error("cached index should be nil after eviction")
	}
}

// TestSegmentMetadata_UpdateStats tests statistics update.
func TestSegmentMetadata_UpdateStats(t *testing.T) {
	segment := newSegmentMetadata(1, "h", "v", "t", "m")

	numDocs := uint32(100)
	sizeBytes := int64(1024 * 1024)

	segment.updateStats(numDocs, sizeBytes)

	if segment.numDocs != numDocs {
		t.Errorf("expected numDocs %d, got %d", numDocs, segment.numDocs)
	}

	if segment.sizeBytes != sizeBytes {
		t.Errorf("expected sizeBytes %d, got %d", sizeBytes, segment.sizeBytes)
	}
}

// TestSegmentManager_Operations tests segment manager operations.
func TestSegmentManager_Operations(t *testing.T) {
	sm := newSegmentManager()

	// Initially empty
	if sm.Count() != 0 {
		t.Errorf("expected count 0, got %d", sm.Count())
	}

	// Add segments
	seg1 := newSegmentMetadata(1, "h1", "v1", "t1", "m1")
	seg2 := newSegmentMetadata(2, "h2", "v2", "t2", "m2")
	seg3 := newSegmentMetadata(3, "h3", "v3", "t3", "m3")

	sm.add(seg1)
	sm.add(seg2)
	sm.add(seg3)

	if sm.Count() != 3 {
		t.Errorf("expected count 3, got %d", sm.Count())
	}

	// Get by ID
	retrieved := sm.get(2)
	if retrieved == nil {
		t.Error("expected to find segment 2")
	} else if retrieved.id != 2 {
		t.Errorf("expected segment ID 2, got %d", retrieved.id)
	}

	// List all
	all := sm.list()
	if len(all) != 3 {
		t.Errorf("expected 3 segments, got %d", len(all))
	}

	// Remove
	removed := sm.remove(2)
	if !removed {
		t.Error("expected remove to succeed")
	}

	if sm.Count() != 2 {
		t.Errorf("expected count 2 after remove, got %d", sm.Count())
	}

	// Verify removed
	retrieved = sm.get(2)
	if retrieved != nil {
		t.Error("segment 2 should be removed")
	}

	// Remove non-existent
	removed = sm.remove(999)
	if removed {
		t.Error("removing non-existent segment should return false")
	}
}

// TestSegmentManager_TotalSize tests total size calculation.
func TestSegmentManager_TotalSize(t *testing.T) {
	sm := newSegmentManager()

	seg1 := newSegmentMetadata(1, "h1", "v1", "t1", "m1")
	seg1.updateStats(10, 1024)

	seg2 := newSegmentMetadata(2, "h2", "v2", "t2", "m2")
	seg2.updateStats(20, 2048)

	seg3 := newSegmentMetadata(3, "h3", "v3", "t3", "m3")
	seg3.updateStats(30, 4096)

	sm.add(seg1)
	sm.add(seg2)
	sm.add(seg3)

	totalSize := sm.TotalSize()
	expectedSize := int64(1024 + 2048 + 4096)

	if totalSize != expectedSize {
		t.Errorf("expected total size %d, got %d", expectedSize, totalSize)
	}
}

// TestSegmentManager_EvictAllCaches tests cache eviction.
func TestSegmentManager_EvictAllCaches(t *testing.T) {
	sm := newSegmentManager()

	vecIdx, _ := NewFlatIndex(4, Cosine)
	txtIdx := NewBM25SearchIndex()
	metaIdx := NewRoaringMetadataIndex()

	seg1 := newSegmentMetadata(1, "h1", "v1", "t1", "m1")
	seg1.cachedIndex = NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	seg2 := newSegmentMetadata(2, "h2", "v2", "t2", "m2")
	seg2.cachedIndex = NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	sm.add(seg1)
	sm.add(seg2)

	// Evict all caches
	sm.EvictAllCaches()

	// Verify caches are evicted
	if seg1.cachedIndex != nil {
		t.Error("seg1 cache should be evicted")
	}
	if seg2.cachedIndex != nil {
		t.Error("seg2 cache should be evicted")
	}
}
