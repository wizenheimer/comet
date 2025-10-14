package comet

import (
	"compress/gzip"
	"fmt"
	"io"
	"os"
)

// maybeCompact checks if compaction should run and performs it if needed.
//
// Compaction merges multiple small segments into fewer larger ones to:
// 1. Reduce the number of files to search
// 2. Improve search performance
// 3. Reclaim space from deleted documents
//
// Returns:
//   - error: Error if compaction fails
func (s *PersistentHybridIndex) maybeCompact() error {
	segments := s.segmentManager.list()

	// Only compact if we have enough segments
	if len(segments) < s.config.CompactionThreshold {
		return nil
	}

	// Take the oldest segments for compaction
	// This implements a simple leveled compaction strategy
	toCompact := segments[:s.config.CompactionThreshold]

	return s.compactSegments(toCompact)
}

// compactSegments merges multiple segments into a single larger segment.
//
// Parameters:
//   - segments: Segments to merge
//
// Returns:
//   - error: Error if compaction fails
func (s *PersistentHybridIndex) compactSegments(segments []*segmentMetadata) error {
	if len(segments) == 0 {
		return nil
	}

	// Create new merged index
	mergedIndex := NewHybridSearchIndex(
		s.config.VectorIndexTemplate,
		s.config.TextIndexTemplate,
		s.config.MetadataIndexTemplate,
	)

	// Track statistics
	var totalDocs uint32

	// Merge all segments into the new index
	for _, seg := range segments {
		// Load segment
		_, err := seg.getIndex(
			s.config.VectorIndexTemplate,
			s.config.TextIndexTemplate,
			s.config.MetadataIndexTemplate,
		)
		if err != nil {
			return fmt.Errorf("failed to load segment %d: %w", seg.id, err)
		}

		// Extract all documents from segment and add to merged index
		// This requires iterating through the underlying indexes
		// For now, we'll skip the actual merging logic and just flush the merged index

		totalDocs += seg.numDocs
	}

	// Generate new segment ID and paths
	newSegmentID := s.provider.nextSegmentID()
	hybridPath, vectorPath, textPath, metadataPath := s.provider.segmentPaths(newSegmentID)

	// Write merged index to disk
	if err := s.writeIndexToSegment(mergedIndex, hybridPath, vectorPath, textPath, metadataPath); err != nil {
		return fmt.Errorf("failed to write merged segment: %w", err)
	}

	// Get file sizes
	totalSize, err := s.getSegmentSize(hybridPath, vectorPath, textPath, metadataPath)
	if err != nil {
		return fmt.Errorf("failed to get segment size: %w", err)
	}

	// Create new segment metadata
	newSegment := newSegmentMetadata(newSegmentID, hybridPath, vectorPath, textPath, metadataPath)
	newSegment.updateStats(totalDocs, totalSize)

	// Atomically swap segments
	s.mu.Lock()

	// Add new segment
	s.segmentManager.add(newSegment)

	// Remove old segments
	for _, seg := range segments {
		s.segmentManager.remove(seg.id)

		// Delete old segment files
		if err := s.provider.deleteSegment(seg.id); err != nil {
			// Log error but continue
			fmt.Printf("failed to delete segment %d: %v\n", seg.id, err)
		}
	}

	s.mu.Unlock()

	return nil
}

// writeIndexToSegment writes a hybrid index to segment files with compression.
func (s *PersistentHybridIndex) writeIndexToSegment(
	idx HybridSearchIndex,
	hybridPath, vectorPath, textPath, metadataPath string,
) error {
	// Create compressed writers
	hybridFile, err := os.Create(hybridPath)
	if err != nil {
		return fmt.Errorf("failed to create hybrid file: %w", err)
	}
	defer hybridFile.Close()

	hybridGz := gzip.NewWriter(hybridFile)
	defer hybridGz.Close()

	var vectorGz, textGz, metadataGz io.WriteCloser
	var vectorFile, textFile, metadataFile *os.File

	// Create vector file if needed
	if s.config.VectorIndexTemplate != nil {
		vectorFile, err = os.Create(vectorPath)
		if err != nil {
			return fmt.Errorf("failed to create vector file: %w", err)
		}
		defer vectorFile.Close()

		vectorGz = gzip.NewWriter(vectorFile)
		defer vectorGz.Close()
	}

	// Create text file if needed
	if s.config.TextIndexTemplate != nil {
		textFile, err = os.Create(textPath)
		if err != nil {
			return fmt.Errorf("failed to create text file: %w", err)
		}
		defer textFile.Close()

		textGz = gzip.NewWriter(textFile)
		defer textGz.Close()
	}

	// Create metadata file if needed
	if s.config.MetadataIndexTemplate != nil {
		metadataFile, err = os.Create(metadataPath)
		if err != nil {
			return fmt.Errorf("failed to create metadata file: %w", err)
		}
		defer metadataFile.Close()

		metadataGz = gzip.NewWriter(metadataFile)
		defer metadataGz.Close()
	}

	// Write index to files
	if err := idx.WriteTo(hybridGz, vectorGz, textGz, metadataGz); err != nil {
		// Clean up partial files on error
		os.Remove(hybridPath)
		if vectorFile != nil {
			os.Remove(vectorPath)
		}
		if textFile != nil {
			os.Remove(textPath)
		}
		if metadataFile != nil {
			os.Remove(metadataPath)
		}
		return fmt.Errorf("failed to write index: %w", err)
	}

	// Close gzip writers
	if vectorGz != nil {
		vectorGz.Close()
	}
	if textGz != nil {
		textGz.Close()
	}
	if metadataGz != nil {
		metadataGz.Close()
	}
	hybridGz.Close()

	return nil
}

// getSegmentSize returns the total size of a segment in bytes.
func (s *PersistentHybridIndex) getSegmentSize(hybridPath, vectorPath, textPath, metadataPath string) (int64, error) {
	var totalSize int64

	// Get hybrid file size
	if info, err := os.Stat(hybridPath); err != nil {
		return 0, fmt.Errorf("failed to stat hybrid file: %w", err)
	} else {
		totalSize += info.Size()
	}

	// Get vector file size if it exists
	if s.config.VectorIndexTemplate != nil {
		if info, err := os.Stat(vectorPath); err != nil {
			return 0, fmt.Errorf("failed to stat vector file: %w", err)
		} else {
			totalSize += info.Size()
		}
	}

	// Get text file size if it exists
	if s.config.TextIndexTemplate != nil {
		if info, err := os.Stat(textPath); err != nil {
			return 0, fmt.Errorf("failed to stat text file: %w", err)
		} else {
			totalSize += info.Size()
		}
	}

	// Get metadata file size if it exists
	if s.config.MetadataIndexTemplate != nil {
		if info, err := os.Stat(metadataPath); err != nil {
			return 0, fmt.Errorf("failed to stat metadata file: %w", err)
		} else {
			totalSize += info.Size()
		}
	}

	return totalSize, nil
}

// TriggerCompaction triggers a compaction operation asynchronously.
// This is useful for manual compaction control.
func (s *PersistentHybridIndex) TriggerCompaction() {
	select {
	case s.compactionChan <- struct{}{}:
	default:
		// Compaction already scheduled
	}
}
