package comet

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
)

// storageProvider manages file system operations for the storage layer.
// It handles segment file creation, naming, and directory management.
//
// Thread-safety: All methods are safe for concurrent use.
type storageProvider struct {
	// Base directory for all storage files
	baseDir string

	// Atomic counter for generating unique segment IDs
	segmentCounter atomic.Uint64

	// Lock file to prevent multiple processes from using same directory
	lockFile *os.File
}

// newStorageProvider creates a new storage provider.
// It creates the base directory if it doesn't exist and acquires a lock file.
//
// Parameters:
//   - baseDir: Base directory for storage files
//
// Returns:
//   - *storageProvider: New provider instance
//   - error: Error if directory creation or lock acquisition fails
func newStorageProvider(baseDir string) (*storageProvider, error) {
	// Create base directory if it doesn't exist
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create base directory: %w", err)
	}

	provider := &storageProvider{
		baseDir: baseDir,
	}

	// Acquire lock file to prevent concurrent access
	if err := provider.acquireLock(); err != nil {
		return nil, fmt.Errorf("failed to acquire lock: %w", err)
	}

	// Scan existing segments to initialize counter
	if err := provider.initSegmentCounter(); err != nil {
		provider.releaseLock()
		return nil, fmt.Errorf("failed to initialize segment counter: %w", err)
	}

	return provider, nil
}

// acquireLock acquires an exclusive lock on the storage directory.
// This prevents multiple processes from writing to the same storage.
func (p *storageProvider) acquireLock() error {
	lockPath := filepath.Join(p.baseDir, "LOCK")

	// Try to create lock file exclusively
	lockFile, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("storage directory is locked by another process")
		}
		return fmt.Errorf("failed to create lock file: %w", err)
	}

	// Write process ID to lock file for debugging
	if _, err := lockFile.WriteString(fmt.Sprintf("%d\n", os.Getpid())); err != nil {
		lockFile.Close()
		os.Remove(lockPath)
		return fmt.Errorf("failed to write lock file: %w", err)
	}

	p.lockFile = lockFile
	return nil
}

// releaseLock releases the storage directory lock.
func (p *storageProvider) releaseLock() error {
	if p.lockFile == nil {
		return nil
	}

	lockPath := filepath.Join(p.baseDir, "LOCK")

	// Close and remove lock file
	if err := p.lockFile.Close(); err != nil {
		return fmt.Errorf("failed to close lock file: %w", err)
	}

	if err := os.Remove(lockPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove lock file: %w", err)
	}

	p.lockFile = nil
	return nil
}

// initSegmentCounter scans existing segments and initializes the counter.
// This ensures newly created segments have unique IDs.
func (p *storageProvider) initSegmentCounter() error {
	entries, err := os.ReadDir(p.baseDir)
	if err != nil {
		return fmt.Errorf("failed to read directory: %w", err)
	}

	var maxSegmentID uint64 = 0

	// Find the highest segment ID
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		// Segment files are named like: hybrid_000001.bin, vector_000001.bin, etc.
		if strings.Contains(name, "_") {
			parts := strings.Split(name, "_")
			if len(parts) >= 2 {
				// Extract segment ID from filename
				idStr := strings.TrimSuffix(parts[1], ".bin.gz")
				idStr = strings.TrimSuffix(idStr, ".bin")
				if id, err := strconv.ParseUint(idStr, 10, 64); err == nil {
					if id > maxSegmentID {
						maxSegmentID = id
					}
				}
			}
		}
	}

	p.segmentCounter.Store(maxSegmentID)
	return nil
}

// nextSegmentID generates the next unique segment ID.
//
// Returns:
//   - uint64: Next segment ID
func (p *storageProvider) nextSegmentID() uint64 {
	return p.segmentCounter.Add(1)
}

// segmentPaths returns the file paths for a segment.
//
// Parameters:
//   - segmentID: Segment identifier
//
// Returns:
//   - hybrid: Path to hybrid metadata file
//   - vector: Path to vector index file
//   - text: Path to text index file
//   - metadata: Path to metadata index file
func (p *storageProvider) segmentPaths(segmentID uint64) (hybrid, vector, text, metadata string) {
	idStr := fmt.Sprintf("%06d", segmentID)
	hybrid = filepath.Join(p.baseDir, fmt.Sprintf("hybrid_%s.bin.gz", idStr))
	vector = filepath.Join(p.baseDir, fmt.Sprintf("vector_%s.bin.gz", idStr))
	text = filepath.Join(p.baseDir, fmt.Sprintf("text_%s.bin.gz", idStr))
	metadata = filepath.Join(p.baseDir, fmt.Sprintf("metadata_%s.bin.gz", idStr))
	return
}

// listSegments lists all segment IDs in the storage directory.
//
// Returns:
//   - []uint64: Sorted list of segment IDs
//   - error: Error if directory reading fails
func (p *storageProvider) listSegments() ([]uint64, error) {
	entries, err := os.ReadDir(p.baseDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read directory: %w", err)
	}

	segmentMap := make(map[uint64]bool)

	// Find all segment IDs by looking for hybrid files
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		if strings.HasPrefix(name, "hybrid_") {
			// Extract segment ID
			parts := strings.Split(name, "_")
			if len(parts) >= 2 {
				idStr := strings.TrimSuffix(parts[1], ".bin.gz")
				idStr = strings.TrimSuffix(idStr, ".bin")
				if id, err := strconv.ParseUint(idStr, 10, 64); err == nil {
					segmentMap[id] = true
				}
			}
		}
	}

	// Convert to sorted slice
	segments := make([]uint64, 0, len(segmentMap))
	for id := range segmentMap {
		segments = append(segments, id)
	}

	// Sort segments (older first)
	for i := 0; i < len(segments); i++ {
		for j := i + 1; j < len(segments); j++ {
			if segments[i] > segments[j] {
				segments[i], segments[j] = segments[j], segments[i]
			}
		}
	}

	return segments, nil
}

// deleteSegment deletes all files associated with a segment.
//
// Parameters:
//   - segmentID: Segment identifier
//
// Returns:
//   - error: Error if deletion fails
func (p *storageProvider) deleteSegment(segmentID uint64) error {
	hybrid, vector, text, metadata := p.segmentPaths(segmentID)

	files := []string{hybrid, vector, text, metadata}
	var errs []error

	for _, file := range files {
		if err := os.Remove(file); err != nil && !os.IsNotExist(err) {
			errs = append(errs, fmt.Errorf("failed to delete %s: %w", file, err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("segment deletion errors: %v", errs)
	}

	return nil
}

// close closes the storage provider and releases resources.
func (p *storageProvider) close() error {
	return p.releaseLock()
}
