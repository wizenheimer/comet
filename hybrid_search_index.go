// Package comet implements a hybrid search index that combines vector, text, and metadata search.
//
// WHAT IS HYBRIDSEARCHINDEX?
// HybridSearchIndex is a facade that provides a unified interface over three specialized indexes:
// 1. VectorIndex: For semantic similarity search using vector embeddings
// 2. TextIndex: For keyword-based BM25 full-text search
// 3. MetadataIndex: For filtering by structured metadata attributes
//
// HOW IT WORKS:
// The index maintains three separate indexes internally and coordinates search across them.
// When searching, it follows this flow:
// 1. Apply metadata filters first (if any) to get candidate document IDs
// 2. Pass candidate IDs to vector and/or text search for relevance ranking
// 3. Combine results from multiple search modes using score aggregation
//
// SEARCH MODES:
// - Vector-only: Semantic similarity search using embeddings
// - Text-only: Keyword-based BM25 search
// - Metadata-only: Pure filtering without ranking
// - Hybrid: Combine any or all of the above with score aggregation
//
// WHEN TO USE:
// Use HybridSearchIndex when:
// 1. You need to combine multiple search modalities
// 2. You want to filter by metadata before expensive vector search
// 3. You need both semantic and keyword-based search
// 4. You want a simple unified API instead of managing multiple indexes
package comet

import (
	"encoding/binary"
	"fmt"
	"io"
	"sort"
	"sync"
)

// Compile-time check to ensure hybridSearchIndex implements HybridSearchIndex
var _ HybridSearchIndex = (*hybridSearchIndex)(nil)

// hybridSearchIndex is the implementation of HybridSearchIndex
type hybridSearchIndex struct {
	mu sync.RWMutex

	vectorIndex   VectorIndex
	textIndex     TextIndex
	metadataIndex MetadataIndex

	// Keep track of which indexes are populated for each document
	docInfo map[uint32]*documentInfo
}

// documentInfo tracks which indexes contain data for a document
type documentInfo struct {
	hasVector   bool
	hasText     bool
	hasMetadata bool
}

// NewHybridSearchIndex creates a new hybrid search index
//
// Parameters:
//   - vectorIndex: The vector index to use (can be nil if not using vector search)
//   - textIndex: The text index to use (can be nil if not using text search)
//   - metadataIndex: The metadata index to use (can be nil if not using metadata search)
//
// Returns:
//   - HybridSearchIndex: A new hybrid search index
//
// Example:
//
//	vecIdx, _ := NewFlatIndex(384, Cosine)
//	txtIdx := NewBM25SearchIndex()
//	metaIdx := NewRoaringMetadataIndex()
//	idx := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)
func NewHybridSearchIndex(vectorIndex VectorIndex, textIndex TextIndex, metadataIndex MetadataIndex) HybridSearchIndex {
	return &hybridSearchIndex{
		vectorIndex:   vectorIndex,
		textIndex:     textIndex,
		metadataIndex: metadataIndex,
		docInfo:       make(map[uint32]*documentInfo),
	}
}

// Add adds a document with its vector, text, and metadata to the index.
// The document ID is auto-generated and returned.
//
// Parameters:
//   - vector: The vector embedding (can be nil if not using vector search)
//   - text: The document text (can be empty if not using text search)
//   - metadata: The document metadata (can be nil if not using metadata search)
//
// Returns:
//   - uint32: The generated document ID
//   - error: Error if any index operation fails
//
// Example:
//
//	id, err := idx.Add(embedding, "the quick brown fox", map[string]interface{}{
//		"category": "animals",
//		"price": 99,
//	})
func (idx *hybridSearchIndex) Add(vector []float32, text string, metadata map[string]interface{}) (uint32, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Create a vector node to get an auto-generated ID
	vectorNode := NewVectorNode(vector)
	id := vectorNode.ID()

	return id, idx.addInternal(id, vector, text, metadata)
}

// AddWithID adds a document with a specific ID to the index.
//
// Parameters:
//   - id: The document ID to use
//   - vector: The vector embedding (can be nil if not using vector search)
//   - text: The document text (can be empty if not using text search)
//   - metadata: The document metadata (can be nil if not using metadata search)
//
// Returns:
//   - error: Error if any index operation fails
//
// Example:
//
//	err := idx.AddWithID(42, embedding, "the quick brown fox", map[string]interface{}{
//		"category": "animals",
//	})
func (idx *hybridSearchIndex) AddWithID(id uint32, vector []float32, text string, metadata map[string]interface{}) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	return idx.addInternal(id, vector, text, metadata)
}

// addInternal adds a document to all relevant indexes.
// Must be called with idx.mu held.
func (idx *hybridSearchIndex) addInternal(id uint32, vector []float32, text string, metadata map[string]interface{}) error {
	info := &documentInfo{}

	// Add to vector index
	if idx.vectorIndex != nil && vector != nil && len(vector) > 0 {
		vectorNode := NewVectorNodeWithID(id, vector)
		if err := idx.vectorIndex.Add(*vectorNode); err != nil {
			return fmt.Errorf("failed to add to vector index: %w", err)
		}
		info.hasVector = true
	}

	// Add to text index
	if idx.textIndex != nil && text != "" {
		if err := idx.textIndex.Add(id, text); err != nil {
			return fmt.Errorf("failed to add to text index: %w", err)
		}
		info.hasText = true
	}

	// Add to metadata index
	if idx.metadataIndex != nil && metadata != nil && len(metadata) > 0 {
		metadataNode := NewMetadataNodeWithID(id, metadata)
		if err := idx.metadataIndex.Add(*metadataNode); err != nil {
			return fmt.Errorf("failed to add to metadata index: %w", err)
		}
		info.hasMetadata = true
	}

	idx.docInfo[id] = info

	return nil
}

// Remove removes a document from all indexes.
//
// Parameters:
//   - id: The document ID to remove
//
// Returns:
//   - error: Error if any index operation fails
func (idx *hybridSearchIndex) Remove(id uint32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	info, exists := idx.docInfo[id]
	if !exists {
		return fmt.Errorf("document %d not found", id)
	}

	// Remove from vector index
	if info.hasVector && idx.vectorIndex != nil {
		vectorNode := NewVectorNodeWithID(id, nil)
		if err := idx.vectorIndex.Remove(*vectorNode); err != nil {
			return fmt.Errorf("failed to remove from vector index: %w", err)
		}
	}

	// Remove from text index
	if info.hasText && idx.textIndex != nil {
		if err := idx.textIndex.Remove(id); err != nil {
			return fmt.Errorf("failed to remove from text index: %w", err)
		}
	}

	// Remove from metadata index
	if info.hasMetadata && idx.metadataIndex != nil {
		metadataNode := NewMetadataNodeWithID(id, nil)
		if err := idx.metadataIndex.Remove(*metadataNode); err != nil {
			return fmt.Errorf("failed to remove from metadata index: %w", err)
		}
	}

	delete(idx.docInfo, id)

	return nil
}

// NewSearch creates a new search builder for this index.
//
// Returns:
//   - HybridSearch: A new search builder ready to be configured
//
// Example:
//
//	results, err := idx.NewSearch().
//		WithVector(queryEmbedding).
//		WithText("quick fox").
//		WithMetadata(Eq("category", "animals")).
//		WithK(10).
//		Execute()
func (idx *hybridSearchIndex) NewSearch() HybridSearch {
	return &hybridSearch{
		index:            idx,
		k:                10,
		scoreAggregation: SumAggregation,
		cutoff:           -1,
		nProbes:          1,
		fusion:           DefaultFusion(),
	}
}

// Train trains the vector index if it requires training (e.g., IVF, PQ, IVFPQ)
//
// Parameters:
//   - vectors: Training vectors
//
// Returns:
//   - error: Error if training fails
func (idx *hybridSearchIndex) Train(vectors [][]float32) error {
	if idx.vectorIndex == nil {
		return fmt.Errorf("no vector index configured")
	}

	// Convert [][]float32 to []VectorNode
	nodes := make([]VectorNode, len(vectors))
	for i, vec := range vectors {
		nodes[i] = *NewVectorNodeWithID(uint32(i), vec)
	}

	return idx.vectorIndex.Train(nodes)
}

// Flush flushes all indexes
func (idx *hybridSearchIndex) Flush() error {
	var errs []error

	if idx.vectorIndex != nil {
		if err := idx.vectorIndex.Flush(); err != nil {
			errs = append(errs, fmt.Errorf("vector index flush failed: %w", err))
		}
	}

	if idx.textIndex != nil {
		if err := idx.textIndex.Flush(); err != nil {
			errs = append(errs, fmt.Errorf("text index flush failed: %w", err))
		}
	}

	if idx.metadataIndex != nil {
		if err := idx.metadataIndex.Flush(); err != nil {
			errs = append(errs, fmt.Errorf("metadata index flush failed: %w", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("flush errors: %v", errs)
	}

	return nil
}

// VectorIndex returns the underlying vector index
func (idx *hybridSearchIndex) VectorIndex() VectorIndex {
	return idx.vectorIndex
}

// TextIndex returns the underlying text index
func (idx *hybridSearchIndex) TextIndex() TextIndex {
	return idx.textIndex
}

// MetadataIndex returns the underlying metadata index
func (idx *hybridSearchIndex) MetadataIndex() MetadataIndex {
	return idx.metadataIndex
}

// HybridSearchResult represents a search result from the hybrid index
type HybridSearchResult struct {
	ID uint32 // Document ID
	// Score is float64 (not float32) because:
	// 1. Fusion combines scores from multiple sources (vector + text) with arithmetic operations
	// 2. RRF uses fractional calculations (1.0 / (k + rank)) requiring higher precision
	// 3. Better numerical stability when sorting results with very similar scores
	// 4. Prevents cumulative rounding errors from weighted sums and aggregations
	Score float64 // Combined relevance score
}

func (r HybridSearchResult) GetId() uint32 {
	return r.ID
}

func (r HybridSearchResult) GetScore() float32 {
	return float32(r.Score)
}

// HybridSearch encapsulates the search context for the hybrid index
type HybridSearch interface {
	// WithVector sets the query vector for vector search
	WithVector(query []float32) HybridSearch

	// WithText sets the query text for text search
	WithText(queries ...string) HybridSearch

	// WithMetadata sets metadata filters
	WithMetadata(filters ...Filter) HybridSearch

	// WithMetadataGroups sets complex metadata filter groups
	WithMetadataGroups(groups ...*FilterGroup) HybridSearch

	// WithK sets the number of results to return
	WithK(k int) HybridSearch

	// WithNProbes sets the number of probes for IVF-based vector indexes
	WithNProbes(nProbes int) HybridSearch

	// WithEfSearch sets the efSearch parameter for HNSW search
	WithEfSearch(efSearch int) HybridSearch

	// WithThreshold sets a score threshold for results
	WithThreshold(threshold float32) HybridSearch

	// WithScoreAggregation sets the strategy for aggregating scores
	WithScoreAggregation(kind ScoreAggregationKind) HybridSearch

	// WithCutoff sets the autocut parameter for result cutoff
	WithCutoff(cutoff int) HybridSearch

	// WithFusion sets the fusion strategy for combining vector and text scores
	WithFusion(fusion Fusion) HybridSearch

	// WithFusionKind sets the fusion strategy by kind with default config
	WithFusionKind(kind FusionKind) HybridSearch

	// Execute performs the search and returns results
	Execute() ([]HybridSearchResult, error)
}

// hybridSearch is the implementation of HybridSearch
type hybridSearch struct {
	index *hybridSearchIndex

	// === Vector Search Parameters ===
	vectorQuery []float32

	// === Text Search Parameters ===
	textQueries []string

	// === Metadata Search Parameters ===
	metadataFilters []Filter
	metadataGroups  []*FilterGroup

	// === Common Search Parameters ===
	k                int
	threshold        float32
	scoreAggregation ScoreAggregationKind
	cutoff           int

	// === Vector Index Specific Parameters ===
	nProbes  int // For IVF-based indexes
	efSearch int // For HNSW index

	// === Fusion Parameters ===
	fusion Fusion // Strategy for combining vector and text scores
}

// WithVector sets the query vector for vector search
func (s *hybridSearch) WithVector(query []float32) HybridSearch {
	s.vectorQuery = query
	return s
}

// WithText sets the query text(s) for text search
func (s *hybridSearch) WithText(queries ...string) HybridSearch {
	s.textQueries = queries
	return s
}

// WithMetadata sets metadata filters (AND logic between filters)
func (s *hybridSearch) WithMetadata(filters ...Filter) HybridSearch {
	s.metadataFilters = filters
	return s
}

// WithMetadataGroups sets complex metadata filter groups
func (s *hybridSearch) WithMetadataGroups(groups ...*FilterGroup) HybridSearch {
	s.metadataGroups = groups
	return s
}

// WithK sets the number of results to return
func (s *hybridSearch) WithK(k int) HybridSearch {
	s.k = k
	return s
}

// WithNProbes sets the number of probes for IVF-based vector indexes
func (s *hybridSearch) WithNProbes(nProbes int) HybridSearch {
	s.nProbes = nProbes
	return s
}

// WithEfSearch sets the efSearch parameter for HNSW search
func (s *hybridSearch) WithEfSearch(efSearch int) HybridSearch {
	s.efSearch = efSearch
	return s
}

// WithThreshold sets a score threshold for results
func (s *hybridSearch) WithThreshold(threshold float32) HybridSearch {
	s.threshold = threshold
	return s
}

// WithScoreAggregation sets the strategy for aggregating scores
func (s *hybridSearch) WithScoreAggregation(kind ScoreAggregationKind) HybridSearch {
	s.scoreAggregation = kind
	return s
}

// WithCutoff sets the autocut parameter for result cutoff
func (s *hybridSearch) WithCutoff(cutoff int) HybridSearch {
	s.cutoff = cutoff
	return s
}

// WithFusion sets the fusion strategy for combining vector and text scores
func (s *hybridSearch) WithFusion(fusion Fusion) HybridSearch {
	s.fusion = fusion
	return s
}

// WithFusionKind sets the fusion strategy by kind with default config
func (s *hybridSearch) WithFusionKind(kind FusionKind) HybridSearch {
	fusion, err := NewFusion(kind, nil)
	if err == nil {
		s.fusion = fusion
	}
	return s
}

// Execute performs the search across all configured indexes
//
// The search flow is:
// 1. Apply metadata filters (if any) to get candidate document IDs
// 2. Perform vector search (if configured) on candidate IDs
// 3. Perform text search (if configured) on candidate IDs
// 4. Combine and aggregate results using fusion strategy
func (s *hybridSearch) Execute() ([]HybridSearchResult, error) {
	// Step 1: Get candidate IDs from metadata filtering
	var candidateIDs []uint32
	if len(s.metadataFilters) > 0 || len(s.metadataGroups) > 0 {
		if s.index.metadataIndex == nil {
			return nil, fmt.Errorf("metadata filters specified but no metadata index configured")
		}

		metaSearch := s.index.metadataIndex.NewSearch()
		if len(s.metadataFilters) > 0 {
			metaSearch = metaSearch.WithFilters(s.metadataFilters...)
		}
		if len(s.metadataGroups) > 0 {
			metaSearch = metaSearch.WithFilterGroups(s.metadataGroups...)
		}

		metaResults, err := metaSearch.Execute()
		if err != nil {
			return nil, fmt.Errorf("metadata search failed: %w", err)
		}

		candidateIDs = make([]uint32, len(metaResults))
		for i, result := range metaResults {
			candidateIDs[i] = result.GetId()
		}

		// Early exit if no candidates
		if len(candidateIDs) == 0 {
			return []HybridSearchResult{}, nil
		}
	}

	// Step 2: Perform vector search
	var vectorResults map[uint32]float64
	if len(s.vectorQuery) > 0 {
		if s.index.vectorIndex == nil {
			return nil, fmt.Errorf("vector query specified but no vector index configured")
		}

		vecSearch := s.index.vectorIndex.NewSearch().
			WithQuery(s.vectorQuery).
			WithK(s.k).
			WithScoreAggregation(s.scoreAggregation).
			WithCutoff(s.cutoff)

		if s.nProbes > 0 {
			vecSearch = vecSearch.WithNProbes(s.nProbes)
		}
		if s.efSearch > 0 {
			vecSearch = vecSearch.WithEfSearch(s.efSearch)
		}
		if s.threshold > 0 {
			vecSearch = vecSearch.WithThreshold(s.threshold)
		}
		if len(candidateIDs) > 0 {
			vecSearch = vecSearch.WithDocumentIDs(candidateIDs...)
		}

		results, err := vecSearch.Execute()
		if err != nil {
			return nil, fmt.Errorf("vector search failed: %w", err)
		}

		vectorResults = make(map[uint32]float64)
		for _, result := range results {
			vectorResults[result.GetId()] = float64(result.GetScore())
		}
	}

	// Step 3: Perform text search
	var textResults map[uint32]float64
	if len(s.textQueries) > 0 {
		if s.index.textIndex == nil {
			return nil, fmt.Errorf("text query specified but no text index configured")
		}

		textSearch := s.index.textIndex.NewSearch().
			WithQuery(s.textQueries...).
			WithK(s.k).
			WithScoreAggregation(s.scoreAggregation).
			WithCutoff(s.cutoff)

		if len(candidateIDs) > 0 {
			textSearch = textSearch.WithDocumentIDs(candidateIDs...)
		}

		results, err := textSearch.Execute()
		if err != nil {
			return nil, fmt.Errorf("text search failed: %w", err)
		}

		textResults = make(map[uint32]float64)
		for _, result := range results {
			textResults[result.GetId()] = float64(result.GetScore())
		}
	}

	// Step 4: Combine results using fusion strategy
	var combinedScores map[uint32]float64

	// Use fusion to combine vector and text results
	if len(vectorResults) > 0 && len(textResults) > 0 {
		combinedScores = s.fusion.Combine(vectorResults, textResults)
	} else if len(vectorResults) > 0 {
		combinedScores = vectorResults
	} else if len(textResults) > 0 {
		combinedScores = textResults
	} else {
		combinedScores = make(map[uint32]float64)
	}

	// If only metadata search was performed (no vector or text)
	if len(combinedScores) == 0 && len(candidateIDs) > 0 {
		for _, id := range candidateIDs {
			combinedScores[id] = 1.0
		}
	}

	// Convert to results and sort by score
	results := make([]HybridSearchResult, 0, len(combinedScores))
	for id, score := range combinedScores {
		results = append(results, HybridSearchResult{
			ID:    id,
			Score: score,
		})
	}

	// Sort by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Limit to k results
	if len(results) > s.k {
		results = results[:s.k]
	}

	return results, nil
}

// WriteTo serializes the HybridSearchIndex to separate writers.
//
// IMPORTANT: This method calls Flush() before serialization to ensure all soft-deleted
// documents are permanently removed from the serialized data.
//
// Parameters:
//   - hybridWriter: Writer for the hybrid index metadata (docInfo map, flags)
//   - vectorWriter: Writer for the vector index data (can be nil if no vector index)
//   - textWriter: Writer for the text index data (can be nil if no text index)
//   - metadataWriter: Writer for the metadata index data (can be nil if no metadata index)
//
// The hybrid metadata serialization format is:
// 1. Magic number (4 bytes) - "HYBR" identifier for validation
// 2. Version (4 bytes) - Format version for backward compatibility
// 3. Index presence flags (3 bytes) - which indexes are present
// 4. docInfo map:
//   - Number of entries (4 bytes)
//   - For each entry:
//   - Document ID (4 bytes)
//   - hasVector flag (1 byte)
//   - hasText flag (1 byte)
//   - hasMetadata flag (1 byte)
//
// Each underlying index is serialized to its respective writer using its own format.
//
// Thread-safety: Acquires read lock during serialization
//
// Returns:
//   - error: Returns error if write fails or flush fails
//
// Example:
//
//	// Save to separate files
//	hybridFile, _ := os.Create("hybrid.bin")
//	vectorFile, _ := os.Create("vector.bin")
//	textFile, _ := os.Create("text.bin")
//	metadataFile, _ := os.Create("metadata.bin")
//	err := idx.WriteTo(hybridFile, vectorFile, textFile, metadataFile)
func (idx *hybridSearchIndex) WriteTo(hybridWriter, vectorWriter, textWriter, metadataWriter io.Writer) error {
	// Flush before serializing to remove soft-deleted documents
	if err := idx.Flush(); err != nil {
		return fmt.Errorf("failed to flush before serialization: %w", err)
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Helper function for writing to hybrid writer
	write := func(data interface{}) error {
		return binary.Write(hybridWriter, binary.LittleEndian, data)
	}

	// 1. Write magic number "HYBR"
	magic := [4]byte{'H', 'Y', 'B', 'R'}
	if _, err := hybridWriter.Write(magic[:]); err != nil {
		return fmt.Errorf("failed to write magic number: %w", err)
	}

	// 2. Write version
	version := uint32(1)
	if err := write(version); err != nil {
		return fmt.Errorf("failed to write version: %w", err)
	}

	// 3. Write index presence flags
	hasVector := uint8(0)
	if idx.vectorIndex != nil {
		hasVector = 1
	}
	hasText := uint8(0)
	if idx.textIndex != nil {
		hasText = 1
	}
	hasMetadata := uint8(0)
	if idx.metadataIndex != nil {
		hasMetadata = 1
	}

	if err := write(hasVector); err != nil {
		return fmt.Errorf("failed to write hasVector flag: %w", err)
	}
	if err := write(hasText); err != nil {
		return fmt.Errorf("failed to write hasText flag: %w", err)
	}
	if err := write(hasMetadata); err != nil {
		return fmt.Errorf("failed to write hasMetadata flag: %w", err)
	}

	// 4. Write docInfo map
	if err := write(uint32(len(idx.docInfo))); err != nil {
		return fmt.Errorf("failed to write docInfo count: %w", err)
	}

	for id, info := range idx.docInfo {
		if err := write(id); err != nil {
			return fmt.Errorf("failed to write document ID: %w", err)
		}

		infoHasVector := uint8(0)
		if info.hasVector {
			infoHasVector = 1
		}
		infoHasText := uint8(0)
		if info.hasText {
			infoHasText = 1
		}
		infoHasMetadata := uint8(0)
		if info.hasMetadata {
			infoHasMetadata = 1
		}

		if err := write(infoHasVector); err != nil {
			return fmt.Errorf("failed to write docInfo hasVector: %w", err)
		}
		if err := write(infoHasText); err != nil {
			return fmt.Errorf("failed to write docInfo hasText: %w", err)
		}
		if err := write(infoHasMetadata); err != nil {
			return fmt.Errorf("failed to write docInfo hasMetadata: %w", err)
		}
	}

	// 5. Write vector index data (if present)
	if idx.vectorIndex != nil && vectorWriter != nil {
		if writerTo, ok := idx.vectorIndex.(io.WriterTo); ok {
			if _, err := writerTo.WriteTo(vectorWriter); err != nil {
				return fmt.Errorf("failed to write vector index: %w", err)
			}
		} else {
			return fmt.Errorf("vector index does not implement io.WriterTo")
		}
	}

	// 6. Write text index data (if present)
	if idx.textIndex != nil && textWriter != nil {
		if writerTo, ok := idx.textIndex.(io.WriterTo); ok {
			if _, err := writerTo.WriteTo(textWriter); err != nil {
				return fmt.Errorf("failed to write text index: %w", err)
			}
		} else {
			return fmt.Errorf("text index does not implement io.WriterTo")
		}
	}

	// 7. Write metadata index data (if present)
	if idx.metadataIndex != nil && metadataWriter != nil {
		if writerTo, ok := idx.metadataIndex.(io.WriterTo); ok {
			if _, err := writerTo.WriteTo(metadataWriter); err != nil {
				return fmt.Errorf("failed to write metadata index: %w", err)
			}
		} else {
			return fmt.Errorf("metadata index does not implement io.WriterTo")
		}
	}

	return nil
}

// ReadFrom deserializes a HybridSearchIndex from a single combined reader.
//
// This method implements the standard io.ReaderFrom interface. The reader should contain
// the serialized data in the following order:
//  1. Hybrid index metadata (docInfo map, flags)
//  2. Vector index data (if present)
//  3. Text index data (if present)
//  4. Metadata index data (if present)
//
// If you have separate readers for each component, use io.MultiReader to combine them:
//
//	combinedReader := io.MultiReader(hybridReader, vectorReader, textReader, metadataReader)
//	idx.ReadFrom(combinedReader)
//
// IMPORTANT: The underlying indexes (vectorIndex, textIndex, metadataIndex) must be
// created and configured before calling ReadFrom. This method will populate them with
// the deserialized data.
//
// Thread-safety: Acquires write lock during deserialization
//
// Returns:
//   - int64: Number of bytes read
//   - error: Returns error if read fails, format is invalid, or data is corrupted
//
// Example:
//
//	// Save hybrid index to separate files
//	hybridFile, _ := os.Create("hybrid.bin")
//	vectorFile, _ := os.Create("vector.bin")
//	textFile, _ := os.Create("text.bin")
//	metadataFile, _ := os.Create("metadata.bin")
//	idx.WriteTo(hybridFile, vectorFile, textFile, metadataFile)
//
//	// Load hybrid index - create underlying indexes first
//	vecIdx, _ := NewFlatIndex(384, Cosine)
//	txtIdx := NewBM25SearchIndex()
//	metaIdx := NewRoaringMetadataIndex()
//	idx2 := NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)
//	hybridFile, _ = os.Open("hybrid.bin")
//	vectorFile, _ = os.Open("vector.bin")
//	textFile, _ = os.Open("text.bin")
//	metadataFile, _ = os.Open("metadata.bin")
//	combined := io.MultiReader(hybridFile, vectorFile, textFile, metadataFile)
//	idx2.(*hybridSearchIndex).ReadFrom(combined)
func (idx *hybridSearchIndex) ReadFrom(r io.Reader) (int64, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	var bytesRead int64

	// Helper function for reading and tracking bytes
	read := func(data interface{}) error {
		err := binary.Read(r, binary.LittleEndian, data)
		if err == nil {
			switch data.(type) {
			case *uint32, *int32:
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
	if string(magic) != "HYBR" {
		return bytesRead, fmt.Errorf("invalid magic number: expected 'HYBR', got '%s'", string(magic))
	}

	// 2. Read version
	var version uint32
	if err := read(&version); err != nil {
		return bytesRead, fmt.Errorf("failed to read version: %w", err)
	}
	if version != 1 {
		return bytesRead, fmt.Errorf("unsupported version: %d", version)
	}

	// 3. Read index presence flags
	var hasVector, hasText, hasMetadata uint8
	if err := read(&hasVector); err != nil {
		return bytesRead, fmt.Errorf("failed to read hasVector flag: %w", err)
	}
	if err := read(&hasText); err != nil {
		return bytesRead, fmt.Errorf("failed to read hasText flag: %w", err)
	}
	if err := read(&hasMetadata); err != nil {
		return bytesRead, fmt.Errorf("failed to read hasMetadata flag: %w", err)
	}

	// Validate that the underlying indexes match what was serialized
	if (hasVector == 1) != (idx.vectorIndex != nil) {
		return bytesRead, fmt.Errorf("vector index presence mismatch: serialized=%v, current=%v", hasVector == 1, idx.vectorIndex != nil)
	}
	if (hasText == 1) != (idx.textIndex != nil) {
		return bytesRead, fmt.Errorf("text index presence mismatch: serialized=%v, current=%v", hasText == 1, idx.textIndex != nil)
	}
	if (hasMetadata == 1) != (idx.metadataIndex != nil) {
		return bytesRead, fmt.Errorf("metadata index presence mismatch: serialized=%v, current=%v", hasMetadata == 1, idx.metadataIndex != nil)
	}

	// 4. Read docInfo map
	var docInfoCount uint32
	if err := read(&docInfoCount); err != nil {
		return bytesRead, fmt.Errorf("failed to read docInfo count: %w", err)
	}

	docInfo := make(map[uint32]*documentInfo, docInfoCount)
	for i := uint32(0); i < docInfoCount; i++ {
		var id uint32
		if err := read(&id); err != nil {
			return bytesRead, fmt.Errorf("failed to read document ID: %w", err)
		}

		var infoHasVector, infoHasText, infoHasMetadata uint8
		if err := read(&infoHasVector); err != nil {
			return bytesRead, fmt.Errorf("failed to read docInfo hasVector: %w", err)
		}
		if err := read(&infoHasText); err != nil {
			return bytesRead, fmt.Errorf("failed to read docInfo hasText: %w", err)
		}
		if err := read(&infoHasMetadata); err != nil {
			return bytesRead, fmt.Errorf("failed to read docInfo hasMetadata: %w", err)
		}

		docInfo[id] = &documentInfo{
			hasVector:   infoHasVector == 1,
			hasText:     infoHasText == 1,
			hasMetadata: infoHasMetadata == 1,
		}
	}

	// 5. Read vector index data (if present)
	if idx.vectorIndex != nil {
		if readerFrom, ok := idx.vectorIndex.(io.ReaderFrom); ok {
			n, err := readerFrom.ReadFrom(r)
			bytesRead += n
			if err != nil {
				return bytesRead, fmt.Errorf("failed to read vector index: %w", err)
			}
		} else {
			return bytesRead, fmt.Errorf("vector index does not implement io.ReaderFrom")
		}
	}

	// 6. Read text index data (if present)
	if idx.textIndex != nil {
		if readerFrom, ok := idx.textIndex.(io.ReaderFrom); ok {
			n, err := readerFrom.ReadFrom(r)
			bytesRead += n
			if err != nil {
				return bytesRead, fmt.Errorf("failed to read text index: %w", err)
			}
		} else {
			return bytesRead, fmt.Errorf("text index does not implement io.ReaderFrom")
		}
	}

	// 7. Read metadata index data (if present)
	if idx.metadataIndex != nil {
		if readerFrom, ok := idx.metadataIndex.(io.ReaderFrom); ok {
			n, err := readerFrom.ReadFrom(r)
			bytesRead += n
			if err != nil {
				return bytesRead, fmt.Errorf("failed to read metadata index: %w", err)
			}
		} else {
			return bytesRead, fmt.Errorf("metadata index does not implement io.ReaderFrom")
		}
	}

	// Update the docInfo map
	idx.docInfo = docInfo

	return bytesRead, nil
}
