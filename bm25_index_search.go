package comet

import (
	"container/heap"
	"fmt"
	"math"
)

// Compile-time checks to ensure bm25TextSearch implements TextSearch
var _ TextSearch = (*bm25TextSearch)(nil)

// bm25TextSearch implements the TextSearch interface for BM25 index.
//
// BM25 search performs relevance-based search:
//   - Uses inverted index for fast term lookup
//   - Scores documents using BM25 ranking function
//   - Supports multi-query search with score aggregation
//   - No training required
type bm25TextSearch struct {
	index           *BM25SearchIndex
	queries         []string
	nodeIDs         []uint32
	documentIDs     []uint32
	k               int
	aggregationKind ScoreAggregationKind
	cutoff          int
}

// WithQuery sets the query text(s) - supports single or batch queries.
// Can be combined with WithNode to search from both direct queries and node-based queries.
//
// Parameters:
//   - queries: One or more query strings
//
// Returns:
//   - TextSearch: The search builder for method chaining
//
// Example:
//
//	search.WithQuery("quick brown fox")
//	search.WithQuery("query1", "query2", "query3")
func (s *bm25TextSearch) WithQuery(queries ...string) TextSearch {
	s.queries = queries
	return s
}

// WithNode sets the node ID(s) to search from - supports single or batch nodes.
// Can be combined with WithQuery to search from both direct queries and node-based queries.
//
// For BM25, this looks up the original text of the specified document IDs and uses
// them as queries. This is useful for "find similar documents" functionality.
//
// Parameters:
//   - nodeIDs: One or more document IDs to use as queries
//
// Returns:
//   - TextSearch: The search builder for method chaining
//
// Example:
//
//	search.WithNode(42)
//	search.WithNode(1, 2, 3)
func (s *bm25TextSearch) WithNode(nodeIDs ...uint32) TextSearch {
	s.nodeIDs = nodeIDs
	return s
}

// WithK sets the number of results to return.
// Defaults to 10 if not set. If k is 0 or negative, returns all results.
//
// Parameters:
//   - k: Maximum number of results to return
//
// Returns:
//   - TextSearch: The search builder for method chaining
//
// Example:
//
//	search.WithK(5)  // Return top 5 results
//	search.WithK(0)  // Return all results
func (s *bm25TextSearch) WithK(k int) TextSearch {
	s.k = k
	return s
}

// WithScoreAggregation sets the strategy for aggregating scores when the same node
// appears in results from multiple queries or nodes.
//
// Available strategies:
//   - SumAggregation (default): Sum all scores
//   - MaxAggregation: Take maximum (worst) score
//   - MeanAggregation: Average all scores
//
// Parameters:
//   - kind: The aggregation strategy to use
//
// Returns:
//   - TextSearch: The search builder for method chaining
//
// Example:
//
//	search.WithScoreAggregation(SumAggregation)
//	search.WithScoreAggregation(MaxAggregation)
func (s *bm25TextSearch) WithScoreAggregation(kind ScoreAggregationKind) TextSearch {
	s.aggregationKind = kind
	return s
}

// WithCutoff sets the autocut parameter for automatically determining result cutoff.
// A value of -1 (default) disables autocut. Otherwise, specifies number of extrema to find.
//
// The autocut algorithm analyzes the score distribution to find natural breakpoints
// in the results, allowing for automatic determination of result quality cutoff.
//
// Parameters:
//   - cutoff: Number of extrema to find before cutting (-1 disables)
//
// Returns:
//   - TextSearch: The search builder for method chaining
//
// Example:
//
//	search.WithCutoff(1)   // Apply autocut with 1 extremum
//	search.WithCutoff(-1)  // Disable autocut (default)
func (s *bm25TextSearch) WithCutoff(cutoff int) TextSearch {
	s.cutoff = cutoff
	return s
}

// WithDocumentIDs sets the eligible document IDs for pre-filtering.
// Only documents with IDs in this set will be considered as candidates.
// If empty, all documents are eligible (default behavior).
//
// This is useful for combining BM25 text search with metadata filters or
// other pre-filtering criteria. For example, you can first filter documents
// by metadata, then perform BM25 search only on the filtered subset.
//
// Parameters:
//   - docIDs: One or more document IDs to restrict search to
//
// Returns:
//   - TextSearch: The search builder for method chaining
//
// Example:
//
//	search.WithDocumentIDs(1, 2, 3)  // Only search in documents 1, 2, 3
//	search.WithDocumentIDs()         // No filtering (default)
func (s *bm25TextSearch) WithDocumentIDs(docIDs ...uint32) TextSearch {
	s.documentIDs = docIDs
	return s
}

// Execute performs the actual search and returns results.
//
// This method validates the search configuration and then executes the search
// using all specified queries (both direct queries and node-based queries).
//
// When multiple queries/nodes are provided, results are aggregated by document ID
// using the configured aggregation strategy (default: Sum).
//
// Returns:
//   - []TextResult: Search results sorted by score with document IDs
//   - error: Returns error if search configuration is invalid
//
// Example:
//
//	results, err := idx.NewSearch().
//		WithQuery("quick brown fox").
//		WithK(10).
//		Execute()
//	if err != nil { ... }
//	for _, result := range results {
//		fmt.Printf("DocID: %d, Score: %.4f\n", result.Id, result.Score)
//	}
func (s *bm25TextSearch) Execute() ([]TextResult, error) {
	// Validate that at least one of queries or nodeIDs is set
	if len(s.queries) == 0 && len(s.nodeIDs) == 0 {
		return nil, fmt.Errorf("must specify either queries or node IDs")
	}

	// Set default aggregation kind if not specified
	aggregationKind := s.aggregationKind
	if aggregationKind == "" {
		aggregationKind = SumAggregation
	}

	// Get aggregation instance
	aggregation, err := NewTextAggregation(aggregationKind)
	if err != nil {
		return nil, err
	}

	// Collect all queries (both direct queries and node-based queries)
	allQueries := make([]string, 0, len(s.queries)+len(s.nodeIDs))

	// Add direct queries
	allQueries = append(allQueries, s.queries...)

	// Convert nodes to queries if specified
	if len(s.nodeIDs) > 0 {
		nodeQueries, err := s.lookupNodeTexts()
		if err != nil {
			return nil, err
		}
		allQueries = append(allQueries, nodeQueries...)
	}

	// Execute search with all queries
	var allResults []TextResult
	for _, query := range allQueries {
		results, err := s.searchSingleQuery(query)
		if err != nil {
			return nil, err
		}
		allResults = append(allResults, results...)
	}

	// Aggregate results (deduplicates by document ID and combines scores)
	aggregatedResults := aggregation.Aggregate(allResults)

	// Apply k limit and autocut
	results := LimitResults(aggregatedResults, s.k)
	results = AutocutResults(results, s.cutoff)

	return results, nil
}

// lookupNodeTexts converts node IDs to their corresponding text tokens.
//
// For BM25, we reconstruct a query from the document's tokens (not perfect,
// but preserves the semantic content for "find similar" queries).
// Returns error if any node ID is not found.
func (s *bm25TextSearch) lookupNodeTexts() ([]string, error) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	queries := make([]string, 0, len(s.nodeIDs))
	for _, nodeID := range s.nodeIDs {
		// SOFT DELETE CHECK: Skip deleted documents
		if s.index.deletedDocs.Contains(nodeID) {
			return nil, fmt.Errorf("node ID %d not found in index (deleted)", nodeID)
		}

		tokens, exists := s.index.docTokens[nodeID]
		if !exists {
			return nil, fmt.Errorf("node ID %d not found in index", nodeID)
		}
		// Reconstruct query from tokens (space-separated)
		// This preserves semantic content for similarity search
		query := ""
		for i, token := range tokens {
			if i > 0 {
				query += " "
			}
			query += token
		}
		queries = append(queries, query)
	}

	return queries, nil
}

// searchSingleQuery performs BM25 search for a single query string.
//
// BM25 SEARCH ALGORITHM:
//  1. Tokenize and normalize the query
//  2. For each query term:
//     a. Calculate IDF (Inverse Document Frequency)
//     b. Find all documents containing the term (using inverted index)
//     c. Calculate BM25 score for each document
//  3. Aggregate scores across all query terms
//  4. Return top k results sorted by score (descending)
//
// Time Complexity: O(q × d + r × log(r)) where:
//   - q is the number of query terms
//   - d is the average number of documents per term
//   - r is the number of results (for sorting or heap operations)
func (s *bm25TextSearch) searchSingleQuery(query string) ([]TextResult, error) {
	qtokens := tokenize(normalize(query))
	if len(qtokens) == 0 {
		return nil, nil
	}

	s.index.mu.RLock()
	defer s.index.mu.RUnlock()

	scores := make(map[uint32]float64)
	N := float64(s.index.numDocs.Load())

	if N == 0 {
		return nil, nil
	}

	// Create document filter for pre-filtering
	docFilter := NewDocumentFilter(s.documentIDs)
	defer ReturnDocumentFilter(docFilter)

	// BM25 scoring
	for _, t := range qtokens {
		bitmap := s.index.postings[t]
		if bitmap == nil {
			continue
		}
		df := float64(bitmap.GetCardinality())
		// BM25 IDF formula
		idf := math.Log((N-df+0.5)/(df+0.5) + 1.0)

		for iter := bitmap.Iterator(); iter.HasNext(); {
			docID := iter.Next()

			// SOFT DELETE CHECK: Skip deleted documents
			if s.index.deletedDocs.Contains(docID) {
				continue
			}

			// Apply document filter
			if docFilter.ShouldSkip(docID) {
				continue
			}

			tfVal := float64(s.index.tf[t][docID])
			docLen := float64(s.index.docLengths[docID])
			// BM25 scoring formula
			score := idf * (tfVal * (K1 + 1)) / (tfVal + K1*(1-B+B*(docLen/s.index.avgDocLen)))
			scores[docID] += score
		}
	}

	// Use min-heap for top-K results instead of sorting everything
	k := s.k
	if k <= 0 || k >= len(scores) {
		// Return all results sorted
		results := make([]SearchResult, 0, len(scores))
		for docID, score := range scores {
			results = append(results, SearchResult{
				DocID: docID,
				Score: score,
			})
		}
		// Sort in descending order using heap
		heap.Init((*resultHeap)(&results))
		sortedResults := make([]SearchResult, len(results))
		for i := len(results) - 1; i >= 0; i-- {
			sortedResults[i] = heap.Pop((*resultHeap)(&results)).(SearchResult)
		}

		// Convert to TextResult
		textResults := make([]TextResult, len(sortedResults))
		for i, r := range sortedResults {
			textResults[i] = TextResult{
				Id:    r.DocID,
				Score: float32(r.Score),
			}
		}
		return textResults, nil
	}

	// Use min-heap to keep only top K results
	h := heapPool.Get().(*resultHeap)
	*h = (*h)[:0] // Reset the heap slice
	defer func() {
		*h = (*h)[:0] // Clear before returning to pool
		heapPool.Put(h)
	}()

	for docID, score := range scores {
		result := SearchResult{
			DocID: docID,
			Score: score,
		}

		if h.Len() < k {
			heap.Push(h, result)
		} else if score > (*h)[0].Score {
			// Replace minimum if new score is higher
			heap.Pop(h)
			heap.Push(h, result)
		}
	}

	// Extract results from heap and reverse to get descending order
	results := make([]SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(SearchResult)
	}

	// Convert to TextResult
	textResults := make([]TextResult, len(results))
	for i, r := range results {
		textResults[i] = TextResult{
			Id:    r.DocID,
			Score: float32(r.Score),
		}
	}

	return textResults, nil
}
