## API Reference

### Core Functions

#### NewFlatIndex

```go
func NewFlatIndex(dim int, distanceKind DistanceKind) (*FlatIndex, error)
```

Creates a flat (brute-force) vector index with guaranteed 100% recall.

**Parameters:**

- `dim`: Vector dimensionality (must be > 0)
- `distanceKind`: Distance metric (Euclidean, L2Squared, or Cosine)

**Returns:**

- `*FlatIndex`: New empty index ready for insertions
- `error`: Returns error if dim <= 0 or invalid distance kind

**Example:**

```go
// Create index for 384-dimensional sentence embeddings
index, err := comet.NewFlatIndex(384, comet.Cosine)
if err != nil {
    log.Fatal(err)
}

// Add vectors
vec := make([]float32, 384)
// ... populate vec with embedding ...
node := comet.NewVectorNode(vec)
index.Add(*node)
```

**Time Complexity:** O(1)  
**Space Complexity:** O(1) initially, grows to O(n × d) where n = vectors, d = dimensions

#### NewHNSWIndex

```go
func NewHNSWIndex(dim int, distanceKind DistanceKind, m, efConstruction, efSearch int) (*HNSWIndex, error)
```

Creates an HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.

**Parameters:**

- `dim`: Vector dimensionality
- `distanceKind`: Distance metric
- `m`: Connections per layer (0 for default=16, typical range: 12-48)
- `efConstruction`: Build-time candidate list size (0 for default=200, typical: 100-500)
- `efSearch`: Search-time candidate list size (0 for default=200, typical: 100-500)

**Returns:**

- `*HNSWIndex`: New HNSW index
- `error`: Returns error if parameters invalid

**Example:**

```go
// Create HNSW index with default parameters
m, efC, efS := comet.DefaultHNSWConfig()
index, err := comet.NewHNSWIndex(384, comet.Cosine, m, efC, efS)
```

**Time Complexity:**

- Insert: O(M × efConstruction × log n)
- Search: O(M × efSearch × log n)

**Space Complexity:** O(n × d + n × M × L) where L = average layers per node

#### NewBM25SearchIndex

```go
func NewBM25SearchIndex() *BM25SearchIndex
```

Creates a BM25 full-text search index using inverted indexes and roaring bitmaps.

**Parameters:** None

**Returns:**

- `*BM25SearchIndex`: New empty text index

**Example:**

```go
index := comet.NewBM25SearchIndex()

// Add documents
index.Add(1, "the quick brown fox jumps over the lazy dog")
index.Add(2, "machine learning tutorial for beginners")

// Search
results, err := index.NewSearch().
    WithQuery("machine learning").
    WithK(10).
    Execute()
```

**Time Complexity:**

- Add: O(m) where m = tokens in document
- Search: O(q × d_avg) where q = query tokens, d_avg = avg docs per term

**Space Complexity:** O(n × m_avg) where n = documents, m_avg = avg tokens per doc

### Advanced Functions

#### NewHybridSearchIndex

```go
func NewHybridSearchIndex(vectorIndex VectorIndex, textIndex TextIndex, metadataIndex MetadataIndex) HybridSearchIndex
```

Creates a unified index combining vector, text, and metadata search.

**Parameters:**

- `vectorIndex`: Vector index (can be nil if not using vector search)
- `textIndex`: Text index (can be nil if not using text search)
- `metadataIndex`: Metadata index (can be nil if not using metadata filtering)

**Returns:**

- `HybridSearchIndex`: Unified search interface

**Example:**

```go
vecIdx, _ := comet.NewFlatIndex(384, comet.Cosine)
txtIdx := comet.NewBM25SearchIndex()
metaIdx := comet.NewRoaringMetadataIndex()

hybrid := comet.NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

// Add documents
id, err := hybrid.Add(
    embedding,                    // []float32
    "machine learning tutorial",  // text
    map[string]interface{}{       // metadata
        "category": "education",
        "price": 49,
    },
)
```

#### NewRoaringMetadataIndex

```go
func NewRoaringMetadataIndex() *RoaringMetadataIndex
```

Creates a metadata filtering index using Roaring Bitmaps (for categorical data) and BSI (for numeric ranges).

**Parameters:** None

**Returns:**

- `*RoaringMetadataIndex`: New metadata index

**Example:**

```go
index := comet.NewRoaringMetadataIndex()

// Add document with metadata
node := comet.NewMetadataNodeWithID(1, map[string]interface{}{
    "category": "electronics",
    "price": 999,
    "in_stock": true,
})
index.Add(*node)

// Query with filters
results, err := index.NewSearch().
    WithFilters(
        comet.Eq("category", "electronics"),
        comet.Gte("price", 500),
        comet.Eq("in_stock", true),
    ).
    Execute()
```

**Time Complexity:**

- Add: O(f) where f = number of fields
- Query: O(f × log n) where f = filters, n = documents

**Space Complexity:** Highly compressed, typically 1-10% of uncompressed size

