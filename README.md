# Comet

![Cover](media/cover.png)

A high-performance hybrid vector store written in Go. Comet brings together multiple indexing strategies and search modalities into a unified, hackable package. Hybrid retrieval with reciprocal rank fusion, autocut, pre-filtering, semantic search, full-text search, and multi-KNN searches, and multi-query operations — all in pure Go.

Understand search internals from the inside out. Built for hackers, not hyperscalers. Tiny enough to fit in your head. Decent enough to blow it.

**Choose from:**

- **Flat** (exact), **HNSW** (graph), **IVF** (clustering), **PQ** (quantization), or **IVFPQ** (hybrid) storage backends
- **Full-Text Search**: BM25 ranking algorithm with tokenization and normalization
- **Metadata Filtering**: Fast filtering using Roaring Bitmaps and Bit-Sliced Indexes
- **Ranking Programmability**: Reciprocal Rank Fusion, Fixed size result sets, Threshold based result sets, Dynamic result sets etc.
- **Hybrid Search**: Unified interface combining vector, text, and metadata search

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Core Concepts](#core-concepts)
- [API Reference](docs/API.md)
- [Examples](docs/EXAMPLE.md)
- [Configuration](#configuration)
- [API Details](#api-details)
- [Use Cases](#use-cases)
- [Contributing](#contributing)
- [License](#license)

## Overview

Everything you need to understand how vector databases actually work—and build one yourself.

**What's inside:**

- **5 Vector Storage Types**: Flat, HNSW, IVF, PQ, IVFPQ
- **3 Distance Metrics**: L2, L2 Squared, Cosine
- **Full-Text Search**: BM25 ranking with Unicode tokenization
- **Metadata Filtering**: Roaring bitmaps + Bit-Sliced Indexes
- **Hybrid Search**: Combine vector + text + metadata with Reciprocal Rank Fusion
- **Advanced Search**: Multi-KNN queries, multi-query operations, autocut result truncation
- **Production Features**: Thread-safe, serialization, soft deletes, configurable parameters

Everything you need to understand how vector databases actually work—and build one yourself.

## Features

### Vector Storage

- **Flat**: Brute-force exact search (100% recall baseline)
- **HNSW**: Hierarchical navigable small world graphs (95-99% recall, O(log n) search)
- **IVF**: Inverted file index with k-means clustering (85-95% recall, 10-20x speedup)
- **PQ**: Product quantization for compression (85-95% recall, 10-500x memory reduction)
- **IVFPQ**: IVF + PQ combined (85-95% recall, 100x speedup + 500x compression)

### Search Modalities

- **Vector Search**: L2, L2 Squared, and Cosine distance metrics
- **Full-Text Search**: BM25 ranking with Unicode-aware tokenization
- **Metadata Filtering**: Boolean queries on structured attributes
- **Hybrid Search**: Combine all three with configurable fusion strategies

### Fusion Strategies

- **Weighted Sum**: Linear combination with configurable weights
- **Reciprocal Rank Fusion (RRF)**: Scale-independent rank-based fusion
- **Max/Min Score**: Simple score aggregation

### Data Structures (The Good Stuff)

- **HNSW Graphs**: Multi-layer skip lists for approximate nearest neighbor search
- **Roaring Bitmaps**: Compressed bitmaps for metadata filtering (array, bitmap, run-length encoding)
- **Bit-Sliced Index (BSI)**: Efficient numeric range queries without full scans
- **Product Quantization Codebooks**: Learned k-means centroids for vector compression
- **Inverted Indexes**: Token-to-document mappings for full-text search

### Other Capabilities

- **Quantization**: Full precision, half precision, int8 precision
- **Soft Deletes**: Fast deletion with lazy cleanup
- **Serialization**: Persist and reload indexes
- **Thread-Safe**: Concurrent read/write operations
- **Autocut**: Automatic result truncation based on score gaps

## Installation

```bash
go get github.com/wizenheimer/comet
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    "github.com/wizenheimer/comet"
)

func main() {
    // Create a vector store (384-dimensional embeddings with cosine distance)
    index, err := comet.NewFlatIndex(384, comet.Cosine)
    if err != nil {
        log.Fatal(err)
    }

    // Add vectors
    vec1 := make([]float32, 384)
    // ... populate vec1 with your embedding ...
    node := comet.NewVectorNode(vec1)
    index.Add(*node)

    // Search for similar vectors
    query := make([]float32, 384)
    // ... populate query vector ...
    results, err := index.NewSearch().
        WithQuery(query).
        WithK(10).
        Execute()

    if err != nil {
        log.Fatal(err)
    }

    // Process results
    for i, result := range results {
        fmt.Printf("%d. ID=%d, Score=%.4f\n", i+1, result.GetId(), result.GetScore())
    }
}
```

Output:

```
1. ID=123, Score=0.0234
2. ID=456, Score=0.0567
3. ID=789, Score=0.0823
...
```

## Architecture

### System Architecture

Comet is organized into three main search engines that can work independently or together:

#### Application Layer

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
│            (Using Comet as a Go Library)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    Vector         Text         Metadata
```

#### Search Engine Layer

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Vector    │    │    Text     │    │  Metadata   │
│   Search    │    │   Search    │    │  Filtering  │
│   Engine    │    │   Engine    │    │   Engine    │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       │ Semantic         │ Keywords         │ Filters
       │ Similarity       │ + Relevance      │ + Boolean Logic
       ▼                  ▼                  ▼
```

#### Index Storage Layer

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ HNSW / IVF  │    │ BM25 Index  │    │  Roaring    │
│ / PQ / Flat │    │ (Inverted)  │    │  Bitmaps    │
└─────────────┘    └─────────────┘    └─────────────┘
   Graph/Trees      Token→DocIDs       Compressed Sets
```

#### Hybrid Coordinator

```
                 All Three Engines
                       │
                       ▼
              ┌─────────────────┐
              │  Hybrid Search   │
              │  Coordinator     │
              │  (Score Fusion)  │
              └─────────────────┘
                       │
                       ▼
              Combined Results
```

### Component Details

#### Component A: Vector Storage Engine

Manages vector storage and similarity search across multiple index types.

**Common Interface:**

```
┌────────────────────────────────────┐
│  VectorIndex Interface             │
│                                    │
│  ├─ Train(vectors)                 │
│  ├─ Add(vector)                    │
│  ├─ Remove(vector)                 │
│  └─ NewSearch()                    │
└────────────────────────────────────┘
```

**Available Implementations:**

```
FlatIndex          → Brute force, 100% recall
HNSWIndex          → Graph-based, O(log n)
IVFIndex           → Clustering, 10-20x faster
PQIndex            → Quantization, 10-500x compression
IVFPQIndex         → Hybrid, best of IVF + PQ
```

**Responsibilities:**

- Vector preprocessing (normalization for cosine distance)
- Distance calculations (Euclidean, L2², Cosine)
- K-nearest neighbor search
- Serialization/deserialization
- Soft delete management with flush mechanism

**Performance Characteristics:**

- Flat: O(n×d) search, 100% recall
- HNSW: O(M×ef×log n) search, 95-99% recall
- IVF: O(nProbes×n/k×d) search, 85-95% recall

#### Component B: Text Search Engine

Full-text search using BM25 ranking algorithm.

**Inverted Index:**

```
┌────────────────────────────────────┐
│  term → RoaringBitmap(docIDs)      │
│                                    │
│  "machine"  →  {1, 5, 12, 45}      │
│  "learning" →  {1, 3, 12, 20}      │
│  "neural"   →  {3, 20, 45}         │
└────────────────────────────────────┘
```

**Term Frequencies:**

```
┌────────────────────────────────────┐
│  term → {docID: count}             │
│                                    │
│  "machine" → {1: 3, 5: 1, 12: 2}   │
└────────────────────────────────────┘
```

**Document Stats:**

```
┌────────────────────────────────────┐
│  docID → (length, token_count)     │
│                                    │
│  1  →  (250 chars, 45 tokens)      │
│  5  →  (180 chars, 32 tokens)      │
└────────────────────────────────────┘
```

**Responsibilities:**

- Text tokenization (UAX#29 word segmentation)
- Unicode normalization (NFKC)
- Inverted index maintenance
- BM25 score calculation
- Top-K retrieval with heap

**Performance Characteristics:**

- Add: O(m) where m = tokens
- Search: O(q×d_avg) where q = query terms, d_avg = avg docs per term
- Memory: Compressed inverted index, no original text stored

#### Component C: Metadata Filter Engine

Fast filtering using compressed bitmaps.

**Categorical Fields (Roaring Bitmaps):**

```
┌────────────────────────────────────┐
│  field:value → Bitmap(docIDs)      │
│                                    │
│  "category:electronics" → {1,5,12} │
│  "category:books"       → {2,8,15} │
│  "in_stock:true"        → {1,2,5}  │
└────────────────────────────────────┘
```

**Numeric Fields (Bit-Sliced Index):**

```
┌────────────────────────────────────┐
│  field → BSI (range queries)       │
│                                    │
│  "price"  → [0-1000, 1000-5000]    │
│  "rating" → [0-5 scale]            │
└────────────────────────────────────┘
```

**Document Universe:**

```
┌────────────────────────────────────┐
│  allDocs → Bitmap(all IDs)         │
│                                    │
│  Used for NOT operations           │
└────────────────────────────────────┘
```

**Responsibilities:**

- Bitmap index maintenance (Roaring compression)
- BSI for numeric range queries
- Boolean query evaluation (AND, OR, NOT)
- Existence checks
- Set operations (IN, NOT IN)

**Performance Characteristics:**

- Add: O(f) where f = fields
- Query: O(f×log n) with bitmap operations
- Memory: Highly compressed, 1-10% of uncompressed

### Data Flow

How a hybrid search request flows through the system:

#### Step 1: Query Input

```
User Query:
├─ Vector:    [0.1, 0.5, ...]
├─ Text:      "machine learning"
└─ Filters:   {category="ai", price<100}
```

#### Step 2: Validation

```
┌────────────┐
│ Validation │ ──✗──> Error Handler
└─────┬──────┘         (Invalid dimension, etc.)
      │
      ✓ Valid
```

#### Step 3: Metadata Pre-filtering

```
┌─────────────────────────┐
│ Metadata Filter Engine  │
│ Apply: category="ai"    │
│        price<100        │
└─────┬───────────────────┘
      │
      ▼
Candidates: {1, 5, 7, 12, 15}
```

#### Step 4: Parallel Search

```
Candidates → Split into both engines

┌────────────┐         ┌────────────┐
│  Vector    │         │   Text     │
│  Search    │         │  Search    │
│ (on cands) │         │ (on cands) │
└─────┬──────┘         └─────┬──────┘
      │                      │
      ▼                      ▼
Vec Results:           Text Results:
{1: 0.2,               {7: 8.5,
 5: 0.3,                1: 7.2,
 7: 0.4}                12: 6.8}
```

#### Step 5: Score Fusion

```
      Both Result Sets
              │
              ▼
      ┌───────────────┐
      │ Score Fusion  │
      │ (RRF/Weighted)│
      └───────┬───────┘
              │
              ▼
      Fused Rankings
```

#### Step 6: Final Ranking

```
      ┌───────────────┐
      │  Rank & Sort  │
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │  Top-K Filter │
      │  (k=10)       │
      └───────┬───────┘
              │
              ▼
      Final Results:
      [{1: 0.8},
       {7: 0.7},
       {5: 0.6}]
```

### Memory Layout

Understanding how different index types use memory:

#### HNSW Index Memory

**File Header (24 bytes):**

```
┌─────────────────────────────┐
│ Magic:      "HNSW" (4 B)    │
│ Version:    1 (4 B)          │
│ Dimensions: 384 (4 B)        │
│ M:          16 (4 B)         │
│ Max Level:  3 (4 B)          │
│ Entry Point: 42 (4 B)        │
└─────────────────────────────┘
```

**Per-Node Storage:**

```
Node ID:       4 bytes
Level:         4 bytes
Vector Data:   1536 bytes (384-dim × 4)
Graph Edges:   320 bytes (M connections × layers)
              ─────────────
Total:         ~1864 bytes per node
```

**Scaling Analysis:**

```
┌───────────────────┬──────────┬───────────┐
│ Component         │ Per Node │ 1M Nodes  │
├───────────────────┼──────────┼───────────┤
│ Vectors (raw)     │ 1536 B   │ 1.46 GB   │
│ Graph structure   │ 320 B    │ 305 MB    │
│ Metadata          │ 8 B      │ 7.6 MB    │
├───────────────────┼──────────┼───────────┤
│ Total             │ 1864 B   │ 1.78 GB   │
└───────────────────┴──────────┴───────────┘
```

#### Product Quantization Memory

**Compression Overview:**

```
Original Vector (384-dim):
384 × 4 bytes = 1536 bytes
         ↓
    Quantization
         ↓
PQ Codes (8 subspaces):
8 × 1 byte = 8 bytes
         ↓
192x smaller!
```

**Codebook Storage:**

```
8 codebooks × 256 centroids × 48 dims × 4 bytes
= 393 KB (shared across all vectors)
```

**1M Vectors Comparison:**

```
┌───────────────────┬────────────┬──────────┐
│ Format            │ Size       │ Ratio    │
├───────────────────┼────────────┼──────────┤
│ Original (float32)│ 1.46 GB    │ 1x       │
│ PQ-8              │ 7.6 MB     │ 192x     │
│ PQ-16             │ 15.3 MB    │ 96x      │
│ PQ-32             │ 30.5 MB    │ 48x      │
│ + Codebooks       │ +393 KB    │ -        │
└───────────────────┴────────────┴──────────┘
```

## Core Concepts

### Concept 1: Vector Storage and Distance Metrics

A vector store maintains high-dimensional embeddings and enables efficient similarity search. The choice of storage type determines the tradeoff between search speed, memory usage, and accuracy.

**Example: How vectors are stored and searched**

Given these input vectors:

```
Vector 1: [0.1, 0.5, 0.3, 0.8]
Vector 2: [0.2, 0.4, 0.7, 0.1]
Query:    [0.15, 0.45, 0.5, 0.5]
```

The index computes distances and returns nearest neighbors:

```
┌─────────────┬────────────────────────────────┬──────────┐
│ Vector ID   │ Distance to Query              │ Rank     │
├─────────────┼────────────────────────────────┼──────────┤
│ 1           │ 0.234 (closest)                │ 1st      │
│ 2           │ 0.567 (further)                │ 2nd      │
└─────────────┴────────────────────────────────┴──────────┘
```

**Visual Representation: Storage Types Comparison**

#### Flat Index (Brute Force)

```
Query → Compare with ALL vectors → Sort → Return K

✓ 100% Accuracy
✗ O(n) time complexity
✗ Slow for large datasets
```

#### HNSW Index (Graph-Based)

```
Layer 2: ●─────────────●    (highways - long jumps)
          │             │
Layer 1: ●───●───●───●    (roads - medium jumps)
          │   │ \ │ \ │
Layer 0: ●─●─●─●─●─●─●    (streets - all nodes)

Search: Start high → Navigate greedily → Descend

✓ O(log n) time complexity
✓ 95-99% recall
✗ 2-3x memory overhead
```

#### IVF Index (Clustering)

```
Cluster 1: ●●●●     Cluster 2: ●●●●
Cluster 3: ●●●●     Cluster 4: ●●●●

Query → Find nearest clusters → Search within

✓ Fast on large datasets
✗ Requires training
~ 85-95% recall
```

#### Product Quantization (Compression)

```
Original:  [0.123, 0.456, 0.789, ...]
                    ↓ Compress
Quantized: [17, 42, 89, ...]

4 bytes each → 1 byte each

✓ 4-32x memory reduction
✗ Slight accuracy loss
```

**Benefits of Different Storage Types:**

- **Flat Index**: Perfect recall, simple, no training. Use for small datasets (<100K vectors)
- **HNSW**: Excellent speed/accuracy tradeoff, no training. Use for most production workloads
- **IVF**: Fast filtering with clusters, scalable. Use for very large datasets (>10M vectors)
- **PQ**: Massive memory savings. Use when storage cost is a concern
- **IVFPQ**: Best of IVF + PQ. Use for extremely large, memory-constrained environments

### Concept 2: BM25 Full-Text Search Algorithm

BM25 (Best Matching 25) ranks documents by relevance to a text query using term frequency and inverse document frequency.

#### Setup

**Test Corpus:**

```
Doc 1: "the quick brown fox jumps over the lazy dog"  (9 words)
Doc 2: "the lazy dog sleeps"                           (4 words)
Doc 3: "quick brown rabbits jump"                      (4 words)
```

**Query:** `"quick brown"`

**Parameters:**

```
Average Document Length: 7 words
K1 = 1.2  (term frequency saturation)
B  = 0.75 (length normalization)
```

#### Step 1: Calculate IDF (Inverse Document Frequency)

**For term "quick":**

```
Appears in: 2 out of 3 documents

IDF = log((N - df + 0.5) / (df + 0.5) + 1)
    = log((3 - 2 + 0.5) / (2 + 0.5) + 1)
        = log(1.5 / 2.5 + 1)
        = log(1.6)
        = 0.470
```

**For term "brown":**

```
Appears in: 2 out of 3 documents
IDF = 0.470  (same calculation)
```

#### Step 2: Calculate TF Component for Each Document

**Doc 1** (9 words - longer than average):

```
Term "quick" (tf=1):
  TF_score = (tf × (K1 + 1)) / (tf + K1 × (1 - B + B × (docLen/avgLen)))
           = (1 × 2.2) / (1 + 1.2 × (1 - 0.75 + 0.75 × (9/7)))
             = 2.2 / 2.457
             = 0.895

Term "brown" (tf=1):
  TF_score = 0.895  (same calculation)
```

**Doc 2** (4 words):

```
Terms "quick" and "brown": tf=0 (not present)
    TF_score = 0
```

**Doc 3** (4 words - shorter than average):

```
Term "quick" (tf=1):
  TF_score = (1 × 2.2) / (1 + 1.2 × (1 - 0.75 + 0.75 × (4/7)))
             = 2.2 / 1.815
             = 1.212

Term "brown" (tf=1):
  TF_score = 1.212  (same calculation)
```

#### Step 3: Calculate Final BM25 Scores

**Combine IDF × TF for each document:**

```
Doc 1: (0.895 × 0.470) + (0.895 × 0.470) = 0.841
Doc 2: 0  (no matching terms)
Doc 3: (1.212 × 0.470) + (1.212 × 0.470) = 1.139
```

#### Final Ranking

```
┌─────────┬──────────┬────────┐
│ Rank    │ Doc ID   │ Score  │
├─────────┼──────────┼────────┤
│ 1st     │ Doc 3    │ 1.139  │
│ 2nd     │ Doc 1    │ 0.841  │
│ 3rd     │ Doc 2    │ 0.000  │
└─────────┴──────────┴────────┘
```

#### Why Doc 3 Ranks Higher

```
Doc 1 vs Doc 3:
├─ Same term frequencies (1 occurrence each)
├─ Doc 3 is shorter (4 words vs 9 words)
├─ BM25 length normalization penalizes longer docs
└─ Result: Doc 3 gets higher TF scores (1.212 vs 0.895)

Key Insight: Shorter documents with same term frequency
             are considered more relevant ✓
```

### Concept 3: Hybrid Search with Score Fusion

Hybrid search combines vector similarity, text relevance, and metadata filtering. Different fusion strategies handle score normalization differently.

#### Query Setup

```
INPUT:
├─ Query:   "machine learning tutorial"
├─ Vector:  [0.12, 0.45, 0.89, ...]  (embedding)
└─ Filter:  category="education" AND price<50
```

#### Step 1: Apply Metadata Filter

```
┌─────────────────────────────────┐
│  Metadata Filtering             │
│  category="education"           │
│  price<50                       │
└─────────────────────────────────┘
                │
                ▼
Candidate Docs: {1, 3, 5, 7, 9, 12, 15, 18, 20}
```

#### Step 2: Vector Search Results

```
Semantic Similarity (on candidates):

┌────────┬──────────┬──────┐
│ Doc ID │ Distance │ Rank │
├────────┼──────────┼──────┤
│   1    │  0.12    │  1   │  ← Closest
│   5    │  0.23    │  2   │
│   7    │  0.34    │  3   │
│  12    │  0.45    │  4   │
└────────┴──────────┴──────┘
```

#### Step 3: Text Search Results

```
BM25 Ranking (on candidates):

┌────────┬────────────┬──────┐
│ Doc ID │ BM25 Score │ Rank │
├────────┼────────────┼──────┤
│   7    │   8.5      │  1   │  ← Most relevant
│   1    │   7.2      │  2   │
│  12    │   6.8      │  3   │
│   5    │   4.1      │  4   │
└────────┴────────────┴──────┘
```

#### Step 4: Reciprocal Rank Fusion (RRF)

**Formula:** `RRF_score = sum(1 / (K + rank_i))` where K=60

**Doc 1 Calculation:**

```
  Vector rank: 1 → 1/(60+1) = 0.0164
Text rank:   2 → 1/(60+2) = 0.0161
                          ────────
Combined RRF score:         0.0325
```

**Doc 5 Calculation:**

```
  Vector rank: 2 → 1/(60+2) = 0.0161
Text rank:   4 → 1/(60+4) = 0.0156
                          ────────
Combined RRF score:         0.0317
```

**Doc 7 Calculation:**

```
  Vector rank: 3 → 1/(60+3) = 0.0159
Text rank:   1 → 1/(60+1) = 0.0164
                          ────────
Combined RRF score:         0.0323
```

**Doc 12 Calculation:**

```
  Vector rank: 4 → 1/(60+4) = 0.0156
Text rank:   3 → 1/(60+3) = 0.0159
                          ────────
Combined RRF score:         0.0315
```

#### Final Ranking

```
┌──────┬────────┬───────────┬──────────────────┐
│ Rank │ Doc ID │ RRF Score │ Why              │
├──────┼────────┼───────────┼──────────────────┤
│ 1st  │   1    │  0.0325   │ Best vector      │
│ 2nd  │   7    │  0.0323   │ Best text        │
│ 3rd  │   5    │  0.0317   │ Balanced         │
│ 4th  │  12    │  0.0315   │ Lower in both    │
└──────┴────────┴───────────┴──────────────────┘
```

#### Why RRF Over Weighted Sum?

```
Problem with Weighted Sum:
├─ Vector distances:    0-2 range
├─ BM25 scores:         0-100+ range
└─ Different scales need manual tuning ✗

RRF Advantages:
├─ ✓ Scale Independent  (uses ranks, not raw scores)
├─ ✓ Robust             (stable across score distributions)
├─ ✓ No Tuning          (no manual weight calibration)
└─ ✓ Industry Standard  (Elasticsearch, Vespa, etc.)
```

## Storage Type Deep Dive

**Choose your poison:** Flat (exact), HNSW (graph), IVF (clustering), PQ (quantization), or IVFPQ (hybrid). Each trades speed, memory, and accuracy differently.

> **Full Deep Dive:** See [INDEX.md](docs/INDEX.md) for complete algorithms, benchmarks, and implementation details.

### Flat Index (Brute Force)

**The Simplest Approach**: Compare query against EVERY vector. 100% recall, zero approximation.

#### How It Works

```
Query Vector
     ↓
┌────────────────────┐
│  Compare to ALL n  │
│     vectors        │  ← Every single one checked
└────────────────────┘   No shortcuts!
        ↓
   Sort by distance
        ↓
   Return top K
```

#### Implementation

```
1. Store vectors → Raw float32 arrays
2. Search step   → Compute distance to all vectors
3. Selection     → Keep top-k closest
4. Index struct  → None (just a list)
```

#### Complexity Analysis

```
┌─────────────┬──────────────────────────┐
│ Operation   │ Complexity               │
├─────────────┼──────────────────────────┤
│ Build       │ O(1) - just store        │
│ Search      │ O(n × d) - check all     │
│ Memory      │ O(n × d) - raw vectors   │
│ Recall      │ 100% - exhaustive        │
└─────────────┴──────────────────────────┘
```

#### When to Use

```
✓ Small datasets (<10K vectors)
✓ 100% recall required (fingerprinting, security)
✓ Benchmarking baseline
✗ Large datasets (too slow)
```

---

### HNSW Index (Hierarchical Graph)

**The Graph Navigator**: Build a multi-layer graph where each node connects to nearby vectors. Search by greedy navigation from layer to layer.

#### Graph Structure

```
Layer 2: [A]─────────[D]           ← Highways (long jumps)
          │           │              Few nodes

Layer 1: [A]──[B]────[D]──[E]      ← Roads (medium jumps)
          │   │ \    │ \  │          More nodes

Layer 0: [A]─[B]─[C]─[D]─[E]─[F]   ← Streets (short links)
                                      ALL vectors here
```

#### Search Process

```
1. Start at top layer    → Long-range navigation
2. Greedy best move      → Move to closest neighbor
3. Drop to next layer    → Refine search
4. Repeat steps 2-3      → Until Layer 0
5. Return k-nearest      → Final results
```

#### How Insertion Works

```
New Vector:
├─ 1. Add to Layer 0 (always)
├─ 2. Randomly assign to higher layers (exponential decay)
├─ 3. Connect to M nearest neighbors per layer
└─ 4. Update neighbors' connections (bidirectional)
```

#### Complexity Analysis

```
┌─────────────┬────────────────────────────────┐
│ Operation   │ Complexity                     │
├─────────────┼────────────────────────────────┤
│ Search      │ O(M × efSearch × log n)        │
│             │ Typically checks <1% of data   │
│ Insert      │ O(M × efConstruction × log n)  │
│ Memory      │ O(n × d + n × M × log n)       │
│ Recall      │ 95-99% with proper tuning      │
└─────────────┴────────────────────────────────┘
```

#### Key Parameters

```
M (Connections per layer):
├─ 4-8:   Low memory, lower recall
├─ 16:    Balanced (default)
└─ 32-48: High recall, more memory

efConstruction (Build quality):
├─ 100:   Fast build, lower quality
├─ 200:   Good balance (default)
└─ 400+:  Better graph, slower build

efSearch (Search quality):
├─ 50:    Fast search, ~85% recall
├─ 200:   Balanced (default), ~96% recall
└─ 400+:  High recall, slower search
```

#### When to Use

```
✓ Large datasets (10K-10M vectors)
✓ Need 95-99% recall
✓ Sub-millisecond latency required
✓ Can afford 2-3x memory overhead
✗ Memory constrained environments
```

---

### IVF Index (Inverted File with Clustering)

**The Clustering Approach**: Partition vectors into clusters using k-means. Search only the nearest clusters.

#### Build Phase (Training)

```
All Vectors (n)
      ↓
   k-means clustering
      ↓
┌────────────────────────────────────┐
│ [C1]  [C2]  [C3]  [C4]  ... [Cn]  │  ← Centroids
└────────────────────────────────────┘
   ↓      ↓     ↓     ↓         ↓
  {v1}   {v8}  {v3}  {v12}    {v7}     ← Vectors assigned
  {v5}   {v9}  {v6}  {v19}    {v11}      to nearest
  {v20}  ...   ...   ...      ...        centroid
```

#### Search Phase

```
Query Vector
      ↓
Find nProbe nearest centroids
      ↓
┌────────────────────────────┐
│ [C2] [C3]  (only these!)   │  ← Search 2 of 100 clusters
└────────────────────────────┘
   ↓     ↓
  {...} {...}  Search within clusters
      ↓
  Top-k results
```

#### Key Insight

```
Instead of:  Check all 1M vectors
Do this:     1. Find 8 nearest clusters  (O(nClusters))
             2. Search 80K vectors total  (10% of data)
             └─> 10-20x faster!
```

#### Complexity Analysis

```
┌─────────────┬────────────────────────────────┐
│ Operation   │ Complexity                     │
├─────────────┼────────────────────────────────┤
│ Build       │ O(iterations × n × k × d)      │
│             │ k-means clustering             │
│ Search      │ O(k × d + (n/k) × nProbe × d)  │
│             │ Typically checks 5-20% of data │
│ Memory      │ O(n × d + k × d)               │
│ Recall      │ 80-95% depending on nProbe     │
└─────────────┴────────────────────────────────┘
```

#### Key Parameters

```
nClusters (number of partitions):
├─ Rule of thumb: sqrt(n) to n/10
├─ 10K vectors   → 100 clusters
├─ 100K vectors  → 316 clusters
├─ 1M vectors    → 1,000 clusters
└─ More clusters → Faster search, slower build

nProbe (clusters to search):
├─ 1:   Fastest, ~60-70% recall
├─ 8:   Good balance, ~85% recall
├─ 16:  Better recall, ~92% recall
└─ 32+: High recall, ~96% recall
```

#### When to Use

```
✓ Large datasets (>100K vectors)
✓ Can tolerate 85-95% recall
✓ Want 10-20x speedup over Flat
✓ Have training data available
✗ Need 100% recall (use Flat)
✗ Dataset too small (<10K)
```

---

### PQ Index (Product Quantization)

**The Compression Master**: Split vectors into subvectors, quantize each subvector to 256 codes. Reduce memory by 16-32×.

#### Compression Process

```
Original Vector (384 dimensions):
  [0.23, 0.91, ..., 0.15, 0.44, ..., 0.73, 0.22, ...]
   \_____48D_____/  \_____48D_____/  \_____48D_____/
   Subspace 1       Subspace 2       Subspace 3
        ↓                ↓                ↓
   K-means on       K-means on       K-means on
   subspace 1       subspace 2       subspace 3
        ↓                ↓                ↓
 Codebook with    Codebook with    Codebook with
 256 centroids    256 centroids    256 centroids
        ↓                ↓                ↓
   Find nearest    Find nearest     Find nearest
   centroid ID     centroid ID      centroid ID
        ↓                ↓                ↓
      [12]             [203]            [45]
       1 byte          1 byte           1 byte
```

#### Result

```
Before: [0.23, 0.91, ...] → 384 × 4 bytes = 1536 bytes
After:  [12, 203, 45, ...]  → 8 × 1 byte  = 8 bytes

192x compression!
```

#### Search Process

```
1. Query arrives
        ↓
2. Split query into subspaces (like training)
        ↓
3. Precompute distance tables for each subspace
   (query_subspace to all 256 codebook centroids)
        ↓
4. For each vector:
   - Look up codes: [12, 203, 45, ...]
   - Table lookup: distances[12] + distances[203] + ...
   - No float operations! Just array lookups
        ↓
5. Return top-k
```

#### Complexity Analysis

```
┌─────────────┬────────────────────────────────┐
│ Operation   │ Complexity                     │
├─────────────┼────────────────────────────────┤
│ Training    │ O(M × iterations × K × n/M)    │
│             │ K-means on each subspace       │
│ Encoding    │ O(M × K × d/M) per vector      │
│ Search      │ O(M × K + n × M)               │
│             │ Super fast, cache-friendly     │
│ Memory      │ O(n × M) - massive savings!    │
│ Recall      │ 70-85% typical                 │
└─────────────┴────────────────────────────────┘
```

#### Memory Comparison

```
1M vectors, 384 dimensions:

Float32:  1M × 384 × 4 bytes = 1.46 GB
PQ-8:     1M × 8 × 1 byte    = 7.6 MB  (192x smaller!)
PQ-16:    1M × 16 × 1 byte   = 15.3 MB (96x smaller!)
PQ-32:    1M × 32 × 1 byte   = 30.5 MB (48x smaller!)

+ Codebooks: ~393 KB (shared across all vectors)
```

#### Key Parameters

```
M (nSubspaces):
├─ 8:   Maximum compression, lower accuracy
├─ 16:  Good balance
├─ 32:  Better accuracy, less compression
└─ 64:  High accuracy, moderate compression

bitsPerCode:
└─ 8:   Standard (256 centroids per subspace)
        Perfect for uint8 storage
```

#### When to Use

```
✓ Massive datasets (millions of vectors)
✓ Memory is the bottleneck
✓ Can tolerate 70-85% recall
✓ Want 30-200x memory reduction
✗ Need high recall (>95%)
✗ Have plenty of memory
```

---

### IVFPQ Index (Hybrid: Clustering + Quantization)

**Best of Both Worlds**: IVF clusters to reduce search space, PQ compression to reduce memory. The ultimate scalability index.

#### Build Process

**Step 1: IVF Clustering**

```
All Vectors
      ↓
  K-means clustering
      ↓
[C1]  [C2]  [C3]  [C4]  ... [Cn]
```

**Step 2: PQ Compression per Cluster**

```
Cluster 1:        Cluster 2:        Cluster 3:
  {vectors}         {vectors}         {vectors}
      ↓                 ↓                 ↓
  Apply PQ          Apply PQ          Apply PQ
      ↓                 ↓                 ↓
[12,203,45]      [91,34,178]       [56,211,19]
[88,9,101]       [23,156,88]       [199,44,73]
[...]            [...]             [...]
           uint8 codes      uint8 codes       uint8 codes
```

#### Search Process

```
Step 1: IVF Stage
  Query → Find nProbe nearest centroids
          ↓
     [C2] [C3] [C7]  (e.g., 3 of 100 clusters)
          ↓
     Search only 3% of the data!

Step 2: PQ Stage
  For each selected cluster:
    ├─ Precompute distance tables
    ├─ Fast table lookups on PQ codes
    └─ No float operations
          ↓
     Top-k results
```

#### The Magic Combination

```
IVF contributes:
└─ Speed:  10-100x faster (search only nProbe clusters)

PQ contributes:
└─ Memory: 30-200x smaller (compressed codes)

Combined:
└─ Fast + Tiny = Billion-scale capability!
```

#### Complexity Analysis

```
┌─────────────┬────────────────────────────────┐
│ Operation   │ Complexity                     │
├─────────────┼────────────────────────────────┤
│ Training    │ O(IVF_kmeans + PQ_kmeans)      │
│ Search      │ O(k×d + (n/k)×nProbe×M)        │
│             │ Searches ~1-10% of data        │
│ Memory      │ O(n×M + k×d)                   │
│             │ Massive compression            │
│ Recall      │ 70-90% depending on params     │
└─────────────┴────────────────────────────────┘
```

#### Memory Savings Example

```
1M vectors, 384 dimensions:

Float32 + IVF:   1.46 GB
IVFPQ (M=8):     7.6 MB + 400 KB (centroids)
                 = ~8 MB total

180x compression!
```

#### Key Parameters

```
nClusters (IVF):
├─ 100:   For 100K vectors
├─ 1K:    For 1M vectors
└─ 10K:   For 100M vectors

nProbe (IVF search):
├─ 1:     Fastest, lower recall
├─ 8:     Good balance
└─ 16:    Better recall

M (PQ subspaces):
├─ 8:     Maximum compression
├─ 16:    Good balance
└─ 32:    Better accuracy
```

#### When to Use

```
✓ Billion-scale datasets
✓ Need <10ms latency
✓ Severe memory constraints
✓ Can tolerate 70-90% recall
✓ Want 100x speed + 100x compression
✗ Need >95% recall (use HNSW)
✗ Small datasets (use Flat or HNSW)
```

#### Real-World Example

```
Use Case: 100M vectors, 768-dim

Flat Index:
├─ Memory: ~288 GB
├─ Search: ~10 seconds
└─ Not feasible! ✗

IVFPQ:
├─ Memory: ~800 MB
├─ Search: ~5 ms
└─ Practical! ✓
```

---

### Decision Matrix

| Index | Recall | Search Speed  | Memory   | Build Time | Best For                        |
| ----- | ------ | ------------- | -------- | ---------- | ------------------------------- |
| Flat  | 100%   | Slow O(n)     | High     | Instant    | <10k vectors, benchmarks        |
| HNSW  | 90-99% | Fast O(log n) | Highest  | Slow       | 10k-10M vectors, low latency    |
| IVF   | 80-95% | Medium        | High     | Medium     | >100k vectors, moderate recall  |
| PQ    | 70-85% | Fast          | Lowest   | Slow       | >1M vectors, memory-constrained |
| IVFPQ | 70-90% | Fastest       | Very Low | Slow       | >10M vectors, billion-scale     |

**Rules of thumb:**

- **Small data (<10k)**: Flat - brute force is fast enough
- **Medium data (10k-1M)**: HNSW - best recall/speed tradeoff
- **Large data (1M-100M)**: IVF or PQ - choose speed (IVF) vs memory (PQ)
- **Massive data (>100M)**: IVFPQ - only option that scales to billions

**Latency targets:**

- Flat: 1-100ms (depends on n)
- HNSW: 0.1-2ms (sub-millisecond possible)
- IVF: 0.5-5ms
- PQ: 1-10ms (fast scan but approximate)
- IVFPQ: 0.5-3ms (fastest for massive datasets)

## API Reference

For detailed API documentation, see [API.md](docs/API.md).

## Examples

For practical examples and code samples, see [EXAMPLE.md](docs/EXAMPLE.md).

## Configuration

### Basic Configuration

```go
// Flat Index - No configuration needed
flatIdx, _ := comet.NewFlatIndex(384, comet.Cosine)

// HNSW Index - Configure build and search parameters
hnswIdx, _ := comet.NewHNSWIndex(
    384,              // dimensions
    comet.Cosine,     // distance metric
    16,               // M: connections per layer
    200,              // efConstruction: build quality
    200,              // efSearch: search quality
)

// IVF Index - Configure clustering
ivfIdx, _ := comet.NewIVFIndex(
    384,              // dimensions
    comet.Cosine,     // distance metric
    100,              // nClusters: number of partitions
)

// Training required for IVF
trainingVectors := []comet.VectorNode{ /* ... */ }
ivfIdx.Train(trainingVectors)
```

### Configuration Options

#### HNSW Parameters

**M (Connections Per Layer)**

- Type: `int`
- Default: `16`
- Range: 4 to 64
- Description: Number of bidirectional connections per node at each layer (except layer 0 which uses 2×M)

Effects of different values:

```
┌─────────┬──────────────────────────────────────────┐
│ Value   │ Effect                                   │
├─────────┼──────────────────────────────────────────┤
│ 4-8     │ Low memory, faster build, lower recall   │
│ 12-16   │ Balanced (recommended for most cases)    │
│ 24-48   │ High recall, slower build, more memory   │
│ 64+     │ Diminishing returns, excessive memory    │
└─────────┴──────────────────────────────────────────┘
```

**efConstruction (Build-Time Candidate List Size)**

- Type: `int`
- Default: `200`
- Range: 100 to 1000
- Description: Size of candidate list during index construction. Higher = better graph quality

Effects:

```
┌─────────┬──────────────────────────────────────────┐
│ Value   │ Effect                                   │
├─────────┼──────────────────────────────────────────┤
│ 100     │ Fast build, lower quality graph          │
│ 200     │ Good balance (default)                   │
│ 400-500 │ Better quality, 2-3x slower build        │
│ 800+    │ Marginal gains, very slow build          │
└─────────┴──────────────────────────────────────────┘
```

**efSearch (Search-Time Candidate List Size)**

- Type: `int`
- Default: `200`
- Range: k to 1000 (must be >= k)
- Description: Size of candidate list during search. Can be adjusted dynamically.

Effects:

```
┌─────────┬──────────────────────────────────────────┐
│ Value   │ Effect                                   │
├─────────┼──────────────────────────────────────────┤
│ 50      │ Very fast, ~85% recall                   │
│ 100     │ Fast, ~92% recall                        │
│ 200     │ Balanced, ~96% recall                    │
│ 400     │ Slower, ~98% recall                      │
│ 800     │ Much slower, ~99% recall                 │
└─────────┴──────────────────────────────────────────┘
```

Example:

```go
// Create HNSW with high quality settings
index, _ := comet.NewHNSWIndex(
    384,
    comet.Cosine,
    32,    // More connections = better recall
    400,   // Higher construction quality
    200,   // Search quality (can adjust later)
)

// Dynamically adjust search quality
index.SetEfSearch(400)  // Trade speed for recall
```

#### IVF Parameters

**nClusters (Number of Clusters)**

- Type: `int`
- Default: sqrt(n) where n = dataset size
- Range: 16 to n/100
- Description: Number of k-means clusters (Voronoi cells) to partition the space

```
┌─────────────┬────────────┬─────────────────────────┐
│ Dataset Size│ nClusters  │ Typical Range           │
├─────────────┼────────────┼─────────────────────────┤
│ 10K         │ 100        │ 50-200                  │
│ 100K        │ 316        │ 200-500                 │
│ 1M          │ 1000       │ 500-2000                │
│ 10M         │ 3162       │ 2000-5000               │
└─────────────┴────────────┴─────────────────────────┘
```

**nProbes (Search-Time Clusters to Probe)**

- Type: `int`
- Default: `1`
- Range: 1 to nClusters
- Description: Number of nearest clusters to search during query

```
┌─────────┬──────────────────────────────────────────┐
│ nProbes │ Effect                                   │
├─────────┼──────────────────────────────────────────┤
│ 1       │ Fastest, ~60-70% recall                  │
│ 8       │ Good balance, ~85% recall                │
│ 16      │ Better recall, ~92% recall               │
│ 32      │ High recall, ~96% recall                 │
│ 64      │ Very high recall, slower                 │
└─────────┴──────────────────────────────────────────┘
```

Example:

```go
// Create IVF index
index, _ := comet.NewIVFIndex(384, comet.Cosine, 256)

// Train with representative sample
trainData := sampleVectors(10000)  // 10K samples for training
index.Train(trainData)

// Search with multiple probes
results, _ := index.NewSearch().
    WithQuery(query).
    WithK(10).
    WithNProbes(16).  // Search 16 nearest clusters
    Execute()
```

### Advanced Configuration

#### Fusion Configuration

```go
// Weighted Sum Fusion (manual weights)
config := &comet.FusionConfig{
    VectorWeight: 0.7,  // 70% weight to semantic similarity
    TextWeight:   0.3,  // 30% weight to keyword relevance
}
fusion, _ := comet.NewFusion(comet.WeightedSumFusion, config)

hybridSearch := hybrid.NewSearch().
    WithVector(queryVec).
    WithText("machine learning").
    WithFusion(fusion).
    Execute()

// Reciprocal Rank Fusion (rank-based, no weights needed)
hybridSearch := hybrid.NewSearch().
    WithVector(queryVec).
    WithText("machine learning").
    WithFusionKind(comet.ReciprocalRankFusion).
    Execute()
```

#### Distance Metrics

```go
// Euclidean (L2): Use when magnitude matters
euclideanIdx, _ := comet.NewFlatIndex(384, comet.Euclidean)

// L2 Squared: Faster than Euclidean (no sqrt), preserves ranking
l2sqIdx, _ := comet.NewFlatIndex(384, comet.L2Squared)

// Cosine: Use for normalized vectors (text embeddings, etc.)
cosineIdx, _ := comet.NewFlatIndex(384, comet.Cosine)
```

## API Details

### FlatIndex

**Constructor:**

```go
func NewFlatIndex(dim int, distanceKind DistanceKind) (*FlatIndex, error)
```

**Parameters:**

| Parameter      | Type           | Required | Description                                            |
| -------------- | -------------- | -------- | ------------------------------------------------------ |
| `dim`          | `int`          | Yes      | Vector dimension (must be > 0)                         |
| `distanceKind` | `DistanceKind` | Yes      | Distance metric: `Euclidean`, `L2Squared`, or `Cosine` |

**Returns:**

- `*FlatIndex`: Initialized flat index
- `error`: Error if parameters are invalid

### HNSWIndex

**Constructor:**

```go
func NewHNSWIndex(dim int, distanceKind DistanceKind, m, efConstruction, efSearch int) (*HNSWIndex, error)
```

**Parameters:**

| Parameter        | Type           | Required | Default          | Description                                                           |
| ---------------- | -------------- | -------- | ---------------- | --------------------------------------------------------------------- |
| `dim`            | `int`          | Yes      | -                | Vector dimension (must be > 0)                                        |
| `distanceKind`   | `DistanceKind` | Yes      | -                | Distance metric: `Euclidean`, `L2Squared`, or `Cosine`                |
| `m`              | `int`          | No       | 16               | Max connections per node (higher = better accuracy, more memory)      |
| `efConstruction` | `int`          | No       | 200              | Build-time search depth (higher = better graph quality, slower build) |
| `efSearch`       | `int`          | No       | `efConstruction` | Query-time search depth (higher = better accuracy, slower search)     |

**Returns:**

- `*HNSWIndex`: Initialized HNSW index
- `error`: Error if parameters are invalid

**Parameter Guidelines:**

```
┌──────────────┬─────┬────────────────┬──────────┬─────────────┐
│ Use Case     │  M  │ efConstruction │ efSearch │ Description │
├──────────────┼─────┼────────────────┼──────────┼─────────────┤
│ Fast         │  8  │      100       │    50    │ Speed first │
│ Balanced     │ 16  │      200       │   100    │ Default     │
│ High Recall  │ 32  │      400       │   200    │ Accuracy    │
│ Memory Eff.  │  4  │       50       │    25    │ Low memory  │
└──────────────┴─────┴────────────────┴──────────┴─────────────┘
```

### IVFIndex

**Constructor:**

```go
func NewIVFIndex(dim int, nlist int, distanceKind DistanceKind) (*IVFIndex, error)
```

**Parameters:**

| Parameter      | Type           | Required | Description                                            |
| -------------- | -------------- | -------- | ------------------------------------------------------ |
| `dim`          | `int`          | Yes      | Vector dimension (must be > 0)                         |
| `nlist`        | `int`          | Yes      | Number of clusters/partitions (must be > 0)            |
| `distanceKind` | `DistanceKind` | Yes      | Distance metric: `Euclidean`, `L2Squared`, or `Cosine` |

**Returns:**

- `*IVFIndex`: Initialized IVF index
- `error`: Error if parameters are invalid

**Parameter Guidelines:**

```
nlist Selection:
├─ Small dataset (10K-100K):    nlist = sqrt(n) = 100-300
├─ Medium dataset (100K-1M):    nlist = sqrt(n) = 300-1000
├─ Large dataset (1M-10M):      nlist = sqrt(n) = 1000-3000
└─ Very large (10M+):           nlist = sqrt(n) = 3000-10000

Rule of thumb: nlist ≈ sqrt(number_of_vectors)
```

**Search Parameters:**

```go
// nProbe: number of clusters to search (1 to nlist)
results, _ := index.NewSearch().
    WithQuery(queryVec).
    WithK(10).
    WithNProbe(8).  // Search top 8 nearest clusters
    Execute()

Speed vs Accuracy:
├─ nProbe = 1:     Fastest, lowest recall
├─ nProbe = 8:     Balanced (typical)
├─ nProbe = nlist: Exhaustive (same as flat)
```

### PQIndex

**Constructor:**

```go
func NewPQIndex(dim int, distanceKind DistanceKind, M int, Nbits int) (*PQIndex, error)
```

**Parameters:**

| Parameter      | Type           | Required | Constraint   | Description                                            |
| -------------- | -------------- | -------- | ------------ | ------------------------------------------------------ |
| `dim`          | `int`          | Yes      | > 0          | Vector dimension                                       |
| `distanceKind` | `DistanceKind` | Yes      | -            | Distance metric: `Euclidean`, `L2Squared`, or `Cosine` |
| `M`            | `int`          | Yes      | dim % M == 0 | Number of subspaces (dim must be divisible by M)       |
| `Nbits`        | `int`          | Yes      | 1-16         | Bits per subspace (typical: 8)                         |

**Returns:**

- `*PQIndex`: Initialized PQ index
- `error`: Error if parameters are invalid or constraints violated

**Parameter Guidelines:**

```
M (Number of Subspaces):
├─ M = 8:    192x compression (typical, good balance)
├─ M = 16:    96x compression (better accuracy)
├─ M = 32:    48x compression (highest accuracy)
└─ Constraint: dim must be divisible by M

Nbits (Codebook size = 2^Nbits):
├─ Nbits = 4:   16 centroids per subspace (very fast)
├─ Nbits = 8:  256 centroids per subspace (typical)
├─ Nbits = 12: 4096 centroids (high accuracy)
└─ Constraint: 1 ≤ Nbits ≤ 16

Common Configurations:
┌──────┬───────┬───────────────┬─────────────────────┐
│  M   │ Nbits │ Compression   │ Use Case            │
├──────┼───────┼───────────────┼─────────────────────┤
│   8  │   8   │ 192x (1 byte) │ Maximum compression │
│  16  │   8   │  96x (2 byte) │ Balanced            │
│  32  │   8   │  48x (4 byte) │ Better accuracy     │
│   8  │  12   │ 128x (1.5 B)  │ Higher quality      │
└──────┴───────┴───────────────┴─────────────────────┘
```

### IVFPQIndex

**Constructor:**

```go
func NewIVFPQIndex(dim int, distanceKind DistanceKind, nlist int, m int, nbits int) (*IVFPQIndex, error)
```

**Parameters:**

| Parameter      | Type           | Required | Constraint   | Description                                            |
| -------------- | -------------- | -------- | ------------ | ------------------------------------------------------ |
| `dim`          | `int`          | Yes      | > 0          | Vector dimension                                       |
| `distanceKind` | `DistanceKind` | Yes      | -            | Distance metric: `Euclidean`, `L2Squared`, or `Cosine` |
| `nlist`        | `int`          | Yes      | > 0          | Number of IVF clusters                                 |
| `m`            | `int`          | Yes      | dim % m == 0 | Number of PQ subspaces                                 |
| `nbits`        | `int`          | Yes      | 1-16         | Bits per PQ subspace                                   |

**Returns:**

- `*IVFPQIndex`: Initialized IVFPQ index
- `error`: Error if parameters are invalid

**Parameter Guidelines:**

```
Combined IVF + PQ Configuration:

IVF Parameters:
├─ nlist ≈ sqrt(n) for number of vectors
└─ See IVFIndex guidelines above

PQ Parameters:
├─ m: Typically 8, 16, or 32
└─ nbits: Typically 8

Common Configurations for 100M vectors (384-dim):
┌────────┬─────┬────────┬──────────┬─────────────────┐
│ nlist  │  m  │ nbits  │ Memory   │ Use Case        │
├────────┼─────┼────────┼──────────┼─────────────────┤
│  4096  │  8  │   8    │  ~800 MB │ Extreme speed   │
│  8192  │  8  │   8    │  ~900 MB │ Balanced        │
│ 16384  │ 16  │   8    │ ~1.6 GB  │ Better accuracy │
│  8192  │  8  │  12    │ ~1.2 GB  │ High quality    │
└────────┴─────┴────────┴──────────┴─────────────────┘

Memory Savings Example (100M vectors, 384-dim):
├─ Original float32:  100M × 384 × 4 = 146 GB
├─ IVF only:          Still ~146 GB (no compression)
├─ PQ only:           100M × 8 × 1 = 760 MB (faster train)
└─ IVFPQ (m=8):       100M × 8 × 1 + centroids ≈ 800 MB (best of both!)
```

**Search Parameters:**

```go
results, _ := index.NewSearch().
    WithQuery(queryVec).
    WithK(10).
    WithNProbe(8).       // IVF: clusters to search
    WithNRefine(100).    // Optional: refine top-100 with original vectors
    Execute()
```

### BM25SearchIndex

**Constructor:**

```go
func NewBM25SearchIndex() *BM25SearchIndex
```

**Parameters:** None (uses default BM25 parameters: k1=1.5, b=0.75)

**Returns:**

- `*BM25SearchIndex`: Initialized BM25 full-text search index

**BM25 Parameters:**

- `k1 = 1.5`: Term frequency saturation (typical range: 1.2-2.0)
- `b = 0.75`: Document length normalization (typical range: 0.5-0.9)

### RoaringMetadataIndex

**Constructor:**

```go
func NewRoaringMetadataIndex() *RoaringMetadataIndex
```

**Parameters:** None

**Returns:**

- `*RoaringMetadataIndex`: Initialized metadata filtering index using Roaring Bitmaps

### HybridSearchIndex

**Constructor:**

```go
func NewHybridSearchIndex(
    vectorIndex VectorIndex,
    textIndex TextIndex,
    metadataIndex MetadataIndex,
) HybridSearchIndex
```

**Parameters:**

| Parameter       | Type            | Required | Description                                   |
| --------------- | --------------- | -------- | --------------------------------------------- |
| `vectorIndex`   | `VectorIndex`   | Yes      | Any vector index (Flat, HNSW, IVF, PQ, IVFPQ) |
| `textIndex`     | `TextIndex`     | Yes      | BM25 text search index                        |
| `metadataIndex` | `MetadataIndex` | Yes      | Roaring metadata filter index                 |

**Returns:**

- `HybridSearchIndex`: Initialized hybrid search combining all three modalities

## Search APIs

### Vector Search (VectorSearch Interface)

**Creating a Search:**

```go
search := index.NewSearch()  // Available on all VectorIndex implementations
```

**Methods:**

| Method                 | Parameters             | Description                                                                               |
| ---------------------- | ---------------------- | ----------------------------------------------------------------------------------------- |
| `WithQuery`            | `queries ...[]float32` | Set query vector(s) for similarity search                                                 |
| `WithNodes`            | `nodes ...uint32`      | Find vectors similar to indexed nodes by ID                                               |
| `WithK`                | `k int`                | Number of results to return (default: 10)                                                 |
| `WithDocumentIDs`      | `ids ...uint32`        | Pre-filter: only search within these document IDs                                         |
| `WithScoreAggregation` | `agg ScoreAggregation` | How to combine multi-query results: `SumAggregation`, `MaxAggregation`, `MeanAggregation` |
| `WithThreshold`        | `threshold float32`    | Only return results with distance ≤ threshold                                             |
| `WithAutocut`          | `enabled bool`         | Automatically determine optimal cutoff point                                              |
| `Execute`              | -                      | Run the search, returns `[]VectorResult, error`                                           |

**HNSW-Specific Methods:**

| Method         | Parameters | Description                              |
| -------------- | ---------- | ---------------------------------------- |
| `WithEfSearch` | `ef int`   | Override default efSearch for this query |

**IVF-Specific Methods:**

| Method       | Parameters   | Description                               |
| ------------ | ------------ | ----------------------------------------- |
| `WithNProbe` | `nprobe int` | Number of clusters to search (1 to nlist) |

**IVFPQ-Specific Methods:**

| Method        | Parameters    | Description                                |
| ------------- | ------------- | ------------------------------------------ |
| `WithNProbe`  | `nprobe int`  | Number of clusters to search               |
| `WithNRefine` | `nrefine int` | Refine top-N results with original vectors |

**Examples:**

```go
// Basic single-query search
results, err := index.NewSearch().
    WithQuery(queryVector).
    WithK(10).
    Execute()

// Multi-query with aggregation
results, _ := index.NewSearch().
    WithQuery(query1, query2, query3).
    WithK(20).
    WithScoreAggregation(comet.MeanAggregation).
    Execute()

// Node-based search (find similar to existing vectors)
results, _ := index.NewSearch().
    WithNodes(1, 5, 10).  // IDs of indexed vectors
    WithK(10).
    Execute()

// Pre-filtered search
eligibleIDs := []uint32{100, 200, 300, 400, 500}
results, _ := index.NewSearch().
    WithQuery(queryVector).
    WithDocumentIDs(eligibleIDs...).
    WithK(5).
    Execute()

// Distance threshold filtering
results, _ := index.NewSearch().
    WithQuery(queryVector).
    WithK(100).
    WithThreshold(0.5).  // Only distances ≤ 0.5
    Execute()

// Autocut (automatic result truncation)
results, _ := index.NewSearch().
    WithQuery(queryVector).
    WithK(100).
    WithAutocut(true).  // Returns fewer if quality drops
    Execute()

// HNSW with custom efSearch
hnswIndex, _ := comet.NewHNSWIndex(384, comet.Cosine, 16, 200, 100)
results, _ := hnswIndex.NewSearch().
    WithQuery(queryVector).
    WithK(10).
    WithEfSearch(200).  // Higher accuracy for this query
    Execute()

// IVF with nProbe
ivfIndex, _ := comet.NewIVFIndex(384, 1000, comet.Cosine)
results, _ := ivfIndex.NewSearch().
    WithQuery(queryVector).
    WithK(10).
    WithNProbe(16).  // Search 16 nearest clusters
    Execute()

// IVFPQ with refinement
ivfpqIndex, _ := comet.NewIVFPQIndex(384, comet.Cosine, 8192, 8, 8)
results, _ := ivfpqIndex.NewSearch().
    WithQuery(queryVector).
    WithK(10).
    WithNProbe(32).
    WithNRefine(100).  // Refine top-100 with original vectors
    Execute()
```

**Result Format:**

```go
type VectorResult struct {
    Node  VectorNode  // The matched vector node
    Score float32     // Distance score (lower = more similar)
}

// Access results
for _, result := range results {
    id := result.GetId()
    score := result.GetScore()
    vector := result.Node.Vector()
    fmt.Printf("ID: %d, Distance: %.4f\n", id, score)
}
```

**Score Aggregation Strategies:**

```go
// When using multiple queries, results are aggregated:

comet.SumAggregation   // Sum of all distances (penalizes far results)
comet.MaxAggregation   // Maximum distance (pessimistic)
comet.MeanAggregation  // Average distance (balanced, recommended)
```

### Text Search (TextSearch Interface)

**Creating a Search:**

```go
search := textIndex.NewSearch()  // BM25SearchIndex
```

**Methods:**

| Method            | Parameters      | Description                                       |
| ----------------- | --------------- | ------------------------------------------------- |
| `WithQuery`       | `query string`  | Text query for keyword search                     |
| `WithK`           | `k int`         | Number of results to return (default: 10)         |
| `WithDocumentIDs` | `ids ...uint32` | Pre-filter: only search within these document IDs |
| `WithAutocut`     | `enabled bool`  | Automatically determine optimal cutoff point      |
| `Execute`         | -               | Run the search, returns `[]TextResult, error`     |

**Examples:**

```go
// Basic text search
results, err := textIndex.NewSearch().
    WithQuery("machine learning algorithms").
    WithK(10).
    Execute()

// Pre-filtered text search
eligibleIDs := []uint32{1, 5, 10, 15, 20}
results, _ := textIndex.NewSearch().
    WithQuery("neural networks").
    WithDocumentIDs(eligibleIDs...).
    WithK(5).
    Execute()

// With autocut
results, _ := textIndex.NewSearch().
    WithQuery("deep learning").
    WithK(50).
    WithAutocut(true).  // Returns fewer if relevance drops
    Execute()
```

**Result Format:**

```go
type TextResult struct {
    Node  TextNode  // The matched text node
    Score float32   // BM25 relevance score (higher = more relevant)
}

// Access results
for _, result := range results {
    id := result.GetId()
    score := result.GetScore()
    text := result.Node.Text()
    fmt.Printf("ID: %d, BM25: %.4f, Text: %s\n", id, score, text)
}
```

**BM25 Scoring:**

```
BM25 score components:
├─ IDF: Term rarity (rare terms score higher)
├─ TF:  Term frequency (with saturation via k1)
└─ DL:  Document length normalization (via b)

Higher score = more relevant
Typical range: 0-10+ (no upper bound)
```

### Metadata Search (MetadataSearch Interface)

**Creating a Search:**

```go
search := metadataIndex.NewSearch()  // RoaringMetadataIndex
```

**Methods:**

| Method        | Parameters          | Description                                       |
| ------------- | ------------------- | ------------------------------------------------- |
| `WithFilters` | `filters ...Filter` | Metadata filter conditions (AND logic)            |
| `Execute`     | -                   | Run the search, returns `[]MetadataResult, error` |

**Filter Functions:**

| Function  | Parameters                           | Description                          |
| --------- | ------------------------------------ | ------------------------------------ |
| `Eq`      | `field string, value interface{}`    | Equality: field == value             |
| `Lt`      | `field string, value interface{}`    | Less than: field < value             |
| `Lte`     | `field string, value interface{}`    | Less than or equal: field ≤ value    |
| `Gt`      | `field string, value interface{}`    | Greater than: field > value          |
| `Gte`     | `field string, value interface{}`    | Greater than or equal: field ≥ value |
| `Between` | `field string, min, max interface{}` | Range: min ≤ field ≤ max             |

**Examples:**

```go
// Single filter
results, err := metadataIndex.NewSearch().
    WithFilters(comet.Eq("category", "electronics")).
    Execute()

// Multiple filters (AND logic)
results, _ := metadataIndex.NewSearch().
    WithFilters(
        comet.Eq("category", "books"),
        comet.Gte("rating", 4.0),
        comet.Lte("price", 50.0),
        comet.Eq("in_stock", true),
    ).
    Execute()

// Range filter
results, _ := metadataIndex.NewSearch().
    WithFilters(
        comet.Between("year", 2020, 2024),
        comet.Eq("status", "published"),
    ).
    Execute()

// Numeric comparisons
results, _ := metadataIndex.NewSearch().
    WithFilters(
        comet.Gt("views", 1000),
        comet.Lt("price", 100),
    ).
    Execute()

// String equality
results, _ := metadataIndex.NewSearch().
    WithFilters(
        comet.Eq("author", "John Doe"),
        comet.Eq("language", "en"),
    ).
    Execute()
```

**Result Format:**

```go
type MetadataResult struct {
    Node MetadataNode  // The matched metadata node
}

// Access results
for _, result := range results {
    id := result.GetId()
    score := result.GetScore()  // Always 0 (binary match)
    metadata := result.Node.Metadata()

    fmt.Printf("ID: %d\n", id)
    fmt.Printf("Category: %v\n", metadata["category"])
    fmt.Printf("Price: %v\n", metadata["price"])
}
```

**Supported Data Types:**

```
Supported field types:
├─ string:   Equality only (Eq)
├─ int/int64: All comparisons (Eq, Lt, Lte, Gt, Gte, Between)
├─ float32/float64: All comparisons
├─ bool:     Equality only (Eq)
└─ nil:      Equality only (Eq)
```

### Hybrid Search (HybridSearch Interface)

**Creating a Search:**

```go
search := hybridIndex.NewSearch()  // HybridSearchIndex
```

**Methods:**

| Method            | Parameters             | Description                                                          |
| ----------------- | ---------------------- | -------------------------------------------------------------------- |
| `WithVectorQuery` | `queries ...[]float32` | Semantic search queries (vector embeddings)                          |
| `WithTextQuery`   | `query string`         | Keyword search query (BM25)                                          |
| `WithFilters`     | `filters ...Filter`    | Metadata filters (pre-filtering before search)                       |
| `WithK`           | `k int`                | Number of results to return per modality before fusion               |
| `WithFusion`      | `fusion FusionKind`    | Score fusion strategy: `ReciprocalRankFusion` or `WeightedSumFusion` |
| `WithDocumentIDs` | `ids ...uint32`        | Pre-filter: only search within these document IDs                    |
| `WithAutocut`     | `enabled bool`         | Automatically determine optimal cutoff point                         |
| `Execute`         | -                      | Run hybrid search, returns `[]HybridResult, error`                   |

**Fusion Strategies:**

| Strategy               | Description                                | When to Use                                                   |
| ---------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| `ReciprocalRankFusion` | Rank-based fusion: `score = Σ(1/(k+rank))` | Default, works without tuning, handles different score scales |
| `WeightedSumFusion`    | Linear combination of normalized scores    | When you want to weight modalities differently                |

**Examples:**

```go
// All three modalities
results, err := hybridIndex.NewSearch().
    WithVectorQuery(queryEmbedding).
    WithTextQuery("machine learning").
    WithFilters(
        comet.Eq("category", "ai"),
        comet.Gte("year", 2020),
    ).
    WithK(20).
    WithFusion(comet.ReciprocalRankFusion).
    Execute()

// Vector + Text only (no metadata filters)
results, _ := hybridIndex.NewSearch().
    WithVectorQuery(queryEmbedding).
    WithTextQuery("neural networks").
    WithK(10).
    Execute()

// Vector + Metadata only (no text)
results, _ := hybridIndex.NewSearch().
    WithVectorQuery(queryEmbedding).
    WithFilters(
        comet.Eq("category", "research"),
        comet.Lte("price", 0),  // Free papers
    ).
    WithK(15).
    Execute()

// Multi-query vector search with text
results, _ := hybridIndex.NewSearch().
    WithVectorQuery(embedding1, embedding2, embedding3).
    WithTextQuery("deep learning transformers").
    WithK(20).
    WithFusion(comet.ReciprocalRankFusion).
    Execute()

// Pre-filtered hybrid search
eligibleIDs := []uint32{1, 5, 10, 15, 20}
results, _ := hybridIndex.NewSearch().
    WithVectorQuery(queryEmbedding).
    WithTextQuery("optimization").
    WithDocumentIDs(eligibleIDs...).
    WithK(5).
    Execute()

// With autocut
results, _ := hybridIndex.NewSearch().
    WithVectorQuery(queryEmbedding).
    WithTextQuery("information retrieval").
    WithK(50).
    WithAutocut(true).
    Execute()
```

**Result Format:**

```go
type HybridResult struct {
    ID          uint32   // Document ID
    FusedScore  float32  // Combined score from all modalities
    VectorScore float32  // Individual vector similarity score
    TextScore   float32  // Individual BM25 score
}

// Access results
for _, result := range results {
    fmt.Printf("ID: %d\n", result.ID)
    fmt.Printf("Fused Score: %.4f\n", result.FusedScore)
    fmt.Printf("Vector: %.4f, Text: %.4f\n",
        result.VectorScore, result.TextScore)
}
```

**How Fusion Works:**

```
Reciprocal Rank Fusion (RRF):
┌──────────────────────────────────────┐
│ Step 1: Get results from each modality
│   Vector: [id=1, id=5, id=7, ...]
│   Text:   [id=7, id=1, id=12, ...]
│
│ Step 2: Compute RRF score per document
│   RRF(doc) = Σ 1/(k + rank_i)
│   where k=60 (default), rank_i is rank in modality i
│
│ Example for doc_id=1:
│   Vector rank: 1 → 1/(60+1) = 0.0164
│   Text rank:   2 → 1/(60+2) = 0.0161
│   RRF score:       0.0164 + 0.0161 = 0.0325
│
│ Step 3: Sort by RRF score (higher = better)
└──────────────────────────────────────┘

Benefits:
├─ No score normalization needed
├─ Robust to different score scales
├─ Emphasizes top-ranked results
└─ Works out-of-the-box (no tuning)
```

**Search Flow:**

```
Hybrid Search Execution:
┌───────────────────────────────────────────┐
│ 1. Metadata Filtering (if filters present)│
│    → Get eligible document IDs            │
└────────────┬──────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
┌─────▼─────┐ ┌─────▼─────┐
│  Vector   │ │   Text    │
│  Search   │ │  Search   │
│ (top-K)   │ │ (top-K)   │
└─────┬─────┘ └─────┬─────┘
      │             │
      └──────┬──────┘
             │
       ┌─────▼─────┐
       │   Fusion  │
       │ (RRF/WS)  │
       └─────┬─────┘
             │
       ┌─────▼─────┐
       │  Results  │
       │ (sorted)  │
       └───────────┘
```

**Use Case Examples:**

```go
// E-commerce: "Show me red dresses under $100 similar to this image"
results, _ := hybridIndex.NewSearch().
    WithVectorQuery(imageEmbedding).
    WithTextQuery("red dress").
    WithFilters(
        comet.Eq("category", "dresses"),
        comet.Lte("price", 100),
        comet.Eq("in_stock", true),
    ).
    WithK(20).
    Execute()

// Academic search: "Papers about 'attention mechanisms' similar to this abstract, published after 2020"
results, _ := hybridIndex.NewSearch().
    WithVectorQuery(abstractEmbedding).
    WithTextQuery("attention mechanisms transformers").
    WithFilters(
        comet.Gte("year", 2020),
        comet.Eq("peer_reviewed", true),
    ).
    WithK(50).
    Execute()

// Job search: "Backend engineer roles in San Francisco paying over $150k"
results, _ := hybridIndex.NewSearch().
    WithVectorQuery(resumeEmbedding).
    WithTextQuery("backend engineer golang kubernetes").
    WithFilters(
        comet.Eq("location", "San Francisco"),
        comet.Gte("salary", 150000),
        comet.Eq("remote", false),
    ).
    WithK(30).
    Execute()
```

## Use Cases

### Use Case 1: Semantic Document Search

**Problem:** A documentation platform needs to find relevant documents based on user queries, going beyond simple keyword matching to understand intent.

**Solution:** Use Comet's vector search with text embeddings from a sentence transformer model.

**Implementation:**

```go
package main

import (
    "github.com/wizenheimer/comet"
    "log"
)

type Document struct {
    ID       uint32
    Title    string
    Content  string
    Embedding []float32  // From sentence-transformers
}

func main() {
    // Create HNSW vector store for 384-dim embeddings
    m, efC, efS := comet.DefaultHNSWConfig()
    index, _ := comet.NewHNSWIndex(384, comet.Cosine, m, efC, efS)

    // Add documents (embeddings generated by your ML model)
    docs := []Document{
        {Title: "Getting Started", Embedding: getEmbedding("getting started guide")},
        {Title: "API Reference", Embedding: getEmbedding("api documentation")},
        {Title: "Troubleshooting", Embedding: getEmbedding("common problems and solutions")},
    }

    for _, doc := range docs {
        node := comet.NewVectorNode(doc.Embedding)
        index.Add(*node)
    }

    // Search with user query
    queryEmbedding := getEmbedding("how do I start using this?")
    results, _ := index.NewSearch().
        WithQuery(queryEmbedding).
        WithK(5).
        Execute()

    // Return relevant documents
    for _, result := range results {
        log.Printf("Found: %s (similarity: %.4f)",
            docs[result.GetId()].Title, 1-result.GetScore())
    }
}

func getEmbedding(text string) []float32 {
    // Call your embedding model API (OpenAI, Cohere, local model, etc.)
    return nil  // Placeholder
}
```

### Use Case 2: E-commerce Product Recommendations

**Problem:** Recommend similar products based on browsing history, with filtering by price, category, and availability.

**Solution:** Hybrid search combining product embeddings with metadata filters.

**Implementation:**

```go
type Product struct {
    ID          uint32
    Name        string
    ImageVector []float32  // From image embedding model
    Price       float64
    Category    string
    InStock     bool
}

func RecommendSimilarProducts(
    productID uint32,
    maxPrice float64,
    category string,
) ([]Product, error) {
    // Setup hybrid store
    vecIdx, _ := comet.NewHNSWIndex(512, comet.Cosine, 16, 200, 200)
    metaIdx := comet.NewRoaringMetadataIndex()
    hybrid := comet.NewHybridSearchIndex(vecIdx, nil, metaIdx)

    // Get the product's embedding
    product := getProduct(productID)

    // Search with filters
    results, err := hybrid.NewSearch().
        WithVector(product.ImageVector).
        WithMetadata(
            comet.Eq("category", category),
            comet.Lte("price", maxPrice),
            comet.Eq("in_stock", true),
        ).
        WithK(10).
        Execute()

    if err != nil {
        return nil, err
    }

    // Convert to Product structs
    products := make([]Product, len(results))
    for i, result := range results {
        products[i] = getProduct(result.ID)
    }

    return products, nil
}
```

### Use Case 3: Question-Answering System with Hybrid Retrieval

**Problem:** Build a QA system that combines semantic understanding (vector search) with keyword precision (BM25) for better answer retrieval.

**Solution:** Use RRF fusion to combine both modalities.

**Implementation:**

```go
func AnswerQuestion(question string) ([]string, error) {
    // Create hybrid store
    vecIdx, _ := comet.NewFlatIndex(768, comet.Cosine)  // BERT embeddings
    txtIdx := comet.NewBM25SearchIndex()
    hybrid := comet.NewHybridSearchIndex(vecIdx, txtIdx, nil)

    // Index knowledge base
    for _, doc := range knowledgeBase {
        embedding := getBERTEmbedding(doc.Text)
        hybrid.Add(embedding, doc.Text, nil)
    }

    // Search with both modalities
    questionEmbedding := getBERTEmbedding(question)
    results, err := hybrid.NewSearch().
        WithVector(questionEmbedding).        // Semantic similarity
        WithText(question).                    // Keyword matching
        WithFusionKind(comet.ReciprocalRankFusion).  // Combine rankings
        WithK(5).
        Execute()

    if err != nil {
        return nil, err
    }

    // Extract answer passages
    answers := make([]string, len(results))
    for i, result := range results {
        answers[i] = knowledgeBase[result.ID].Text
    }

    return answers, nil
}
```

## Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Clone repository
git clone https://github.com/wizenheimer/comet.git
cd comet

# Install dependencies
make deps

# Run tests
make test

# Run linter
make lint

# Check everything
make check
```

### Code Style

- Follow standard Go conventions (`gofmt`, `go vet`)
- Write descriptive comments for exported functions
- Include examples in documentation comments
- Add tests for new features (maintain >95% coverage)
- Keep functions focused and composable
- Use meaningful variable names

### Commit Messages

Follow conventional commits format:

**Good commit messages:**

- `feat: Add IVF index with k-means clustering`
- `fix: Handle zero vectors in cosine distance`
- `perf: Optimize HNSW search with heap pooling`
- `docs: Update API reference for hybrid search`
- `test: Add benchmarks for BM25 search`

**Bad commit messages:**

- `Update code`
- `Fix bug`
- `WIP`
- `asdf`

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Write your code** following the style guidelines
3. **Add tests** for all new functionality
4. **Run the full test suite** and ensure all tests pass
5. **Update documentation** (README, godoc comments)
6. **Commit with descriptive messages** using conventional commits
7. **Push to your fork** and open a Pull Request
8. **Respond to review feedback** promptly

### Code Review Checklist

Before submitting a PR, ensure:

- [ ] All tests pass (`make test`)
- [ ] Linter shows no issues (`make lint`)
- [ ] Coverage remains >95% (`make test-coverage`)
- [ ] Documentation is updated
- [ ] Benchmarks show no regression (if applicable)
- [ ] No breaking changes (unless discussed)

## License

MIT License - see LICENSE file for details

Copyright (c) 2025 wizenheimer

## References

- **HNSW Algorithm**: [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)
- **Product Quantization**: [Product quantization for nearest neighbor search](https://ieeexplore.ieee.org/document/5432202)
- **BM25 Ranking**: [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- **Roaring Bitmaps**: [Better bitmap performance with Roaring bitmaps](https://arxiv.org/abs/1603.06549)
- **Reciprocal Rank Fusion**: [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
