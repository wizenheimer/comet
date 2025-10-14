/*
Package comet provides a high-performance hybrid vector search library for Go.

Comet combines multiple indexing strategies and search modalities into a unified,
efficient package. It supports semantic search (vector embeddings), full-text search
(BM25), metadata filtering, and hybrid search with score fusion.

# Overview

Comet is built for developers who want to understand how vector databases work
from the inside out. It provides production-ready implementations of modern
vector search algorithms with comprehensive documentation and examples.

# Quick Start

Create a vector index and perform similarity search:

	package main

	import (
	    "fmt"
	    "log"

	    "github.com/wizenheimer/comet"
	)

	func main() {
	    // Create a flat index for 384-dimensional vectors using cosine distance
	    index, err := comet.NewFlatIndex(384, comet.Cosine)
	    if err != nil {
	        log.Fatal(err)
	    }

	    // Add vectors to the index
	    vec1 := make([]float32, 384)
	    // ... populate vec1 with your embedding ...
	    node := comet.NewVectorNode(vec1)
	    if err := index.Add(*node); err != nil {
	        log.Fatal(err)
	    }

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

# Vector Storage Types

Comet provides five vector index implementations, each with different tradeoffs:

FlatIndex: Brute-force exhaustive search with 100% recall. Best for small datasets
(<10K vectors) or when perfect accuracy is required.

	index, _ := comet.NewFlatIndex(384, comet.Cosine)

HNSWIndex: Hierarchical graph-based search with 95-99% recall and O(log n)
performance. Best for most production workloads (10K-10M vectors).

	m, efConstruction, efSearch := comet.DefaultHNSWConfig()
	index, _ := comet.NewHNSWIndex(384, comet.Cosine, m, efConstruction, efSearch)

IVFIndex: Inverted file index using k-means clustering with 85-95% recall.
Best for large datasets (>100K vectors) with moderate accuracy requirements.

	index, _ := comet.NewIVFIndex(384, comet.Cosine, 100) // 100 clusters
	index.Train(trainingVectors) // Training required

PQIndex: Product quantization for massive memory compression (10-500x smaller)
with 70-85% recall. Best for memory-constrained environments.

	m, nbits := comet.CalculatePQParams(384)
	index, _ := comet.NewPQIndex(384, comet.Cosine, m, nbits)
	index.Train(trainingVectors) // Training required

IVFPQIndex: Combines IVF and PQ for maximum scalability with 70-90% recall.
Best for billion-scale datasets.

	m, nbits := comet.CalculatePQParams(384)
	index, _ := comet.NewIVFPQIndex(384, comet.Cosine, 100, m, nbits)
	index.Train(trainingVectors) // Training required

# Distance Metrics

Three distance metrics are supported:

Euclidean (L2): Measures absolute spatial distance. Use when magnitude matters.

	index, _ := comet.NewFlatIndex(384, comet.Euclidean)

L2Squared: Squared Euclidean distance (faster, preserves ordering). Use for
better performance when only relative distances matter.

	index, _ := comet.NewFlatIndex(384, comet.L2Squared)

Cosine: Measures angular similarity, independent of magnitude. Use for normalized
vectors like text embeddings.

	index, _ := comet.NewFlatIndex(384, comet.Cosine)

# Full-Text Search

BM25-based full-text search with Unicode tokenization:

	index := comet.NewBM25SearchIndex()
	index.Add(1, "the quick brown fox jumps over the lazy dog")
	index.Add(2, "pack my box with five dozen liquor jugs")

	results, _ := index.NewSearch().
	    WithQuery("quick fox").
	    WithK(10).
	    Execute()

# Metadata Filtering

Fast filtering using Roaring Bitmaps and Bit-Sliced Indexes:

	index := comet.NewRoaringMetadataIndex()

	// Add documents with metadata
	node := comet.NewMetadataNode(map[string]interface{}{
	    "category": "electronics",
	    "price": 999,
	    "in_stock": true,
	})
	index.Add(*node)

	// Filter by metadata
	results, _ := index.NewSearch().
	    WithFilter(
	        comet.Eq("category", "electronics"),
	        comet.Lte("price", 1000),
	        comet.Eq("in_stock", true),
	    ).
	    Execute()

# Hybrid Search

Combine vector, text, and metadata search with score fusion:

	// Create indexes
	vecIdx, _ := comet.NewHNSWIndex(384, comet.Cosine, 16, 200, 200)
	txtIdx := comet.NewBM25SearchIndex()
	metaIdx := comet.NewRoaringMetadataIndex()

	// Create hybrid index
	hybrid := comet.NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

	// Add documents
	id, _ := hybrid.Add(
	    embedding,  // 384-dim vector
	    "machine learning tutorial",  // text
	    map[string]interface{}{  // metadata
	        "category": "education",
	        "price": 29.99,
	    },
	)

	// Hybrid search with all modalities
	results, _ := hybrid.NewSearch().
	    WithVector(queryEmbedding).
	    WithText("machine learning").
	    WithMetadata(
	        comet.Eq("category", "education"),
	        comet.Lt("price", 50),
	    ).
	    WithFusionKind(comet.ReciprocalRankFusion).
	    WithK(10).
	    Execute()

# Score Fusion Strategies

When combining results from multiple search modalities, different fusion
strategies are available:

WeightedSumFusion: Linear combination with configurable weights

	config := &comet.FusionConfig{
	    VectorWeight: 0.7,
	    TextWeight: 0.3,
	}
	fusion, _ := comet.NewFusion(comet.WeightedSumFusion, config)

ReciprocalRankFusion: Rank-based fusion (scale-independent, recommended)

	fusion, _ := comet.NewFusion(comet.ReciprocalRankFusion, nil)

MaxFusion/MinFusion: Simple maximum or minimum across modalities

	fusion, _ := comet.NewFusion(comet.MaxFusion, nil)

# Serialization

All indexes support persistence:

	// Save index
	file, _ := os.Create("index.bin")
	defer file.Close()
	index.WriteTo(file)

	// Load index
	file, _ := os.Open("index.bin")
	defer file.Close()
	index, _ := comet.NewFlatIndex(384, comet.Cosine)
	index.ReadFrom(file)

# Performance Tuning

HNSW parameters for tuning search quality:

	// Higher M = better recall, more memory
	// Lower M = faster search, less memory
	m := 16  // connections per layer (default: 16)

	// Higher efConstruction = better graph quality, slower build
	efConstruction := 200  // build quality (default: 200)

	// Higher efSearch = better recall, slower search
	efSearch := 200  // search quality (default: 200)

	index, _ := comet.NewHNSWIndex(384, comet.Cosine, m, efConstruction, efSearch)

IVF parameters for tuning speed/accuracy:

	nClusters := 100  // more clusters = faster search, lower recall
	index, _ := comet.NewIVFIndex(384, comet.Cosine, nClusters)

	// At search time
	nProbes := 8  // more probes = better recall, slower search
	results, _ := index.NewSearch().
	    WithQuery(query).
	    WithK(10).
	    WithNProbes(nProbes).
	    Execute()

# Thread Safety

All indexes are safe for concurrent use. Multiple goroutines can search
simultaneously while one goroutine adds or removes vectors.

# Use Cases

Document Search: Use vector embeddings for semantic search in documentation,
knowledge bases, or content management systems.

Product Recommendations: Combine product image embeddings with metadata filters
for personalized recommendations.

Question Answering: Use hybrid search (vector + BM25) for retrieval-augmented
generation (RAG) systems.

Duplicate Detection: Use high-recall vector search to find near-duplicate
documents or images.

Multi-modal Search: Combine text, image embeddings, and structured metadata
for comprehensive search experiences.

# Best Practices

Choose the right index type:
  - <10K vectors: Use FlatIndex
  - 10K-10M vectors: Use HNSWIndex
  - >10M vectors: Use IVFIndex or IVFPQIndex
  - Memory constrained: Use PQIndex or IVFPQIndex

Use appropriate distance metrics:
  - Text embeddings: Use Cosine
  - Image features: Use L2 or L2Squared
  - Custom embeddings: Depends on how they were trained

Batch operations:
  - Add vectors in batches for better performance
  - Call Flush() periodically after Remove() operations

Training indexes:
  - Use representative sample (10K-100K vectors) for IVF/PQ training
  - Training sample should cover the data distribution

Metadata filtering:
  - Apply filters before vector search to reduce candidates
  - Use equality filters for categorical data
  - Use range filters for numeric data

# Documentation and Examples

For detailed API documentation, see the godoc comments on each type and function.

For more examples and use cases, visit:
https://github.com/wizenheimer/comet

# License

MIT License - Copyright (c) 2025 wizenheimer
*/
package comet
