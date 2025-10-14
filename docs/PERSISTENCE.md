# Persistent Hybrid Search Index

This document explains the persistent storage layer for Comet's hybrid search index, which provides durability and scalability through an LSM-tree inspired architecture.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Performance](#performance)
- [Operations](#operations)
- [File Format](#file-format)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The persistent storage layer provides:

- **Durability**: Data survives process restarts and crashes
- **Scalability**: Handle datasets larger than available RAM
- **Fast Writes**: Buffered writes to in-memory memtables
- **Efficient Reads**: Merged results from memory and disk
- **Automatic Compaction**: Background consolidation of segments
- **100% API Parity**: Implements `HybridSearchIndex` interface with builder-style search

### Key Features

- **Thread-Safe**: All operations safe for concurrent use
- **Lock Files**: Prevents multiple processes from corrupting data
- **Compression**: gzip compression for storage efficiency
- **Crash Recovery**: Loads existing segments on startup
- **Graceful Shutdown**: Final flush before closing
- **Concurrent Search**: Segments searched in parallel
- **Result Merging**: Efficient deduplication by highest score
- **Background Workers**: Non-blocking flush and compaction

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                         WRITES                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  Active Memtable│  ← In-memory write buffer
         │   (Mutable)     │
         └────────┬────────┘
                  │ (fills up)
                  ▼
         ┌─────────────────┐
         │ Frozen Memtables│  ← Immutable, waiting flush
         │     (Queue)     │
         └────────┬────────┘
                  │ (background flush)
                  ▼
         ┌─────────────────┐
         │   Segments      │  ← Compressed on disk
         │   (Immutable)   │
         └────────┬────────┘
                  │ (background compaction)
                  ▼
         ┌─────────────────┐
         │ Merged Segments │  ← Larger, consolidated
         └─────────────────┘
```

### Components

#### 1. Memtables

- **Active Memtable**: Current write buffer accepting new documents
- **Frozen Memtables**: Immutable memtables waiting to be flushed to disk
- **Size Limit**: Default 100MB per memtable
- **Automatic Rotation**: Creates new memtable when current is full

#### 2. Segments

- **Immutable**: Once written, never modified
- **Compressed**: gzip compression for storage efficiency
- **Lazy Loading**: Loaded from disk on first access
- **Cached**: Kept in memory for fast access

#### 3. Background Workers

- **Flush Worker**: Writes frozen memtables to disk segments
- **Compaction Worker**: Merges small segments into larger ones

## Understanding Segments

### What Are Segments?

Segments are **immutable, compressed files** on disk that store a snapshot of the hybrid search index. Each segment represents a batch of documents that were flushed from memory at a specific point in time.

Think of segments as **frozen snapshots** of your data:

- Once created, they are **never modified**
- They are **compressed with gzip** (level 9) for storage efficiency
- They are **lazy-loaded** from disk only when needed
- They are **cached in memory** after first access

### Segment Structure

Each segment consists of **4 separate files** that together represent a complete hybrid search index:

```
Segment #000001
├── hybrid_000001.bin.gz    → Document IDs and hybrid index metadata
├── vector_000001.bin.gz    → Vector index (HNSW/IVF/Flat/etc.)
├── text_000001.bin.gz      → BM25 inverted index
└── metadata_000001.bin.gz  → Roaring bitmap metadata index
```

This split structure allows:

- **Selective loading**: Load only the index types needed for a query
- **Parallel I/O**: Read different index types concurrently
- **Independent compression**: Each index type compressed separately
- **Flexible schemas**: Optional indexes (e.g., no text index if not needed)

### Segment Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SEGMENT LIFECYCLE                             │
└─────────────────────────────────────────────────────────────────────┘

1. CREATION (from Memtable Flush)
   ┌──────────────┐
   │ Frozen       │
   │ Memtable     │ → Serialize → Compress → Write to disk
   │ (100MB)      │
   └──────────────┘
                         ↓
                  ┌──────────────┐
                  │  Segment     │ (e.g., 20-40MB compressed)
                  │  Files       │
                  └──────────────┘

2. ACTIVE STATE (Searchable)
   ┌──────────────┐
   │  Segment     │ ← Lazy load on first search
   │  Metadata    │ ← Cache in memory
   └──────────────┘
        ↓
   [Searched by queries]
        ↓
   [Cached index used for subsequent queries]

3. COMPACTION (Merging Multiple Segments)
   ┌────────┐   ┌────────┐   ┌────────┐
   │Segment │ + │Segment │ + │Segment │
   │   #1   │   │   #2   │   │   #3   │
   └────────┘   └────────┘   └────────┘
        ↓            ↓            ↓
        └────────────┴────────────┘
                     ↓
              ┌─────────────┐
              │   Merged    │ (Larger, consolidated)
              │  Segment    │
              └─────────────┘

4. DELETION (After Compaction)
   Old segments deleted from disk
   └→ Space reclaimed
```

### Memory vs Disk: The Complete Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PERSISTENT STORAGE LAYERS                         │
└─────────────────────────────────────────────────────────────────────┘

                           MEMORY (Fast, Volatile)
    ┌──────────────────────────────────────────────────────────┐
    │                                                           │
    │  ┌─────────────────────────────────────────────────┐    │
    │  │  Active Memtable (Mutable)                      │    │
    │  │  • Accepts all writes                           │    │
    │  │  • Size: 0-100MB                                │    │
    │  │  • State: HOT - always searched first           │    │
    │  └─────────────────────────────────────────────────┘    │
    │                                                           │
    │  ┌─────────────────────────────────────────────────┐    │
    │  │  Frozen Memtables (Immutable Queue)             │    │
    │  │  • Waiting to be flushed                        │    │
    │  │  • Size: 0-200MB total                          │    │
    │  │  • State: Pending flush                         │    │
    │  └─────────────────────────────────────────────────┘    │
    │                                                           │
    │  ┌─────────────────────────────────────────────────┐    │
    │  │  Segment Cache (Lazy-loaded indexes)            │    │
    │  │  • Recently accessed segments                   │    │
    │  │  • Size: Variable (LRU eviction)                │    │
    │  │  • State: WARM - cached for performance         │    │
    │  └─────────────────────────────────────────────────┘    │
    │                                                           │
    └───────────────────────────────────────────────────────────┘
                              ↕ Flush
    ┌──────────────────────────────────────────────────────────┐
    │                                                           │
    │                   DISK (Durable, Slower)                  │
    │                                                           │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
    │  │  Segment    │  │  Segment    │  │  Segment    │     │
    │  │   #00001    │  │   #00002    │  │   #00003    │ ... │
    │  │  (20-40MB)  │  │  (20-40MB)  │  │  (20-40MB)  │     │
    │  └─────────────┘  └─────────────┘  └─────────────┘     │
    │                                                           │
    │  • Compressed with gzip                                  │
    │  • Immutable (never modified)                            │
    │  • State: COLD - loaded on demand                        │
    │                                                           │
    └──────────────────────────────────────────────────────────┘
                              ↕ Compaction
    ┌──────────────────────────────────────────────────────────┐
    │                                                           │
    │  ┌─────────────────────────────────────────────────┐    │
    │  │  Compacted Segments (Larger, fewer files)       │    │
    │  │  • Merged from multiple small segments          │    │
    │  │  • Size: 100-500MB each                         │    │
    │  │  • State: Optimized for search                  │    │
    │  └─────────────────────────────────────────────────┘    │
    │                                                           │
    └──────────────────────────────────────────────────────────┘
```

## How Operations Work

### Search Operation

Searches scan **all data sources** from newest to oldest, then merge results:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          SEARCH FLOW                                 │
└─────────────────────────────────────────────────────────────────────┘

Query: WithVector(v).WithText("query").WithK(10)
  ↓
  ├─→ Step 1: Search Active Memtable (in-memory, newest data)
  │   ┌──────────────┐
  │   │   Active     │
  │   │  Memtable    │ → Search → Results: [doc_100: 0.95, doc_99: 0.89]
  │   └──────────────┘
  │
  ├─→ Step 2: Search Frozen Memtables (in-memory, pending flush)
  │   ┌──────────────┐
  │   │   Frozen     │
  │   │  Memtable 1  │ → Search → Results: [doc_95: 0.92, doc_94: 0.87]
  │   └──────────────┘
  │   ┌──────────────┐
  │   │   Frozen     │
  │   │  Memtable 2  │ → Search → Results: [doc_90: 0.88, doc_89: 0.85]
  │   └──────────────┘
  │
  ├─→ Step 3: Search Segments (on disk, in parallel)
  │   ┌──────────────┐
  │   │  Segment #1  │ ─┐
  │   └──────────────┘  │
  │   ┌──────────────┐  ├─→ Parallel Search → Results aggregated
  │   │  Segment #2  │ ─┤    [doc_80: 0.91, doc_50: 0.86, ...]
  │   └──────────────┘  │
  │   ┌──────────────┐  │
  │   │  Segment #3  │ ─┘
  │   └──────────────┘
  │
  ├─→ Step 4: Merge Results (deduplicate by highest score)
  │   ┌──────────────────────────────────────────────────┐
  │   │  All Results Combined:                            │
  │   │  doc_100: 0.95 (from active)                     │
  │   │  doc_95:  0.92 (from frozen)                     │
  │   │  doc_80:  0.91 (from segment)                    │
  │   │  doc_99:  0.89 (from active)                     │
  │   │  doc_90:  0.88 (from frozen)                     │
  │   │  doc_94:  0.87 (from frozen)                     │
  │   │  doc_50:  0.86 (from segment)                    │
  │   │  ...                                              │
  │   └──────────────────────────────────────────────────┘
  │
  └─→ Step 5: Sort and Limit
      ┌──────────────────────────────────────────────────┐
      │  Top K=10 Results:                                │
      │  1. doc_100: 0.95                                │
      │  2. doc_95:  0.92                                │
      │  3. doc_80:  0.91                                │
      │  4. doc_99:  0.89                                │
      │  5. doc_90:  0.88                                │
      │  ...                                              │
      └──────────────────────────────────────────────────┘
```

#### Search Performance Characteristics

```
Performance Trade-offs:
┌─────────────────┬────────────────┬──────────────────────────┐
│ Component       │ Search Speed   │ Data Freshness           │
├─────────────────┼────────────────┼──────────────────────────┤
│ Active Memtable │ Very Fast      │ Real-time (most recent)  │
│ Frozen Memtable │ Very Fast      │ Recent (pending flush)   │
│ Segments (few)  │ Fast           │ Older (flushed data)     │
│ Segments (many) │ Slower*        │ Historical               │
└─────────────────┴────────────────┴──────────────────────────┘

* This is why compaction is important!
  More segments = more parallel searches = more overhead
  Compaction reduces segment count → improves search speed
```

#### Lazy Loading and Caching

```
First Search to a Segment:
┌────────────┐
│   Query    │
└──────┬─────┘
       ↓
┌────────────────────────────────────┐
│  Segment Metadata (always in RAM)  │
│  • ID: 00001                       │
│  • Paths: hybrid_*.gz, vector_*.gz │
│  • Stats: 1000 docs, 25MB          │
│  • Cached Index: nil ←─────────────│← Not loaded yet
└──────────────┬─────────────────────┘
               ↓
    Load from disk (first time)
               ↓
┌────────────────────────────────────┐
│  1. Open hybrid_000001.bin.gz     │
│  2. Open vector_000001.bin.gz     │
│  3. Open text_000001.bin.gz       │
│  4. Open metadata_000001.bin.gz   │
│  5. Decompress with gzip          │
│  6. Deserialize index             │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  Segment Metadata                  │
│  • Cached Index: ✓ [Loaded] ←─────│← Now cached in memory
└────────────────────────────────────┘
               ↓
       [Search the index]


Subsequent Searches to Same Segment:
┌────────────┐
│   Query    │
└──────┬─────┘
       ↓
┌────────────────────────────────────┐
│  Segment Metadata                  │
│  • Cached Index: ✓ [Loaded] ←─────│← Already in memory!
└──────────────┬─────────────────────┘
               ↓
       [Search the cached index]
       (No disk I/O needed!)
```

### Write Operation (Add Document)

Writes go to the **active memtable** only, ensuring fast write performance:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          WRITE FLOW                                  │
└─────────────────────────────────────────────────────────────────────┘

Add(vector, text, metadata)
  ↓
  ┌─────────────────────────────────────────┐
  │ Step 1: Check if storage is closed      │
  └─────────────────┬───────────────────────┘
                    ↓
  ┌─────────────────────────────────────────┐
  │ Step 2: Add to Active Memtable          │
  │                                          │
  │  ┌────────────────────────────────┐    │
  │  │  Active Memtable               │    │
  │  │  • Check space available       │    │
  │  │  • If full → rotate to new one │    │
  │  │  • Add document                │    │
  │  │  • Update size estimate        │    │
  │  │  • Return document ID          │    │
  │  └────────────────────────────────┘    │
  └─────────────────┬───────────────────────┘
                    ↓
  ┌─────────────────────────────────────────┐
  │ Step 3: Check Flush Threshold           │
  │  Total memtable size ≥ 200MB?           │
  │  • Yes → Schedule background flush      │
  │  • No  → Continue                       │
  └─────────────────┬───────────────────────┘
                    ↓
               [Write complete]
               (Returned to caller immediately)

Meanwhile, in background:
  ↓
  ┌─────────────────────────────────────────┐
  │ Background Flush Worker                  │
  │                                          │
  │  1. Freeze current memtable             │
  │  2. Create new active memtable          │
  │  3. Serialize frozen memtable           │
  │  4. Compress with gzip                  │
  │  5. Write to disk as segment            │
  │  6. Update segment manager              │
  └──────────────────────────────────────────┘
```

#### Memtable Rotation

```
When a memtable fills up:

BEFORE Rotation:
┌──────────────────────────────────────────────────────┐
│  Active Memtable (Size: 100MB / Limit: 100MB) FULL! │
└──────────────────────────────────────────────────────┘

DURING Rotation:
  ↓
  ├─→ Freeze current memtable (mark immutable)
  │   ┌──────────────────────────────────────┐
  │   │  Frozen Memtable (100MB)             │
  │   │  • No more writes accepted           │
  │   │  • Queued for flush                  │
  │   └──────────────────────────────────────┘
  │
  └─→ Create new active memtable
      ┌──────────────────────────────────────┐
      │  Active Memtable (Size: 0MB)         │
      │  • Ready for writes                  │
      └──────────────────────────────────────┘

AFTER Rotation:
┌──────────────────────────────────────────────────────┐
│  Memtable Queue:                                      │
│  1. Frozen #1 (100MB) ← waiting flush                │
│  2. Active    (0MB)   ← accepting writes             │
└──────────────────────────────────────────────────────┘
```

### Delete Operation

Deletes are **eventually consistent** due to the immutable nature of segments:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DELETE FLOW                                 │
└─────────────────────────────────────────────────────────────────────┘

Remove(docID = 42)
  ↓
  ┌─────────────────────────────────────────────────────┐
  │ Step 1: Remove from Active Memtable                 │
  │  ┌────────────────────────────────────┐            │
  │  │  Active Memtable                   │            │
  │  │  • docID=42 found? → Remove it ✓   │            │
  │  │  • docID=42 not found? → Skip      │            │
  │  └────────────────────────────────────┘            │
  └─────────────────┬───────────────────────────────────┘
                    ↓
  ┌─────────────────────────────────────────────────────┐
  │ Step 2: Frozen Memtables                            │
  │  • CANNOT be modified (immutable)                   │
  │  • Document still present if it exists              │
  │  • Will be handled during compaction                │
  └─────────────────┬───────────────────────────────────┘
                    ↓
  ┌─────────────────────────────────────────────────────┐
  │ Step 3: Segments (on disk)                          │
  │  • CANNOT be modified (immutable)                   │
  │  • Document still present if it exists              │
  │  • Will be removed during compaction                │
  └─────────────────────────────────────────────────────┘

KEY INSIGHT: Deletes are NOT immediately consistent!
- Deleted from active memtable: ✓ Immediate
- Still in frozen memtables:    ✗ Until flushed
- Still in old segments:        ✗ Until compacted

Timeline:
  T0: Remove(42) called
  T1: Doc 42 removed from active memtable ✓
      [Searches still may find doc 42 in frozen/segments]
  T2: Frozen memtables flushed to disk
      [Searches still may find doc 42 in old segments]
  T3: Compaction runs
      [Doc 42 excluded from compacted segment]
  T4: Old segments deleted
      [Doc 42 completely removed] ✓
```

#### Delete Visibility

```
Scenario: Document 42 exists in multiple places

BEFORE Delete:
┌─────────────────────────────────────────────────────┐
│  Active Memtable:  [doc 42 ✓]                      │
│  Frozen Memtable:  [doc 42 ✓]  ← Pending flush     │
│  Segment #001:     [doc 42 ✓]  ← On disk           │
│  Segment #002:     [doc 42 ✓]  ← On disk           │
└─────────────────────────────────────────────────────┘
Search for doc 42 → Found (score: 0.95)

AFTER Remove(42):
┌─────────────────────────────────────────────────────┐
│  Active Memtable:  [doc 42 ✗]  ← Removed!          │
│  Frozen Memtable:  [doc 42 ✓]  ← Still there       │
│  Segment #001:     [doc 42 ✓]  ← Still there       │
│  Segment #002:     [doc 42 ✓]  ← Still there       │
└─────────────────────────────────────────────────────┘
Search for doc 42 → Still found! (from frozen/segments)

AFTER Flush:
┌─────────────────────────────────────────────────────┐
│  Active Memtable:  [doc 42 ✗]                      │
│  Segment #001:     [doc 42 ✓]  ← Still there       │
│  Segment #002:     [doc 42 ✓]  ← Still there       │
│  Segment #003:     [doc 42 ✓]  ← From frozen flush │
└─────────────────────────────────────────────────────┘
Search for doc 42 → Still found! (from segments)

AFTER Compaction:
┌─────────────────────────────────────────────────────┐
│  Active Memtable:  [doc 42 ✗]                      │
│  Segment #004:     [doc 42 ✗]  ← Merged, excluded  │
└─────────────────────────────────────────────────────┘
Search for doc 42 → Not found ✓ (finally removed)
```

### Compaction Operation

Compaction **merges multiple segments** into fewer, larger segments:

```
┌─────────────────────────────────────────────────────────────────────┐
│                       COMPACTION FLOW                                │
└─────────────────────────────────────────────────────────────────────┘

Trigger: Segment count ≥ CompactionThreshold (default: 5)
  ↓
  ┌─────────────────────────────────────────────────────┐
  │ Step 1: Select Segments to Compact                  │
  │  • Strategy: Oldest N segments (leveled)            │
  │  • Example: Take oldest 5 segments                  │
  └─────────────────┬───────────────────────────────────┘
                    ↓
  ┌─────────────────────────────────────────────────────┐
  │ Step 2: Load Segments into Memory                   │
  │                                                      │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
  │  │Segment 1 │  │Segment 2 │  │Segment 3 │ ...     │
  │  │ 1K docs  │  │ 1K docs  │  │ 1K docs  │         │
  │  │ 25MB     │  │ 25MB     │  │ 25MB     │         │
  │  └──────────┘  └──────────┘  └──────────┘         │
  └─────────────────┬───────────────────────────────────┘
                    ↓
  ┌─────────────────────────────────────────────────────┐
  │ Step 3: Merge into New Segment                      │
  │                                                      │
  │  • Create new hybrid index                          │
  │  • Iterate through all source segments              │
  │  • Add documents to merged index                    │
  │  • Deduplicate: keep latest version                 │
  │  • Exclude: deleted documents                       │
  └─────────────────┬───────────────────────────────────┘
                    ↓
  ┌─────────────────────────────────────────────────────┐
  │ Step 4: Write Merged Segment to Disk                │
  │                                                      │
  │  ┌────────────────────────────────┐                │
  │  │  Merged Segment #006            │                │
  │  │  • 5K docs (deduplicated)       │                │
  │  │  • 100MB (compressed)           │                │
  │  │  • Deleted docs excluded        │                │
  │  └────────────────────────────────┘                │
  └─────────────────┬───────────────────────────────────┘
                    ↓
  ┌─────────────────────────────────────────────────────┐
  │ Step 5: Atomic Swap                                 │
  │  • Add new segment to manager                       │
  │  • Remove old segments from manager                 │
  │  • Delete old segment files from disk               │
  │  • Space reclaimed!                                 │
  └──────────────────────────────────────────────────────┘
```

#### Benefits of Compaction

```
BEFORE Compaction:
┌────────────────────────────────────────────────────────────┐
│  Segment #001: 1,000 docs (25MB) age: 2 hours             │
│  Segment #002: 1,000 docs (25MB) age: 1.5 hours           │
│  Segment #003: 1,000 docs (25MB) age: 1 hour              │
│  Segment #004: 1,000 docs (25MB) age: 30 min              │
│  Segment #005: 1,000 docs (25MB) age: 10 min              │
│                                                             │
│  Total: 5 segments, 5,000 docs, 125MB                     │
│  Search Performance: 5 parallel searches + merge           │
│  Deleted Docs: Still taking up space                       │
└────────────────────────────────────────────────────────────┘

AFTER Compaction:
┌────────────────────────────────────────────────────────────┐
│  Segment #006: 4,500 docs (95MB) age: 0 min (just created)│
│                                                             │
│  Total: 1 segment, 4,500 docs, 95MB                       │
│  Search Performance: 1 search (faster!)                    │
│  Deleted Docs: Removed (500 docs cleaned up)              │
│  Space Saved: 30MB reclaimed                               │
└────────────────────────────────────────────────────────────┘

Performance Impact:
┌─────────────────────┬──────────────┬──────────────┐
│ Metric              │ Before       │ After        │
├─────────────────────┼──────────────┼──────────────┤
│ Segments to search  │ 5            │ 1            │
│ Parallel overhead   │ High         │ None         │
│ Disk space          │ 125MB        │ 95MB         │
│ Search latency      │ Slower       │ Faster ✓     │
│ Deleted docs        │ Present      │ Removed ✓    │
└─────────────────────┴──────────────┴──────────────┘
```

### Flush Operation

Flushing converts **in-memory memtables** to **durable disk segments**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FLUSH FLOW                                  │
└─────────────────────────────────────────────────────────────────────┘

Trigger: Total memtable size ≥ FlushThreshold (200MB) OR Manual Flush()
  ↓
  ┌─────────────────────────────────────────────────────┐
  │ Step 1: Identify Frozen Memtables                   │
  │  ┌────────────────────────────────────┐            │
  │  │  Frozen Queue:                     │            │
  │  │  • Frozen #1 (100MB) ← Ready       │            │
  │  │  • Frozen #2 (100MB) ← Ready       │            │
  │  └────────────────────────────────────┘            │
  └─────────────────┬───────────────────────────────────┘
                    ↓
  ┌─────────────────────────────────────────────────────┐
  │ Step 2: For Each Frozen Memtable                    │
  │                                                      │
  │  Memtable → Serialize → Compress → Write to Disk    │
  │                                                      │
  │  ┌────────────────┐                                 │
  │  │ Frozen Memtable│                                 │
  │  │ 1,000 docs     │                                 │
  │  │ 100MB (RAM)    │                                 │
  │  └────────┬───────┘                                 │
  │           ↓                                          │
  │  ┌────────────────────────────────┐                │
  │  │ Serialize to bytes             │                │
  │  │ • Hybrid index metadata        │                │
  │  │ • Vector index data            │                │
  │  │ • Text inverted index          │                │
  │  │ • Metadata bitmap index        │                │
  │  └────────┬───────────────────────┘                │
  │           ↓                                          │
  │  ┌────────────────────────────────┐                │
  │  │ Compress with gzip (level 9)   │                │
  │  │ 100MB → 25MB (75% reduction)   │                │
  │  └────────┬───────────────────────┘                │
  │           ↓                                          │
  │  ┌────────────────────────────────┐                │
  │  │ Write to disk                  │                │
  │  │ • hybrid_000007.bin.gz         │                │
  │  │ • vector_000007.bin.gz         │                │
  │  │ • text_000007.bin.gz           │                │
  │  │ • metadata_000007.bin.gz       │                │
  │  └────────┬───────────────────────┘                │
  └───────────┼────────────────────────────────────────┘
              ↓
  ┌─────────────────────────────────────────────────────┐
  │ Step 3: Update Segment Manager                      │
  │  • Add new segment metadata                         │
  │  • Track segment ID, paths, size, doc count         │
  └─────────────────┬───────────────────────────────────┘
                    ↓
  ┌─────────────────────────────────────────────────────┐
  │ Step 4: Remove from Memtable Queue                  │
  │  • Flushed memtables removed from queue             │
  │  • Memory reclaimed                                 │
  └──────────────────────────────────────────────────────┘
```

#### Compression Details

```
Compression Example:

BEFORE Compression (In Memory):
┌────────────────────────────────────────────────────┐
│  Hybrid Index (Memtable)                           │
│  ├─ Document IDs:     100 KB                       │
│  ├─ Vector Index:     70 MB (1000 docs × 384 dims) │
│  ├─ Text Index:       25 MB (inverted index)       │
│  └─ Metadata Index:    5 MB (roaring bitmaps)      │
│                                                     │
│  Total: ~100 MB in RAM                             │
└────────────────────────────────────────────────────┘
                    ↓ gzip level 9
AFTER Compression (On Disk):
┌────────────────────────────────────────────────────┐
│  Segment Files:                                     │
│  ├─ hybrid_*.bin.gz:    25 KB (4x reduction)       │
│  ├─ vector_*.bin.gz:    18 MB (4x reduction)       │
│  ├─ text_*.bin.gz:       5 MB (5x reduction)       │
│  └─ metadata_*.bin.gz:   1 MB (5x reduction)       │
│                                                     │
│  Total: ~24 MB on disk (75% compression)           │
└────────────────────────────────────────────────────┘

Compression ratios vary by index type:
- Vector indexes: ~3-4x (float32 arrays compress well)
- Text indexes: ~5-6x (strings and posting lists compress well)
- Metadata indexes: ~5-10x (sparse bitmaps compress extremely well)
```

### Complete Operation Timeline

Here's how all operations interact over time:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OPERATIONS OVER TIME                              │
└─────────────────────────────────────────────────────────────────────┘

T=0: System Start
     ┌──────────────┐
     │  Active MT   │ (empty, 0MB)
     └──────────────┘
     No segments on disk

T=1-5: Heavy writes (20MB/sec)
     ┌──────────────┐
     │  Active MT   │ (100MB) ← FULL!
     └──────────────┘

     Rotation triggered:
     ┌──────────────┐
     │  Frozen MT#1 │ (100MB)
     ├──────────────┤
     │  Active MT   │ (0MB) ← New
     └──────────────┘

T=6-10: More writes (20MB/sec)
     ┌──────────────┐
     │  Frozen MT#1 │ (100MB)
     ├──────────────┤
     │  Active MT   │ (100MB) ← FULL!
     └──────────────┘

     Rotation + Flush triggered (total ≥ 200MB):
     ┌──────────────┐
     │  Frozen MT#1 │ (100MB) ← Flushing...
     ├──────────────┤
     │  Frozen MT#2 │ (100MB) ← Just frozen
     ├──────────────┤
     │  Active MT   │ (0MB) ← New
     └──────────────┘

T=11: Flush completes
     ┌──────────────┐
     │  Active MT   │ (0MB)
     └──────────────┘
     Disk: Segment #001 (25MB compressed)

T=15: More writes, another flush
     ┌──────────────┐
     │  Active MT   │ (0MB)
     └──────────────┘
     Disk: Segment #001, #002 (50MB total)

T=20-50: Continuous writes
     Many flushes occur...
     ┌──────────────┐
     │  Active MT   │ (variable)
     └──────────────┘
     Disk: Segments #001, #002, #003, #004, #005 (125MB total)

T=51: Compaction triggered (5 segments ≥ threshold)
      ┌────────────────────────────────────┐
      │  Compaction Worker                 │
      │  • Load segments #001-#005         │
      │  • Merge into segment #006         │
      │  • Write #006 to disk              │
      │  • Delete #001-#005                │
      └────────────────────────────────────┘

T=52: After compaction
      ┌──────────────┐
      │  Active MT   │ (variable)
      └──────────────┘
      Disk: Segment #006 (95MB) ← Single merged segment

      30MB disk space reclaimed!
      Future searches are faster (1 segment vs 5)!

T=100: Delete operations
       Remove(doc_42)
       ┌──────────────┐
       │  Active MT   │ ← doc_42 removed immediately
       └──────────────┘
       Disk: Segment #006 ← doc_42 still present

       Search for doc_42 → Still found (from disk segment)

T=150: Next compaction
       doc_42 excluded from merged segment
       Search for doc_42 → Not found ✓
```

## Getting Started

### Basic Example

```go
package main

import (
    "log"
    "time"

    "github.com/your-org/comet"
)

func main() {
    // Create configuration
    config := comet.DefaultStorageConfig("./data")

    // Set up index templates
    config.VectorIndexTemplate, _ = comet.NewFlatIndex(384, comet.Cosine)
    config.TextIndexTemplate = comet.NewBM25SearchIndex()
    config.MetadataIndexTemplate = comet.NewRoaringMetadataIndex()

    // Open persistent storage
    store, err := comet.OpenPersistentHybridIndex(config)
    if err != nil {
        log.Fatal(err)
    }
    defer store.Close()

    // Add documents
    vector := []float32{0.1, 0.2, 0.3, /* ... */}
    text := "the quick brown fox jumps over the lazy dog"
    metadata := map[string]interface{}{
        "category": "animals",
        "year": 2024,
        "tags": []string{"fox", "dog"},
    }

    id, err := store.Add(vector, text, metadata)
    if err != nil {
        log.Fatal(err)
    }

    // Search using builder pattern
    results, err := store.NewSearch().
        WithVector(vector).
        WithText("quick fox").
        WithMetadata(comet.Eq("category", "animals")).
        WithK(10).
        Execute()
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Found %d results", len(results))
}
```

### Drop-in Replacement

`PersistentHybridIndex` implements the `HybridSearchIndex` interface, making it a drop-in replacement for in-memory storage:

```go
// Define your storage interface
var idx comet.HybridSearchIndex

// Use in-memory during development
idx = comet.NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

// Use persistent in production (same API!)
idx, _ = comet.OpenPersistentHybridIndex(config)

// Everything works identically
results, _ := idx.NewSearch().
    WithVector(queryVector).
    WithText("query").
    WithK(10).
    Execute()
```

## Configuration

### StorageConfig

```go
type StorageConfig struct {
    // BaseDir is the directory for storing segments
    BaseDir string

    // MemtableSizeLimit is the maximum size of each memtable
    // Default: 100MB
    MemtableSizeLimit int64

    // FlushThreshold is the total memtable size that triggers a flush
    // Default: 200MB
    FlushThreshold int64

    // CompactionInterval is how often to check for compaction
    // Default: 5 minutes
    CompactionInterval time.Duration

    // CompactionThreshold is the minimum number of segments before compaction
    // Default: 5
    CompactionThreshold int

    // Index templates for creating new memtables
    VectorIndexTemplate   VectorIndex
    TextIndexTemplate     TextIndex
    MetadataIndexTemplate MetadataIndex
}
```

### Default Configuration

```go
config := comet.DefaultStorageConfig("./data")
// Returns configuration with sensible defaults:
// - MemtableSizeLimit: 100MB
// - FlushThreshold: 200MB
// - CompactionInterval: 5 minutes
// - CompactionThreshold: 5 segments
```

### Custom Configuration

```go
config := &comet.StorageConfig{
    BaseDir:             "./my-data",
    MemtableSizeLimit:   50 * 1024 * 1024,   // 50MB per memtable
    FlushThreshold:      150 * 1024 * 1024,  // Flush at 150MB total
    CompactionInterval:  10 * time.Minute,   // Check every 10 min
    CompactionThreshold: 10,                 // Compact when 10+ segments
}
config.VectorIndexTemplate, _ = comet.NewFlatIndex(384, comet.Cosine)
config.TextIndexTemplate = comet.NewBM25SearchIndex()
config.MetadataIndexTemplate = comet.NewRoaringMetadataIndex()
```

## API Reference

### Core Operations

```go
// Add document with auto-generated ID
id, err := store.Add(vector, text, metadata)

// Add document with specific ID
err := store.AddWithID(id, vector, text, metadata)

// Remove document
err := store.Remove(id)

// Train vector index (for IVF, PQ, etc.)
err := store.Train(trainingVectors)

// Force flush to disk
err := store.Flush()

// Close and release resources
err := store.Close()
```

### Builder-Style Search

All search parameters from `HybridSearchIndex` are supported:

```go
results, err := store.NewSearch().
    WithVector(queryVector).              // Vector search
    WithText("query", "terms").           // Text search
    WithMetadata(filters...).             // Metadata filters
    WithMetadataGroups(groups...).        // Complex filter groups
    WithK(10).                            // Number of results
    WithThreshold(0.5).                   // Score threshold
    WithFusion(fusion).                   // Fusion strategy
    WithFusionKind(comet.RRFFusion).      // Or use predefined fusion
    WithNProbes(10).                      // For IVF indexes
    WithEfSearch(100).                    // For HNSW indexes
    WithScoreAggregation(comet.SumAggregation).  // Score aggregation
    WithCutoff(5).                        // Autocut parameter
    Execute()
```

### Management Operations

```go
// Trigger compaction manually
store.TriggerCompaction()

// Monitor segment count
count := store.segmentManager.Count()

// Monitor disk usage
size := store.segmentManager.TotalSize()

// Free memory under pressure
store.segmentManager.EvictAllCaches()

// Check queue depth
queueDepth := store.memtableQueue.Count()

// Force memtable rotation
store.memtableQueue.Rotate()
```

## Examples

### Vector + Text Hybrid Search

```go
results, _ := store.NewSearch().
    WithVector(queryEmbedding).
    WithText("machine learning", "deep learning").
    WithK(20).
    WithFusionKind(comet.RRFFusion).  // Reciprocal Rank Fusion
    Execute()
```

### Metadata Filtering

```go
results, _ := store.NewSearch().
    WithVector(queryEmbedding).
    WithMetadata(
        comet.Eq("category", "tech"),
        comet.Gt("year", 2020),
        comet.In("tags", "AI"),
    ).
    WithK(10).
    Execute()
```

### Complex Metadata Queries

```go
// (category = "tech" AND year > 2020) OR (category = "science" AND verified = true)
results, _ := store.NewSearch().
    WithVector(queryEmbedding).
    WithMetadataGroups(
        comet.And(
            comet.Eq("category", "tech"),
            comet.Gt("year", 2020),
        ),
        comet.And(
            comet.Eq("category", "science"),
            comet.Eq("verified", true),
        ),
    ).
    WithK(10).
    Execute()
```

### IVF Index Configuration

```go
// Configure IVF vector index
config.VectorIndexTemplate, _ = comet.NewIVFIndex(384, 100, comet.Cosine)

// Search with nProbes parameter
results, _ := store.NewSearch().
    WithVector(queryEmbedding).
    WithK(10).
    WithNProbes(10).  // Search 10 clusters
    Execute()
```

### HNSW Index Configuration

```go
// Configure HNSW vector index
config.VectorIndexTemplate, _ = comet.NewHNSWIndex(384, 16, 200, comet.Cosine)

// Search with efSearch parameter
results, _ := store.NewSearch().
    WithVector(queryEmbedding).
    WithK(10).
    WithEfSearch(100).  // Increase search quality
    Execute()
```

### Persistence Across Restarts

```go
// First run - add data and close
func firstRun() {
    store, _ := comet.OpenPersistentHybridIndex(config)
    store.Add(vec, text, meta)
    store.Flush()  // Ensure data is written to disk
    store.Close()
}

// Second run - data is still there
func secondRun() {
    store, _ := comet.OpenPersistentHybridIndex(config)
    defer store.Close()

    results, _ := store.NewSearch().
        WithText("original query").
        WithK(10).
        Execute()

    // Returns results from first run
}
```

### Concurrent Usage

```go
var wg sync.WaitGroup

// Concurrent writers
for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()
        vec := generateVector(id)
        store.Add(vec, fmt.Sprintf("doc %d", id), nil)
    }(i)
}

// Concurrent readers
for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()
        query := generateQuery(id)
        store.NewSearch().WithVector(query).WithK(10).Execute()
    }(i)
}

wg.Wait()
```

### Production Monitoring

```go
// Monitor segment accumulation
if store.segmentManager.Count() > 50 {
    log.Warn("Too many segments, triggering compaction")
    store.TriggerCompaction()
}

// Monitor disk usage
diskUsage := store.segmentManager.TotalSize()
if diskUsage > threshold {
    log.Warn("High disk usage", "bytes", diskUsage)
}

// Monitor queue depth
queueDepth := store.memtableQueue.Count()
if queueDepth > 10 {
    log.Warn("Deep memtable queue", "depth", queueDepth)
}

// Handle memory pressure
if memoryPressure {
    store.segmentManager.EvictAllCaches()
}
```

## Performance

### Write Path

1. **Add to Active Memtable**: New documents added to active memtable in memory
2. **Check Size**: After each write, check if memtable size exceeds limit
3. **Rotate if Full**: If memtable is full, freeze it and create new active memtable
4. **Background Flush**: When total memtable size exceeds threshold, flush to disk
5. **Create Segment**: Frozen memtables serialized and compressed to disk segments

### Read Path

1. **Search Memtables**: Search all memtables from newest to oldest
2. **Search Segments**: Concurrently search all disk segments
3. **Merge Results**: Combine results from all sources
4. **Deduplicate**: Keep highest score for each document ID
5. **Sort and Limit**: Return top-k results by score

### Performance Characteristics

- **Write Latency**: O(log n) in memtable (milliseconds)
- **Read Latency**: O(memtables + segments) with parallelization
- **Memory Usage**: ~100-200MB for memtables + segment cache
- **Disk Usage**: Compressed segments with gzip level 9
- **Scalability**: Dataset size limited by disk, not RAM

### Memory Management

- **Memtables**: In-memory write buffers (~100-200MB total)
- **Segment Cache**: Loaded segments cached in memory
- **Cache Eviction**: Manually evict segment caches under pressure

### Write Performance

- **Fast Writes**: O(log n) writes to in-memory index
- **Background Flush**: Writes don't block on disk I/O
- **Batching**: Multiple memtables flushed together

### Read Performance

- **Concurrent Reads**: Segments searched in parallel
- **Cache Benefit**: Hot segments stay in memory
- **Trade-off**: More segments = slower reads (mitigated by compaction)

### Disk Usage

- **Compression**: gzip compression reduces disk space
- **Compaction**: Merges small segments into larger ones
- **Cleanup**: Old segments deleted after compaction

## Operations

### Compaction

Compaction merges multiple small segments into fewer larger ones to:

1. **Reduce File Count**: Fewer files to search during queries
2. **Improve Performance**: Fewer index merges needed
3. **Reclaim Space**: Remove deleted documents
4. **Optimize Layout**: Better disk I/O patterns

#### Compaction Strategy

- **Threshold Based**: Triggered when segment count exceeds threshold (default: 5)
- **Leveled**: Merges oldest segments first
- **Background**: Runs asynchronously without blocking writes
- **Manual**: Can be triggered via `TriggerCompaction()`

#### Manual Compaction

```go
// Check segment count
segmentCount := store.segmentManager.Count()
log.Printf("Current segments: %d", segmentCount)

// Trigger compaction if needed
if segmentCount > 20 {
    store.TriggerCompaction()
}
```

### Graceful Shutdown

Always close the storage properly to ensure all data is flushed:

```go
// Ensures final flush before closing
if err := store.Close(); err != nil {
    log.Fatal(err)
}
```

### Manual Flush

Force flush all pending memtables to disk:

```go
if err := store.Flush(); err != nil {
    log.Printf("flush failed: %v", err)
}
```

## File Format

### Directory Structure

```
data/
├── LOCK                      # Process lock file
├── hybrid_000001.bin.gz      # Hybrid metadata (segment 1)
├── vector_000001.bin.gz      # Vector index (segment 1)
├── text_000001.bin.gz        # Text index (segment 1)
├── metadata_000001.bin.gz    # Metadata index (segment 1)
├── hybrid_000002.bin.gz      # Segment 2
├── vector_000002.bin.gz
├── text_000002.bin.gz
└── metadata_000002.bin.gz
```

### Segment Files

Each segment consists of 4 files:

1. **hybrid\_\*.bin.gz**: Hybrid index metadata and document tracking
2. **vector\_\*.bin.gz**: Vector index data (HNSW, IVF, etc.)
3. **text\_\*.bin.gz**: Text index data (BM25 inverted index)
4. **metadata\_\*.bin.gz**: Metadata index data (Roaring bitmaps)

All files are compressed with gzip (level 9) for maximum space efficiency.

### Lock File

The `LOCK` file prevents multiple processes from using the same storage directory:

- Created when storage is opened
- Contains the process ID
- Automatically removed on clean shutdown
- Must be manually removed if process crashes

## Best Practices

### Configuration

1. **Memory Budget**: Set `FlushThreshold` based on available RAM
2. **Write Load**: Lower `MemtableSizeLimit` for write-heavy workloads
3. **Compaction**: Adjust `CompactionInterval` based on segment growth rate
4. **Index Selection**: Choose appropriate vector index for your dataset size

### Operations

1. **Graceful Shutdown**: Always call `Close()` to flush pending data
2. **Monitoring**: Track segment count and total size
3. **Backups**: Backup the entire data directory
4. **Testing**: Test recovery by killing and restarting process
5. **Compaction**: Monitor segment count and trigger compaction when needed

### Production Deployment

1. **Persistent Storage**: Use SSD for better I/O performance
2. **Memory**: Allocate enough RAM for memtables + segment cache
3. **Monitoring**: Monitor flush/compaction latencies
4. **Alerts**: Alert on high segment count or disk usage
5. **Backups**: Regular backups of data directory
6. **Capacity Planning**: Monitor disk usage growth rate

## Troubleshooting

### Common Issues

#### "Storage directory is locked by another process"

Another process is using the storage directory.

**Solutions:**

- Close the other process
- Manually remove the `LOCK` file (if process crashed)
- Verify no stale processes are running

#### "Out of disk space"

Not enough disk space for flushing memtables.

**Solutions:**

- Free up disk space
- Reduce `MemtableSizeLimit` and `FlushThreshold`
- Trigger compaction to merge segments
- Delete old backups

#### Slow Search Performance

Too many segments or memory pressure.

**Solutions:**

- Trigger compaction to reduce segment count
- Increase `CompactionThreshold` for more aggressive compaction
- Reduce `MemtableSizeLimit` for more frequent flushes
- Add more RAM for segment caching

#### High Memory Usage

Segment caches consuming too much memory.

**Solutions:**

- Call `store.segmentManager.EvictAllCaches()` to free memory
- Reduce memtable sizes
- Trigger compaction to reduce segment count

### Error Handling

#### Crash Recovery

On restart, the storage:

- Loads existing segments from disk
- Creates new active memtable
- Continues normal operation

Note: Data in memtables that weren't flushed before crash is lost (no WAL yet).

#### Partial Writes

Incomplete segment files are detected and ignored during startup.

#### Lock Cleanup

If a process crashes, manually remove the `LOCK` file:

```bash
rm data/LOCK
```

### Debugging

Check the contents of the data directory:

```bash
# List all files
ls -lh data/

# Check segment count
ls data/hybrid_*.bin.gz | wc -l

# Check total disk usage
du -sh data/
```

## Comparison with In-Memory Index

| Feature      | In-Memory      | Persistent        |
| ------------ | -------------- | ----------------- |
| Durability   | Lost on crash  | Survives restarts |
| Dataset Size | Limited by RAM | Limited by disk   |
| Write Speed  | Fast           | Fast (buffered)   |
| Read Speed   | Fastest        | Fast (cached)     |
| Memory Usage | High           | Lower             |
| Disk I/O     | None           | Background        |
| Complexity   | Simple         | More complex      |

## Limitations

### Current Implementation

1. **Simple Compaction**: Basic leveled compaction (can be improved)
2. **No WAL**: No write-ahead log (data in memtable can be lost on crash)
3. **Manual Recovery**: No automatic corruption detection/recovery
4. **No Replication**: Single-node only (no distributed support)

### Future Improvements

- Write-ahead log (WAL) for durability
- Bloom filters for faster segment filtering
- More sophisticated compaction strategies
- Distributed replication
- Snapshot and restore functionality
- Metrics and observability hooks

## References

- LSM-Tree: https://www.cloudcentric.dev/exploring-memtables/
- LevelDB: https://github.com/google/leveldb
- RocksDB: https://rocksdb.org/

## Support

For issues or questions:

- Check existing tests in `storage_test.go`
- Review this documentation
- File an issue on GitHub
