## Examples

### Example 1: Basic Vector Search

```go
package main

import (
    "fmt"
    "log"
    "math/rand"

    "github.com/wizenheimer/comet"
)

func main() {
    // Step 1: Create index
    index, err := comet.NewFlatIndex(128, comet.Cosine)
    if err != nil {
        log.Fatal(err)
    }

    // Step 2: Add vectors
    for i := 0; i < 1000; i++ {
        vec := make([]float32, 128)
        for j := range vec {
            vec[j] = rand.Float32()
        }
        node := comet.NewVectorNode(vec)
        if err := index.Add(*node); err != nil {
            log.Fatal(err)
        }
    }

    // Step 3: Search
    query := make([]float32, 128)
    for i := range query {
        query[i] = rand.Float32()
    }

    results, err := index.NewSearch().
        WithQuery(query).
        WithK(10).
        Execute()

    if err != nil {
        log.Fatal(err)
    }

    // Step 4: Process results
    fmt.Printf("Found %d results:\n", len(results))
    for i, result := range results {
        fmt.Printf("%d. ID=%d, Distance=%.4f\n",
            i+1, result.GetId(), result.GetScore())
    }
}
```

Output:

```
Found 10 results:
1. ID=342, Distance=0.1234
2. ID=789, Distance=0.2345
3. ID=156, Distance=0.3456
...
```

### Example 2: BM25 Full-Text Search

```go
package main

import (
    "fmt"
    "log"

    "github.com/wizenheimer/comet"
)

func main() {
    // Create text index
    index := comet.NewBM25SearchIndex()

    // Add documents
    docs := map[uint32]string{
        1: "Introduction to machine learning and artificial intelligence",
        2: "Deep learning tutorial for beginners",
        3: "Natural language processing with transformers",
        4: "Computer vision and image recognition",
        5: "Reinforcement learning in robotics",
    }

    for id, text := range docs {
        if err := index.Add(id, text); err != nil {
            log.Fatal(err)
        }
    }

    // Search for documents
    results, err := index.NewSearch().
        WithQuery("machine learning tutorial").
        WithK(3).
        Execute()

    if err != nil {
        log.Fatal(err)
    }

    // Display results
    fmt.Println("Search Results:")
    for i, result := range results {
        fmt.Printf("%d. Doc %d: \"%s\" (Score: %.2f)\n",
            i+1,
            result.GetId(),
            docs[result.GetId()],
            result.GetScore())
    }
}
```

Output:

```
Search Results:
1. Doc 2: "Deep learning tutorial for beginners" (Score: 12.45)
2. Doc 1: "Introduction to machine learning and artificial intelligence" (Score: 10.23)
3. Doc 5: "Reinforcement learning in robotics" (Score: 5.67)
```

### Example 3: Hybrid Search with Metadata Filtering

```go
package main

import (
    "fmt"
    "log"
    "math/rand"

    "github.com/wizenheimer/comet"
)

type Document struct {
    ID       uint32
    Text     string
    Vector   []float32
    Metadata map[string]interface{}
}

func main() {
    // Step 1: Create indexes
    vecIdx, _ := comet.NewFlatIndex(128, comet.Cosine)
    txtIdx := comet.NewBM25SearchIndex()
    metaIdx := comet.NewRoaringMetadataIndex()

    // Create hybrid index
    hybrid := comet.NewHybridSearchIndex(vecIdx, txtIdx, metaIdx)

    // Step 2: Add documents
    docs := []Document{
        {
            Text: "Introduction to Python programming",
            Metadata: map[string]interface{}{
                "category": "programming",
                "level": "beginner",
                "price": 29,
            },
        },
        {
            Text: "Advanced machine learning techniques",
            Metadata: map[string]interface{}{
                "category": "ai",
                "level": "advanced",
                "price": 99,
            },
        },
        {
            Text: "Python for machine learning",
            Metadata: map[string]interface{}{
                "category": "ai",
                "level": "intermediate",
                "price": 59,
            },
        },
    }

    for _, doc := range docs {
        // Generate random embedding
        vec := make([]float32, 128)
        for i := range vec {
            vec[i] = rand.Float32()
        }

        id, err := hybrid.Add(vec, doc.Text, doc.Metadata)
        if err != nil {
            log.Fatal(err)
        }
        doc.ID = id
    }

    // Step 3: Hybrid search with filters
    queryVec := make([]float32, 128)
    for i := range queryVec {
        queryVec[i] = rand.Float32()
    }

    results, err := hybrid.NewSearch().
        WithVector(queryVec).                  // Semantic search
        WithText("machine learning python").   // Keyword search
        WithMetadata(                          // Metadata filters
            comet.Eq("category", "ai"),
            comet.Lte("price", 70),
        ).
        WithK(5).
        WithFusionKind(comet.ReciprocalRankFusion).
        Execute()

    if err != nil {
        log.Fatal(err)
    }

    // Step 4: Display results
    fmt.Println("Hybrid Search Results:")
    for i, result := range results {
        fmt.Printf("%d. ID=%d, Score=%.4f\n",
            i+1, result.ID, result.Score)
    }
}
```

Output:

```
Hybrid Search Results:
1. ID=3, Score=0.0645
2. ID=2, Score=0.0312
```

