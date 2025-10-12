package comet

import (
	"fmt"
	"reflect"
	"testing"
)

// TestMetadataSearchBasicFilters tests basic filtering operations
func TestMetadataSearchBasicFilters(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add test documents
	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"category": "electronics", "price": 100, "in_stock": true}),
		NewMetadataNodeWithID(2, map[string]interface{}{"category": "electronics", "price": 200, "in_stock": false}),
		NewMetadataNodeWithID(3, map[string]interface{}{"category": "books", "price": 15, "in_stock": true}),
		NewMetadataNodeWithID(4, map[string]interface{}{"category": "books", "price": 25, "in_stock": true}),
		NewMetadataNodeWithID(5, map[string]interface{}{"category": "clothing", "price": 50, "in_stock": false}),
	}

	for _, node := range nodes {
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Failed to add node %d: %v", node.ID(), err)
		}
	}

	tests := []struct {
		name     string
		filters  []Filter
		expected []uint32
	}{
		{
			name:     "Filter by category",
			filters:  []Filter{Eq("category", "electronics")},
			expected: []uint32{1, 2},
		},
		{
			name:     "Filter by price greater than",
			filters:  []Filter{Gt("price", 50)},
			expected: []uint32{1, 2},
		},
		{
			name:     "Filter by price less than or equal",
			filters:  []Filter{Lte("price", 25)},
			expected: []uint32{3, 4},
		},
		{
			name:     "Filter by in_stock",
			filters:  []Filter{Eq("in_stock", "true")},
			expected: []uint32{1, 3, 4},
		},
		{
			name:     "Multiple filters (AND)",
			filters:  []Filter{Eq("category", "books"), Gt("price", 15)},
			expected: []uint32{4},
		},
		{
			name:     "Price range",
			filters:  []Filter{Range("price", 20, 150)},
			expected: []uint32{1, 4, 5},
		},
		{
			name:     "Not equal",
			filters:  []Filter{Ne("category", "electronics")},
			expected: []uint32{3, 4, 5},
		},
		{
			name:     "Empty filters",
			filters:  []Filter{},
			expected: []uint32{1, 2, 3, 4, 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().WithFilters(tt.filters...).Execute()
			if err != nil {
				t.Fatalf("Query failed: %v", err)
			}

			gotIDs := extractIDs(results)

			if !reflect.DeepEqual(gotIDs, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, gotIDs)
			}
		})
	}
}

// TestMetadataSearchExpressiveQueries tests complex query scenarios
func TestMetadataSearchExpressiveQueries(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Create a realistic e-commerce dataset
	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"category": "laptop", "brand": "Apple", "price": 1500, "rating": 4.5, "verified": true}),
		NewMetadataNodeWithID(2, map[string]interface{}{"category": "laptop", "brand": "Dell", "price": 800, "rating": 4.2, "verified": true}),
		NewMetadataNodeWithID(3, map[string]interface{}{"category": "laptop", "brand": "HP", "price": 600, "rating": 3.8, "verified": false}),
		NewMetadataNodeWithID(4, map[string]interface{}{"category": "phone", "brand": "Apple", "price": 1000, "rating": 4.7, "verified": true}),
		NewMetadataNodeWithID(5, map[string]interface{}{"category": "phone", "brand": "Samsung", "price": 900, "rating": 4.5, "verified": true}),
		NewMetadataNodeWithID(6, map[string]interface{}{"category": "phone", "brand": "Google", "price": 700, "rating": 4.3, "verified": false}),
		NewMetadataNodeWithID(7, map[string]interface{}{"category": "tablet", "brand": "Apple", "price": 800, "rating": 4.6, "verified": true}),
		NewMetadataNodeWithID(8, map[string]interface{}{"category": "tablet", "brand": "Samsung", "price": 500, "rating": 4.1, "verified": true}),
	}

	for _, node := range nodes {
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Failed to add node %d: %v", node.ID(), err)
		}
	}

	t.Run("Complex query with MetadataFilterQueryBuilder - Apple products OR Samsung phones", func(t *testing.T) {
		// (brand = 'Apple') OR (category = 'phone' AND brand = 'Samsung')
		results, err := NewMetadataFilterQuery().
			Where(Eq("brand", "Apple")).
			Or(Eq("category", "phone"), Eq("brand", "Samsung")).
			Execute(idx)

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		gotIDs := extractIDs(results)
		expected := []uint32{1, 4, 5, 7} // Apple: 1,4,7 + Samsung phone: 5

		if !reflect.DeepEqual(gotIDs, expected) {
			t.Errorf("Expected %v, got %v", expected, gotIDs)
		}
	})

	t.Run("Find premium verified products", func(t *testing.T) {
		// price >= 900 AND verified = true AND rating >= 4.5
		results, err := NewMetadataFilterQuery().
			Where(
				Gte("price", 900),
				Eq("verified", "true"),
				Gte("rating", 4.5),
			).
			Execute(idx)

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		gotIDs := extractIDs(results)
		expected := []uint32{1, 4, 5}

		if !reflect.DeepEqual(gotIDs, expected) {
			t.Errorf("Expected %v, got %v", expected, gotIDs)
		}
	})

	t.Run("Find affordable options across multiple categories", func(t *testing.T) {
		// price <= 700 AND category IN ('phone', 'tablet')
		results, err := NewMetadataFilterQuery().
			Where(
				Lte("price", 700),
				In("category", "phone", "tablet"),
			).
			Execute(idx)

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		gotIDs := extractIDs(results)
		expected := []uint32{6, 8}

		if !reflect.DeepEqual(gotIDs, expected) {
			t.Errorf("Expected %v, got %v", expected, gotIDs)
		}
	})

	t.Run("Exclude specific brands", func(t *testing.T) {
		// category = 'laptop' AND brand NOT IN ('HP')
		results, err := NewMetadataFilterQuery().
			Where(
				Eq("category", "laptop"),
				NotIn("brand", "HP"),
			).
			Execute(idx)

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		gotIDs := extractIDs(results)
		expected := []uint32{1, 2}

		if !reflect.DeepEqual(gotIDs, expected) {
			t.Errorf("Expected %v, got %v", expected, gotIDs)
		}
	})

	t.Run("Price range with verified filter", func(t *testing.T) {
		// price BETWEEN 600 AND 900 AND verified = true
		results, err := NewMetadataFilterQuery().
			Where(
				Between("price", 600, 900),
				Eq("verified", "true"),
			).
			Execute(idx)

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		gotIDs := extractIDs(results)
		expected := []uint32{2, 5, 7}

		if !reflect.DeepEqual(gotIDs, expected) {
			t.Errorf("Expected %v, got %v", expected, gotIDs)
		}
	})
}

// TestMetadataSearchAdvancedExpressions tests advanced filter expressions
func TestMetadataSearchAdvancedExpressions(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Create a dataset for a movie recommendation system
	movies := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"genre": "action", "year": 2020, "rating": 8.5, "language": "en", "director": "Nolan"}),
		NewMetadataNodeWithID(2, map[string]interface{}{"genre": "action", "year": 2019, "rating": 7.8, "language": "en", "director": "Bay"}),
		NewMetadataNodeWithID(3, map[string]interface{}{"genre": "comedy", "year": 2021, "rating": 7.2, "language": "en", "director": "Wright"}),
		NewMetadataNodeWithID(4, map[string]interface{}{"genre": "drama", "year": 2020, "rating": 9.0, "language": "en", "director": "Nolan"}),
		NewMetadataNodeWithID(5, map[string]interface{}{"genre": "drama", "year": 2018, "rating": 8.8, "language": "fr", "director": "Dumont"}),
		NewMetadataNodeWithID(6, map[string]interface{}{"genre": "comedy", "year": 2022, "rating": 6.5, "language": "en", "director": "Apatow"}),
		NewMetadataNodeWithID(7, map[string]interface{}{"genre": "action", "year": 2022, "rating": 7.5, "language": "en", "director": "Nolan"}),
		NewMetadataNodeWithID(8, map[string]interface{}{"genre": "scifi", "year": 2021, "rating": 8.2, "language": "en", "director": "Villeneuve"}),
	}

	for _, movie := range movies {
		err := idx.Add(*movie)
		if err != nil {
			t.Fatalf("Failed to add node %d: %v", movie.ID(), err)
		}
	}

	t.Run("Find recent highly-rated action or sci-fi movies", func(t *testing.T) {
		// (genre = 'action' OR genre = 'scifi') AND year >= 2020 AND rating >= 8.0
		results, err := NewMetadataFilterQuery().
			Where(
				AnyOf("genre", "action", "scifi"),
				Gte("year", 2020),
				Gte("rating", 8.0),
			).
			Execute(idx)

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		gotIDs := extractIDs(results)
		expected := []uint32{1, 8}

		if !reflect.DeepEqual(gotIDs, expected) {
			t.Errorf("Expected %v, got %v", expected, gotIDs)
		}
	})

	t.Run("Find Nolan films OR highly-rated dramas", func(t *testing.T) {
		// director = 'Nolan' OR (genre = 'drama' AND rating >= 8.5)
		results, err := NewMetadataFilterQuery().
			Where(Eq("director", "Nolan")).
			Or(Eq("genre", "drama"), Gte("rating", 8.5)).
			Execute(idx)

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		gotIDs := extractIDs(results)
		expected := []uint32{1, 4, 5, 7}

		if !reflect.DeepEqual(gotIDs, expected) {
			t.Errorf("Expected %v, got %v", expected, gotIDs)
		}
	})

	t.Run("Exclude low-rated recent comedies", func(t *testing.T) {
		// genre = 'comedy' AND NOT (rating < 7.0)
		results, err := NewMetadataFilterQuery().
			Where(Eq("genre", "comedy")).
			And(Not(Lt("rating", 7.0))).
			Execute(idx)

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		gotIDs := extractIDs(results)
		expected := []uint32{3}

		if !reflect.DeepEqual(gotIDs, expected) {
			t.Errorf("Expected %v, got %v", expected, gotIDs)
		}
	})

	t.Run("Complex multi-condition query", func(t *testing.T) {
		// (year >= 2020 AND rating >= 8.0 AND language = 'en') OR (director = 'Nolan' AND genre = 'action')
		results, err := NewMetadataFilterQuery().
			Where(Gte("year", 2020), Gte("rating", 8.0), Eq("language", "en")).
			Or(Eq("director", "Nolan"), Eq("genre", "action")).
			Execute(idx)

		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		gotIDs := extractIDs(results)
		expected := []uint32{1, 4, 7, 8}

		if !reflect.DeepEqual(gotIDs, expected) {
			t.Errorf("Expected %v, got %v", expected, gotIDs)
		}
	})
}

// TestMetadataSearchExistenceQueries tests field existence operators
func TestMetadataSearchExistenceQueries(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add documents with varying fields
	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"name": "Product A", "price": 100, "category": "electronics"}),
		NewMetadataNodeWithID(2, map[string]interface{}{"name": "Product B", "price": 200}),
		NewMetadataNodeWithID(3, map[string]interface{}{"name": "Product C", "category": "books"}),
		NewMetadataNodeWithID(4, map[string]interface{}{"name": "Product D", "price": 50, "category": "clothing", "discount": 10}),
		NewMetadataNodeWithID(5, map[string]interface{}{"name": "Product E"}),
	}

	for _, node := range nodes {
		err := idx.Add(*node)
		if err != nil {
			t.Fatalf("Failed to add node %d: %v", node.ID(), err)
		}
	}

	tests := []struct {
		name     string
		filters  []Filter
		expected []uint32
	}{
		{
			name:     "Documents with price field",
			filters:  []Filter{Exists("price")},
			expected: []uint32{1, 2, 4},
		},
		{
			name:     "Documents without category",
			filters:  []Filter{NotExists("category")},
			expected: []uint32{2, 5},
		},
		{
			name:     "Documents with discount (IsNotNull)",
			filters:  []Filter{IsNotNull("discount")},
			expected: []uint32{4},
		},
		{
			name:     "Documents without discount (IsNull)",
			filters:  []Filter{IsNull("discount")},
			expected: []uint32{1, 2, 3, 5},
		},
		{
			name:     "Has price but no category",
			filters:  []Filter{Exists("price"), IsNull("category")},
			expected: []uint32{2},
		},
		{
			name:     "Has category but no price",
			filters:  []Filter{Exists("category"), NotExists("price")},
			expected: []uint32{3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().WithFilters(tt.filters...).Execute()
			if err != nil {
				t.Fatalf("Query failed: %v", err)
			}

			gotIDs := extractIDs(results)

			if !reflect.DeepEqual(gotIDs, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, gotIDs)
			}
		})
	}
}

// TestMetadataSearchInOperator tests IN and NOT IN operators
func TestMetadataSearchInOperator(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add test data
	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"color": "red", "size": "small"}),
		NewMetadataNodeWithID(2, map[string]interface{}{"color": "blue", "size": "medium"}),
		NewMetadataNodeWithID(3, map[string]interface{}{"color": "green", "size": "large"}),
		NewMetadataNodeWithID(4, map[string]interface{}{"color": "red", "size": "large"}),
		NewMetadataNodeWithID(5, map[string]interface{}{"color": "yellow", "size": "small"}),
	}

	for _, node := range nodes {
		idx.Add(*node)
	}

	tests := []struct {
		name     string
		filters  []Filter
		expected []uint32
	}{
		{
			name:     "Color in red or blue",
			filters:  []Filter{In("color", "red", "blue")},
			expected: []uint32{1, 2, 4},
		},
		{
			name:     "Color not in red or blue",
			filters:  []Filter{NotIn("color", "red", "blue")},
			expected: []uint32{3, 5},
		},
		{
			name:     "Size in small or large",
			filters:  []Filter{AnyOf("size", "small", "large")},
			expected: []uint32{1, 3, 4, 5},
		},
		{
			name:     "Color in red AND size not in small",
			filters:  []Filter{In("color", "red"), NotIn("size", "small")},
			expected: []uint32{4},
		},
		{
			name:     "None of yellow or green",
			filters:  []Filter{NoneOf("color", "yellow", "green")},
			expected: []uint32{1, 2, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().WithFilters(tt.filters...).Execute()
			if err != nil {
				t.Fatalf("Query failed: %v", err)
			}

			gotIDs := extractIDs(results)

			if !reflect.DeepEqual(gotIDs, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, gotIDs)
			}
		})
	}
}

// TestMetadataSearchNotOperator tests the Not operator
func TestMetadataSearchNotOperator(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add test data
	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"status": "active", "score": 100}),
		NewMetadataNodeWithID(2, map[string]interface{}{"status": "inactive", "score": 50}),
		NewMetadataNodeWithID(3, map[string]interface{}{"status": "active", "score": 75}),
		NewMetadataNodeWithID(4, map[string]interface{}{"status": "pending", "score": 90}),
	}

	for _, node := range nodes {
		idx.Add(*node)
	}

	tests := []struct {
		name     string
		filter   Filter
		expected []uint32
	}{
		{
			name:     "Not equal to inactive",
			filter:   Not(Eq("status", "inactive")),
			expected: []uint32{1, 3, 4},
		},
		{
			name:     "Not greater than 75 (becomes less than or equal)",
			filter:   Not(Gt("score", 75)),
			expected: []uint32{2, 3},
		},
		{
			name:     "Not less than 75 (becomes greater than or equal)",
			filter:   Not(Lt("score", 75)),
			expected: []uint32{1, 3, 4},
		},
		{
			name:     "Not in active or pending",
			filter:   Not(In("status", "active", "pending")),
			expected: []uint32{2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().WithFilters(tt.filter).Execute()
			if err != nil {
				t.Fatalf("Query failed: %v", err)
			}

			gotIDs := extractIDs(results)

			if !reflect.DeepEqual(gotIDs, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, gotIDs)
			}
		})
	}
}

// TestMetadataSearchEmptyResults tests queries that return no results
func TestMetadataSearchEmptyResults(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add test data
	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"category": "electronics", "price": 100}),
		NewMetadataNodeWithID(2, map[string]interface{}{"category": "books", "price": 20}),
	}

	for _, node := range nodes {
		idx.Add(*node)
	}

	tests := []struct {
		name    string
		filters []Filter
	}{
		{
			name:    "Non-existent category",
			filters: []Filter{Eq("category", "nonexistent")},
		},
		{
			name:    "Impossible price range",
			filters: []Filter{Gt("price", 1000)},
		},
		{
			name:    "Contradictory filters",
			filters: []Filter{Eq("category", "electronics"), Eq("category", "books")},
		},
		{
			name:    "Non-existent field",
			filters: []Filter{Exists("nonexistent_field")},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().WithFilters(tt.filters...).Execute()
			if err != nil {
				t.Fatalf("Query failed: %v", err)
			}

			if len(results) != 0 {
				t.Errorf("Expected 0 results, got %d: %v", len(results), extractIDs(results))
			}
		})
	}
}

// TestMetadataSearchFilterGroups tests complex filter groups
func TestMetadataSearchFilterGroups(t *testing.T) {
	idx := NewRoaringMetadataIndex()

	// Add test data
	nodes := []*MetadataNode{
		NewMetadataNodeWithID(1, map[string]interface{}{"type": "A", "value": 10}),
		NewMetadataNodeWithID(2, map[string]interface{}{"type": "B", "value": 20}),
		NewMetadataNodeWithID(3, map[string]interface{}{"type": "A", "value": 30}),
		NewMetadataNodeWithID(4, map[string]interface{}{"type": "C", "value": 15}),
	}

	for _, node := range nodes {
		idx.Add(*node)
	}

	tests := []struct {
		name     string
		groups   []*FilterGroup
		expected []uint32
	}{
		{
			name: "Type A OR Type B",
			groups: []*FilterGroup{
				{Filters: []Filter{Eq("type", "A")}, Logic: AND},
				{Filters: []Filter{Eq("type", "B")}, Logic: AND},
			},
			expected: []uint32{1, 2, 3},
		},
		{
			name: "(Type A AND value > 20) OR (Type B)",
			groups: []*FilterGroup{
				{Filters: []Filter{Eq("type", "A"), Gt("value", 20)}, Logic: AND},
				{Filters: []Filter{Eq("type", "B")}, Logic: AND},
			},
			expected: []uint32{2, 3},
		},
		{
			name: "Multiple OR groups",
			groups: []*FilterGroup{
				{Filters: []Filter{Eq("type", "A"), Lt("value", 20)}, Logic: AND},
				{Filters: []Filter{Eq("type", "B")}, Logic: AND},
				{Filters: []Filter{Eq("type", "C")}, Logic: AND},
			},
			expected: []uint32{1, 2, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().WithFilterGroups(tt.groups...).Execute()
			if err != nil {
				t.Fatalf("Query failed: %v", err)
			}

			gotIDs := extractIDs(results)

			if !reflect.DeepEqual(gotIDs, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, gotIDs)
			}
		})
	}
}

// Benchmark tests
func BenchmarkMetadataSearchSimpleQuery(b *testing.B) {
	idx := NewRoaringMetadataIndex()

	// Add 10000 documents
	for i := uint32(1); i <= 10000; i++ {
		node := NewMetadataNodeWithID(i, map[string]interface{}{
			"category": fmt.Sprintf("cat%d", i%10),
			"price":    int(i % 1000),
			"in_stock": i%2 == 0,
		})
		idx.Add(*node)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = idx.NewSearch().WithFilters(
			Eq("category", "cat5"),
			Gt("price", 500),
		).Execute()
	}
}

func BenchmarkMetadataSearchComplexQuery(b *testing.B) {
	idx := NewRoaringMetadataIndex()

	// Add 10000 documents
	for i := uint32(1); i <= 10000; i++ {
		node := NewMetadataNodeWithID(i, map[string]interface{}{
			"category": fmt.Sprintf("cat%d", i%10),
			"brand":    fmt.Sprintf("brand%d", i%5),
			"price":    int(i % 1000),
			"rating":   float64((i % 50) / 10.0),
			"in_stock": i%2 == 0,
		})
		idx.Add(*node)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = NewMetadataFilterQuery().
			Where(In("category", "cat1", "cat2", "cat3"), Gte("price", 300)).
			Or(Eq("brand", "brand0"), Gte("rating", 4.0)).
			Execute(idx)
	}
}

func BenchmarkMetadataSearchRangeQuery(b *testing.B) {
	idx := NewRoaringMetadataIndex()

	// Add 10000 documents
	for i := uint32(1); i <= 10000; i++ {
		node := NewMetadataNodeWithID(i, map[string]interface{}{
			"price":  int(i % 1000),
			"rating": float64((i % 50) / 10.0),
		})
		idx.Add(*node)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = idx.NewSearch().WithFilters(
			Between("price", 200, 800),
			Gte("rating", 3.0),
		).Execute()
	}
}

func BenchmarkMetadataSearchExistenceQuery(b *testing.B) {
	idx := NewRoaringMetadataIndex()

	// Add 10000 documents with varying fields
	for i := uint32(1); i <= 10000; i++ {
		metadata := map[string]interface{}{
			"name": fmt.Sprintf("doc%d", i),
		}
		if i%2 == 0 {
			metadata["category"] = "test"
		}
		if i%3 == 0 {
			metadata["price"] = int(i)
		}
		node := NewMetadataNodeWithID(i, metadata)
		idx.Add(*node)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = idx.NewSearch().WithFilters(
			Exists("category"),
			Exists("price"),
		).Execute()
	}
}

func BenchmarkMetadataSearchInOperator(b *testing.B) {
	idx := NewRoaringMetadataIndex()

	// Add 10000 documents
	for i := uint32(1); i <= 10000; i++ {
		node := NewMetadataNodeWithID(i, map[string]interface{}{
			"category": fmt.Sprintf("cat%d", i%20),
			"value":    int(i),
		})
		idx.Add(*node)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = idx.NewSearch().WithFilters(
			In("category", "cat1", "cat2", "cat3", "cat4", "cat5"),
			Gt("value", 1000),
		).Execute()
	}
}
