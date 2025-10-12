package comet

import (
	"testing"
)

// TestBM25TextSearchInterface tests that bm25TextSearch implements TextSearch
func TestBM25TextSearchInterface(t *testing.T) {
	var _ TextSearch = (*bm25TextSearch)(nil)
}

// TestBM25TextSearchWithQuery tests the WithQuery builder method
func TestBM25TextSearchWithQuery(t *testing.T) {
	idx := NewBM25SearchIndex()
	idx.Add(1, "the quick brown fox")
	idx.Add(2, "the lazy dog")

	search := idx.NewSearch().WithQuery("fox")

	if search == nil {
		t.Fatal("WithQuery() returned nil")
	}

	// Verify it's chainable
	search = search.WithK(5)
	if search == nil {
		t.Fatal("WithK() after WithQuery() returned nil")
	}
}

// TestBM25TextSearchWithNode tests the WithNode builder method
func TestBM25TextSearchWithNode(t *testing.T) {
	idx := NewBM25SearchIndex()
	idx.Add(1, "the quick brown fox")
	idx.Add(2, "the lazy dog")
	idx.Add(3, "quick brown rabbit")

	// Search using node 1 as query (should find similar documents)
	results, err := idx.NewSearch().
		WithNode(1).
		WithK(5).
		Execute()

	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	// Should find at least the document itself
	if len(results) == 0 {
		t.Error("WithNode() search returned no results")
	}
}

// TestBM25TextSearchWithK tests the WithK builder method
func TestBM25TextSearchWithK(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add multiple documents
	for i := uint32(1); i <= 10; i++ {
		idx.Add(i, "the quick brown fox jumps")
	}

	tests := []struct {
		name       string
		k          int
		wantLength int
	}{
		{"k=3", 3, 3},
		{"k=5", 5, 5},
		{"k=10", 10, 10},
		{"k=0 (all)", 0, 10},
		{"k=-1 (all)", -1, 10},
		{"k=100 (more than available)", 100, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().
				WithQuery("quick").
				WithK(tt.k).
				Execute()

			if err != nil {
				t.Fatalf("Execute() error = %v", err)
			}

			if len(results) != tt.wantLength {
				t.Errorf("got %d results, want %d", len(results), tt.wantLength)
			}
		})
	}
}

// TestBM25TextSearchWithScoreAggregation tests the WithScoreAggregation builder method
func TestBM25TextSearchWithScoreAggregation(t *testing.T) {
	idx := NewBM25SearchIndex()

	idx.Add(1, "fox dog cat")
	idx.Add(2, "fox dog")
	idx.Add(3, "cat mouse")
	idx.Add(4, "dog")

	tests := []struct {
		name            string
		aggregationKind ScoreAggregationKind
		wantErr         bool
	}{
		{"sum aggregation", SumAggregation, false},
		{"max aggregation", MaxAggregation, false},
		{"mean aggregation", MeanAggregation, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().
				WithQuery("fox", "dog"). // Multiple queries
				WithScoreAggregation(tt.aggregationKind).
				WithK(5).
				Execute()

			if (err != nil) != tt.wantErr {
				t.Fatalf("Execute() error = %v, wantErr %v", err, tt.wantErr)
			}

			if !tt.wantErr && len(results) == 0 {
				t.Error("expected some results")
			}

			// Verify scores are still sorted
			for i := 1; i < len(results); i++ {
				if results[i].Score > results[i-1].Score {
					t.Errorf("results not sorted: results[%d].Score > results[%d].Score",
						i, i-1)
				}
			}
		})
	}
}

// TestBM25TextSearchWithCutoff tests the WithCutoff builder method
func TestBM25TextSearchWithCutoff(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add documents with varying relevance
	idx.Add(1, "fox fox fox fox")       // Very relevant
	idx.Add(2, "fox fox")               // Relevant
	idx.Add(3, "the lazy dog sleeps")   // Not relevant
	idx.Add(4, "cat and mouse")         // Not relevant
	idx.Add(5, "quick brown fox jumps") // Relevant

	tests := []struct {
		name   string
		cutoff int
	}{
		{"no cutoff", -1},
		{"cutoff=1", 1},
		{"cutoff=2", 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().
				WithQuery("fox").
				WithK(10).
				WithCutoff(tt.cutoff).
				Execute()

			if err != nil {
				t.Fatalf("Execute() error = %v", err)
			}

			// With cutoff, we might get fewer results
			if tt.cutoff == -1 {
				// Should get all matching results (documents containing "fox")
				if len(results) == 0 {
					t.Error("expected results without cutoff")
				}
			}
		})
	}
}

// TestBM25TextSearchExecute tests the Execute method
func TestBM25TextSearchExecute(t *testing.T) {
	idx := NewBM25SearchIndex()

	docs := map[uint32]string{
		1: "the quick brown fox jumps over the lazy dog",
		2: "the lazy cat sleeps under the warm sun",
		3: "quick brown rabbits run through the forest",
		4: "the forest is dark and mysterious",
		5: "dogs and cats are popular pets",
	}

	for id, text := range docs {
		idx.Add(id, text)
	}

	tests := []struct {
		name       string
		query      string
		k          int
		minResults int
		wantErr    bool
	}{
		{
			name:       "basic query",
			query:      "fox",
			k:          5,
			minResults: 1,
			wantErr:    false,
		},
		{
			name:       "multi-term query",
			query:      "quick brown",
			k:          5,
			minResults: 2,
			wantErr:    false,
		},
		{
			name:       "no matches",
			query:      "elephant",
			k:          5,
			minResults: 0,
			wantErr:    false,
		},
		{
			name:       "empty query",
			query:      "",
			k:          5,
			minResults: 0,
			wantErr:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := idx.NewSearch().
				WithQuery(tt.query).
				WithK(tt.k).
				Execute()

			if (err != nil) != tt.wantErr {
				t.Fatalf("Execute() error = %v, wantErr %v", err, tt.wantErr)
			}

			if !tt.wantErr && len(results) < tt.minResults {
				t.Errorf("got %d results, want at least %d", len(results), tt.minResults)
			}

			// Verify result structure
			for i, result := range results {
				if result.Id == 0 && len(results) > 0 {
					t.Errorf("results[%d].Id = 0, want non-zero", i)
				}
				if result.Score < 0 {
					t.Errorf("results[%d].Score = %.4f, want >= 0", i, result.Score)
				}
			}

			// Verify results are sorted by score (descending)
			for i := 1; i < len(results); i++ {
				if results[i].Score > results[i-1].Score {
					t.Errorf("results not sorted: results[%d].Score > results[%d].Score",
						i, i-1)
				}
			}
		})
	}
}

// TestBM25TextSearchMultiQuery tests searching with multiple queries
func TestBM25TextSearchMultiQuery(t *testing.T) {
	idx := NewBM25SearchIndex()

	idx.Add(1, "fox and dog")
	idx.Add(2, "fox and cat")
	idx.Add(3, "dog and cat")
	idx.Add(4, "rabbit and mouse")

	results, err := idx.NewSearch().
		WithQuery("fox", "dog").
		WithK(5).
		Execute()

	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	if len(results) == 0 {
		t.Fatal("multi-query search returned no results")
	}

	// Documents 1, 2, and 3 should be in results
	foundDocs := make(map[uint32]bool)
	for _, result := range results {
		foundDocs[result.Id] = true
	}

	if !foundDocs[1] {
		t.Error("expected document 1 in results")
	}
}

// TestBM25TextSearchWithNodeAndQuery tests combining WithNode and WithQuery
func TestBM25TextSearchWithNodeAndQuery(t *testing.T) {
	idx := NewBM25SearchIndex()

	idx.Add(1, "quick brown fox")
	idx.Add(2, "lazy brown dog")
	idx.Add(3, "quick rabbit")
	idx.Add(4, "slow turtle")

	// Search using both node and direct query
	results, err := idx.NewSearch().
		WithNode(1).           // Use doc 1 as query
		WithQuery("lazy dog"). // Also search for "lazy dog"
		WithK(5).
		Execute()

	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	if len(results) == 0 {
		t.Fatal("combined node+query search returned no results")
	}
}

// TestBM25TextSearchNoQueryOrNode tests error when neither query nor node is provided
func TestBM25TextSearchNoQueryOrNode(t *testing.T) {
	idx := NewBM25SearchIndex()
	idx.Add(1, "test document")

	_, err := idx.NewSearch().
		WithK(5).
		Execute()

	if err == nil {
		t.Error("Execute() expected error when no query or node provided, got nil")
	}
}

// TestBM25TextSearchNodeNotFound tests error when node ID doesn't exist
func TestBM25TextSearchNodeNotFound(t *testing.T) {
	idx := NewBM25SearchIndex()
	idx.Add(1, "test document")

	_, err := idx.NewSearch().
		WithNode(999). // Non-existent node
		WithK(5).
		Execute()

	if err == nil {
		t.Error("Execute() expected error for non-existent node, got nil")
	}
}

// TestBM25TextSearchEmptyIndex tests searching an empty index
func TestBM25TextSearchEmptyIndex(t *testing.T) {
	idx := NewBM25SearchIndex()

	results, err := idx.NewSearch().
		WithQuery("test").
		WithK(5).
		Execute()

	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	if len(results) != 0 {
		t.Errorf("got %d results from empty index, want 0", len(results))
	}
}

// TestBM25TextSearchScoreOrdering tests that scores are properly ordered
func TestBM25TextSearchScoreOrdering(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add documents with different relevance levels
	idx.Add(1, "fox fox fox fox fox")       // Very relevant
	idx.Add(2, "fox fox fox")               // Highly relevant
	idx.Add(3, "fox")                       // Relevant
	idx.Add(4, "the quick brown fox jumps") // Less relevant (more words)
	idx.Add(5, "cat and dog")               // Not relevant

	results, err := idx.NewSearch().
		WithQuery("fox").
		WithK(10).
		Execute()

	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	// Should have results for documents containing "fox"
	if len(results) < 4 {
		t.Errorf("got %d results, want at least 4", len(results))
	}

	// Document 1 should have highest score
	if len(results) > 0 && results[0].Id != 1 {
		t.Errorf("expected document 1 to rank first, got document %d", results[0].Id)
	}

	// Verify descending order
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("scores not in descending order at index %d: %.4f > %.4f",
				i, results[i].Score, results[i-1].Score)
		}
	}
}

// TestTextAggregations tests the text aggregation interface
func TestTextAggregations(t *testing.T) {
	tests := []struct {
		name            string
		aggregationKind ScoreAggregationKind
	}{
		{"sum aggregation", SumAggregation},
		{"max aggregation", MaxAggregation},
		{"mean aggregation", MeanAggregation},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results := []TextResult{
				{Id: 1, Score: 1.0},
				{Id: 2, Score: 2.0},
			}

			agg, err := NewTextAggregation(tt.aggregationKind)
			if err != nil {
				t.Fatalf("NewTextAggregation() error = %v", err)
			}

			aggregated := agg.Aggregate(results)

			if len(aggregated) != 2 {
				t.Errorf("got %d results, want 2", len(aggregated))
			}
		})
	}
}

// TestTextSumAggregation tests sum aggregation for text results
func TestTextSumAggregation(t *testing.T) {
	results := []TextResult{
		{Id: 1, Score: 1.0},
		{Id: 2, Score: 2.0},
		{Id: 1, Score: 1.5}, // Duplicate ID
		{Id: 3, Score: 0.5},
		{Id: 2, Score: 1.0}, // Duplicate ID
	}

	agg, err := NewTextAggregation(SumAggregation)
	if err != nil {
		t.Fatalf("NewTextAggregation() error = %v", err)
	}

	aggregated := agg.Aggregate(results)

	// Should have 3 unique documents
	if len(aggregated) != 3 {
		t.Errorf("got %d results, want 3", len(aggregated))
	}

	// Verify scores are summed correctly
	scoreMap := make(map[uint32]float32)
	for _, result := range aggregated {
		scoreMap[result.Id] = result.Score
	}

	if scoreMap[1] != 2.5 { // 1.0 + 1.5
		t.Errorf("document 1 score = %.2f, want 2.5", scoreMap[1])
	}
	if scoreMap[2] != 3.0 { // 2.0 + 1.0
		t.Errorf("document 2 score = %.2f, want 3.0", scoreMap[2])
	}
	if scoreMap[3] != 0.5 {
		t.Errorf("document 3 score = %.2f, want 0.5", scoreMap[3])
	}
}

// TestTextMaxAggregation tests max aggregation for text results
func TestTextMaxAggregation(t *testing.T) {
	results := []TextResult{
		{Id: 1, Score: 1.0},
		{Id: 2, Score: 2.0},
		{Id: 1, Score: 1.5}, // Duplicate ID
		{Id: 3, Score: 0.5},
		{Id: 2, Score: 1.0}, // Duplicate ID
	}

	agg, err := NewTextAggregation(MaxAggregation)
	if err != nil {
		t.Fatalf("NewTextAggregation() error = %v", err)
	}

	aggregated := agg.Aggregate(results)

	// Should have 3 unique documents
	if len(aggregated) != 3 {
		t.Errorf("got %d results, want 3", len(aggregated))
	}

	// Verify scores are max correctly
	scoreMap := make(map[uint32]float32)
	for _, result := range aggregated {
		scoreMap[result.Id] = result.Score
	}

	if scoreMap[1] != 1.5 { // max(1.0, 1.5)
		t.Errorf("document 1 score = %.2f, want 1.5", scoreMap[1])
	}
	if scoreMap[2] != 2.0 { // max(2.0, 1.0)
		t.Errorf("document 2 score = %.2f, want 2.0", scoreMap[2])
	}
	if scoreMap[3] != 0.5 {
		t.Errorf("document 3 score = %.2f, want 0.5", scoreMap[3])
	}
}

// TestTextMeanAggregation tests mean aggregation for text results
func TestTextMeanAggregation(t *testing.T) {
	results := []TextResult{
		{Id: 1, Score: 1.0},
		{Id: 2, Score: 2.0},
		{Id: 1, Score: 2.0}, // Duplicate ID
		{Id: 3, Score: 0.5},
		{Id: 2, Score: 4.0}, // Duplicate ID
	}

	agg, err := NewTextAggregation(MeanAggregation)
	if err != nil {
		t.Fatalf("NewTextAggregation() error = %v", err)
	}

	aggregated := agg.Aggregate(results)

	// Should have 3 unique documents
	if len(aggregated) != 3 {
		t.Errorf("got %d results, want 3", len(aggregated))
	}

	// Verify scores are averaged correctly
	scoreMap := make(map[uint32]float32)
	for _, result := range aggregated {
		scoreMap[result.Id] = result.Score
	}

	if scoreMap[1] != 1.5 { // (1.0 + 2.0) / 2
		t.Errorf("document 1 score = %.2f, want 1.5", scoreMap[1])
	}
	if scoreMap[2] != 3.0 { // (2.0 + 4.0) / 2
		t.Errorf("document 2 score = %.2f, want 3.0", scoreMap[2])
	}
	if scoreMap[3] != 0.5 {
		t.Errorf("document 3 score = %.2f, want 0.5", scoreMap[3])
	}
}

// TestRealisticWikipediaSearch simulates searching Wikipedia-style articles
func TestRealisticWikipediaSearch(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add Wikipedia-style article summaries (50+ articles covering various topics)
	articles := map[uint32]string{
		// Programming Languages
		1:  "Go is a statically typed, compiled programming language designed at Google. It is syntactically similar to C, but with memory safety, garbage collection, structural typing, and CSP-style concurrency.",
		2:  "Python is an interpreted, high-level, general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant indentation.",
		3:  "JavaScript, often abbreviated JS, is a programming language that is one of the core technologies of the World Wide Web. JavaScript enables interactive web pages and is an essential part of web applications.",
		4:  "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. Rust enforces memory safety without requiring garbage collection.",
		5:  "C is a general-purpose computer programming language. It was created in the 1970s and is still widely used. C has been standardized by ANSI and ISO.",
		6:  "Java is a high-level, class-based, object-oriented programming language. Java applications are typically compiled to bytecode that can run on any Java virtual machine regardless of the underlying computer architecture.",
		7:  "TypeScript is a strongly typed programming language that builds on JavaScript. TypeScript adds optional static typing to JavaScript, helping to catch errors at compile time.",
		8:  "Ruby is an interpreted, high-level, general-purpose programming language. It was designed with an emphasis on programming productivity and simplicity. Ruby is dynamically typed and uses garbage collection.",
		9:  "C++ is a general-purpose programming language created as an extension of the C programming language. C++ has object-oriented, generic, and functional features in addition to facilities for low-level memory manipulation.",
		10: "Swift is a general-purpose, multi-paradigm, compiled programming language developed by Apple. Swift is designed to work with Apple's Cocoa and Cocoa Touch frameworks and the large body of existing Objective-C code.",
		11: "Kotlin is a cross-platform, statically typed, general-purpose programming language with type inference. Kotlin is designed to interoperate fully with Java and runs on the Java Virtual Machine.",
		12: "PHP is a general-purpose scripting language geared towards web development. It was originally created by Danish-Canadian programmer Rasmus Lerdorf in 1994. The PHP reference implementation is now produced by The PHP Group.",

		// Computer Science Topics
		13: "Machine Learning is a branch of artificial intelligence focused on building applications that learn from data and improve their accuracy over time without being programmed to do so.",
		14: "Artificial Intelligence refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind.",
		15: "Deep Learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
		16: "Neural Networks are computing systems inspired by the biological neural networks that constitute animal brains. A neural network is based on a collection of connected units or nodes called artificial neurons.",
		17: "Data Structures are a way of organizing and storing data so that they can be accessed and worked with efficiently. They define the relationship between the data, and the operations that can be performed on the data.",
		18: "Algorithms are step-by-step procedures for calculations, data processing, and automated reasoning tasks. Algorithms are essential to the way computers process data.",
		19: "Database Management Systems are software systems used to store, retrieve, and run queries on data. A DBMS serves as an interface between an end-user and a database, allowing users to create, read, update, and delete data.",
		20: "Cloud Computing is the delivery of computing services including servers, storage, databases, networking, software, analytics, and intelligence over the Internet to offer faster innovation, flexible resources, and economies of scale.",

		// Web Technologies
		21: "HTML (HyperText Markup Language) is the standard markup language for documents designed to be displayed in a web browser. It can be assisted by technologies such as Cascading Style Sheets and scripting languages such as JavaScript.",
		22: "CSS (Cascading Style Sheets) is a style sheet language used for describing the presentation of a document written in a markup language such as HTML. CSS is designed to enable the separation of presentation and content.",
		23: "React is a free and open-source front-end JavaScript library for building user interfaces based on UI components. It is maintained by Meta and a community of individual developers and companies.",
		24: "Angular is a TypeScript-based free and open-source web application framework led by the Angular Team at Google and by a community of individuals and corporations.",
		25: "Vue.js is an open-source model–view–viewmodel front end JavaScript framework for building user interfaces and single-page applications. It was created by Evan You and is maintained by him and the rest of the active core team members.",
		26: "Node.js is an open-source, cross-platform, back-end JavaScript runtime environment that runs on the V8 engine and executes JavaScript code outside a web browser.",

		// Operating Systems
		27: "Linux is a family of open-source Unix-like operating systems based on the Linux kernel, an operating system kernel first released by Linus Torvalds on September 17, 1991.",
		28: "Windows is a group of several proprietary graphical operating system families developed and marketed by Microsoft. Each family caters to a certain sector of the computing industry.",
		29: "macOS is a Unix operating system developed and marketed by Apple Inc. since 2001. It is the primary operating system for Apple's Mac computers.",
		30: "Unix is a family of multitasking, multiuser computer operating systems that derive from the original AT&T Unix, whose development started in the 1960s at the Bell Labs research center.",

		// Databases
		31: "PostgreSQL is a free and open-source relational database management system emphasizing extensibility and SQL compliance. It was originally named POSTGRES, referring to its origins as a successor to the Ingres database.",
		32: "MySQL is an open-source relational database management system. Its name is a combination of My, the name of co-founder Michael Widenius's daughter, and SQL, the abbreviation for Structured Query Language.",
		33: "MongoDB is a source-available cross-platform document-oriented database program. Classified as a NoSQL database program, MongoDB uses JSON-like documents with optional schemas.",
		34: "Redis is an in-memory data structure store, used as a distributed, in-memory key–value database, cache and message broker, with optional durability.",
		35: "SQLite is a relational database management system contained in a C library. In contrast to many other database management systems, SQLite is not a client–server database engine.",

		// Networking
		36: "TCP/IP (Transmission Control Protocol/Internet Protocol) is the suite of communications protocols used to connect hosts on the Internet. TCP/IP uses several protocols, the two main ones being TCP and IP.",
		37: "HTTP (Hypertext Transfer Protocol) is an application-layer protocol for transmitting hypermedia documents, such as HTML. It was designed for communication between web browsers and web servers.",
		38: "HTTPS (Hypertext Transfer Protocol Secure) is an extension of HTTP. It is used for secure communication over a computer network, and is widely used on the Internet.",
		39: "DNS (Domain Name System) is a hierarchical and decentralized naming system for computers, services, or other resources connected to the Internet or a private network.",
		40: "REST (Representational State Transfer) is a software architectural style that defines a set of constraints to be used for creating Web services. Web services that conform to the REST architectural style, called RESTful Web services.",

		// Security
		41: "Cryptography is the practice and study of techniques for secure communication in the presence of adversarial behavior. It is about constructing and analyzing protocols that prevent third parties or the public from reading private messages.",
		42: "Blockchain is a type of distributed ledger technology that consists of growing list of records, called blocks, that are securely linked together using cryptography.",
		43: "Encryption is the process of converting information or data into a code, especially to prevent unauthorized access. Encryption does not itself prevent interference but denies the intelligible content to a would-be interceptor.",
		44: "Authentication is the act of proving an assertion, such as the identity of a computer system user. In contrast with identification, the act of indicating a person or thing's identity, authentication is the process of verifying that identity.",

		// Software Development
		45: "Git is software for tracking changes in any set of files, usually used for coordinating work among programmers collaboratively developing source code during software development.",
		46: "Docker is a set of platform as a service products that use OS-level virtualization to deliver software in packages called containers. Containers are isolated from one another and bundle their own software, libraries and configuration files.",
		47: "Kubernetes is an open-source container-orchestration system for automating computer application deployment, scaling, and management. It was originally designed by Google and is now maintained by the Cloud Native Computing Foundation.",
		48: "Microservices are a software development technique—a variant of the service-oriented architecture architectural style that structures an application as a collection of loosely coupled services.",
		49: "CI/CD (Continuous Integration/Continuous Deployment) is a method to frequently deliver apps to customers by introducing automation into the stages of app development.",
		50: "Agile Software Development is an approach to software development under which requirements and solutions evolve through the collaborative effort of self-organizing and cross-functional teams and their customers.",
	}

	for id, text := range articles {
		idx.Add(id, text)
	}

	// Test case 1: Single term search
	t.Run("SingleTerm_Programming", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("programming").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		if len(results) < 5 {
			t.Errorf("Expected at least 5 results for 'programming', got %d", len(results))
		}
		// Results should be programming-related documents (IDs 1-12 are programming languages)
		for _, r := range results {
			if r.Id < 1 || r.Id > 12 {
				t.Logf("Result ID %d may not be a programming language article", r.Id)
			}
		}
	})

	// Test case 2: Multi-word query
	t.Run("MultiWord_MemorySafety", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("memory safety").WithK(5).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find Go and Rust articles (both mention memory safety)
		if len(results) < 2 {
			t.Errorf("Expected at least 2 results for 'memory safety', got %d", len(results))
		}
		// Top results should be Go (id=1) or Rust (id=4)
		topIDs := make(map[uint32]bool)
		for _, r := range results[:min(2, len(results))] {
			topIDs[r.Id] = true
		}
		if !topIDs[1] && !topIDs[4] {
			t.Error("Expected Go or Rust articles in top 2 results for 'memory safety'")
		}
	})

	// Test case 3: Specific language features
	t.Run("SpecificFeature_GarbageCollection", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("garbage collection").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find Go, Python, Ruby (have garbage collection)
		// Should NOT rank Rust highly (explicitly says "without requiring garbage collection")
		expectedIDs := map[uint32]bool{1: true, 2: true, 8: true}
		foundExpected := 0
		for _, r := range results[:min(3, len(results))] {
			if expectedIDs[r.Id] {
				foundExpected++
			}
		}
		if foundExpected < 2 {
			t.Errorf("Expected to find at least 2 articles with garbage collection in top 3")
		}
	})
}

// TestRealisticEcommerceSearch simulates product search
func TestRealisticEcommerceSearch(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add product descriptions (60+ products across various categories)
	products := map[uint32]string{
		// Smartphones
		1: "Apple iPhone 14 Pro Max - 256GB - Space Black - 5G smartphone with A16 Bionic chip and ProRAW camera",
		2: "Samsung Galaxy S23 Ultra - 512GB - Phantom Black - Android flagship with 200MP camera and S Pen",
		3: "Google Pixel 7 Pro - 128GB - Snow - Pure Android experience with Tensor G2 chip and amazing camera",
		4: "Apple iPhone 13 - 128GB - Midnight - A15 Bionic chip - Dual camera system - 5G capable",
		5: "Samsung Galaxy Z Fold 4 - 256GB - Graygreen - Foldable smartphone with 7.6-inch display",
		6: "OnePlus 11 5G - 256GB - Titan Black - Snapdragon 8 Gen 2 - Hasselblad camera - 100W fast charging",

		// Laptops
		7:  "Apple MacBook Pro 16-inch M2 Max - 32GB RAM - 1TB SSD - Space Gray laptop for professionals",
		8:  "Dell XPS 15 9530 - Intel Core i7 - 16GB RAM - 512GB SSD - Windows 11 laptop",
		9:  "Apple MacBook Air M2 - 13-inch - 8GB RAM - 256GB SSD - Midnight - Ultra portable laptop",
		10: "Lenovo ThinkPad X1 Carbon Gen 11 - Intel i7 - 16GB RAM - 512GB SSD - Business laptop",
		11: "HP Spectre x360 14 - Intel Evo i7 - 16GB RAM - 1TB SSD - 2-in-1 convertible laptop",
		12: "ASUS ROG Zephyrus G14 - AMD Ryzen 9 - RTX 4060 - 16GB RAM - Gaming laptop",
		13: "Microsoft Surface Laptop 5 - Intel i7 - 16GB RAM - 512GB SSD - Platinum - Touch screen",

		// Tablets
		14: "Apple iPad Pro 12.9-inch M2 chip - 256GB - Wi-Fi - Space Gray tablet",
		15: "Samsung Galaxy Tab S8 Ultra - 128GB - Graphite - 14.6-inch Super AMOLED display",
		16: "Apple iPad Air 5th Gen - 64GB - Starlight - M1 chip - 10.9-inch Liquid Retina display",
		17: "Microsoft Surface Pro 9 - Intel i7 - 16GB RAM - 256GB SSD - 2-in-1 tablet with keyboard",

		// Headphones & Earbuds
		18: "Sony WH-1000XM5 Wireless Noise-Canceling Over-Ear Headphones - Black - Premium audio",
		19: "Apple AirPods Pro 2nd Generation - USB-C - Active Noise Cancellation wireless earbuds",
		20: "Bose QuietComfort 45 Wireless Bluetooth Noise-Cancelling Headphones - White",
		21: "Sennheiser Momentum 4 Wireless - Adaptive Noise Cancellation - 60-hour battery life",
		22: "Sony WF-1000XM4 - True Wireless Noise Canceling Earbuds - LDAC support",
		23: "Apple AirPods Max - Over-ear headphones - Spatial Audio - Premium build quality",
		24: "Beats Studio Pro - Wireless Noise Cancelling Headphones - Lossless audio via USB-C",

		// TVs
		25: "Samsung 65-inch OLED 4K Smart TV - HDR10+ - Gaming Hub - Voice Control",
		26: "LG C3 55-inch OLED evo 4K Smart TV - NVIDIA G-SYNC - Dolby Vision",
		27: "Sony Bravia XR A95K 65-inch QD-OLED 4K TV - Cognitive Processor XR - Google TV",
		28: "Samsung 55-inch QLED 4K Smart TV - Quantum Dot technology - 120Hz refresh rate",
		29: "TCL 65-inch 6-Series 4K QLED Roku TV - Mini-LED - Dolby Vision - Budget flagship",
		30: "LG 77-inch G3 OLED evo Gallery Edition - Ultra bright - Wall mount design",

		// Smart Home
		31: "Amazon Echo Dot 5th Gen - Smart speaker with Alexa - Improved audio - Charcoal",
		32: "Google Nest Hub Max - 10-inch Smart Display - Built-in Nest Cam - Voice control",
		33: "Apple HomePod mini - Space Gray - Siri integration - Multi-room audio",
		34: "Ring Video Doorbell Pro 2 - 1536p HD video - 3D Motion Detection - Alexa compatible",
		35: "Philips Hue White and Color Ambiance Starter Kit - Smart LED bulbs - 16 million colors",
		36: "Nest Learning Thermostat - 3rd Gen - Smart temperature control - Energy saving",

		// Cameras
		37: "Sony Alpha a7 IV - Full-frame mirrorless camera - 33MP - 4K60p video - Hybrid AF",
		38: "Canon EOS R6 Mark II - Full-frame mirrorless - 24.2MP - Dual Pixel AF II - 4K60p",
		39: "Nikon Z9 - Professional mirrorless camera - 45.7MP - 8K video - Flagship performance",
		40: "Fujifilm X-T5 - APS-C mirrorless - 40MP - Film simulations - Retro design",
		41: "GoPro HERO 11 Black - Action camera - 5.3K60 video - HyperSmooth stabilization",

		// Smartwatches & Fitness
		42: "Apple Watch Series 9 - GPS + Cellular - 45mm - Midnight Aluminum - Always-On display",
		43: "Samsung Galaxy Watch 6 - 44mm - Graphite - Wear OS - Advanced health tracking",
		44: "Garmin Fenix 7X - Solar - Multisport GPS watch - Ultra-long battery life",
		45: "Fitbit Sense 2 - Advanced health smartwatch - Stress management - Sleep tracking",
		46: "Apple Watch Ultra - 49mm Titanium - Action button - Diving computer - Rugged design",

		// Gaming
		47: "Sony PlayStation 5 - Digital Edition - 825GB SSD - 4K gaming - Ray tracing",
		48: "Xbox Series X - 1TB SSD - 4K gaming at 120fps - Quick Resume",
		49: "Nintendo Switch OLED - 7-inch OLED screen - Enhanced audio - Handheld console",
		50: "Steam Deck - 512GB - Handheld gaming PC - Runs PC games natively",
		51: "Sony DualSense Edge - Wireless controller - Customizable - Replaceable stick modules",
		52: "Xbox Elite Wireless Controller Series 2 - Adjustable components - 40-hour battery",

		// Computer Peripherals
		53: "Logitech MX Master 3S - Wireless mouse - 8K DPI - Quiet clicks - Multi-device",
		54: "Apple Magic Keyboard with Touch ID - Wireless - USB-C - Scissor mechanism",
		55: "Keychron Q6 - Full-size mechanical keyboard - QMK/VIA support - Hot-swappable",
		56: "Dell UltraSharp U2723DE - 27-inch QHD monitor - IPS Black - USB-C hub",
		57: "LG 27GN950-B - 27-inch 4K 144Hz gaming monitor - Nano IPS - G-SYNC Compatible",

		// Storage
		58: "Samsung 980 PRO - 2TB NVMe SSD - PCIe 4.0 - 7000MB/s read - For PS5 and PC",
		59: "WD Black SN850X - 1TB NVMe SSD - PCIe 4.0 - Gaming optimized - Heatsink included",
		60: "Seagate IronWolf Pro - 18TB NAS HDD - 7200 RPM - For RAID systems",
		61: "SanDisk Extreme Portable SSD - 2TB - USB 3.2 - IP55 water resistant - 1050MB/s",
	}

	for id, text := range products {
		idx.Add(id, text)
	}

	// Test case 1: Brand search
	t.Run("BrandSearch_Apple", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("apple").WithK(20).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find Apple products (IDs 1, 4, 7, 9, 14, 16, 19, 23, 33, 42, 46, 54)
		if len(results) < 8 {
			t.Errorf("Expected at least 8 Apple product results, got %d", len(results))
		}
	})

	// Test case 2: Product category search
	t.Run("CategorySearch_Headphones", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("wireless headphones noise cancelling").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find multiple headphone products
		if len(results) < 5 {
			t.Errorf("Expected at least 5 headphone/earbud results, got %d", len(results))
		}
		// Check that we got actual headphone products (IDs 18-24 are headphones/earbuds)
		headphoneCount := 0
		for _, r := range results {
			if r.Id >= 18 && r.Id <= 24 {
				headphoneCount++
			}
		}
		if headphoneCount < 3 {
			t.Errorf("Expected at least 3 headphone products in top results, got %d", headphoneCount)
		}
	})

	// Test case 3: Specific feature search
	t.Run("FeatureSearch_OLED", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("OLED 4K TV").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should prioritize TVs with OLED (IDs 25-30 are TVs)
		if len(results) < 3 {
			t.Errorf("Expected at least 3 OLED TV results, got %d", len(results))
		}
		// Top results should include OLED TVs
		tvCount := 0
		for _, r := range results[:min(5, len(results))] {
			if r.Id >= 25 && r.Id <= 30 {
				tvCount++
			}
		}
		if tvCount < 2 {
			t.Error("Expected at least 2 TVs in top 5 results")
		}
	})

	// Test case 4: Specific model search
	t.Run("ModelSearch_MacBookPro", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("MacBook Pro").WithK(5).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		if len(results) == 0 {
			t.Fatal("Expected at least 1 result for 'MacBook Pro'")
		}
		// Top result should be the MacBook Pro (id=7)
		if results[0].Id != 7 {
			t.Errorf("Expected MacBook Pro (id=7) as top result, got id=%d", results[0].Id)
		}
	})

	// Test case 5: Gaming category search
	t.Run("CategorySearch_Gaming", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("gaming 4K").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find gaming consoles, monitors, and related products
		if len(results) < 3 {
			t.Errorf("Expected at least 3 gaming-related results, got %d", len(results))
		}
	})

	// Test case 6: Storage search
	t.Run("CategorySearch_SSD", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("NVMe SSD PCIe").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find SSDs (IDs 58-61 are storage)
		ssdCount := 0
		for _, r := range results {
			if r.Id >= 58 && r.Id <= 61 {
				ssdCount++
			}
		}
		if ssdCount < 2 {
			t.Errorf("Expected at least 2 SSD products, got %d", ssdCount)
		}
	})
}

// TestRealisticCodeSearch simulates searching code documentation
func TestRealisticCodeSearch(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add code documentation entries (30+ functions from various Go packages)
	docs := map[uint32]string{
		// File I/O
		1: "func ReadFile(filename string) ([]byte, error) - Reads the entire file and returns its contents as a byte slice. Returns an error if the file cannot be read.",
		2: "func WriteFile(filename string, data []byte, perm os.FileMode) error - Writes data to a file. Creates the file if it doesn't exist, truncates it if it does.",
		3: "func Open(name string) (*File, error) - Opens a file for reading. Returns a file descriptor and an error if the operation fails.",
		4: "func Create(name string) (*File, error) - Creates or truncates a file. Returns a file descriptor and an error if the operation fails.",
		5: "func OpenFile(name string, flag int, perm FileMode) (*File, error) - Opens a file with specified flags and permissions. More flexible than Open.",
		6: "func Remove(name string) error - Removes the named file or empty directory. Returns an error if the operation fails.",
		7: "func RemoveAll(path string) error - Removes path and any children it contains. It removes everything it can but returns the first error it encounters.",

		// JSON Encoding
		8:  "func Marshal(v interface{}) ([]byte, error) - Converts Go data structure to JSON format. Returns serialized bytes or an error.",
		9:  "func Unmarshal(data []byte, v interface{}) error - Parses JSON data and stores the result in the value pointed to by v.",
		10: "func MarshalIndent(v interface{}, prefix, indent string) ([]byte, error) - Marshals Go value to JSON with indentation for readability.",
		11: "func NewEncoder(w io.Writer) *Encoder - Returns a new JSON encoder that writes to w. Useful for streaming encoding.",
		12: "func NewDecoder(r io.Reader) *Decoder - Returns a new JSON decoder that reads from r. Useful for streaming decoding.",

		// I/O Operations
		13: "func NewReader(r io.Reader) *Reader - Returns a new Reader that reads from r. The Reader buffers input from the underlying reader.",
		14: "func NewWriter(w io.Writer) *Writer - Returns a new Writer that writes to w. The Writer buffers output to the underlying writer.",
		15: "func Copy(dst Writer, src Reader) (written int64, err error) - Copies from src to dst until EOF or error. Returns bytes written and error.",
		16: "func ReadAll(r Reader) ([]byte, error) - Reads from r until an error or EOF and returns the data it read. Useful for reading entire streams.",
		17: "func WriteString(w Writer, s string) (n int, err error) - Writes the contents of the string s to w, which accepts a slice of bytes.",

		// String Operations
		18: "func Contains(s, substr string) bool - Reports whether substr is within s. Returns true if substr is present.",
		19: "func Split(s, sep string) []string - Slices s into all substrings separated by sep and returns a slice of the substrings between those separators.",
		20: "func Join(elems []string, sep string) string - Concatenates the elements of slice to create a single string. The separator string sep is placed between elements.",
		21: "func Replace(s, old, new string, n int) string - Returns a copy of string s with the first n non-overlapping instances of old replaced by new.",
		22: "func ToLower(s string) string - Returns s with all Unicode letters mapped to their lower case.",
		23: "func ToUpper(s string) string - Returns s with all Unicode letters mapped to their upper case.",
		24: "func TrimSpace(s string) string - Returns a slice of string s with all leading and trailing white space removed as defined by Unicode.",

		// HTTP Operations
		25: "func Get(url string) (resp *Response, err error) - Issues a GET request to the specified URL. Returns response or error.",
		26: "func Post(url, contentType string, body io.Reader) (resp *Response, err error) - Issues a POST request to the specified URL with given body.",
		27: "func NewRequest(method, url string, body io.Reader) (*Request, error) - Returns a new HTTP Request given a method, URL, and optional body.",
		28: "func ListenAndServe(addr string, handler Handler) error - Listens on the TCP network address addr and then serves requests using handler.",
		29: "func HandleFunc(pattern string, handler func(ResponseWriter, *Request)) - Registers the handler function for the given pattern in the DefaultServeMux.",

		// Context
		30: "func WithCancel(parent Context) (ctx Context, cancel CancelFunc) - Returns a copy of parent with a new Done channel. The returned context's Done channel is closed when cancel is called.",
		31: "func WithTimeout(parent Context, timeout time.Duration) (Context, CancelFunc) - Returns WithDeadline(parent, time.Now().Add(timeout)).",
		32: "func WithValue(parent Context, key, val interface{}) Context - Returns a copy of parent in which the value associated with key is val.",

		// Time Operations
		33: "func Now() Time - Returns the current local time.",
		34: "func Sleep(d Duration) - Pauses the current goroutine for at least the duration d. A negative or zero duration causes Sleep to return immediately.",
		35: "func Since(t Time) Duration - Returns the time elapsed since t. It is shorthand for time.Now().Sub(t).",
		36: "func After(d Duration) <-chan Time - Waits for the duration to elapse and then sends the current time on the returned channel.",
	}

	for id, text := range docs {
		idx.Add(id, text)
	}

	// Test case 1: Function name search
	t.Run("FunctionName_ReadFile", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("ReadFile").WithK(5).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		if len(results) == 0 {
			t.Fatal("Expected results for 'ReadFile'")
		}
		if results[0].Id != 1 {
			t.Errorf("Expected ReadFile (id=1) as top result, got id=%d", results[0].Id)
		}
	})

	// Test case 2: Operation search
	t.Run("Operation_WriteData", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("write data file").WithK(5).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find WriteFile and NewWriter
		if len(results) < 2 {
			t.Errorf("Expected at least 2 results for write operations, got %d", len(results))
		}
		// WriteFile should rank highly
		foundWriteFile := false
		for _, r := range results[:min(2, len(results))] {
			if r.Id == 2 {
				foundWriteFile = true
				break
			}
		}
		if !foundWriteFile {
			t.Error("Expected WriteFile in top results for 'write data file'")
		}
	})

	// Test case 3: Data format search
	t.Run("DataFormat_JSON", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("JSON").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find Marshal and Unmarshal (IDs 8-12 are JSON encoding functions)
		if len(results) < 3 {
			t.Errorf("Expected at least 3 JSON-related results, got %d", len(results))
		}
		// Verify we found JSON-related functions
		jsonCount := 0
		for _, r := range results[:min(5, len(results))] {
			if r.Id >= 8 && r.Id <= 12 {
				jsonCount++
			}
		}
		if jsonCount < 2 {
			t.Errorf("Expected to find at least 2 JSON encoding functions, got %d", jsonCount)
		}
	})
}

// TestRealisticEmailSearch simulates email/document search
func TestRealisticEmailSearch(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Add email-like documents (40+ emails simulating an inbox)
	emails := map[uint32]string{
		// Q4 Planning
		1: "Subject: Quarterly Meeting Schedule - From: john@company.com - The Q4 planning meeting is scheduled for next Tuesday at 2 PM in Conference Room A. Please review the attached agenda.",
		2: "Subject: Q4 Budget Review - From: finance@company.com - Please submit your Q4 budget proposals by end of week. Include detailed breakdowns for personnel, equipment, and operational costs.",
		3: "Subject: Q4 Sales Targets - From: sales@company.com - Q4 targets have been updated based on market conditions. Please review the revised quotas and let me know if you have concerns.",

		// Project Alpha
		4: "Subject: Project Alpha Update - From: sarah@company.com - Great progress on Project Alpha! The frontend team completed the dashboard implementation. Backend API integration is 80% complete.",
		5: "Subject: Project Alpha - Sprint Review - From: sarah@company.com - Sprint 5 review for Project Alpha is scheduled for Friday. We'll demo the new features and discuss the next sprint planning.",
		6: "Subject: Project Alpha - Launch Date - From: sarah@company.com - After review with stakeholders, Project Alpha launch is confirmed for November 15th. All teams need to be ready by Nov 10th.",
		7: "Subject: Project Alpha - Testing Results - From: qa@company.com - Completed regression testing for Project Alpha. Found 3 critical bugs that need to be fixed before launch. Details in JIRA.",

		// Project Beta
		8:  "Subject: Project Beta Kickoff - From: david@company.com - Project Beta kickoff meeting scheduled for Monday 10 AM. We'll discuss requirements, timeline, and team assignments.",
		9:  "Subject: Project Beta - Resource Request - From: david@company.com - Project Beta needs 2 additional frontend developers for Q1. Please review the job descriptions I've drafted.",
		10: "Subject: Project Beta - Architecture Review - From: tech@company.com - Scheduled architecture review for Project Beta on Wednesday. Please come prepared with your design proposals.",

		// Invoices and Finance
		11: "Subject: Invoice #12345 - Payment Due - From: billing@vendor.com - This is a reminder that invoice #12345 for $5,000 is due on March 15th. Please process payment at your earliest convenience.",
		12: "Subject: Invoice #12346 - Office Supplies - From: supplies@vendor.com - Invoice for October office supplies: $2,350. Payment terms net 30. Thank you for your business.",
		13: "Subject: Invoice #12347 - Cloud Services - From: aws@amazon.com - Your AWS bill for October is $8,450. Includes EC2, S3, and RDS services. Auto-pay scheduled for Nov 1st.",
		14: "Subject: Expense Report Approved - From: finance@company.com - Your expense report #ER-1045 for $1,234.56 has been approved. Reimbursement will be in your next paycheck.",
		15: "Subject: Annual Budget Planning - From: cfo@company.com - It's time for annual budget planning. Please submit your 2024 budget requests by December 1st with justifications.",

		// Team Events
		16: "Subject: Team Lunch Tomorrow - From: mike@company.com - Hey team! Let's grab lunch tomorrow at the new Italian restaurant downtown. Meet in the lobby at 12:30 PM.",
		17: "Subject: Team Building Event - From: hr@company.com - Save the date! Company-wide team building event on December 8th. We're going bowling and having dinner. RSVP by Nov 30th.",
		18: "Subject: Birthday Celebration - From: admin@company.com - Join us in celebrating Lisa's birthday tomorrow at 3 PM in the break room. Cake and refreshments will be provided!",
		19: "Subject: Holiday Party - From: events@company.com - Annual holiday party is scheduled for December 15th at the Grand Hotel. Dinner, dancing, and prizes. Bring your spouse or guest!",

		// Security
		20: "Subject: Security Alert: Password Reset Required - From: security@company.com - We detected suspicious activity on your account. Please reset your password immediately using the link below.",
		21: "Subject: Security Training Mandatory - From: security@company.com - All employees must complete security awareness training by Nov 30th. Link to training portal in email. Takes 1 hour.",
		22: "Subject: Phishing Test Results - From: security@company.com - Thanks to all who identified last week's phishing test. 85% pass rate! Those who clicked will receive additional training.",
		23: "Subject: VPN Access Update - From: it@company.com - Updated VPN client available. Please install version 3.2 for improved security and performance. Instructions attached.",

		// Conferences and Events
		24: "Subject: Conference Registration Confirmation - From: events@conference.com - Your registration for TechConf 2024 is confirmed. The conference runs March 20-22 at the Convention Center.",
		25: "Subject: Webinar Invitation: AI in Practice - From: marketing@techcompany.com - Join our free webinar on November 18th at 2 PM. Learn how AI is transforming software development.",
		26: "Subject: Speaking Opportunity - From: conference@devcon.com - We'd love to have you speak at DevCon 2024. Topic: Microservices at Scale. Conference is April 5-7 in San Francisco.",

		// Reports and Analytics
		27: "Subject: Monthly Report - February 2024 - From: analytics@company.com - Attached is the monthly performance report for February. Revenue increased 15% compared to last month.",
		28: "Subject: Weekly Metrics - From: analytics@company.com - Weekly metrics show user engagement up 8%, conversion rate at 3.2%, and customer satisfaction at 4.5/5 stars.",
		29: "Subject: Dashboard Access - From: bi@company.com - You now have access to the new analytics dashboard. Login credentials in separate email. Training session tomorrow at 11 AM.",
		30: "Subject: Traffic Spike Analysis - From: analytics@company.com - Investigating yesterday's traffic spike. Appears to be organic from viral social media post. No infrastructure issues.",

		// HR and Admin
		31: "Subject: Open Enrollment Reminder - From: hr@company.com - Health insurance open enrollment ends Nov 30th. Review your options and make selections in the HR portal.",
		32: "Subject: Performance Review Cycle - From: hr@company.com - Annual performance reviews start next week. Please complete self-assessments by Nov 20th. Manager meetings follow.",
		33: "Subject: New Hire Announcement - From: hr@company.com - Please welcome Jennifer Chen, our new Senior DevOps Engineer. She starts Monday and will sit in the engineering pod.",
		34: "Subject: Office Closure Notice - From: facilities@company.com - Building will be closed Dec 24-26 for holidays. Badge access disabled. Emergency contact info attached.",
		35: "Subject: Parking Permit Renewal - From: facilities@company.com - Parking permits expire Dec 31st. Renew online by Dec 15th to avoid service interruption. Cost: $50/month.",

		// Product and Customer Updates
		36: "Subject: Customer Feedback Summary - From: support@company.com - October customer feedback compiled. Overall NPS score: 42. Top request: dark mode. Top complaint: slow loading.",
		37: "Subject: Product Roadmap Update - From: product@company.com - Updated product roadmap for Q1 2024. Focus areas: mobile app, API improvements, and integration with popular tools.",
		38: "Subject: Feature Request Priority - From: product@company.com - After analyzing feedback, top 3 feature requests: SSO integration, bulk operations, and advanced filtering.",
		39: "Subject: Beta Program Invitation - From: product@company.com - You're invited to join our beta program for the new mobile app. Early access starts Nov 1st. Sign up link below.",
		40: "Subject: Service Outage Post-Mortem - From: sre@company.com - Post-mortem for Oct 25th outage. Root cause: database connection pool exhaustion. Mitigations implemented.",
	}

	for id, text := range emails {
		idx.Add(id, text)
	}

	// Test case 1: Find emails from specific person
	t.Run("SenderSearch_Sarah", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("sarah@company.com").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find emails from Sarah (IDs 4, 5, 6 are from sarah@company.com)
		if len(results) < 3 {
			t.Errorf("Expected at least 3 emails from Sarah, got %d", len(results))
		}
		// Sarah's emails are about Project Alpha (IDs 4, 5, 6)
		alphaIDs := map[uint32]bool{4: true, 5: true, 6: true}
		alphaCount := 0
		for _, r := range results[:min(3, len(results))] {
			if alphaIDs[r.Id] {
				alphaCount++
			}
		}
		if alphaCount < 2 {
			t.Error("Expected at least 2 Project Alpha emails (IDs 4,5,6) from Sarah")
		}
	})

	// Test case 2: Project-specific search
	t.Run("ProjectSearch_ProjectAlpha", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("Project Alpha").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find emails about Project Alpha (IDs 4, 5, 6, 7)
		if len(results) < 4 {
			t.Errorf("Expected at least 4 emails about Project Alpha, got %d", len(results))
		}
		// Verify top results are Project Alpha emails (IDs 4, 5, 6, 7)
		alphaIDs := map[uint32]bool{4: true, 5: true, 6: true, 7: true}
		alphaCount := 0
		for _, r := range results[:min(4, len(results))] {
			if alphaIDs[r.Id] {
				alphaCount++
			}
		}
		if alphaCount < 3 {
			t.Error("Expected at least 3 Project Alpha emails (IDs 4-7) in top 4 results")
		}
	})

	// Test case 3: Urgent/security search
	t.Run("UrgentSearch_Security", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("security password reset").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		if len(results) < 3 {
			t.Fatalf("Expected at least 3 security-related results, got %d", len(results))
		}
		// Should find multiple security emails (IDs 20-23 are security-related)
		securityCount := 0
		for _, r := range results {
			if r.Id >= 20 && r.Id <= 23 {
				securityCount++
			}
		}
		if securityCount < 2 {
			t.Errorf("Expected at least 2 security emails, got %d", securityCount)
		}
	})

	// Test case 4: Date/time-specific search
	t.Run("TimeSearch_Meeting", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("meeting schedule").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find meeting-related emails
		if len(results) < 3 {
			t.Errorf("Expected at least 3 meeting-related results, got %d", len(results))
		}
	})

	// Test case 5: Invoice search
	t.Run("FinanceSearch_Invoice", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("invoice payment").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find invoice emails (IDs 11-15 are finance/invoice related)
		if len(results) < 3 {
			t.Errorf("Expected at least 3 invoice results, got %d", len(results))
		}
		// Check that we got invoice-related results (IDs 11-15 are invoices/finance)
		invoiceIDs := map[uint32]bool{11: true, 12: true, 13: true, 14: true, 15: true}
		invoiceCount := 0
		for _, r := range results[:min(5, len(results))] {
			if invoiceIDs[r.Id] {
				invoiceCount++
			}
		}
		if invoiceCount < 2 {
			t.Error("Expected at least 2 invoice emails (IDs 11-15) in top 5 results")
		}
	})

	// Test case 6: Q4 planning search
	t.Run("TopicSearch_Q4", func(t *testing.T) {
		results, err := idx.NewSearch().WithQuery("Q4 budget planning").WithK(10).Execute()
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		// Should find Q4-related emails (IDs 1, 2, 3)
		if len(results) < 2 {
			t.Errorf("Expected at least 2 Q4-related results, got %d", len(results))
		}
	})
}

// TestRealisticRankingCorrectness verifies BM25 ranking makes sense
func TestRealisticRankingCorrectness(t *testing.T) {
	idx := NewBM25SearchIndex()

	// Documents with varying term frequencies and lengths
	docs := map[uint32]string{
		1: "cat",                                                                                                                                                                                                                      // Very short, one occurrence
		2: "cat cat cat",                                                                                                                                                                                                              // Short, multiple occurrences
		3: "cat dog bird fish turtle rabbit hamster",                                                                                                                                                                                  // Medium, one occurrence among many words
		4: "The cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae and is often referred to as the domestic cat to distinguish it from the wild members of the family.", // Long document, multiple occurrences
		5: "A dog is a domestic animal. Dogs are great pets.",                                                                                                                                                                         // No "cat", should not match
	}

	for id, text := range docs {
		idx.Add(id, text)
	}

	results, err := idx.NewSearch().WithQuery("cat").WithK(10).Execute()
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}

	// Test: Document 5 should not be in results (doesn't contain "cat")
	for _, r := range results {
		if r.Id == 5 {
			t.Error("Document 5 (dog) should not appear in results for 'cat'")
		}
	}

	// Test: Document with multiple occurrences should rank well
	// Doc 2 (cat cat cat) should be in top 2 results
	foundDoc2InTop := false
	for i := 0; i < min(2, len(results)); i++ {
		if results[i].Id == 2 {
			foundDoc2InTop = true
			break
		}
	}
	if !foundDoc2InTop {
		t.Error("Document 2 with high term frequency should be in top 2 results")
	}

	// Test: Verify scores are in descending order
	for i := 0; i < len(results)-1; i++ {
		if results[i].Score < results[i+1].Score {
			t.Errorf("Results not in descending score order: result[%d].Score=%.4f < result[%d].Score=%.4f",
				i, results[i].Score, i+1, results[i+1].Score)
		}
	}

	// Test: All scores should be positive
	for _, r := range results {
		if r.Score <= 0 {
			t.Errorf("Document %d has non-positive score: %.4f", r.Id, r.Score)
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
