# Color definitions
RED     := \033[0;31m
GREEN   := \033[0;32m
YELLOW  := \033[0;33m
BLUE    := \033[0;34m
MAGENTA := \033[0;35m
CYAN    := \033[0;36m
RESET   := \033[0m

# Project configuration
PKG_NAME := comet
GO_FILES := $(shell find . -type f -name '*.go' -not -path "./vendor/*")

.PHONY: all test clean fmt lint vet help deps tidy bench test-coverage check

# Default target
all: help

# Run tests
test:
	@echo "$(CYAN)Running tests...$(RESET)"
	@go test -v -race -coverprofile=coverage.out ./...
	@echo "$(GREEN)✓ Tests complete$(RESET)"

# Run tests with coverage report
test-coverage: test
	@echo "$(CYAN)Generating coverage report...$(RESET)"
	@go tool cover -html=coverage.out -o coverage.html
	@echo "$(GREEN)✓ Coverage report: coverage.html$(RESET)"

# Run benchmarks
bench:
	@echo "$(CYAN)Running benchmarks...$(RESET)"
	@go test -bench=. -benchmem ./...
	@echo "$(GREEN)✓ Benchmarks complete$(RESET)"

# Format code
fmt:
	@echo "$(CYAN)Formatting code...$(RESET)"
	@gofmt -s -w $(GO_FILES)
	@echo "$(GREEN)✓ Code formatted$(RESET)"

# Lint code
lint:
	@echo "$(CYAN)Running linter...$(RESET)"
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
		echo "$(GREEN)✓ Linting complete$(RESET)"; \
	else \
		echo "$(YELLOW)⚠ golangci-lint not installed. Run: brew install golangci-lint$(RESET)"; \
	fi

# Run go vet
vet:
	@echo "$(CYAN)Running go vet...$(RESET)"
	@go vet ./...
	@echo "$(GREEN)✓ Vet complete$(RESET)"

# Check for common issues
check: fmt vet lint test
	@echo "$(GREEN)✓ All checks passed$(RESET)"

# Download dependencies
deps:
	@echo "$(CYAN)Downloading dependencies...$(RESET)"
	@go mod download
	@go mod verify
	@echo "$(GREEN)✓ Dependencies downloaded$(RESET)"

# Tidy dependencies
tidy:
	@echo "$(CYAN)Tidying dependencies...$(RESET)"
	@go mod tidy
	@echo "$(GREEN)✓ Dependencies tidied$(RESET)"

# Clean generated files and build artifacts
clean:
	@echo "$(CYAN)Cleaning...$(RESET)"
	@rm -f coverage.out coverage.html
	@echo "$(GREEN)✓ Clean complete$(RESET)"

# Display help
help:
	@echo "$(CYAN)"
	@echo "╔═╗╔═╗╔╦╗╔═╗╔╦╗"
	@echo "║  ║ ║║║║║╣  ║ "
	@echo "╚═╝╚═╝╩ ╩╚═╝ ╩ "
	@echo "$(RESET)"
	@echo "$(MAGENTA)A Tiny Vector Database$(RESET)"
	@echo ""
	@echo "$(MAGENTA)═══════════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(YELLOW)Development Commands:$(RESET)"
	@echo "  $(GREEN)make fmt$(RESET)            - Format Go code"
	@echo "  $(GREEN)make vet$(RESET)            - Run go vet"
	@echo "  $(GREEN)make lint$(RESET)           - Run golangci-lint"
	@echo "  $(GREEN)make check$(RESET)          - Run fmt, vet, lint, and test"
	@echo ""
	@echo "$(YELLOW)Testing Commands:$(RESET)"
	@echo "  $(GREEN)make test$(RESET)           - Run tests with race detector"
	@echo "  $(GREEN)make test-coverage$(RESET)  - Run tests and generate coverage report"
	@echo "  $(GREEN)make bench$(RESET)          - Run benchmarks"
	@echo ""
	@echo "$(YELLOW)Dependency Commands:$(RESET)"
	@echo "  $(GREEN)make deps$(RESET)           - Download dependencies"
	@echo "  $(GREEN)make tidy$(RESET)           - Tidy dependencies"
	@echo ""
	@echo "$(YELLOW)Utility Commands:$(RESET)"
	@echo "  $(GREEN)make clean$(RESET)          - Remove generated files and artifacts"
	@echo "  $(GREEN)make help$(RESET)           - Display this help message"
	@echo ""
	@echo "$(MAGENTA)═══════════════════════════════════════════════$(RESET)"

