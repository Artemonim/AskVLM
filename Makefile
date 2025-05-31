# Makefile for Artemonim's Speech Kit
# Provides convenient commands for development workflow

.PHONY: help install install-dev clean test test-quick lint format type-check security check-all fix pre-commit setup-dev

# * Default target
help:
	@echo "Artemonim's Speech Kit - Development Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  setup-dev      Complete development environment setup"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  test           Run all tests with coverage"
	@echo "  test-quick     Run fast tests only"
	@echo "  lint           Run all linters and static analysis"
	@echo "  format         Auto-format code with ruff"
	@echo "  type-check     Run mypy type checking"
	@echo "  security       Run security analysis with bandit"
	@echo "  check-all      Run comprehensive code quality checks"
	@echo "  fix            Auto-fix code issues where possible"
	@echo ""
	@echo "Git Hooks:"
	@echo "  pre-commit     Setup and run pre-commit hooks"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean          Clean up generated files"

# ! Installation commands
install:
	python -m pip install -e .

install-dev:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .

setup-dev: install-dev
	pre-commit install
	@echo "✅ Development environment setup complete!"
	@echo "💡 Run 'make check-all' to verify everything is working"

# ! Code quality commands
test:
	python test.py --tests

test-quick:
	python -m pytest -x --tb=short

lint:
	python -m ruff check .
	python -m pylint core gui editing utils main.py test.py 2>/dev/null || true

format:
	python test.py --format-only

type-check:
	python test.py --type-check

security:
	python test.py --security

check-all:
	python test.py

fix:
	python -m ruff check --fix .
	python -m ruff format .

# ! Git hooks
pre-commit:
	pre-commit install
	pre-commit run --all-files

# ! Utility commands
clean:
	@echo "🧹 Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	@echo "✅ Cleanup complete!"

# ! Advanced commands for CI/CD
ci-test:
	python -m pytest --cov=core --cov=gui --cov=editing --cov=utils --cov-report=xml --cov-report=term-missing --cov-fail-under=80

ci-lint:
	python -m ruff check --output-format=github .
	python -m mypy . --no-error-summary
	python -m bandit -r . -f json -o bandit-report.json || true

# ! Documentation commands (future use)
docs:
	@echo "📚 Documentation generation not yet implemented"
	@echo "💡 Will use Sphinx when documentation is added"

docs-serve:
	@echo "📚 Documentation serving not yet implemented" 