# Makefile for AskVLM
# Provides convenient commands for development workflow

.PHONY: help install install-dev clean clean-all clean-verbose clean-deep-verbose test test-quick lint format type-check security check-all fix setup-dev

# * Default target
help:
	@echo "AskVLM - Development Commands"
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
	@echo "Utility Commands:"
	@echo "  clean          Clean up generated files and pip cache"
	@echo "  clean-all      Deep clean including temp files"
	@echo "  clean-verbose  Clean with detailed output"
	@echo "  clean-deep-verbose  Deep clean with detailed output"

# ! Installation commands
install:
	python -m pip install -e .

install-dev: clean
	python.exe -m pip install --upgrade pip
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .

setup-dev: install-dev
	@echo "✅ Development environment setup complete!"
	@echo "💡 Run 'make check-all' to verify everything is working"
	@echo "⚠️  Note: This project requires Python 3.11 or 3.12 for ML libraries compatibility"

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

# ! Code analysis (removed pre-commit - using test.py instead)
# pre-commit functionality is integrated into test.py

# ! Utility commands
clean:
	python utils/cleanup.py

clean-all:
	python utils/cleanup.py --deep

clean-verbose:
	python utils/cleanup.py --verbose

clean-deep-verbose:
	python utils/cleanup.py --deep --verbose

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
