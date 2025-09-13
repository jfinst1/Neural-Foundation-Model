# Neural Foundation Model - Makefile
# Comprehensive build and deployment automation

.PHONY: help install install-dev test lint format type-check clean build docker-build docker-run train inference benchmark docs deploy-k8s

# Default target
help:
	@echo "Neural Foundation Model - Available Commands:"
	@echo ""
	@echo "🚀 Setup & Installation:"
	@echo "  install           Install production dependencies"
	@echo "  install-dev       Install development dependencies"
	@echo "  install-cuda      Install CUDA-accelerated packages"
	@echo ""
	@echo "🧪 Development & Testing:"
	@echo "  test              Run test suite"
	@echo "  test-gpu          Run GPU-specific tests"
	@echo "  lint              Run linting checks"
	@echo "  format            Format code with black and isort"
	@echo "  type-check        Run type checking with mypy"
	@echo "  clean             Clean build artifacts and cache"
	@echo ""
	@echo "🏗️  Building & Docker:"
	@echo "  build             Build Python package"
	@echo "  docker-build      Build Docker images"
	@echo "  docker-run        Run complete stack with Docker Compose"
	@echo "  docker-dev        Run development environment"
	@echo ""
	@echo "🎯 Training & Inference:"
	@echo "  train             Train foundation model (local)"
	@echo "  train-distributed Train with distributed setup"
	@echo "  inference         Start real-time inference server"
	@echo "  benchmark         Run performance benchmarks"
	@echo ""
	@echo "📚 Documentation & Deployment:"
	@echo "  docs              Build documentation"
	@echo "  docs-serve        Serve documentation locally"
	@echo "  deploy-k8s        Deploy to Kubernetes"
	@echo "  deploy-local      Deploy locally with Docker Compose"

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl

# Project settings
PROJECT_NAME := neural-foundation-model
VERSION := $(shell grep -E "^version" pyproject.toml | cut -d'"' -f2)
IMAGE_NAME := $(PROJECT_NAME)
IMAGE_TAG := $(VERSION)

# Paths
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
CONFIG_DIR := configs
SCRIPTS_DIR := scripts

# Setup & Installation
install:
	@echo "🚀 Installing production dependencies..."
	$(PIP) install -e .
	@echo "✅ Installation complete!"

install-dev:
	@echo "🚀 Installing development dependencies..."
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "✅ Development environment ready!"

install-cuda:
	@echo "🚀 Installing CUDA-accelerated packages..."
	$(PIP) install cupy-cuda11x>=12.2.0
	$(PIP) install cudf-cu11>=23.8.0
	$(PIP) install cugraph-cu11>=23.8.0
	@echo "✅ CUDA packages installed!"

# Development & Testing
test:
	@echo "🧪 Running test suite..."
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "✅ Tests completed!"

test-gpu:
	@echo "🧪 Running GPU-specific tests..."
	pytest $(TEST_DIR) -v -m gpu --tb=short
	@echo "✅ GPU tests completed!"

test-integration:
	@echo "🧪 Running integration tests..."
	pytest $(TEST_DIR) -v -m integration --tb=short
	@echo "✅ Integration tests completed!"

lint:
	@echo "🔍 Running linting checks..."
	flake8 $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	@echo "✅ Linting completed!"

format:
	@echo "🎨 Formatting code..."
	black $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	isort $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	@echo "✅ Code formatted!"

type-check:
	@echo "🔍 Running type checks..."
	mypy $(SRC_DIR) --ignore-missing-imports
	@echo "✅ Type checking completed!"

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup completed!"

# Building & Docker
build:
	@echo "🏗️ Building Python package..."
	$(PYTHON) -m build
	@echo "✅ Package built successfully!"

docker-build:
	@echo "🐳 Building Docker images..."
	$(DOCKER) build -f docker/Dockerfile.training -t $(IMAGE_NAME):$(IMAGE_TAG) .
	$(DOCKER) build -f docker/Dockerfile.training -t $(IMAGE_NAME):latest .
	$(DOCKER) build -f docker/Dockerfile.inference -t $(IMAGE_NAME)-inference:$(IMAGE_TAG) .
	$(DOCKER) build -f docker/Dockerfile.inference -t $(IMAGE_NAME)-inference:latest .
	@echo "✅ Docker images built!"

docker-run:
	@echo "🐳 Starting complete stack with Docker Compose..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml up -d
	@echo "✅ Stack running! Access:"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Inference API: http://localhost:8080"
	@echo "  - Jupyter: http://localhost:8888"
	@echo "  - Grafana: http://localhost:3000"

docker-dev:
	@echo "🐳 Starting development environment..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
	@echo "✅ Development environment ready!"

docker-stop:
	@echo "🛑 Stopping Docker services..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml down
	@echo "✅ Services stopped!"

docker-logs:
	@echo "📋 Showing Docker logs..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml logs -f

# Training & Inference
train:
	@echo "🎯 Starting foundation model training (local)..."
	$(PYTHON) $(SCRIPTS_DIR)/train_foundation_model.py \
		--config $(CONFIG_DIR)/training/foundation_model.yaml \
		--data_path ./data/processed \
		--output_dir ./outputs/$(shell date +%Y%m%d_%H%M%S)
	@echo "✅ Training completed!"

train-distributed:
	@echo "🎯 Starting distributed training..."
	torchrun --nproc_per_node=2 --nnodes=1 \
		$(SCRIPTS_DIR)/train_foundation_model.py \
		--config $(CONFIG_DIR)/training/distributed.yaml \
		--distributed
	@echo "✅ Distributed training completed!"

train-hpo:
	@echo "🎯 Starting hyperparameter optimization..."
	$(PYTHON) $(SCRIPTS_DIR)/train_foundation_model.py \
		--config $(CONFIG_DIR)/training/hpo_config.yaml \
		training.hyperparameter_optimization.enabled=true
	@echo "✅ HPO completed!"

inference:
	@echo "🚀 Starting real-time inference server..."
	$(PYTHON) $(SCRIPTS_DIR)/real_time_inference.py \
		--model-path ./models/foundation_model.pt \
		--task motor_control \
		--port 8080 \
		--log-level INFO
	@echo "Server running at http://localhost:8080"

benchmark:
	@echo "📊 Running performance benchmarks..."
	$(PYTHON) $(SCRIPTS_DIR)/real_time_inference.py \
		--model-path ./models/foundation_model.pt \
		--benchmark \
		--task motor_control
	@echo "✅ Benchmarks completed!"

# Data Processing
prepare-data:
	@echo "📊 Preparing training data..."
	$(PYTHON) $(SCRIPTS_DIR)/data_preprocessing.py \
		--input ./data/raw \
		--output ./data/processed \
		--config $(CONFIG_DIR)/data/preprocessing.yaml
	@echo "✅ Data preparation completed!"

validate-data:
	@echo "🔍 Validating data quality..."
	$(PYTHON) $(SCRIPTS_DIR)/validate_data.py \
		--data_path ./data/processed
	@echo "✅ Data validation completed!"

# Documentation
docs:
	@echo "📚 Building documentation..."
	cd $(DOCS_DIR) && sphinx-build -b html . _build/html
	@echo "✅ Documentation built in $(DOCS_DIR)/_build/html/"

docs-serve:
	@echo "📚 Serving documentation locally..."
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000
	@echo "Documentation available at http://localhost:8000"

docs-clean:
	@echo "🧹 Cleaning documentation..."
	rm -rf $(DOCS_DIR)/_build/
	@echo "✅ Documentation cleaned!"

# Deployment
deploy-k8s:
	@echo "☸️ Deploying to Kubernetes..."
	$(KUBECTL) apply -f kubernetes/namespace.yaml
	$(KUBECTL) apply -f kubernetes/training/
	$(KUBECTL) apply -f kubernetes/inference/
	$(KUBECTL) apply -f kubernetes/monitoring/
	@echo "✅ Kubernetes deployment completed!"

deploy-local:
	@echo "🚀 Deploying locally with Docker Compose..."
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml up -d --build
	@echo "✅ Local deployment completed!"

undeploy-k8s:
	@echo "☸️ Removing Kubernetes deployment..."
	$(KUBECTL) delete -f kubernetes/training/ --ignore-not-found=true
	$(KUBECTL) delete -f kubernetes/inference/ --ignore-not-found=true
	$(KUBECTL) delete -f kubernetes/monitoring/ --ignore-not-found=true
	@echo "✅ Kubernetes resources removed!"

# Monitoring & Debugging
logs-training:
	@echo "📋 Showing training logs..."
	$(KUBECTL) logs -f -l app=neural-foundation-model,component=training

logs-inference:
	@echo "📋 Showing inference logs..."
	$(KUBECTL) logs -f -l app=neural-foundation-model,component=inference

monitor:
	@echo "📊 Opening monitoring dashboard..."
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "MLflow: http://localhost:5000"

# Security & Compliance
security-scan:
	@echo "🔒 Running security scan..."
	safety check
	bandit -r $(SRC_DIR)/
	@echo "✅ Security scan completed!"

# Model Management
export-model:
	@echo "📦 Exporting model for deployment..."
	$(PYTHON) $(SCRIPTS_DIR)/export_model.py \
		--model_path ./models/foundation_model.pt \
		--output_dir ./models/exported/ \
		--formats pytorch onnx torchscript
	@echo "✅ Model exported!"

validate-model:
	@echo "✅ Validating model..."
	$(PYTHON) $(SCRIPTS_DIR)/validate_model.py \
		--model_path ./models/foundation_model.pt \
		--test_data ./data/test/
	@echo "✅ Model validation completed!"

# Performance & Profiling
profile:
	@echo "🔍 Profiling training performance..."
	$(PYTHON) -m cProfile -o profile_output.prof \
		$(SCRIPTS_DIR)/train_foundation_model.py \
		--config $(CONFIG_DIR)/training/profile_config.yaml
	$(PYTHON) -c "import pstats; pstats.Stats('profile_output.prof').sort_stats('cumulative').print_stats(20)"
	@echo "✅ Profiling completed!"

memory-profile:
	@echo "🔍 Memory profiling..."
	mprof run $(PYTHON) $(SCRIPTS_DIR)/train_foundation_model.py \
		--config $(CONFIG_DIR)/training/foundation_model.yaml
	mprof plot
	@echo "✅ Memory profiling completed!"

# Development Helpers
setup-dev:
	@echo "🛠️ Setting up development environment..."
	make install-dev
	make pre-commit-install
	@echo "✅ Development environment ready!"

pre-commit-install:
	@echo "🔧 Installing pre-commit hooks..."
	pre-commit install
	@echo "✅ Pre-commit hooks installed!"

pre-commit-run:
	@echo "🔧 Running pre-commit on all files..."
	pre-commit run --all-files
	@echo "✅ Pre-commit completed!"

jupyter:
	@echo "📓 Starting Jupyter Lab..."
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
	@echo "Jupyter Lab available at http://localhost:8888"

# CI/CD Helpers
ci-test:
	@echo "🤖 Running CI test suite..."
	make lint
	make type-check
	make test
	make test-integration
	@echo "✅ CI tests completed!"

ci-build:
	@echo "🤖 Running CI build..."
	make clean
	make build
	make docker-build
	@echo "✅ CI build completed!"

# Quick Commands
quick-train:
	@echo "⚡ Quick training run (small dataset)..."
	$(PYTHON) $(SCRIPTS_DIR)/train_foundation_model.py \
		--config $(CONFIG_DIR)/training/quick_test.yaml \
		data.train_path=./data/samples/train \
		training.num_epochs=5
	@echo "✅ Quick training completed!"

quick-test:
	@echo "⚡ Quick test run..."
	pytest $(TEST_DIR) -x -v --tb=short
	@echo "✅ Quick tests completed!"

# Environment Info
info:
	@echo "ℹ️ Environment Information:"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "PyTorch: $(shell $(PYTHON) -c 'import torch; print(torch.__version__)')"
	@echo "CUDA Available: $(shell $(PYTHON) -c 'import torch; print(torch.cuda.is_available())')"
	@echo "GPU Count: $(shell $(PYTHON) -c 'import torch; print(torch.cuda.device_count())')"
	@echo "Package Version: $(VERSION)"
	@echo "Docker: $(shell $(DOCKER) --version)"
	@echo "Kubernetes: $(shell $(KUBECTL) version --client --short 2>/dev/null || echo 'Not available')"

# All-in-one commands
setup: install-dev prepare-data
	@echo "✅ Complete setup finished!"

build-all: clean build docker-build docs
	@echo "✅ Complete build finished!"

test-all: lint type-check test test-gpu test-integration
	@echo "✅ Complete testing finished!"

deploy-all: build-all deploy-local
	@echo "✅ Complete deployment finished!"

# Help for specific sections
help-dev:
	@echo "🛠️ Development Commands:"
	@echo "  setup-dev         Complete development setup"
	@echo "  quick-test        Run essential tests quickly"
	@echo "  quick-train       Fast training run for testing"
	@echo "  jupyter           Start Jupyter development environment"
	@echo "  format            Format all code"
	@echo "  profile           Profile training performance"

help-docker:
	@echo "🐳 Docker Commands:"
	@echo "  docker-build      Build all Docker images"
	@echo "  docker-run        Start complete stack"
	@echo "  docker-dev        Start development containers"
	@echo "  docker-stop       Stop all containers"
	@echo "  docker-logs       View container logs"

help-k8s:
	@echo "☸️ Kubernetes Commands:"
	@echo "  deploy-k8s        Deploy to Kubernetes cluster"
	@echo "  undeploy-k8s      Remove Kubernetes deployment"
	@echo "  logs-training     View training pod logs"
	@echo "  logs-inference    View inference pod logs"