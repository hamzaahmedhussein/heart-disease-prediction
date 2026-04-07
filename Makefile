.PHONY: help install train serve serve-dev dashboard test lint docker-build docker-up clean

PYTHON := venv/bin/python
PIP := venv/bin/pip

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "%-15s %s\n", $$1, $$2}'

install: ## Install dependencies into venv
	$(PIP) install -r requirements.txt

train: ## Train the model and generate evaluation artifacts
	$(PYTHON) -m src.train

serve: ## Start the FastAPI server (production mode)
	$(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 2

serve-dev: ## Start the FastAPI server with hot-reload (development only)
	$(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Launch the Streamlit dashboard
	$(PYTHON) -m streamlit run src/dashboard.py --server.port 8501

test: ## Run the test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

lint: ## Run basic code quality checks
	$(PYTHON) -m py_compile src/config.py
	$(PYTHON) -m py_compile src/data_loader.py
	$(PYTHON) -m py_compile src/model.py
	$(PYTHON) -m py_compile src/train.py
	$(PYTHON) -m py_compile src/evaluate.py
	$(PYTHON) -m py_compile src/api.py
	@echo "All modules compile successfully."

docker-build: ## Build the Docker image (includes model training)
	docker build -t heart-disease-api .

docker-up: ## Start API + Dashboard via docker-compose
	docker-compose up --build

clean: ## Remove generated artifacts
	rm -rf results/ models/*.keras models/*.pkl logs/ tuner_results/ jupyter_tuner/
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	@echo "Cleaned generated files."
