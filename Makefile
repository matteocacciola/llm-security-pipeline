UID := $(shell id -u)
GID := $(shell id -g)
PWD = $(shell pwd)

args=

help:  ## Show help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

install: ## Update the local virtual environment with the latest requirements.
	@uv sync --link-mode=copy --frozen --no-install-project --no-upgrade --no-cache --all-extras
	@uv cache clean
	@pip cache purge

update: ## Update and compile requirements for the local virtual environment.
	@uv sync --upgrade --link-mode=copy --no-install-project --no-cache --all-extras
	@uv cache clean
	@pip cache purge

check: ## Check requirements for the local virtual environment.
	@uv sync --check --no-install-project --all-extras

lint: ## Run the lint and type-check gates (same as CI).
	@uv sync --group dev --all-extras; \
	uv run ruff check .; \
	uv run mypy

format: ## Apply ruff's automatic fixes.
	@uv sync --group dev --all-extras; \
	uv run ruff check . --fix

test:  ## Run tests.
	@uv sync --group dev --all-extras; \
	rm -f .coverage .coverage.*; \
	COVERAGE_PROCESS_START=pyproject.toml uv run python -m pytest --color=yes -W ignore --disable-warnings ${args}

test-backends: ## Run the full test suite, including the postgres/mysql/redis integration tests, against real throwaway Docker containers (torn down automatically on exit).
	@trap 'docker stop llm-sec-test-redis llm-sec-test-redis-cluster llm-sec-test-pg llm-sec-test-mysql >/dev/null 2>&1 || true' EXIT; \
	set -e; \
	docker rm -f llm-sec-test-redis llm-sec-test-redis-cluster llm-sec-test-pg llm-sec-test-mysql >/dev/null 2>&1 || true; \
	docker run -d --rm --name llm-sec-test-redis -p 6379:6379 redis:7-alpine >/dev/null; \
	docker run -d --rm --name llm-sec-test-redis-cluster -p 7000:7000 redis:7-alpine \
		redis-server --port 7000 --cluster-enabled yes --cluster-config-file nodes.conf \
		--cluster-announce-ip 127.0.0.1 >/dev/null; \
	docker run -d --rm --name llm-sec-test-pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16-alpine >/dev/null; \
	docker run -d --rm --name llm-sec-test-mysql -e MYSQL_ALLOW_EMPTY_PASSWORD=yes -e MYSQL_DATABASE=test -p 3306:3306 mysql:8 >/dev/null; \
	echo "Assigning all hash slots to the single-node test cluster..."; \
	timeout 30 sh -c 'until docker exec llm-sec-test-redis-cluster redis-cli -p 7000 ping >/dev/null 2>&1; do sleep 1; done'; \
	docker exec llm-sec-test-redis-cluster redis-cli -p 7000 cluster addslotsrange 0 16383 >/dev/null 2>&1 || true; \
	timeout 30 sh -c 'until docker exec llm-sec-test-redis-cluster redis-cli -p 7000 cluster info 2>/dev/null | grep -q "cluster_state:ok"; do sleep 1; done'; \
	echo "Waiting for Postgres to accept connections..."; \
	timeout 60 sh -c 'until docker exec llm-sec-test-pg pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done'; \
	echo "Waiting for MySQL to accept connections..."; \
	timeout 120 sh -c 'until docker exec llm-sec-test-mysql mysqladmin ping -h 127.0.0.1 --silent >/dev/null 2>&1; do sleep 2; done'; \
	uv sync --group dev --all-extras; \
	rm -f .coverage .coverage.*; \
	REDIS_URL=redis://localhost:6379/0 \
	REDIS_CLUSTER_URL=redis://127.0.0.1:7000 \
	POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/postgres \
	MYSQL_HOST=127.0.0.1 MYSQL_PORT=3306 MYSQL_USER=root MYSQL_PASSWORD= MYSQL_DB=test \
	COVERAGE_PROCESS_START=pyproject.toml \
	uv run python -m pytest --color=yes -W ignore --disable-warnings ${args}
