UID := $(shell id -u)
GID := $(shell id -g)
PWD = $(shell pwd)

args=

help:  ## Show help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

install: ## Update the local virtual environment with the latest requirements.
	@uv sync --link-mode=copy --frozen --no-install-project --no-upgrade --no-cache
	@uv cache clean
	@pip cache purge

update: ## Update and compile requirements for the local virtual environment.
	@uv sync --upgrade --link-mode=copy --no-install-project --no-cache
	@uv cache clean
	@pip cache purge

check: ## Check requirements for the local virtual environment.
	@uv sync --check

test:  ## Run tests.
	@uv sync --group dev && uv run python -m pytest --color=yes -W ignore --disable-warnings ${args}
