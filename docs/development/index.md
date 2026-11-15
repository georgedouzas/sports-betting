# Development

## Environment setup

This project uses [PDM](https://github.com/pdm-project/pdm) to manage the various dependencies. To setup the development
environment you can follow the next steps:

- Install [PDM](https://github.com/pdm-project/pdm) and [uv](https://github.com/astral-sh/uv). PDM resolves and installs
  the dependencies through uv, as `pdm.toml` sets, and stops if uv is missing.

- Fork and clone the repository.

- `pdm install` from the root of the project to install the main and development dependencies.
