# geom_helpers

Basic utility functions for cell code and reg code data and derived constructs

Based on [ChatGPT](https://chatgpt.com/c/69a1dba3-ae90-8325-a164-37453193599c)

## Requirements

- Pandas>=2.2.2
- Numpy>=1.26.4
- Numba>=0.51.2
- Shapely>=2.0.0
- basic_helpers

## Notes

### Linter
-   Der Linter (ruff) mag keine ungenutzten Importe.

### Type checker
-   Mypy
-   Make sure mypy is installed in the conda environment.
-   Install the new package in the conda environment like so:
    - pip install -e .[dev,typecheck] => installs the package with optional dependencies which includes mypy
	- pip install -e .  => installs only the package with its core dependencies
-   Run mypy locally: **mypy src/**

### Venv

There are still problems with the venv setup.

-   The only Python version available 3.8.x.
-   GitHub Copilot found mainly conda, not venv or uv.

### Pytest

- Run **pytest** from package root

#### Check test coverage: 
- Run pytest --cov=geom_helpers.osm_reader_helper --cov-report=term-missing
- shows % of "statements" checked
- list all statements (by line number) which are missing in any given module

### CI

- Local CI is done when changes are committed locally (**only if pre-commit install was run before**)
- Remote CI (on GitHub) is done when committed changes are pushed upstream thanks to GitHub Actions
  - see settings in .github/workflows/ci.yml
  - see also settings in pyproject.toml

#### Remote: GitHub Actions

-   First make sure that Pytest works correctly locally, otherwise all CI runs on GitHub will fail.

-   File names with tests **must** start with "test_".

-   on every push to GitHub.com

#### Local CI

Setting it up

```bash
pip install pre-commit

Create .pre-commit-config.yaml (see example in this template repo)

pre-commit install
```

**!!! pre-commit install** has to be installed **per repository**, i.e. write the required configuration in the .git files. **!!!**

pre-commit install must be run with bash in the osmox environment, because only there "pre-commit" is currently installed.

### GitHub

-   Create empty repo on github.com

-   Either clone new repo to desired location

-   **or better push an existing repository from local machine**
    ```bash
    git remote add origin https://github.com/hgmuc/sudoku2.git
    git branch -M main
    git push -u origin main
    ```

This links the local repository with the remote repository, creates a branch "main" and sets upstream default to "main".
More branches can be created and pushed to by specifying branch names.

**There were issue with this approach when I tried to push changes from my laptop to the server for the first time. Git complained that there was 1 commit on the server and 3 locally, but they were from different bases. A Forced-Update Push fixed this.**

```bash
git push origin main
git push origin develop
git push origin feature/new-ui
```

#### Branching

Clean Branching Strategy (Recommended)

- main → stable releases
- feature/* → development work

Workflow:

1. Branch from main
2. Do work
3. Commit
4. Push branch
5. Open Pull Request
6. Merge into main
7. Tag release

Create and swich

-   Create: git checkout -b feature/type-checking
-   Switch: git switch -c feature/type-checking
-   Work
-   Commit
-   Push: git push -u origin feature/type-checking

Switching

-   git switch main

#### Semantic versioning

MAJOR.MINOR.PATCH

| Type  | When to increment                |
| ----- | -------------------------------- |
| MAJOR | Breaking change                  |
| MINOR | New feature, backward compatible |
| PATCH | Bugfix                           |

When you're ready to release:
Tagging

```git
git tag v0.2.0
git push origin v0.2.0
```

Releasing
Bump version in pyproject.toml:
[project]
version = "0.2.0"

Commit

```git
git commit -am "Release 0.2.0"
```

Tagging

```git
git tag v0.2.0
```

Push

```git
git push
git push --tags
```
