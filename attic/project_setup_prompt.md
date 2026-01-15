# Project Initialization Prompt

## Request
Please initialize a new Python project with the following specifications:

1.  **Project Name**: `[INSERT PROJECT NAME]`
2.  **Package Manager**: Use `poetry` to initialize the project.
3.  **Directory Structure**: Use the standard `src` layout:
    *   `src/[package_name]/__init__.py` (package marker)
    *   `tests/` (test directory)
    *   `README.md`
    *   Standard Python `.gitignore`
4.  **Dependencies**:
    *   Add `torch`.
        *   *Note*: Ensure `pyproject.toml` has `requires-python` set to a range compatible with `torch` (e.g., `">=3.10,<3.15"`) to avoid version solving errors.
    *   Add any other specified dependencies: `[INSERT OTHER DEPENDENCIES]`
5.  **Version Control (GitHub)**:
    *   Initialize a git repository (`git init`).
    *   Commit the initial structure (`git add . && git commit -m "Initial commit"`).
    *   Use `gh` to create a public repository under the user `murbard` (or the current user):
        *   `gh repo create murbard/[project_name] --public --source=. --remote=origin --push`

## Execution Steps
1.  Run `poetry init -n` with appropriate name and description.
2.  Create the directory structure (`mkdir -p src/[name] tests`).
3.  Create `README.md`, `.gitignore`, and `src/[name]/__init__.py`.
4.  Update `pyproject.toml` Python version constraints.
5.  Run `poetry add torch` (and other dependencies).
6.  Initialize Git and push to GitHub.
