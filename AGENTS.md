# AGENTS.md for Python Projects

This AGENTS.md file provides guidance for AI coding agents working on Python projects. It includes detailed instructions on setting up the development environment, running tests, and preparing pull requests (PRs). These are tailored for automation and may include steps not typically emphasized in the human-focused README.md.

## Dev Environment Tips

- To create a new virtual environment for the project, run `python -m venv .venv` from the project root, then activate it with `source .venv/bin/activate` (on Unix) or `.venv\Scripts\activate` (on Windows).
- Install dependencies using `pip install -r requirements.txt` or, if using Poetry, `poetry install`. For development dependencies, add `--dev` if using pip-tools or specify groups in Poetry.
- To add a new package, use `pip install <package_name>` and then freeze the requirements with `pip freeze > requirements.txt`, or `poetry add <package_name>` for Poetry-managed projects.
- Navigate to submodules or packages: Use `cd src/<package_name>` or explore with `ls -l` to check Python files (.py) and ensure they are importable (e.g., via `__init__.py` files).

## Testing Instructions

- Lint the code after changes: Use `black .` for formatting, `flake8 .` for style checks, or `pylint src/` for deeper analysis. Fix any errors until clean.
- Fix any test failures or type errors (e.g., via `mypy .` if type hints are used) until everything is green.

## MCP
- use MCP Context7 to manage context effectively and ensure relevant information is available for decision-making, up-to-date documentation for a library
- use MCP sequential-thinking to break down complex tasks into manageable steps, ensuring clarity and focus on each part of the task.
 

## Additional Best Practices

### Code Quality
- Use type hints consistently throughout the codebase
- Follow PEP 8 style guidelines
- Write docstrings for all public functions and classes
- Keep functions small and focused on a single responsibility

### Documentation
- Update docstrings when modifying function signatures
- Add inline comments for complex logic
- Update the main README.md if adding new features
- Initialize the Memory Bank with memory-bank.md
- Update Memory Bank in memory-bank/ : Include new information or insights gained during development
- Consider adding examples in docstrings for complex functions

### Performance
- Use appropriate data structures for the task
- Consider memory usage for large datasets
- Cache expensive computations when appropriate