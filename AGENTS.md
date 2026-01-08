# AGENTS.md - Development Guidelines for LangChain Research Agent

This file contains essential information for agentic coding assistants working on this LangChain-based research agent project.

## Project Overview

This is a Python 3.12+ research assistant application that combines LangChain, Chroma vector database, and LiveKit voice agents. It provides both text and voice interfaces for research queries, with citation support and local PDF document ingestion.

## Build/Lint/Test Commands

### Package Management
- **Install dependencies**: `uv sync`
- **Run with dependencies**: `uv run python <script.py>`
- **Add dependency**: `uv add <package>`

### Development Commands
- **Run main application**: `uv run python main.py`
- **Voice mode**: Select option 2 in main.py (requires LiveKit server)
- **Text mode**: Select option 1 in main.py for interactive CLI

### Testing
No test framework is currently configured. Recommended setup:
- **Install pytest**: `uv add --dev pytest pytest-asyncio`
- **Run all tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/test_file.py::test_function_name`
- **Run with coverage**: `uv run pytest --cov=.`

### Linting and Formatting
No linters/formatters are currently configured. Recommended setup:
- **Install tools**: `uv add --dev ruff black mypy`
- **Format code**: `uv run black .`
- **Lint code**: `uv run ruff check .`
- **Fix linting issues**: `uv run ruff check --fix .`
- **Type check**: `uv run mypy .`

## Code Style Guidelines

### Python Version
- Target: Python 3.12+
- Use modern Python features (match statements, f-strings, type hints)

### Imports
```python
# Standard library imports (alphabetical)
from os import getenv
from pathlib import Path
from typing import List, Optional, Union

# Third-party imports (alphabetical, grouped by package)
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Local imports (alphabetical)
from chroma import create_chroma_client
from models.citation import Citation
```

### Naming Conventions
- **Functions/Methods**: `snake_case` (e.g., `init_tools`, `normalize_messages`)
- **Classes**: `PascalCase` (e.g., `ResearchAnswer`, `VoiceResearchAgent`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `SYSTEM_PROMPT`, `RESEARCH_PAPERS_DIR`)
- **Variables**: `snake_case` (e.g., `confidence_score`, `mcp_client`)
- **Modules**: `snake_case` (e.g., `research_agent.py`, `voice_research_agent.py`)

### Type Hints
- Use comprehensive type hints for all function parameters and return values
- Use `Union` for multiple possible types
- Use `Optional` for nullable types
- Prefer specific types over generic `Any`

```python
def normalize_messages(
    messages: Union[List[BaseMessage], List[dict], List[Union[BaseMessage, dict]]],
) -> List[BaseMessage]:
    # Function implementation
```

### Data Models
- Use Pydantic `BaseModel` for all data structures
- Include descriptive docstrings for classes
- Use `Field` with descriptions for model fields
- Use appropriate default values

```python
class ResearchAnswer(BaseModel):
    """A research answer generated from multiple research papers."""

    answer: str = Field(description="The generated research answer.")
    citations: List[Citation] = Field(
        description="List of citations supporting the research answer."
    )
    confidence_score: float = Field(description="Confidence level from 0 to 1.")
```

### Documentation
- Use triple-quoted docstrings for all modules, classes, and functions
- Follow Google-style docstring format
- Include parameter descriptions and return value descriptions
- Document side effects and exceptions

### Error Handling
- Use specific exception types when possible
- Provide meaningful error messages
- Use context managers for resource management
- Handle async exceptions appropriately

```python
try:
    result = await some_async_operation()
except ValueError as e:
    raise ValueError(f"Invalid input for operation: {e}") from e
```

### Async/Await
- Use `async`/`await` for all I/O operations
- Prefer async versions of libraries when available
- Use `asyncio.gather()` for concurrent operations
- Handle cancellation and timeouts appropriately

### String Formatting
- Use f-strings for string interpolation
- Use triple-quoted strings for multi-line strings
- Prefer named format parameters for complex formatting

```python
print(f"Response:\n{response.answer}\n")
print(f"Confidence: {response.confidence_score:.2%}\n")
```

### Control Flow
- Use `match` statements for pattern matching (Python 3.10+)
- Prefer list/dict comprehensions over explicit loops when appropriate
- Use early returns to reduce nesting
- Keep functions focused on single responsibilities

### Constants and Configuration
- Define constants at module level
- Use environment variables for configuration (via `python-dotenv`)
- Group related constants together

```python
SYSTEM_PROMPT = """You are a research assistant..."""

RESEARCH_PAPERS_DIR = Path("./data/research_papers")
```

### File Structure
- Group related functionality into modules
- Use `models/` directory for Pydantic models
- Use `libs/` directory for third-party integrations
- Keep utility functions in dedicated modules

### Logging and Tracing
- Use LangSmith `@traceable` decorator for important functions
- Include meaningful print statements for debugging
- Consider structured logging for production code

### Dependencies
- Use specific version pins in `pyproject.toml`
- Prefer well-maintained, actively developed packages
- Minimize dependencies by using built-in functionality when possible

### Performance Considerations
- Use async operations for I/O bound tasks
- Consider memory usage with large documents
- Use appropriate data structures for performance-critical code
- Profile code before optimizing

## Testing Guidelines

### Test Structure
- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names

### Test Categories
- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

### Async Testing
- Use `pytest-asyncio` for async test functions
- Use `pytest.mark.asyncio` decorator

### Test Data
- Use fixtures for reusable test data
- Mock external dependencies
- Use realistic but minimal test data

## Deployment and Production

### Environment Variables
- Use `.env` files for local development
- Document all required environment variables
- Never commit secrets to version control

### Docker Considerations
- Consider containerizing the application
- Use multi-stage builds for optimization
- Include health checks and proper shutdown handling

### Monitoring
- Implement proper error tracking
- Add performance monitoring
- Use structured logging

## Development Workflow

1. Create feature branch from `main`
2. Make changes following style guidelines
3. Run linting and type checking
4. Write/update tests
5. Test changes locally
6. Commit with descriptive messages
7. Create pull request for review

## Common Patterns

### LangChain Integration
- Use LangGraph functional API for complex workflows
- Leverage LangChain tools and retrievers
- Handle message normalization properly

### Vector Database
- Use Chroma for document storage and retrieval
- Configure appropriate embedding models
- Handle document chunking and metadata

### Voice Integration
- Use LiveKit for voice agent functionality
- Configure STT/TTS appropriately
- Handle real-time communication patterns

This document should be updated as the project evolves and new patterns emerge.</content>
<parameter name="filePath">/home/khoinguyxn/Documents/Personal/langchain-for-llm-app-dev-capstone-proj/AGENTS.md