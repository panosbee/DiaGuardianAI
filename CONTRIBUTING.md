# Contributing to DiaGuardianAI

We welcome contributions from the diabetes and AI communities! This document provides guidelines for contributing to DiaGuardianAI.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We are committed to providing a welcoming and inclusive environment for all contributors.

## How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Check if the issue already exists
2. Use the issue templates provided
3. Include detailed information about your environment
4. Provide steps to reproduce the problem

### Suggesting Features

We welcome feature suggestions! Please:
1. Check existing feature requests
2. Describe the use case clearly
3. Explain how it benefits the diabetes community
4. Consider implementation complexity

### Contributing Code

#### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/diaguardianai.git
   cd diaguardianai
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```
5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

#### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Write or update tests
4. Update documentation if needed
5. Run tests:
   ```bash
   pytest tests/
   ```
6. Run code quality checks:
   ```bash
   black DiaGuardianAI/
   flake8 DiaGuardianAI/
   mypy DiaGuardianAI/
   ```

#### Submitting Changes

1. Commit your changes:
   ```bash
   git commit -m "Add feature: description"
   ```
2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
3. Create a Pull Request

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting
- Use type hints where appropriate
- Write descriptive variable and function names
- Keep functions focused and small

### Testing

- Write unit tests for new functionality
- Maintain test coverage above 80%
- Use pytest for testing framework
- Include integration tests for major features

### Documentation

- Update docstrings for new functions/classes
- Follow Google-style docstring format
- Update README.md for new features
- Include code examples in documentation

### Medical Accuracy

- Ensure medical accuracy in diabetes-related calculations
- Cite relevant medical literature
- Consider safety implications
- Include appropriate disclaimers

## Project Structure

```
DiaGuardianAI/
├── core/                   # Core system components
├── data_generation/        # Patient and data generation
├── models/                 # AI models and predictors
├── agents/                 # Multi-agent system
├── pattern_repository/     # Pattern storage and analysis
├── clinical_demo/          # Clinical dashboard
├── training/               # Model training utilities
├── utils/                  # Utility functions
└── tests/                  # Test suite
```

## Commit Message Guidelines

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

Examples:
```
feat(models): add transformer attention mechanism
fix(agents): resolve decision loop in edge cases
docs(api): update prediction method documentation
```

## Review Process

1. All submissions require review
2. Maintainers will review within 48 hours
3. Address feedback promptly
4. Ensure CI checks pass
5. Squash commits before merge

## Release Process

1. Version bumping follows semantic versioning
2. Releases are tagged and published automatically
3. Changelog is updated for each release
4. Breaking changes are clearly documented

## Community

- Join our discussions on GitHub
- Follow us for updates
- Participate in community calls
- Share your use cases and feedback

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Annual contributor highlights
- Conference presentations

## Questions?

If you have questions about contributing:
- Open a discussion on GitHub
- Email us at contact@diaguardianai.com
- Check our FAQ section

Thank you for contributing to DiaGuardianAI and helping improve diabetes care through AI!
