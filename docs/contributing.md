# Contributing to OrthoRoute

## Getting Started

### Development Environment Setup

1. **Clone Repository:**
```bash
git clone https://github.com/bbenchoff/OrthoRoute.git
cd OrthoRoute
```

2. **Install Development Dependencies:**
```bash
pip install -e .[dev]
```

3. **Install Pre-commit Hooks:**
```bash
pre-commit install
```

## Development Workflow

### 1. Creating a New Feature

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following our coding standards
3. Write tests for new functionality
4. Run test suite locally:
```bash
pytest tests/
```

### 2. Code Style

We follow PEP 8 with these additions:

- Maximum line length: 100 characters
- Use type hints for function parameters and return values
- Use docstrings in Google format

Example:
```python
def route_net(start: Point, end: Point) -> Path:
    """
    Route a single net between two points.
    
    Args:
        start (Point): Starting point coordinates
        end (Point): Ending point coordinates
        
    Returns:
        Path: Routed path object
        
    Raises:
        RoutingError: If no valid path is found
    """
```

### 3. Testing

#### Writing Tests

1. Place tests in `tests/` directory
2. Name test files with `test_` prefix
3. Use pytest fixtures for common setup
4. Include both unit and integration tests

Example:
```python
def test_route_simple_net():
    # Arrange
    start = Point(0, 0)
    end = Point(10, 10)
    
    # Act
    path = route_net(start, end)
    
    # Assert
    assert len(path.points) > 0
    assert path.points[0] == start
    assert path.points[-1] == end
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_gpu_engine.py

# Run with coverage
pytest --cov=orthoroute tests/
```

### 4. Documentation

1. Keep docstrings up to date
2. Update API reference for new features
3. Add examples for significant functionality
4. Include performance considerations for GPU code

### 5. Performance Considerations

#### GPU Code

1. Minimize host-device transfers
2. Use proper memory alignment
3. Optimize kernel launch configurations
4. Profile with NVIDIA tools:
```bash
nvprof python -m pytest tests/test_gpu_engine.py
```

#### Memory Management

1. Use memory pools for repeated allocations
2. Implement proper cleanup in destructors
3. Monitor GPU memory usage

## Pull Request Process

1. **Before Submitting:**
   - Run full test suite
   - Update documentation
   - Check code style
   - Verify GPU memory handling

2. **PR Template:**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing performed
   
   ## Documentation
   - [ ] API documentation updated
   - [ ] Examples added/updated
   - [ ] Performance notes added
   ```

3. **Review Process:**
   - Two approvals required
   - All tests must pass
   - Documentation must be complete

## Release Process

1. **Version Bump:**
   - Update version in `setup.py`
   - Update CHANGELOG.md
   
2. **Release Checklist:**
   - [ ] All tests passing
   - [ ] Documentation updated
   - [ ] CHANGELOG updated
   - [ ] Version bumped
   - [ ] Release notes prepared

3. **Tag and Release:**
```bash
git tag -a v1.x.x -m "Release v1.x.x"
git push origin v1.x.x
```

## Additional Resources

- [GPU Programming Guide](docs/gpu_programming.md)
- [Algorithm Documentation](docs/algorithms.md)
- [Performance Tuning](docs/performance.md)

## Questions and Support

- Create an issue for bugs or feature requests
- Join our discussion forum for questions
- Check existing issues before creating new ones
