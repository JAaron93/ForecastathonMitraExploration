# Type Hinting Documentation for Serialization Utilities

This document provides comprehensive documentation for the type hinting improvements made to `src/utils/serialization.py`. It includes detailed explanations of type hints, usage examples, type inference demonstrations, and best practices for type hinting in serialization.

## Overview

The serialization module provides utilities for saving and loading data in various formats (JSON, Parquet, Pickle/Joblib) with comprehensive type hints to ensure type safety and improve code maintainability.

## Type Hinting Improvements

### 1. Generic Type Variables

```python
from typing import TypeVar

T = TypeVar('T')
```

**Usage**: The `T` type variable is used in the `load_joblib` function to provide generic type inference:

```python
def load_joblib(
    path: Union[str, Path], 
    signature: Optional[str] = None, 
    hmac_key: Optional[bytes] = None
) -> T:
    """Load object from joblib with optional HMAC signature verification."""
    # Implementation
```

**Type Inference Example**:

```python
# Without explicit type annotation
model = load_joblib('model.joblib')  # Type: Any (inferred as T)

# With explicit type annotation
model: RandomForestClassifier = load_joblib('model.joblib')  # Type: RandomForestClassifier
```

### 2. Union Types for Flexible Input

```python
from typing import Union
from pathlib import Path
```

**Usage**: Multiple functions accept both `str` and `Path` objects:

```python
def save_json(
    data: Union[Dict[str, Any], List[Any], str, int, float, bool, None], 
    path: Union[str, Path], 
    **kwargs
) -> None:
    """Save data to JSON with datetime support."""
    # Implementation
```

**Type Inference Example**:

```python
# Both calls are valid
save_json(data, 'data.json')
save_json(data, Path('data.json'))
```

### 3. Complex Union Types for Return Values

```python
def load_json(
    path: Union[str, Path]
) -> Union[Dict[str, Any], List[Any], str, int, float, bool, None]:
    """Load data from JSON."""
    # Implementation
```

**Type Inference Example**:

```python
# The return type depends on the JSON content
data = load_json('config.json')
# Type: Union[Dict[str, Any], List[Any], str, int, float, bool, None]
```

### 4. Optional Parameters with Type Hints

```python
def load_joblib(
    path: Union[str, Path], 
    signature: Optional[str] = None, 
    hmac_key: Optional[bytes] = None
) -> T:
    """Load object from joblib with optional HMAC signature verification."""
    # Implementation
```

**Type Inference Example**:

```python
# Without signature verification
model = load_joblib('model.joblib')

# With signature verification
signature = 'abc123'
key = b'secret_key'
model = load_joblib('model.joblib', signature=signature, hmac_key=key)
```

### 5. Custom Type Hints for Domain-Specific Types

```python
from src.data.structs import TimeSeriesData
```

**Usage**: The `save_timeseries_data` and `load_timeseries_data` functions use the `TimeSeriesData` type:

```python
def save_timeseries_data(data: TimeSeriesData, path: Union[str, Path]) -> None:
    """Save TimeSeriesData to a directory structure."""
    # Implementation

def load_timeseries_data(path: Union[str, Path]) -> TimeSeriesData:
    """Load TimeSeriesData from a directory."""
    # Implementation
```

**Type Inference Example**:

```python
# Create TimeSeriesData instance
timeseries = TimeSeriesData(
    timestamp=pd.date_range('2020-01-01', periods=100),
    features=pd.DataFrame(np.random.randn(100, 5)),
    targets=pd.Series(np.random.randn(100)),
    metadata={'source': 'simulation'},
    split_indices={'train': slice(0, 80), 'test': slice(80, 100)}
)

save_timeseries_data(timeseries, 'timeseries_data')

# Load with type inference
loaded_timeseries = load_timeseries_data('timeseries_data')
# Type: TimeSeriesData
```

## Usage Examples

### Basic Serialization

```python
import pandas as pd
from src.utils.serialization import save_json, load_json

# Create sample data
data = {
    'name': 'John Doe',
    'age': 30,
    'timestamp': pd.Timestamp('2023-01-01'),
    'scores': [95.5, 88.0, 92.3]
}

# Save with type hints
save_json(data, 'data.json')

# Load with type hints
loaded_data = load_json('data.json')
# Type: Union[Dict[str, Any], List[Any], str, int, float, bool, None]
```

### Model Serialization

```python
from sklearn.ensemble import RandomForestClassifier
from src.utils.serialization import save_joblib, load_joblib

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save with type hints
save_joblib(model, 'model.joblib')

# Load with type hints
loaded_model = load_joblib('model.joblib')
# Type: Any (inferred as T)

# With explicit type annotation
loaded_model: RandomForestClassifier = load_joblib('model.joblib')
```

### DataFrame Serialization

```python
import pandas as pd
from src.utils.serialization import save_parquet, load_parquet

# Create DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4.5, 5.5, 6.5],
    'C': ['x', 'y', 'z']
})

# Save with type hints
save_parquet(df, 'data.parquet')

# Load with type hints
loaded_df = load_parquet('data.parquet')
# Type: pd.DataFrame
```

### TimeSeriesData Serialization

```python
from src.data.structs import TimeSeriesData
from src.utils.serialization import save_timeseries_data, load_timeseries_data

# Create TimeSeriesData instance
timeseries = TimeSeriesData(
    timestamp=pd.date_range('2020-01-01', periods=100),
    features=pd.DataFrame(np.random.randn(100, 5)),
    targets=pd.Series(np.random.randn(100)),
    metadata={'source': 'simulation'},
    split_indices={'train': slice(0, 80), 'test': slice(80, 100)}
)

# Save with type hints
save_timeseries_data(timeseries, 'timeseries_data')

# Load with type hints
loaded_timeseries = load_timeseries_data('timeseries_data')
# Type: TimeSeriesData
```

## Type Inference Examples

### Generic Type Inference

```python
# Without explicit type annotation
model = load_joblib('model.joblib')
# Type: Any (inferred as T)

# With explicit type annotation
model: RandomForestClassifier = load_joblib('model.joblib')
# Type: RandomForestClassifier

# Type checking works
if isinstance(model, RandomForestClassifier):
    predictions = model.predict(X_test)
```

### Union Type Inference

```python
# Path type inference
path1 = 'data.json'  # Type: str
path2 = Path('data.json')  # Type: Path

# Both are valid for save_json
save_json(data, path1)
save_json(data, path2)
```

### Return Type Inference

```python
# JSON return type depends on content
data1 = load_json('config.json')
# Type: Union[Dict[str, Any], List[Any], str, int, float, bool, None]

# DataFrame return type is specific
df = load_parquet('data.parquet')
# Type: pd.DataFrame
```

## Best Practices for Type Hinting in Serialization

### 1. Use Generic Type Variables for Flexible Return Types

```python
# Good: Generic type variable for flexible return
T = TypeVar('T')

def load_joblib(path: Union[str, Path]) -> T:
    """Load object with generic type."""
    return joblib.load(path)

# Bad: Too specific return type
# def load_joblib(path: Union[str, Path]) -> Any:
#     """Less type-safe."""
```

### 2. Use Union Types for Flexible Input

```python
# Good: Accept both str and Path
from pathlib import Path

def save_json(data: Any, path: Union[str, Path]) -> None:
    """Accept both string paths and Path objects."""
    # Implementation

# Bad: Only accept one type
# def save_json(data: Any, path: str) -> None:
#     """Less flexible."""
```

### 3. Use Optional for Parameters That May Be None

```python
# Good: Optional parameters with defaults
def load_joblib(
    path: Union[str, Path], 
    signature: Optional[str] = None, 
    hmac_key: Optional[bytes] = None
) -> T:
    """Optional signature verification."""
    # Implementation

# Bad: No type hints for optional parameters
# def load_joblib(path, signature=None, hmac_key=None):
#     """No type safety."""
```

### 4. Use Domain-Specific Types for Complex Data Structures

```python
# Good: Use custom types for domain-specific data
from src.data.structs import TimeSeriesData

def save_timeseries_data(data: TimeSeriesData, path: Union[str, Path]) -> None:
    """Type-safe for TimeSeriesData."""
    # Implementation

# Bad: Use generic types
# def save_timeseries_data(data: Dict[str, Any], path: str) -> None:
#     """Less type-safe."""
```

### 5. Document Type Hints with Clear Docstrings

```python
# Good: Comprehensive docstrings with type information
def load_joblib(
    path: Union[str, Path], 
    signature: Optional[str] = None, 
    hmac_key: Optional[bytes] = None
) -> T:
    """
    Load object from joblib with optional HMAC signature verification.
    
    Args:
        path: Path to the joblib file
        signature: Optional HMAC signature to verify (hex string)
        hmac_key: Optional HMAC key for verification (bytes)
        
    Returns:
        Deserialized object
        
    Raises:
        ValueError: If signature verification fails
        FileNotFoundError: If file doesn't exist
        Exception: For other deserialization errors
    """
    # Implementation
```

## Common Type Hinting Patterns

### 1. Function with Multiple Return Types

```python
def parse_data(
    data: str
) -> Union[int, float, str, bool, None]:
    """Parse string data into appropriate type."""
    if data.isdigit():
        return int(data)
    try:
        return float(data)
    except ValueError:
        if data.lower() in ['true', 'false']:
            return data.lower() == 'true'
        if data.lower() == 'none':
            return None
        return data
```

### 2. Generic Container Types

```python
from typing import Dict, List, Tuple

def process_data(
    items: List[Dict[str, Any]],
    config: Dict[str, Union[int, float, str]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process list of items with configuration."""
    # Implementation
```

### 3. Type Hints with Default Values

```python
def configure(
    timeout: int = 30,
    retries: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """Configure system with default values."""
    return {
        'timeout': timeout,
        'retries': retries,
        'verbose': verbose
    }
```

### 4. Type Hints for Callbacks

```python
from typing import Callable, Any

def process_with_callback(
    data: List[Any],
    callback: Callable[[Any], Any]
) -> List[Any]:
    """Process data with callback function."""
    return [callback(item) for item in data]
```

## Type Hinting Guide for the Project

### 1. Follow PEP 484 Type Hints

- Use `typing` module for all type hints
- Use `TypeVar` for generic types
- Use `Union` for multiple possible types
- Use `Optional` for parameters that can be None

### 2. Be Specific When Possible

```python
# Good: Specific types
from typing import Dict, List

def process_data(
    items: List[Dict[str, Any]],
    config: Dict[str, Union[int, float, str]]
) -> Dict[str, Any]:
    """Process data with specific types."""
    # Implementation

# Bad: Too generic
# def process_data(items, config):
#     """No type safety."""
```

### 3. Use Domain-Specific Types

```python
# Good: Use custom types from the project
from src.data.structs import TimeSeriesData

def save_timeseries_data(data: TimeSeriesData, path: Union[str, Path]) -> None:
    """Type-safe for TimeSeriesData."""
    # Implementation

# Bad: Use generic types
# def save_timeseries_data(data: Dict[str, Any], path: str) -> None:
#     """Less type-safe."""
```

### 4. Document Type Hints in Docstrings

```python
# Good: Comprehensive docstrings

def load_joblib(
    path: Union[str, Path], 
    signature: Optional[str] = None, 
    hmac_key: Optional[bytes] = None
) -> T:
    """
    Load object from joblib with optional HMAC signature verification.
    
    Args:
        path: Path to the joblib file
        signature: Optional HMAC signature to verify (hex string)
        hmac_key: Optional HMAC key for verification (bytes)
        
    Returns:
        Deserialized object
        
    Raises:
        ValueError: If signature verification fails
        FileNotFoundError: If file doesn't exist
        Exception: For other deserialization errors
    """
    # Implementation
```

### 5. Use Type Hints for Error Handling

```python
# Good: Type hints for error handling
from typing import NoReturn

def critical_error(message: str) -> NoReturn:
    """Log critical error and exit."""
    logger.critical(message)
    sys.exit(1)
```

## Conclusion

This documentation provides a comprehensive overview of the type hinting improvements made to the serialization utilities. By following these patterns and best practices, you can ensure type safety, improve code maintainability, and provide better developer experience throughout the project.

Remember to:
- Use specific types when possible
- Leverage generic type variables for flexible return types
- Use Union types for flexible input
- Document type hints with clear docstrings
- Use domain-specific types for complex data structures

These practices will help maintain a robust and type-safe codebase for the forecasting pipeline.