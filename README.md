# LLM Bridge
 
## Process

1. **Message Preprocessor**: Preprocess Messages
    1. **Message Preprocessor**: Extract Text Files to Message
2. **Chat Client Factory**: Create Chat Client
    1. **Model Message Converter**: Convert Message to Model
        1. **Media Processor**: Get Model Image Content from URL
3. **Chat Client**: Generate Response

## Test

```bash
pytest ./tests/
```

## Usage

See ./usage/

## Build

### Dependencies

```bash
pip install --upgrade setuptools wheel twine
```

### Build

```bash
python setup.py sdist bdist_wheel
```

## Publish

### TestPyPI

```bash
twine upload --repository testpypi dist/*
```

### PyPI

```bash
twine upload dist/*
```

## Installation

### TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ --upgrade llm_bridge
```

### PyPI

```bash
pip install --upgrade llm_bridge
```

