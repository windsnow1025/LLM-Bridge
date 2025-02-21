# LLM Bridge

A Bridge for LLMs. Support OpenAI, OpenAI-Azure, OpenAI-GitHub, Gemini, Claude, Grok.
 
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

See `./usage/`

## Installation

### PyPI

```bash
pip install --upgrade llm_bridge
```

