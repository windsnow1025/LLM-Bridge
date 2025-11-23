# LLM Bridge

LLM Bridge is a unified Python interface for interacting with LLMs, including OpenAI, OpenAI-Azure, OpenAI-GitHub, Gemini, Claude, and Grok.

GitHub: [https://github.com/windsnow1025/LLM-Bridge](https://github.com/windsnow1025/LLM-Bridge)

PyPI: [https://pypi.org/project/LLM-Bridge/](https://pypi.org/project/LLM-Bridge/)

## Workflow and Features

1. **Message Preprocessor**: extracts text content from documents (Word, Excel, PPT, Code files, PDFs) which are not natively supported by the target model.
2. **Chat Client Factory**: create a client for the specific LLM API with model parameters
    1. **Model Message Converter**: convert general messages to model messages
        1. **Media Processor**: converts media (Image, Audio, Video, PDF) which are natively supported by the target model into compatible formats.
3. **Chat Client**: generate stream or non-stream responses
    1. **Model Thoughts**: captures and formats the model's thinking process
    2. **Search Citations**: extracts and formats citations from search results
    3. **Token Counter**: tracks and reports input and output token usage

### Model Features

The features listed represent the maximum capabilities of each API type supported by LLM Bridge.

| API Type | Input Format                   | Capabilities                                     | Output Format     |
|----------|--------------------------------|--------------------------------------------------|-------------------|
| OpenAI   | Text, Image                    | Thinking, Web Search, Code Execution             | Text              |
| Gemini   | Text, Image, Video, Audio, PDF | Thinking, Web Search + Citations, Code Execution | Text, Image, File |
| Claude   | Text, Image, PDF               | Thinking, Web Search, Code Execution             | Text              |
| Grok     | Text, Image                    |                                                  | Text              |

#### Planned Features

- OpenAI: Web Search: Citations, Image Output
- Gemini: Code Execution: Code, Code Output
- Claude: Code Execution, File Output

## Installation

```bash
pip install --upgrade llm_bridge
```

## Test

```bash
pytest
```

## Quick Start

### Setup

1. Copy `./usage/.env.example` and rename it to `./usage/.env`, then fill in the environment variables.
2. Install requirements: `pip install -r requirements.txt`
3. In PyCharm, add a new Python configuration:
   - script: `./usage/main.py`
   - Paths to ".env" files: `./usage/.env`
