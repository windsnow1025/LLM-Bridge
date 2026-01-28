# LLM Bridge

LLM Bridge is a unified API wrapper for native interactions with various LLM providers.

GitHub: [https://github.com/windsnow1025/LLM-Bridge](https://github.com/windsnow1025/LLM-Bridge)

PyPI: [https://pypi.org/project/LLM-Bridge/](https://pypi.org/project/LLM-Bridge/)

## Workflow and Features

1. **Message Preprocessor**: extracts text content from documents (Word, Excel, PPT, Code files, PDFs) which are not natively supported by the target model.
2. **Chat Client Factory**: creates a client for the specific LLM API with model parameters
    1. **Model Message Converter**: converts general messages to model messages
        1. **Media Processor**: converts general media (Image, Audio, Video, PDF) to model compatible formats.
3. **Chat Client**: generate stream or non-stream responses
    - **Model Thoughts**: captures the model's thinking process
    - **Code Execution**: generates and executes Python code
    - **Web Search**: generates response from search results
    - **Token Counter**: tracks and reports input and output token usage

### Supported Features for API Types

The features listed represent the maximum capabilities of each API type supported by LLM Bridge.

| API Type | Input Format                   | Capabilities                                            | Output Format     |
|----------|--------------------------------|---------------------------------------------------------|-------------------|
| OpenAI   | Text, Image, PDF               | Thinking, Web Search, Code Execution                    | Text, Image       |
| Gemini   | Text, Image, Video, Audio, PDF | Thinking, Web Search, Code Execution, Structured Output | Text, Image, File |
| Claude   | Text, Image, PDF               | Thinking, Web Search, Code Execution                    | Text, File        |
| Grok     | Text, Image                    |                                                         | Text              |

#### Planned Features

- Structured Output
- More features for API Types
- Native support for Grok

## Development

### Python uv

1. Install uv: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Install Python in uv: `uv python install 3.12`; upgrade Python in uv: `uv python install 3.12`
3. Configure requirements:
  ```bash
  uv sync --refresh
  ```

### Pycharm

1. Add New Interpreter >> Add Local Interpreter
  - Environment: Select existing
  - Type: uv
2. Add New Configuration >> uv run >> script: `./usage/main.py`

### Usage

Copy `./usage/.env.example` and rename it to `./usage/.env`, then fill in the environment variables.

### Build

```bash
uv build
```
