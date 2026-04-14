# LLM Bridge

LLM Bridge is a Python library that wraps multiple LLM providers into a consistent API while using each provider's native SDK internally, supporting multimodal I/O, file processing, and stream output.

GitHub: [https://github.com/windsnow1025/LLM-Bridge](https://github.com/windsnow1025/LLM-Bridge)

PyPI: [https://pypi.org/project/LLM-Bridge/](https://pypi.org/project/LLM-Bridge/)

## Workflow and Features

1. **Chat Client Factory**: creates a client for the specific LLM API with model parameters
    1. **Model Message Converter**: converts general messages to model messages
        1. **Media Processor**: converts general media to model compatible formats.
2. **Chat Client**: generate stream or non-stream responses

### Supported Features for API Types

The features listed represent the maximum capabilities of each API type supported by LLM Bridge.

| API Type              | Input Format                                     | Capabilities                                            | Output Format     |
|-----------------------|--------------------------------------------------|---------------------------------------------------------|-------------------|
| OpenAI Completion API | Text, Image, PDF                                 | Thinking, Structured Output                             | Text              |
| OpenAI Responses API  | Text, Image, PDF                                 | Thinking, Web Search, Code Execution, Structured Output | Text, Image       |
| Google GenAI          | Text, Image, PDF, Audio, Video                   | Thinking, Web Search, Code Execution, Structured Output | Text, Image, File |
| Anthropic             | Text, Image, PDF                                 | Thinking, Web Search, Code Execution, Structured Output | Text, File        |
| xAI                   | Text, Image, PDF, Audio, Video, docx, xlsx, pptx | Thinking, Web Search, Code Execution, Structured Output | Text              |

## Development

### Python uv

1. Install uv: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Install Python in uv: `uv python install 3.12`; upgrade Python in uv: `uv python upgrade 3.12`
3. Configure requirements:
  ```bash
  uv sync --refresh
  ```

### PyCharm

Add New Interpreter >> Add Local Interpreter
  - Environment: Select existing
  - Type: uv

### Usage

Copy `./usage/.env.example` and rename it to `./usage/.env`, then fill in the environment variables.

### Build

```bash
uv build
```
