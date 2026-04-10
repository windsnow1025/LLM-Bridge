from llm_bridge import *

# Standard JSON Schema
structured_output_schema = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {
      "description": "The unique identifier for a product",
      "type": "integer"
    },
    "productName": {
      "description": "Name of the product",
      "type": "string"
    },
    "price": {
      "description": "The price of the product",
      "type": "number",
      "exclusiveMinimum": 0
    },
    "tags": {
      "description": "Tags for the product",
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "uniqueItems": True
    }
  },
  "required": [
    "productId",
    "productName",
    "price"
  ],
  "additionalProperties": False
}

# OpenAI Responses API JSON Schema
# structured_output_schema = {
#   "$schema": "https://json-schema.org/draft/2020-12/schema",
#   "$id": "https://example.com/product.schema.json",
#   "title": "Product",
#   "description": "A product from Acme's catalog",
#   "type": "object",
#   "properties": {
#     "productId": {
#       "description": "The unique identifier for a product",
#       "type": "integer"
#     },
#     "productName": {
#       "description": "Name of the product",
#       "type": "string"
#     },
#     "price": {
#       "description": "The price of the product",
#       "type": "number",
#       "exclusiveMinimum": 0
#     },
#     "tags": {
#       "description": "Tags for the product",
#       "type": "array",
#       "items": {
#         "type": "string"
#       },
#       "minItems": 1,
#     }
#   },
#   "required": [
#     "productId",
#     "productName",
#     "price",
#     "tags"
#   ],
#   "additionalProperties": False
# }

structured_output_schema = None

messages = [
    Message(
        role=Role.System,
        contents=[
            Content(type=ContentType.Text, data="You are a helpful assistant."),

            # PDF Input in System Message
            # Content(type=ContentType.File, data="https://pdfobject.com/pdf/sample.pdf")
        ]
    ),
    # Message History

    # Message(
    #     role=Role.User,
    #     contents=[
    #         Content(type=ContentType.Text, data="Hello")
    #     ]
    # ),
    # Message(
    #     role=Role.Assistant,
    #     contents=[
    #         Content(type=ContentType.Text, data="Hello! How can I assist you today?")
    #     ]
    # ),

    # Capabilities
    # Message(
    #     role=Role.User,
    #     contents=[
    #         # Simple Question
    #         # Content(type=ContentType.Text, data="What's this?"),
    #
    #         # Thinking
    #         # Content(type=ContentType.Text, data="Explain the concept of Occam's Razor and provide a simple, everyday example."),
    #
    #         # Web Search
    #         # Content(type=ContentType.Text, data="What's the weather in NYC today?"),
    #
    #         # URL Context
    #         # Content(type=ContentType.Text, data="What is in https://www.windsnow1025.com/"),
    #
    #         # Code Execution
    #         # Content(type=ContentType.Text, data="What is the sum of the first 50 prime numbers? Generate and run code for the calculation, and make sure you get all 50."),
    #
    #         # Structured Output
    #         # Content(type=ContentType.Text, data="Please generate a product."),
    #
    #         # Image Output
    #         # Content(type=ContentType.Text, data="Please generate an image of a cat."),
    #
    #         # File Output
    #         # Content(type=ContentType.Text, data="Create a matplotlib visualization and save it as output.png"),
    #     ]
    # ),

    # File Inputs
    Message(
        role=Role.User,
        contents=[
            # Text Input
            # Content(type=ContentType.File, data="https://example-files.online-convert.com/document/txt/example.txt"),

            # PDF Input
            # Content(type=ContentType.File, data="https://pdfobject.com/pdf/sample.pdf")

            # Image Input
            Content(type=ContentType.File, data="https://www.gstatic.com/webp/gallery3/1.png"),

            # Audio Input
            # Content(type=ContentType.File, data="https://samplelib.com/lib/preview/mp3/sample-3s.mp3"),

            # Video Input
            # Content(type=ContentType.File, data="https://examplefiles.org/files/video/mp4-example-video-download-640x480.mp4"),

            Content(type=ContentType.Text, data="What's this?"),
        ]
    ),
]

# See /llm_bridge/resources/model_prices.json for available models
# model = "gpt-5.4-mini"
# model = "gpt-5.2"
# model = "gpt-5.1"
model = "gpt-5.4"
# model = "gpt-5"
# model = "gpt-4.1"
# model = "gpt-5-codex"
# model = "gemini-3-pro-preview"
# model = "gemini-3-pro-image-preview"
# model = "gemini-3.1-flash-image-preview"
# model = "gemini-3-flash-preview"
# model = "claude-sonnet-4-6"
# model = "grok-4.20-reasoning"

# api_type = "Vertex AI"
# api_type = "Google AI Studio Free Tier"
# api_type = "Google AI Studio"
api_type = "OpenAI"
# api_type = "OpenAI-Azure"
# api_type = "OpenAI-GitHub"
# api_type = "Claude"
# api_type = "Grok"

temperature = 0
stream = True
# stream = False
thought = True
# thought = False
# web_search = True
web_search = False
# code_execution = True
code_execution = False
