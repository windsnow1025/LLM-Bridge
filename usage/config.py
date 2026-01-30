from llm_bridge import *

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
  ]
}
structured_output_schema = None

messages = [
    Message(
        role=Role.System,
        contents=[
            # Content(type=ContentType.Text, data="You are a helpful assistant."),
            Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746209841847-A%20Tutorial%20on%20Spectral%20Clustering.pdf")
        ]
    ),
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
    Message(
        role=Role.User,
        contents=[
            # Simple Question
            Content(type=ContentType.Text, data="What's this?"),

            # Thinking
            # Content(type=ContentType.Text, data="Explain the concept of Occam's Razor and provide a simple, everyday example."),

            # Web Search
            # Content(type=ContentType.Text, data="What's the weather in NYC today?"),

            # Image Understanding
            # Content(type=ContentType.File, data="https://www.gstatic.com/webp/gallery3/1.png"),
            # Content(type=ContentType.Text, data="What is in this image?"),

            # Image Generation
            # Content(type=ContentType.Text, data="Please generate an image of a cat."),

            # URL Context
            # Content(type=ContentType.Text, data="What is in https://www.windsnow1025.com/"),

            # Code Execution
            # Content(type=ContentType.Text, data="What is the sum of the first 50 prime numbers? Generate and run code for the calculation, and make sure you get all 50."),

            # File Output
            # Content(type=ContentType.Text, data="Create a matplotlib visualization and save it as output.png"),

            # Structured Output
            # Content(type=ContentType.Text, data="Please generate a product."),
        ]
    ),
    # Message(
    #     role=Role.User,
    #     contents=[
    #         Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1769429581512-Test.txt"),
    #         # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746208707489-image.png"),
    #         # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746209841847-A%20Tutorial%20on%20Spectral%20Clustering.pdf"),
    #         # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746212253473-file_example_MP3_700KB.mp3"),
    #         # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746212980820-file_example_MP4_480_1_5MG.mp4"),
    #         Content(type=ContentType.Text, data="What's this?"),
    #     ]
    # ),
]
# See /llm_bridge/resources/model_prices.json for available models
model = "gpt-5.2"
# model = "gpt-5.1"
# model = "gpt-5-pro"
# model = "gpt-5"
# model = "gpt-4.1"
# model = "gpt-5-codex"
# model = "gemini-3-pro-preview"
# model = "gemini-3-pro-image-preview"
# model = "gemini-3-flash-preview"
# model = "grok-4-1-fast-reasoning"
# model = "claude-sonnet-4-5"
# model = "claude-opus-4-5"
# api_type = "Gemini-Vertex"
# api_type = "Gemini-Free"
# api_type = "Gemini-Paid"
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
# code_execution = True
code_execution = False
