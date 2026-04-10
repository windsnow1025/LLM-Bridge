import os
from xai_sdk import Client
from xai_sdk.chat import user, system
client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    management_api_key=os.getenv("XAI_MANAGEMENT_API_KEY"),
    timeout=3600,
)
chat = client.chat.create(model="grok-4.20-reasoning")
chat.append(system("You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."))
chat.append(user("What is the meaning of life, the universe, and everything?"))
response = chat.sample()
print(response)
# The response ID that can be used to continue the conversation later
print(response.id)