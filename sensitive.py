import os

# Single Azure key and endpoint for all models
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
azure_endpoint = os.getenv(
    "AZURE_ENDPOINT",
    "https://bjayt-mffvuqlc-eastus2.cognitiveservices.azure.com/",
)
# Optional: base URL for Mistral OCR and OpenAI-compatible chat (e.g. services.ai.azure.com)
azure_serverless_endpoint = os.getenv(
    "AZURE_SERVERLESS_ENDPOINT",
    "https://bjayt-mffvuqlc-eastus2.services.ai.azure.com",
)
# Azure Responses API (/openai/v1/responses). Date-stamped versions often return
# "API version not supported" on many resources; `preview` is the usual value in MS docs.
azure_responses_api_version = os.getenv(
    "AZURE_RESPONSES_API_VERSION",
    "preview",
)
