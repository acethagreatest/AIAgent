"""Settings for Agent."""

from pydantic import BaseModel, Field, SecretStr # type: ignore


class AgentSettingsModel(BaseModel): # type: ignore
    """Configuration specific to the Flare ecosystem interactions."""

    gemini_api_key: SecretStr = Field( # type: ignore
        ...,
        description="API key for using Google Gemini (https://aistudio.google.com/app/apikey).",
    )
    gemini_model: str = Field( # type: ignore
        ..., description="Gemini model to use (e.g. gemini-2.0-flash)"
    )
     chatgpt_model: str = Field( # type: ignore
        ..., description="ChatGPT model to use (e.g. 03-mini)"
    )
    chatgpt_api_key: SecretStr = Field( # type: ignore
        ...,
        description="API key for using ChatGPT (https://platform.openai.com/api-keys).",
    )
    openrouter_api_key: SecretStr | None = Field( # type: ignore
        None,
        description="API key for OpenRouter.",
    )
