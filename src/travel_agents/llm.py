from langchain_core.language_models import BaseChatModel

from travel_agents.config import get_settings


def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """Default chat model: Llama 3.3 70B on Groq. Swappable — every agent takes a BaseChatModel."""
    from langchain_groq import ChatGroq

    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=temperature)
