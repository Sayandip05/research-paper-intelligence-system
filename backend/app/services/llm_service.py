from llama_index.llms.groq import Groq
from llama_index.core.llms import LLM
from app.config import get_settings

settings = get_settings()

def get_llm() -> LLM:
    """
    Get LlamaIndex LLM instance
    
    Using Groq as the provider for fast and free inference.
    Even if the model name is 'openai/gpt-oss-120b', we use the Groq 
    infrastructure if the user provided a Groq API key.
    """
    # If the user specifically wants the openai/ prefix but uses Groq API,
    # we might need to strip or map it, but let's try passing it directly
    # as requested.
    model_name = settings.llm_model
    
    # Common Groq models for fallback if needed:
    # - llama-3.1-70b-versatile
    # - llama-3.1-8b-instant
    # - mixtral-8x7b-32768
    
    print(f"🧠 Initializing LLM: {model_name}")
    
    return Groq(
        model=model_name,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature
    )
