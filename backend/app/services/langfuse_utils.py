"""
Langfuse Observability — Central tracing utility

Provides a shared Langfuse client and the @observe decorator
for deep tracing across the entire system.
"""
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from app.config import get_settings

settings = get_settings()

# ── Shared Langfuse client ─────────────────────────────────────
_langfuse_client = None


def get_langfuse() -> Langfuse:
    """Get singleton Langfuse client"""
    global _langfuse_client
    if _langfuse_client is None and settings.enable_langfuse:
        try:
            _langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host
            )
            print("✅ Langfuse client initialized")
        except Exception as e:
            print(f"⚠️ Langfuse client init failed: {e}")
    return _langfuse_client


def flush_langfuse():
    """Flush pending events to Langfuse"""
    client = get_langfuse()
    if client:
        client.flush()
