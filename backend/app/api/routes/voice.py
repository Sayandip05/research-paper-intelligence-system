"""
🎤 Voice Query API — Sarvam AI Speech-to-Text + RAG

Accepts audio, transcribes via Sarvam STT, then queries the RAG engine.
"""

import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from langfuse.decorators import observe
from sarvamai import SarvamAI

from app.config import get_settings
from app.services.query_engine import get_query_engine
from app.services.langfuse_utils import flush_langfuse

settings = get_settings()
router = APIRouter()


@router.post("/query/voice")
@observe(name="Voice_Query")
async def voice_query(
    audio: UploadFile = File(...),
    search_mode: str = Form("hybrid"),
    similarity_top_k: int = Form(5),
):
    """
    🎤 Voice Query Endpoint

    1. Receive audio file (WAV/MP3/WebM)
    2. Transcribe with Sarvam AI STT (saaras:v3)
    3. Query RAG engine with transcribed text
    4. Return transcription + answer + sources + images
    """
    if not settings.sarvam_api_key:
        raise HTTPException(
            status_code=503,
            detail="Sarvam AI API key not configured. Add SARVAM_API_KEY to .env"
        )

    # ── 1. Save audio to temp file ────────────────────────────────
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {e}")

    # ── 2. Transcribe with Sarvam AI ──────────────────────────────
    try:
        client = SarvamAI(api_subscription_key=settings.sarvam_api_key)

        with open(tmp_path, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=f,
                model="saaras:v3",
                mode="transcribe",
            )

        transcribed_text = response.transcript
        if not transcribed_text or not transcribed_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not transcribe audio. Try speaking more clearly."
            )

        print(f"🎤 Transcribed: {transcribed_text}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sarvam STT failed: {e}")
    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except:
            pass

    # ── 3. Query RAG engine ───────────────────────────────────────
    try:
        query_engine = get_query_engine()
        result = query_engine.query(
            question=transcribed_text,
            similarity_top_k=similarity_top_k,
            search_mode=search_mode,
        )

        flush_langfuse()

        return {
            "transcribed_text": transcribed_text,
            "question": result["question"],
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "images": result.get("images", []),
            "num_sources": result.get("num_sources", 0),
            "search_mode": search_mode,
        }

    except Exception as e:
        import traceback
        print(f"❌ Voice Query Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
