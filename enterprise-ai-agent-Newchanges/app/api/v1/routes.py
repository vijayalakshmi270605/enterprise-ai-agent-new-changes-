import json
import asyncio
import logging
import time
from typing import AsyncGenerator
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from prometheus_client import Counter
from app.api.v1.schemas import (
    AudioSentimentResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    LeadScoringRequest,
    LeadScoringResponse,
    SentimentRequest,
    SentimentResponse,
    SpeechToTextResponse,
    VoiceChatResponse,
    VoiceEmotionResponse,
)
from app.fusion.engine import EmotionFusionEngine
from app.speech.preprocessing import SpeechPreprocessor
from app.speech.whisper_service import WhisperService
from app.services.rag_service import RAGService
from app.services.memory_service import MemoryService
from app.services.tool_service import ToolService
from app.services.model_service import ModelService
from app.services.quick_assistant_service import QuickAssistantService
from app.services.writing_service import WritingService
from app.services.sentiment_service import SentimentService
from app.services.lead_scoring_service import LeadScoringService
from app.services.tts_service import TTSService
from app.services.voice_emotion_service import VoiceEmotionService
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

request_counter = Counter("assistant_requests_total", "Total API requests")
error_counter = Counter("assistant_errors_total", "Total API errors")
speech_preprocessor = SpeechPreprocessor(
    target_sr=settings.audio_sample_rate,
    enable_noise_reduction=settings.enable_noise_reduction,
    max_audio_seconds=settings.max_voice_audio_seconds,
)
whisper_service = WhisperService(model_name=settings.whisper_model, language=settings.whisper_language)
fusion_engine = EmotionFusionEngine()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", version="1.1.0")


@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(payload: SentimentRequest):
    request_counter.inc()
    try:
        return SentimentResponse(**SentimentService.analyze(payload.text))
    except (ValueError, RuntimeError) as exc:
        error_counter.inc()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        error_counter.inc()
        logger.exception("Sentiment analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


async def _read_audio_upload(request: Request) -> tuple[bytes, dict]:
    try:
        form = await request.form()
    except (AssertionError, RuntimeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="python-multipart is required for multipart audio uploads",
        ) from exc
    file = form.get("file")
    if file is None or not hasattr(file, "read"):
        raise HTTPException(status_code=400, detail="multipart field 'file' is required")
    return await file.read(), dict(form)


async def _preprocess_upload(request: Request):
    audio_bytes, form = await _read_audio_upload(request)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="uploaded audio file is empty")
    audio, sample_rate = speech_preprocessor.preprocess(audio_bytes)
    return audio, sample_rate, form


async def _preprocess_file_upload(file: UploadFile):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="uploaded audio file is empty")
    audio, sample_rate = speech_preprocessor.preprocess(audio_bytes)
    return audio, sample_rate


def _voice_emotion_response(result) -> VoiceEmotionResponse:
    return VoiceEmotionResponse(
        voice_emotion=result.emotion,
        confidence=result.confidence,
        probabilities=result.probabilities,
        model=result.model,
        inference_time_ms=result.inference_time_ms,
        memory_mb=result.memory_mb,
    )


def _analyze_transcript(transcript: str) -> dict:
    if transcript.strip():
        return SentimentService.analyze(transcript)
    return {
        "text": transcript,
        "sentiment": "neutral",
        "confidence": 1.0,
        "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
        "model": "empty_transcript_fallback",
    }


@router.post("/lead-scoring", response_model=LeadScoringResponse)
async def lead_scoring(payload: LeadScoringRequest):
    request_counter.inc()
    try:
        result = LeadScoringService.score(
            payload.text,
            voice_emotion=payload.voice_emotion,
            emotion_confidence=payload.emotion_confidence,
            history=[{"role": item.role, "content": item.content} for item in (payload.history or [])],
            metadata=payload.metadata,
            crm_context=payload.crm_context,
            business_docs=payload.business_docs,
        )
        if payload.session_id:
            LeadScoringService.update_customer_profile(
                payload.session_id,
                profile=payload.metadata,
                metadata=payload.metadata,
            )
            LeadScoringService.store_lead_history(
                payload.session_id,
                {
                    "session_id": payload.session_id,
                    "lead_score": result.get("lead_score"),
                    "lead_category": result.get("lead_category"),
                    "recommendation": result.get("recommendation"),
                },
                text=payload.text,
            )
        return LeadScoringResponse(**result)
    except Exception as exc:
        error_counter.inc()
        logger.exception("Lead scoring failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(file: UploadFile = File(...)):
    request_counter.inc()
    start = time.perf_counter()
    try:
        audio, sample_rate = await _preprocess_file_upload(file)
        transcript = whisper_service.transcribe(audio, sample_rate)
        return SpeechToTextResponse(
            transcript=transcript,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
        )
    except HTTPException:
        raise
    except Exception as exc:
        error_counter.inc()
        logger.exception("Speech-to-text failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/emotion", response_model=VoiceEmotionResponse)
async def recognize_emotion(file: UploadFile = File(...)):
    request_counter.inc()
    try:
        audio, _ = await _preprocess_file_upload(file)
        return _voice_emotion_response(VoiceEmotionService.predict(audio))
    except HTTPException:
        raise
    except Exception as exc:
        error_counter.inc()
        logger.exception("Voice emotion recognition failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/audio-sentiment", response_model=AudioSentimentResponse)
async def audio_sentiment(file: UploadFile = File(...)):
    request_counter.inc()
    start = time.perf_counter()
    try:
        audio, sample_rate = await _preprocess_file_upload(file)
        transcript = whisper_service.transcribe(audio, sample_rate)
        text_sentiment = _analyze_transcript(transcript)
        voice_emotion = VoiceEmotionService.predict(audio)
        fused = fusion_engine.fuse(
            text_sentiment=text_sentiment["sentiment"],
            text_confidence=float(text_sentiment["confidence"]),
            voice_emotion=voice_emotion.emotion,
            voice_confidence=voice_emotion.confidence,
        )
        lead_scoring = LeadScoringService.score(
            transcript,
            voice_emotion=voice_emotion.emotion,
            emotion_confidence=voice_emotion.confidence,
        )
        return AudioSentimentResponse(
            transcript=transcript,
            text_sentiment=SentimentResponse(**text_sentiment),
            voice_emotion=_voice_emotion_response(voice_emotion),
            final_emotion=fused.final_emotion,
            confidence=fused.confidence,
            reason=fused.reason,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            lead_scoring=LeadScoringResponse(**lead_scoring),
        )
    except HTTPException:
        raise
    except Exception as exc:
        error_counter.inc()
        logger.exception("Audio sentiment failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/voice-chat", response_model=VoiceChatResponse)
async def voice_chat(
    file: UploadFile = File(...),
    session_id: str = Form(default="voice-session"),
    use_rag: bool = Form(default=True),
    tts: bool = Form(default=False),
):
    request_counter.inc()
    start = time.perf_counter()
    try:
        audio, sample_rate = await _preprocess_file_upload(file)
        use_server_tts = bool(tts or settings.enable_server_tts)
        transcript = whisper_service.transcribe(audio, sample_rate)
        text_sentiment = _analyze_transcript(transcript)
        voice_emotion = VoiceEmotionService.predict(audio)
        fused = fusion_engine.fuse(
            text_sentiment=text_sentiment["sentiment"],
            text_confidence=float(text_sentiment["confidence"]),
            voice_emotion=voice_emotion.emotion,
            voice_confidence=voice_emotion.confidence,
        )
        retrieved_docs = await RAGService.retrieve(transcript) if use_rag and transcript else []
        emotion_context = (
            f"Detected user emotion: {fused.final_emotion} "
            f"(confidence {fused.confidence}). Reply naturally, directly, and briefly in 2 to 4 sentences."
        )
        llm_response = await ModelService.generate_response(
            session_id=session_id,
            prompt=transcript,
            history=[],
            retrieved_docs=[emotion_context] + retrieved_docs,
            tools=None,
        )
        lead_scoring = LeadScoringService.score(
            transcript,
            voice_emotion=voice_emotion.emotion,
            emotion_confidence=voice_emotion.confidence,
        )
        await MemoryService.append_user_message(session_id, transcript)
        await MemoryService.append_assistant_message(session_id, llm_response)
        return VoiceChatResponse(
            transcript=transcript,
            text_sentiment=SentimentResponse(**text_sentiment),
            voice_emotion=_voice_emotion_response(voice_emotion),
            final_emotion=fused.final_emotion,
            confidence=fused.confidence,
            reason=fused.reason,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            lead_scoring=LeadScoringResponse(**lead_scoring),
            llm_response=llm_response,
            tts=(
                TTSService.synthesize(llm_response, output_dir=settings.tts_output_dir).to_dict()
                if use_server_tts
                else {"engine": "browser", "status": "skipped", "text": llm_response}
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        error_counter.inc()
        logger.exception("Voice chat failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/infer")
async def infer(payload: InferenceRequest, request: Request):
    request_counter.inc()
    session_id = payload.session_id
    prompt = payload.prompt
    history = payload.history or []

    try:
        await MemoryService.append_user_message(session_id, prompt)
        lead_result = LeadScoringService.score(
            prompt,
            history=[{"role": message.role, "content": message.content} for message in history],
            metadata={"session_id": session_id},
        )
        if session_id:
            LeadScoringService.update_customer_profile(session_id, metadata={"session_id": session_id})
            LeadScoringService.store_lead_history(
                session_id,
                {
                    "session_id": session_id,
                    "lead_score": lead_result.get("lead_score"),
                    "lead_category": lead_result.get("lead_category"),
                    "recommendation": lead_result.get("recommendation"),
                },
                text=prompt,
            )

        if settings.enable_fast_assistant:
            quick_response = QuickAssistantService.maybe_answer(prompt)
            if quick_response:
                await MemoryService.append_assistant_message(session_id, quick_response)
                return InferenceResponse(
                    session_id=session_id,
                    response=quick_response,
                    metadata={
                        "handled_by": "fast_assistant",
                        "target_latency": "under_2_seconds",
                        "lead_score": lead_result.get("lead_score"),
                        "lead_category": lead_result.get("lead_category"),
                        "recommendation": lead_result.get("recommendation"),
                    },
                )
            if settings.strict_fast_mode:
                result = QuickAssistantService.timeout_fallback(prompt)
                await MemoryService.append_assistant_message(session_id, result)
                return InferenceResponse(
                    session_id=session_id,
                    response=result,
                    metadata={
                        "handled_by": "strict_fast_fallback",
                        "target_latency": "under_2_seconds",
                        "lead_score": lead_result.get("lead_score"),
                        "lead_category": lead_result.get("lead_category"),
                        "recommendation": lead_result.get("recommendation"),
                    },
                )

        if payload.use_rag:
            retrieved_docs = await RAGService.retrieve(prompt)
        else:
            retrieved_docs = []

        writing_result = WritingService.maybe_handle(prompt)
        if writing_result:
            await MemoryService.append_assistant_message(session_id, writing_result)
            return InferenceResponse(
                session_id=session_id,
                response=writing_result,
                metadata={
                    "retrieved_docs": len(retrieved_docs),
                    "handled_by": "writing_template",
                    "lead_score": lead_result.get("lead_score"),
                    "lead_category": lead_result.get("lead_category"),
                    "recommendation": lead_result.get("recommendation"),
                },
            )

        if payload.stream and settings.enable_streaming:
            async def stream_response() -> AsyncGenerator[bytes, None]:
                async for chunk in ModelService.stream_response(
                    session_id=session_id,
                    prompt=prompt,
                    history=history,
                    retrieved_docs=retrieved_docs,
                    tools=payload.tools,
                ):
                    yield chunk

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
            )

        try:
            result = await asyncio.wait_for(
                ModelService.generate_response(
                    session_id=session_id,
                    prompt=prompt,
                    history=history,
                    retrieved_docs=retrieved_docs,
                    tools=payload.tools,
                ),
                timeout=settings.llm_timeout_seconds,
            )
        except asyncio.TimeoutError:
            result = QuickAssistantService.timeout_fallback(prompt)

        await MemoryService.append_assistant_message(session_id, result)
        return InferenceResponse(
            session_id=session_id,
            response=result,
            metadata={
                "retrieved_docs": len(retrieved_docs),
                "lead_score": lead_result.get("lead_score"),
                "lead_category": lead_result.get("lead_category"),
                "recommendation": lead_result.get("recommendation"),
            },
        )
    except Exception as exc:
        error_counter.inc()
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/tool-callback")
async def tool_callback(payload: dict):
    tool_name = payload.get("tool_name")
    args = payload.get("args", {})
    result = await ToolService.execute(tool_name, args)
    return {"status": "ok", "result": result}


# ============ ADVANCED METRICS ENDPOINTS ============

@router.post("/advanced-metrics")
async def get_advanced_metrics(payload: LeadScoringRequest):
    """
    Get comprehensive advanced metrics including:
    - Economic value (ROI potential, deal size)
    - Economic buyer identification
    - Decision criteria extraction
    - Decision process analysis with recommended followups
    - Pain point identification with cosine similarity clustering
    """
    request_counter.inc()
    try:
        result = LeadScoringService.get_advanced_metrics(
            payload.text,
            history=[{"role": item.role, "content": item.content} for item in (payload.history or [])],
            metadata=payload.metadata,
            crm_context=payload.crm_context,
        )
        return result
    except Exception as exc:
        error_counter.inc()
        logger.exception("Advanced metrics generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/economic-value")
async def get_economic_value(payload: LeadScoringRequest):
    """Get quantifiable economic value score and ROI potential."""
    request_counter.inc()
    try:
        result = LeadScoringService.get_economic_value(
            payload.text,
            history=[{"role": item.role, "content": item.content} for item in (payload.history or [])],
            metadata=payload.metadata,
        )
        return result
    except Exception as exc:
        error_counter.inc()
        logger.exception("Economic value calculation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/economic-buyer")
async def get_economic_buyer(payload: LeadScoringRequest):
    """Identify economic buyer and decision authority level."""
    request_counter.inc()
    try:
        result = LeadScoringService.get_economic_buyer(
            payload.text,
            history=[{"role": item.role, "content": item.content} for item in (payload.history or [])],
            metadata=payload.metadata,
        )
        return result
    except Exception as exc:
        error_counter.inc()
        logger.exception("Economic buyer identification failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/decision-criteria")
async def get_decision_criteria(payload: LeadScoringRequest):
    """Extract and categorize decision criteria (must-have, nice-to-have)."""
    request_counter.inc()
    try:
        result = LeadScoringService.get_decision_criteria(
            payload.text,
            history=[{"role": item.role, "content": item.content} for item in (payload.history or [])],
        )
        return result
    except Exception as exc:
        error_counter.inc()
        logger.exception("Decision criteria extraction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/decision-process")
async def get_decision_process(payload: LeadScoringRequest):
    """
    Analyze decision process stage and recommended followups.
    Returns current stage, next steps, and timeline urgency.
    """
    request_counter.inc()
    try:
        result = LeadScoringService.get_decision_process(
            payload.text,
            history=[{"role": item.role, "content": item.content} for item in (payload.history or [])],
            metadata=payload.metadata,
        )
        return result
    except Exception as exc:
        error_counter.inc()
        logger.exception("Decision process analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/pain-points")
async def get_pain_points(payload: LeadScoringRequest):
    """
    Identify and analyze pain points.
    Uses cosine similarity for intelligent clustering of similar pain points.
    """
    request_counter.inc()
    try:
        result = LeadScoringService.get_pain_points(
            payload.text,
            history=[{"role": item.role, "content": item.content} for item in (payload.history or [])],
        )
        return result
    except Exception as exc:
        error_counter.inc()
        logger.exception("Pain point identification failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/compare-leads")
async def compare_leads(payload: dict):
    """
    Compare two leads using cosine similarity.
    Returns similarity score and matching/diverging aspects.
    """
    request_counter.inc()
    try:
        lead_1 = payload.get("lead_text_1", "")
        lead_2 = payload.get("lead_text_2", "")
        
        if not lead_1 or not lead_2:
            raise HTTPException(status_code=400, detail="Both lead_text_1 and lead_text_2 are required")
        
        result = LeadScoringService.compare_leads(lead_1, lead_2)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        error_counter.inc()
        logger.exception("Lead comparison failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/compare-pain-profiles")
async def compare_pain_profiles(payload: dict):
    """
    Compare pain point profiles between two leads.
    Uses cosine similarity for intelligent comparison.
    """
    request_counter.inc()
    try:
        pain_1 = payload.get("pain_points_1", [])
        pain_2 = payload.get("pain_points_2", [])
        
        if not pain_1 or not pain_2:
            raise HTTPException(status_code=400, detail="Both pain_points_1 and pain_points_2 are required")
        
        result = LeadScoringService.compare_pain_profiles(pain_1, pain_2)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        error_counter.inc()
        logger.exception("Pain profile comparison failed")
        raise HTTPException(status_code=500, detail=str(exc))
