from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    name: str
    description: str
    inputs: Dict[str, str]


class Message(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., min_length=1, max_length=20_000)


class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


class InferenceRequest(BaseModel):
    session_id: str
    prompt: str
    history: Optional[List[Message]] = None
    tools: Optional[List[ToolDefinition]] = None
    use_rag: bool = True
    stream: Optional[bool] = True


class InferenceResponse(BaseModel):
    session_id: str
    response: str
    metadata: Optional[Dict[str, Any]] = None


class ToolInvocation(BaseModel):
    tool_name: str
    args: Dict[str, Any]
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    model: str


class SpeechToTextResponse(BaseModel):
    transcript: str
    latency_ms: float


class LeadScoringRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000, description="Lead conversation text")
    voice_emotion: Optional[str] = Field(default=None, max_length=64)
    emotion_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    session_id: Optional[str] = Field(default=None, max_length=256)
    history: Optional[List[Message]] = Field(default=None, max_length=100)
    metadata: Optional[Dict[str, Any]] = None
    crm_context: Optional[Dict[str, Any]] = None
    business_docs: Optional[List[str]] = Field(default=None, max_length=50)


class VoiceEmotionResponse(BaseModel):
    voice_emotion: str
    confidence: float
    probabilities: Dict[str, float]
    model: str
    inference_time_ms: float
    memory_mb: Optional[float] = None


class LeadScoringResponse(BaseModel):
    text: str
    scores: Dict[str, float]
    icp_score: Optional[float] = None
    intent_score: Optional[float] = None
    engagement_score: Optional[float] = None
    qualification_score: Optional[float] = None
    buying_signal_score: Optional[float] = None
    relationship_score: Optional[float] = None
    bant_score: Optional[float] = None
    bant_scores: Optional[Dict[str, float]] = None
    bant_category: Optional[str] = None
    bant_recommendation: Optional[str] = None
    meddic_score: Optional[float] = None
    meddic_scores: Optional[Dict[str, float]] = None
    meddic_category: Optional[str] = None
    meddic_recommendation: Optional[str] = None
    conversion_probability: float
    conversion_probability_percent: Optional[float] = None
    xgboost_probability: Optional[float] = None
    xgboost_probability_percent: Optional[float] = None
    conversion_category: str
    lead_score: Optional[float] = None
    lead_category: Optional[str] = None
    recommendation: Optional[str] = None
    sentiment: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    engineered_features: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    model: str
    probabilities: Dict[str, float]
    shap_available: Optional[bool] = None


class AudioSentimentResponse(BaseModel):
    transcript: str
    text_sentiment: SentimentResponse
    voice_emotion: VoiceEmotionResponse
    final_emotion: str
    confidence: float
    reason: str
    latency_ms: float
    lead_scoring: Optional[LeadScoringResponse] = None


class VoiceChatResponse(AudioSentimentResponse):
    llm_response: str
    tts: Dict[str, Any]
