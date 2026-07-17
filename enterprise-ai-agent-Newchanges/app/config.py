import os
from pydantic_settings import BaseSettings # type: ignore
from pydantic import Field, ConfigDict # type: ignore


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    app_name: str = "Enterprise AI Assistant"
    environment: str = Field(default="production", env="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    model_base: str = Field(default="HuggingFaceTB/SmolLM2-360M-Instruct", env="MODEL_BASE")
    lora_output_dir: str = Field(default="./models/lora-360m", env="LORA_OUTPUT_DIR")
    adapter_name: str = Field(default="qwen-7b-lora-final", env="LORA_ADAPTER_NAME")
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", env="EMBEDDING_MODEL")
    chroma_persist_dir: str = Field(default="./chroma_store_bge", env="CHROMA_PERSIST_DIR")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    api_key: str = Field(default="", env="API_KEY")
    require_api_key: bool = Field(default=False, env="REQUIRE_API_KEY")
    cors_origins: str = Field(default="", env="CORS_ORIGINS")
    allowed_hosts: str = Field(default="*", env="ALLOWED_HOSTS")
    max_request_body_bytes: int = Field(default=1_000_000, env="MAX_REQUEST_BODY_BYTES")
    max_history: int = Field(default=20, env="MAX_HISTORY")
    max_tokens: int = Field(default=1024, env="MAX_TOKENS")
    max_new_tokens: int = Field(default=512, env="MAX_NEW_TOKENS")
    enable_fast_assistant: bool = Field(default=True, env="ENABLE_FAST_ASSISTANT")
    strict_fast_mode: bool = Field(default=False, env="STRICT_FAST_MODE")
    llm_timeout_seconds: float = Field(default=60.0, env="LLM_TIMEOUT_SECONDS")
    fine_tune_max_steps: int = Field(default=10, env="FINE_TUNE_MAX_STEPS")
    fine_tune_learning_rate: float = Field(default=2e-5, env="FINE_TUNE_LEARNING_RATE")
    sentiment_model_path: str = Field(
        default="./models/sentiment/naive_bayes.json",
        env="SENTIMENT_MODEL_PATH",
    )
    text_sentiment_model_path: str = Field(
        default="./models/sentiment/tfidf_naive_bayes.joblib",
        env="TEXT_SENTIMENT_MODEL_PATH",
    )
    sentiment_dataset_path: str = Field(
        default="./training/examples/sentiment_dataset.json",
        env="SENTIMENT_DATASET_PATH",
    )
    sentiment_alpha: float = Field(default=0.1, env="SENTIMENT_ALPHA")
    slue_voxceleb_dataset: str = Field(
        default="DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb",
        env="SLUE_VOXCELEB_DATASET",
    )
    voice_emotion_model_path: str = Field(
        default="./models/emotion/cnn_mfcc.pt",
        env="VOICE_EMOTION_MODEL_PATH",
    )
    wav2vec_emotion_model_path: str = Field(
        default="./models/emotion/wav2vec2",
        env="WAV2VEC_EMOTION_MODEL_PATH",
    )
    whisper_model: str = Field(default="base", env="WHISPER_MODEL")
    whisper_language: str = Field(default="en", env="WHISPER_LANGUAGE")
    warmup_whisper: bool = Field(default=True, env="WARMUP_WHISPER")
    audio_sample_rate: int = Field(default=16000, env="AUDIO_SAMPLE_RATE")
    enable_noise_reduction: bool = Field(default=False, env="ENABLE_NOISE_REDUCTION")
    max_voice_audio_seconds: float = Field(default=15.0, env="MAX_VOICE_AUDIO_SECONDS")
    enable_server_tts: bool = Field(default=False, env="ENABLE_SERVER_TTS")
    kokoro_model_path: str = Field(default="", env="KOKORO_MODEL_PATH")
    kokoro_voices_path: str = Field(default="", env="KOKORO_VOICES_PATH")
    kokoro_voice: str = Field(default="af_sarah", env="KOKORO_VOICE")
    tts_output_dir: str = Field(default="./outputs/tts", env="TTS_OUTPUT_DIR")
    temperature: float = Field(default=0.3, env="TEMPERATURE")
    top_p: float = Field(default=0.95, env="TOP_P")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    service_timeout: int = Field(default=60, env="SERVICE_TIMEOUT")
    enable_streaming: bool = Field(default=True, env="ENABLE_STREAMING")


settings = Settings()
