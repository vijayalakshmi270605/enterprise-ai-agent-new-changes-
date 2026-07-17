# Enterprise AI Voice Emotion Assistant

A FastAPI backend for the Follei Enterprise AI Voice Assistant. It combines a fine-tuned LoRA LLM, RAG, Redis memory, Whisper speech-to-text, TF-IDF Multinomial Naive Bayes text sentiment, voice emotion recognition, weighted emotion fusion, Prometheus monitoring, and a Kokoro TTS handoff.

## Architecture

- `app/` - production API and service code
- `app/speech/` - noise reduction, VAD, silence removal, mono conversion, 16kHz resampling, normalization, Whisper STT
- `app/nlp/` - complete text preprocessing and TF-IDF Multinomial Naive Bayes sentiment
- `app/feature_extraction/` - MFCC, delta MFCC, Mel, pitch, energy, spectral, chroma, ZCR, RMS, tempo
- `app/emotion/` - CNN-MFCC voice emotion inference with an acoustic fallback
- `app/fusion/` - weighted text + voice emotion fusion
- `app/api/` - FastAPI routes and schemas
- `training/` - fine-tuning dataset and trainer pipelines
- `training/train_text.py` - trains TF-IDF + MultinomialNB from Hugging Face
- `training/train_audio.py` - trains CNN using MFCC tensors
- `training/train_wav2vec.py` - fine-tunes Wav2Vec2 for audio classification
- `training/evaluate.py` - metrics and plots
- `training/predict.py` - local inference CLI
- `deployment/` - Docker deployment manifests
- `scripts/` - run scripts for training and serving

## Production Flow

Microphone audio -> noise reduction -> VAD -> silence removal -> mono conversion -> 16kHz resampling -> normalization -> Whisper Base/Small -> transcript -> NLP preprocessing -> TF-IDF MultinomialNB -> audio feature extraction -> CNN-MFCC or Wav2Vec2 emotion model -> weighted fusion -> final emotion -> fine-tuned 1B LLM -> Kokoro TTS.

## Setup

1. Create a Python environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Configure environment variables in `.env`:
   - `MODEL_BASE` (e.g. `Qwen2.5-1.5B-Instruct` or your Hugging Face path)
   - `LORA_OUTPUT_DIR`
   - `CHROMA_PERSIST_DIR`
   - `REDIS_URL`
   - `EMBEDDING_MODEL`
   - `TEXT_SENTIMENT_MODEL_PATH`
   - `VOICE_EMOTION_MODEL_PATH`
   - `WAV2VEC_EMOTION_MODEL_PATH`
   - `WHISPER_MODEL=base` or `small`
   - `API_HOST`, `API_PORT`

3. Start Redis and ChromaDB services.

4. Train or load adapters and run the API.

## Dataset

Only this dataset is used for the production sentiment/emotion pipeline:

`DynamicSuperb/Sentiment_Analysis_SLUE-VoxCeleb`

The scripts load it directly with `datasets.load_dataset` and create deterministic 80% train, 10% validation, and 10% test splits. No custom dataset is created.

## NLP Sentiment

The text path uses Unicode normalization, HTML removal, URL removal, email removal, contraction expansion, lowercase conversion, number removal, special character cleanup, extra space cleanup, sentence tokenization, word tokenization, stopword removal, lemmatization, negation handling, TF-IDF vectorization, and Multinomial Naive Bayes.

Multinomial Naive Bayes is a strong production baseline for text sentiment because sparse non-negative TF-IDF features match the multinomial event model, training is fast, inference is CPU-friendly, and `predict_proba` gives positive, negative, and neutral confidence scores.

Train text sentiment from Hugging Face:

```bash
python training/train_text.py
```

The older JSON Naive Bayes trainer remains available for compatibility:

```bash
python training/train_sentiment.py
```

## Speech Emotion

Speech preprocessing includes:

- Voice Activity Detection using frame RMS energy
- Noise reduction with `noisereduce` when installed
- Silence removal
- Mono conversion
- 16kHz resampling with `librosa`
- Peak normalization
- Augmentations: time shift, pitch shift, background noise

Audio features:

- MFCC: DCT of log Mel filterbank energies
- Delta MFCC: first temporal derivative of MFCC
- Delta-Delta MFCC: second temporal derivative
- Mel Spectrogram: short-time power spectrum on the Mel scale
- Pitch: estimated fundamental frequency
- Energy: `sum(x[n]^2) / N`
- Spectral Centroid: `sum(f * magnitude) / sum(magnitude)`
- Spectral Contrast: peak-valley contrast per band
- Chroma: octave-invariant 12 pitch-class energy
- Zero Crossing Rate: sign-change rate
- RMS Energy: `sqrt(mean(x[n]^2))`
- Tempo: BPM estimate from onset strength

Train CNN-MFCC:

```bash
python training/train_audio.py --epochs 10 --batch-size 16
```

Fine-tune Wav2Vec2:

```bash
python training/train_wav2vec.py --base-model facebook/wav2vec2-base --epochs 3
```

Compare both models using accuracy, precision, recall, F1, confusion matrix, ROC, average inference time, and memory usage. In general, Wav2Vec2 is expected to win on accuracy and robustness when GPU and enough data are available. CNN-MFCC is recommended when CPU latency, memory, and operational simplicity matter more.

## Emotion Fusion

The fusion engine combines text sentiment and voice emotion with default weights:

- Text sentiment: `0.45`
- Voice emotion: `0.55`

Examples:

- Text positive + voice happy -> happy
- Text positive + voice angry -> frustrated
- Text neutral + voice sad -> sad
- Text negative + voice angry -> angry

Contradiction rules handle mixed signals, and the confidence is the weighted sum of the selected text and voice probabilities.

## See Output

Run the real-time client workflow in three terminals:

```bash
# Terminal 1: API server
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload

# Terminal 2: live transcript + assistant answers
.\.venv\Scripts\python.exe -m scripts.live_assistant_terminal --session-id demo

# Terminal 3: live ICP, intent, engagement, qualification,
# buying signal, and relationship scores
.\.venv\Scripts\python.exe -m scripts.live_score_monitor --session-id demo
```

For microphone mode, install `sounddevice` and run:

```bash
.\.venv\Scripts\python.exe -m scripts.live_assistant_terminal --session-id demo --mode mic
```

Mic mode listens until you pause, up to 10 seconds by default. For longer call-style sentences:

```bash
.\.venv\Scripts\python.exe -m scripts.live_assistant_terminal --session-id demo --mode mic --chunk-seconds 15 --silence-seconds 1.4
```

Typed mode is the fallback when microphone capture is not installed.

Text sentiment:

```bash
python training/predict.py --text "I am very happy with your service"
```

Export a real audio sample from the required Hugging Face dataset:

```bash
python scripts/export_real_audio_sample.py --index 0 --output outputs/audio_samples/slue_sample.wav
```

Run the complete local audio pipeline:

```bash
python training/predict.py --audio outputs/audio_samples/slue_sample.wav
```

Create a real local TTS WAV response:

```bash
python training/predict.py --text "I am very happy with your service" --tts --response-text "I am glad to help you."
```

The TTS file is saved under `outputs/tts/`. If Kokoro is configured, Kokoro is used. Otherwise the project uses the offline `pyttsx3` fallback.

## APIs

Analyze text:

```bash
curl -X POST http://localhost:8000/api/v1/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"The support team was excellent"}'
```

Speech-to-text:

```bash
curl -X POST http://localhost:8000/api/v1/speech-to-text \
  -F "file=@outputs/audio_samples/slue_sample.wav"
```

Voice emotion:

```bash
curl -X POST http://localhost:8000/api/v1/emotion \
  -F "file=@outputs/audio_samples/slue_sample.wav"
```

Audio sentiment and fusion:

```bash
curl -X POST http://localhost:8000/api/v1/audio-sentiment \
  -F "file=@outputs/audio_samples/slue_sample.wav"
```

Voice chat:

```bash
curl -X POST http://localhost:8000/api/v1/voice-chat \
  -F "file=@outputs/audio_samples/slue_sample.wav" \
  -F "session_id=demo"
```

Prometheus metrics:

```bash
curl http://localhost:8000/metrics
```

`/audio-sentiment` and `/voice-chat` return transcript, text sentiment, voice emotion, confidence, final emotion, reason, latency, and the LLM response where applicable.

## Evaluation and Visualization

Generate metrics and plots:

```bash
python training/evaluate.py --task text --output-dir reports/evaluation
```

Generated artifacts include confusion matrix, ROC, precision-recall, distribution plots, and JSON metrics. CNN training stores loss and accuracy history in the checkpoint for loss/accuracy curve plotting.

## Optimization

- GPU: use CUDA, mixed precision for Wav2Vec2/LLM, gradient checkpointing, larger batches, pinned dataloaders, and model warmup.
- CPU fallback: CNN-MFCC and MultinomialNB remain lightweight; Whisper Base is preferred over Small on CPU.
- Serving: load models once at startup, avoid per-request model creation, cap audio duration, validate upload size, and record latency/memory metrics.
- Reliability: keep Redis optional, log degraded services, expose Prometheus metrics, and return structured JSON errors.
- Security: restrict upload types and sizes at the gateway, do not execute uploaded content, keep secrets in `.env`, and pin model versions for production.

## Deployment

Build and run:

```bash
docker build -f deployment/Dockerfile -t enterprise-ai-voice-assistant .
docker compose -f deployment/docker-compose.yml up --build
```

## Notes

This repository includes:
- LoRA fine-tuning pipeline
- Instruction-style dataset creation
- Tokenization pipeline
- RAG retrieval service
- Redis memory store
- Tool-calling architecture
- Streaming REST API with monitoring
- TF-IDF Multinomial Naive Bayes sentiment training and prediction API
- Whisper speech-to-text
- Voice emotion recognition
- Weighted emotion fusion
