import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.model_service import ModelService


async def main():
    print("loading model...", flush=True)
    await ModelService.initialize()
    print("loaded", ModelService.model is not None, ModelService.tokenizer is not None)
    print("generating...", flush=True)
    response = await ModelService.generate_response(
        session_id="test1",
        prompt="Say hello in one short sentence.",
        history=[],
        retrieved_docs=[],
        tools=None,
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
