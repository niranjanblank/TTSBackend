from fastapi import FastAPI
from contextlib import asynccontextmanager
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
import numpy as np
import io
from fastapi.responses import StreamingResponse

# dictionary to store assets of the tts model
assets = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # initialize the tts synthesizer with speecht5
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
    # loading a dataset with speaker embeddings
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    # select a specific speaker embedding and format it for use
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # store initialized assets of tts in the assets dictionary
    assets["synthesiser"] = synthesiser
    assets["embeddings_dataset"] = embeddings_dataset
    assets["speaker_embedding"] = speaker_embedding

    yield

    # clean up the assets after the application shuts down
    assets.clear()

# initialize fastAPI app
app = FastAPI(lifespan=lifespan)



@app.get("/")
def hello(text):
    # simple test endpoint
    return "This is a tts application"

# generates speech from text provided and returns an audio clip
@app.get("/generate")
def generate(text: str):
    # generate speech form text using the assets and the text provided from user
    speech = assets["synthesiser"](text,
                         forward_params={"speaker_embeddings": assets["speaker_embedding"]})
    audio_bytes = speech["audio"]
    samplerate = speech["sampling_rate"]

    # Convert audio array to bytes
    # creates in-memory buffer to store audio
    buffer = io.BytesIO()
    # writes the audio data to the buffer as a wav file
    sf.write(buffer, audio_bytes, samplerate, format='WAV')
    # resets the buffer's cursor to the beginning
    buffer.seek(0)

    # return the audio file as a streaming response
    return StreamingResponse(buffer, media_type="audio/wav")