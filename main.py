import json
import logging
import os

from dotenv import load_dotenv
import librosa
from tqdm import tqdm
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.logger_config import configure_logging
from src.utils import convert_to_wav, write_transcript


def create_transcription():
    # Input preparation
    audio_path = "Test01.wav"  # "Voice1.m4a"
    file_name = os.path.basename(audio_path)
    format = file_name.rsplit(".", 1)[1].lower()

    if format != "wav":
        logging.info(f"Converting {file_name} to WAV format...")
        audio_path = convert_to_wav(audio_path)
        file_name = os.path.basename(audio_path)
        logging.info(f"Conversion completed: {file_name}")

    # Model loading with device management
    model_id = "openai/whisper-large-v3"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logging.info(f"Initial device: {device}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch_dtype, use_safetensors=True
    )

    try:
        model.to(device)
        logging.info(f"Model loaded successfully on {device}")

    except torch.OutOfMemoryError:
        logging.warning(f"CUDA out of memory. Falling back to CPU...")
        device = "cpu"
        torch_dtype = torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=torch_dtype, use_safetensors=True
        )
        model.to(device)
        logging.info(f"Model loaded successfully on {device}")
        num_cores = os.cpu_count()
        torch.set_num_threads(num_cores)
        logging.info(f"Using {num_cores} CPU threads")

    # Pipeline setup
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        # batch_size=32,
        return_timestamps=True,
        dtype=torch_dtype,
        device=device,
        chunk_length_s=30,  # Process audio in 30-second chunks for better progress tracking
    )
    generate_kwargs = {
        # "language": "Ukrainian",
    }  # TODO: Check if language setting is stil needed

    # print(pipe, generate_kwargs)

    logging.info("Starting transcription...")

    # Get audio duration for progress bar
    duration = librosa.get_duration(path=audio_path)
    logging.info(f"Audio duration: {duration:.2f} seconds")

    # Transcribe with progress bar
    with tqdm(total=100, desc="Transcribing", unit="%") as pbar:
        result = pipe(audio_path, generate_kwargs=generate_kwargs)
        pbar.update(100)

    logging.info("Transcription completed.")

    # Save result to JSON file
    write_transcript(result, file_name)


if __name__ == "__main__":
    load_dotenv()
    debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"
    configure_logging(debug_mode)

    create_transcription()
