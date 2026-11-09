import argparse
import logging
import os
import sys

# Let MPS run ops it doesn't implement on the CPU instead of failing.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dotenv import load_dotenv
import librosa
from tqdm import tqdm
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.device import (
    accelerator_hint,
    describe_device,
    detect_device,
    free_memory_gb,
    is_out_of_memory,
    release_memory,
)
from src.logger_config import configure_logging
from src.utils import convert_to_wav, write_transcript

DEFAULT_MODEL = "openai/whisper-large-v3"
CPU = torch.device("cpu")

# Rough per-sample activation memory for whisper-large in fp16 (GB).
# Only used to pick a starting batch size; OOM errors shrink it further.
GB_PER_BATCH_ITEM = 0.75
MAX_BATCH_SIZE = 16
DEFAULT_ACCELERATOR_BATCH_SIZE = 4


def initial_batch_size(device: torch.device) -> int:
    if device.type == "cpu":
        return 1
    free_gb = free_memory_gb(device)
    if free_gb is None:
        return DEFAULT_ACCELERATOR_BATCH_SIZE
    batch = 1
    while batch * 2 <= MAX_BATCH_SIZE and (batch * 2) * GB_PER_BATCH_ITEM < free_gb - 1:
        batch *= 2
    logging.info(f"{free_gb:.1f} GB free on {device.type}, starting batch size {batch}")
    return batch


def move_model(model, device: torch.device):
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    model.to(device=device, dtype=dtype)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count())
    logging.info(f"Model on {describe_device(device)} ({dtype})")
    return dtype


def transcribe(
    model, processor, audio_path: str, device: torch.device, batch_size: int
):
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=model.dtype,
        device=device,
        chunk_length_s=30,  # Whisper's native window; smaller values don't save memory
        batch_size=batch_size,
        return_timestamps=True,
    )
    generate_kwargs = {
        # "language": "Ukrainian",
    }  # TODO: Check if language setting is stil needed
    with tqdm(
        total=100, desc=f"Transcribing ({device.type}, batch {batch_size})", unit="%"
    ) as pbar:
        result = pipe(audio_path, generate_kwargs=generate_kwargs)
        pbar.update(100)
    return result


def transcribe_with_fallback(model, processor, audio_path: str, device: torch.device):
    """Run on the accelerator, halving the batch size on OOM, then fall back to CPU."""
    if device.type != "cpu":
        try:
            move_model(model, device)
        except Exception as error:
            logging.warning(
                f"Could not load model on {device.type} ({error}). Falling back to CPU..."
            )
            if hint := accelerator_hint(error):
                logging.warning(hint)
            release_memory(device)
            device = CPU
            move_model(model, device)

    batch_size = initial_batch_size(device)
    while True:
        try:
            return transcribe(model, processor, audio_path, device, batch_size)
        except (RuntimeError, NotImplementedError) as error:
            if device.type == "cpu":
                raise
            release_memory(device)
            if is_out_of_memory(error) and batch_size > 1:
                batch_size //= 2
                logging.warning(
                    f"{device.type} out of memory. Retrying with batch size {batch_size}..."
                )
                continue
            reason = "out of memory" if is_out_of_memory(error) else f"failed: {error}"
            logging.warning(f"{device.type} {reason}. Falling back to CPU...")
            if hint := accelerator_hint(error):
                logging.warning(hint)
            device = CPU
            move_model(model, device)
            release_memory(device)
            batch_size = initial_batch_size(device)


def create_transcription(audio_path: str):
    file_name = os.path.basename(audio_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Input file not found: {audio_path}")

    if file_name.rsplit(".", 1)[-1].lower() != "wav":
        logging.info(f"Converting {file_name} to WAV format...")
        audio_path = convert_to_wav(audio_path)
        file_name = os.path.basename(audio_path)
        logging.info(f"Conversion completed: {file_name}")

    model_id = os.getenv("MODEL_NAME") or DEFAULT_MODEL
    device = detect_device(os.getenv("DEVICE"))
    logging.info(f"Model: {model_id}")
    logging.info(f"Detected device: {describe_device(device)}")

    # Load on CPU first; transcribe_with_fallback moves it to the accelerator.
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=torch.float32 if device.type == "cpu" else torch.float16,
        use_safetensors=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    duration = librosa.get_duration(path=audio_path)
    logging.info(f"Audio duration: {duration:.2f} seconds")
    logging.info("Starting transcription...")

    result = transcribe_with_fallback(model, processor, audio_path, device)

    logging.info("Transcription completed.")
    write_transcript(result, file_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio/video file with Whisper."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=os.getenv("AUDIO_PATH"),
        help="Path to the audio file (or set AUDIO_PATH)",
    )
    args = parser.parse_args()
    if not args.audio:
        parser.error("no audio file given: pass a path or set AUDIO_PATH")
    return args


if __name__ == "__main__":
    load_dotenv()
    debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"
    configure_logging(debug_mode)

    try:
        create_transcription(parse_args().audio)
    except FileNotFoundError as error:
        logging.error(error)
        sys.exit(1)
