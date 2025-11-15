import json
from pathlib import Path
import logging
import os
import uuid

from pydub import AudioSegment


def convert_to_wav(file_path: str, output_path: str = None) -> str:
    """
    Convert any audio or video file to WAV format.

    Args:
        file_path: Path to the input audio/video file
        output_path: Optional path for the output WAV file.

    Returns:
        str: Path to the converted WAV file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if output_path is None:
        output_path = file_path.rsplit(".", 1)[0] + ".wav"

    # Load audio/video file (pydub auto-detects format)
    audio = AudioSegment.from_file(file_path)

    # Export as WAV
    audio.export(output_path, format="wav")

    print(f"File {file_path} converted to {output_path}!")
    return output_path


def write_transcript(transcript: dict, file_name: str) -> None:
    output_file = "data"+  file_name.rsplit(".", 1)[0] + uuid.uuid4() +".json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)

    logging.info(f"Transcript written to {output_file}")


def write_transcript(transcript: dict, file_name: str) -> None:
    """
    Write transcript to a JSON file with a unique identifier.
    
    Args:
        transcript: Dictionary containing transcript data
        file_name: Original file name to derive output name from
        
    Returns:
        Path object of the written file
    """
    base_name = Path(file_name).stem
    output_file = Path("data") / f"{base_name}_{uuid.uuid4().hex[:5]}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
    
    logging.info(f"Transcript written to {output_file}")
