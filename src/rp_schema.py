INPUT_VALIDATIONS = {
    "audio_files": {
        "type": list,
        "required": True        # URLs to audio files
    },
    "language": {
        "type": str,
        "required": False,
        "default": None         # Optional: Two letter language code e.g. "en"
    },
    "task": {
        "type": str,
        "required": False,
        "default": "transcribe" # Optional: "transcribe" or "translate"
    },
    "diarization": {
        "type": bool,
        "required": False,
        "default": True         # Optional: Run diarization
    },
    "min_speakers": {
        "type": int,
        "required": False,
        "default": None         # Optional: Min speakers
    },
    "max_speakers": {
        "type": int,
        "required": False,
        "default": None         # Optional: Max speakers
    }
}