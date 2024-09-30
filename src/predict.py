import gc
import logging
import time
import torch
import whisperx

from typing import Optional


logger = logging.getLogger(__name__)


class Predictor:

    def __init__(self):
        pass

    def setup(self):
        pass

    def predict(
        self,
        audio_file_path: Optional[str],
        language: Optional[str],
        task: str,
        diarization: bool,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        hf_token: Optional[str] = None,
        batch_size: int = 8,
        compute_type: str = "int8",
    ):
        if audio_file_path is None:
            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sample_rate = 16000
        gpu_price_per_hour = 0.16
        gpu_price_per_sec = gpu_price_per_hour/(60*60)

        print(f"Device: {device}, audio file: {audio_file_path}, batch size: {batch_size}, compute type: {compute_type}")
        s_tot_time = time.time()

        # 1. Transcribe with original whisper (batched)
        s_time = time.time()
        model = whisperx.load_model(
            "medium", #large-v2
            device=device,
            compute_type=compute_type,
            language=language
        )
        print(f"Loaded model in: {time.time() - s_time} sec")

        # save model to local path (optional)
        # model_dir = "/path/"
        # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

        s_time = time.time()
        audio = whisperx.load_audio(audio_file_path, sr=sample_rate)
        audio_length_sec = len(audio)/sample_rate
        print(f"Loaded {audio_length_sec} seconds of audio in: {time.time() - s_time} sec")

        s_time = time.time()
        result = model.transcribe(
            audio=audio,
            batch_size=batch_size,
            language=language,
            task=task,
        )
        print(f"Transcribed audio in: {time.time() - s_time} sec")
        #print(result["segments"]) # before alignment

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model

        # 2. Align whisper output
        s_time = time.time()
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        print(f"Loaded alignment model in: {time.time() - s_time} sec")
        s_time = time.time()
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        print(f"Executed forced alignment in: {time.time() - s_time} sec")
        #print(result["segments"]) # after alignment

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

        # 3. (Optional) Assign speaker labels
        if diarization and hf_token is None:
            logger.error("Cannot load DiarizationPipeline due to missing HuggingFace API key - skipping diarization")
        elif diarization and hf_token is not None:
            s_time = time.time()
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            print(f"Loaded diarization model in: {time.time() - s_time} sec")
            
            s_time = time.time()
            diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(f"Executed diarization in: {time.time() - s_time} sec")

        # 4. Calculate price
        total_time = time.time() - s_tot_time
        print(f"Transcription price per hour: {(total_time/audio_length_sec) * gpu_price_per_sec * 60 * 60} $")

        # 5. Display results
        return result

    