import os
import logging
import runpod

from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from rp_schema import INPUT_VALIDATIONS
from predict import Predictor


logger = logging.getLogger(__name__)

logger.info("Setting up transcription model ..")
model = Predictor()
model.setup()

if not os.environ.get("HF_TOKEN"):
    logger.warning("HF_TOKEN is missing from environment - diarization will be unavailable")


@rp_debugger.FunctionTimer
def run_transcription_job(job):
    try:
        job_input = job["input"]

        # Validate input
        with rp_debugger.LineTimer("validation_step"):
            input_validation = validate(job_input, INPUT_VALIDATIONS)
            if "errors" in input_validation:
                return {"error": input_validation["errors"]}
        
        job_input = input_validation["validated_input"]

        # Prepare audio files
        audio_file_urls = job_input["audio_files"]

        with rp_debugger.LineTimer("download_audio_files_step"):
            audio_file_paths = download_files_from_urls(
                job_id=job["id"],
                urls=audio_file_urls
            )

            # Log any download failures
            for i in range(len(audio_file_paths)):
                if audio_file_paths[i] == None:
                    logger.warning(f"Failed to download audio file from URL: {audio_file_urls[i]}")

        # Run transcription
        with rp_debugger.LineTimer("transcription_step"):
            results = {}

            for i in range(len(audio_file_paths)):
                try:
                    result = model.predict(
                        audio_file_path=audio_file_paths[i],
                        language=job_input["language"],
                        task=job_input["task"],
                        diarization=job_input["diarization"],
                        min_speakers=job_input["min_speakers"],
                        max_speakers=job_input["max_speakers"],
                        hf_token=os.environ.get("HF_TOKEN")
                    )
                except Exception as ex:
                    result = None
                
                results[audio_file_urls[i]] = result

        # Clean up
        with rp_debugger.LineTimer("cleanup_step"):
            rp_cleanup.clean(["input_objects"])

        return results
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": run_transcription_job})