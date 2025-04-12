"""Event handler for clients of the server."""
import argparse
import asyncio
import logging
import os
import wave
from scipy.io import wavfile
from typing import Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from transformers import Wav2Vec2ProcessorWithLM
import torchaudio.transforms as T
import torch
import numpy as np
from transformers import AutoModelForCTC

_LOGGER = logging.getLogger(__name__)

class HajoProcessor(Wav2Vec2ProcessorWithLM):
    @staticmethod
    def get_missing_alphabet_tokens(decoder, tokenizer):
        return []

class HausKlaus:
    """Dummy class to represent the model & processor together."""

    def __init__(self, modelPath) -> None:
        self.model = AutoModelForCTC.from_pretrained(modelPath)
        self.processor = HajoProcessor.from_pretrained(modelPath)
        self.model.to('cuda')
    
    # this function will be called for each WAV file
    def predict_single_audio(self, batch):    
        audio = batch['audio']['array']
        # resample, if needed
        if batch['audio']['sampling_rate'] != 16000:
            transform = T.Resample(orig_freq=batch['audio']['sampling_rate'], new_freq=16000)
            resampled_tensor = transform(torch.from_numpy(audio).float())
            audio = resampled_tensor.numpy()
        # normalize
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        # ask HF processor to prepare audio for GPU eval
        input_values = self.processor(audio, return_tensors="pt", sampling_rate=16_000).input_values
        # call model on GPU
        with torch.no_grad():
            logits = self.model(input_values.to('cuda')).logits.cpu().numpy()[0]
            _LOGGER.debug("Logits: %s", logits)
        # ask HF processor to decode logits
        decoded = self.processor.decode(logits, beam_width=500)
        # return as dictionary
        return decoded.text

class HausKlausEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: HausKlaus,
        model_lock: asyncio.Lock,
        *args,
        initial_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt
        self._language = self.cli_args.language
        self._wav_dir = "./data/unlabeled"
        self._wav_path = os.path.join(self._wav_dir, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None
        self._wav_idx = 0

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if self._wav_file is None:
                # Count existing files in the directory
                if not os.path.exists(self._wav_dir):
                    os.makedirs(self._wav_dir)
                for file in os.listdir(self._wav_dir):
                    if file.endswith(".wav") and file.startswith("speech-"):
                        self._wav_idx += 1
                self._wav_path = os.path.join(self._wav_dir, f"speech-{self._wav_idx}.wav")
                _LOGGER.debug("Audio chunk received, creating WAV file: %s", self._wav_path)
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with initial prompt=%s",
                self.initial_prompt,
            )
            assert self._wav_file is not None

            self._wav_file.close()
            self._wav_idx += 1
            self._wav_file = None

            async with self.model_lock:
                text = self.model.predict_single_audio(self.load_single_wav(self._wav_path))
            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            # Reset
            self._language = self.cli_args.language

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        _LOGGER.debug("Event type received: %s", event.type)
        return True
    
    def load_single_wav(self, file_path):
        # Read the WAV file
        sampling_rate, data = wavfile.read(file_path)
        # Return a dictionary with the audio data and sampling rate
        return {'audio': {'array': data, 'sampling_rate': sampling_rate}}
