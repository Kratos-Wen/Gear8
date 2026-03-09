"""Speech functions equivalent to the original implementation behavior."""

import threading

import numpy as np
import sounddevice as sd


def text_to_speech(text, tts_engine, should_stop_audio):
    """
    Convert text to speech and read it aloud line by line.

    The function splits input into lines and queues each non-empty line.
    During synthesis, playback can be interrupted when should_stop_audio()
    returns True.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip() != ""]
    for line in lines:
        if should_stop_audio():
            break
        tts_engine.say(line)
    tts_engine.runAndWait()


class VoiceQueryHandler:
    """Voice query handler equivalent to the original lock and callback semantics."""

    def __init__(self, speech_model, callback, samplerate, duration=8):
        self.speech_model = speech_model
        self.callback = callback
        self.samplerate = samplerate
        self.duration = duration
        self.lock = threading.Lock()

    def record_audio(self):
        """Record one fixed-length audio clip."""
        print("Starting audio recording...")
        audio_data = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("Recording completed.")
        audio_array = audio_data.flatten()
        if audio_array.size < int(self.samplerate * self.duration * 0.5):
            raise ValueError("Audio is too short or recording failed.")
        return audio_array

    def recognize_and_handle(self):
        """Run ASR and dispatch callback in original try/finally structure."""
        if not self.lock.acquire(blocking=False):
            print("Recognition already in progress. Please wait...")
            return
        try:
            audio_array = self.record_audio()
            audio_array = audio_array.astype("float32")
            audio_array = np.clip(audio_array, -1.0, 1.0)
            result = self.speech_model.transcribe(audio_array, language="en")
            query = result["text"].strip()
            print(f"Speech recognition result: {query}")
        except Exception as e:
            print(f"Speech recognition failed: {e}")
            query = ""
        try:
            if query:
                self.callback(query)
        except Exception as e:
            print(f"Error processing query: {e}")
        finally:
            self.lock.release()

