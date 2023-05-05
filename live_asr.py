import pyaudio
import webrtcvad
from wav2vec2_inference import Wave2Vec2Inference
import numpy as np
import threading
import time
from sys import exit
from queue import Queue
import audioop


class LiveWav2Vec2:
    pause_event = threading.Event()
    exit_event = threading.Event()

    def __init__(self, model_name, device_name="default"):
        self.model_name = model_name
        self.device_name = device_name
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()
        self.asr_process = threading.Thread(target=LiveWav2Vec2.asr_process, args=(
            self.model_name, self.asr_input_queue, self.asr_output_queue,))
        self.asr_process.start()
        self.vad_process = threading.Thread(target=LiveWav2Vec2.vad_process, args=(self.asr_input_queue,))
        self.stopped = False

    def pause(self):
        """stop the asr process"""
        LiveWav2Vec2.pause_event.set()

    def stop(self):
        self.asr_input_queue.put("close")
        LiveWav2Vec2.exit_event.set()


    def start(self):
        """start the asr process"""
        if not self.vad_process.is_alive():
            self.vad_process.start()
        LiveWav2Vec2.pause_event.clear()

    @staticmethod
    def vad_process(asr_input_queue):
        vad = webrtcvad.Vad()
        vad.set_mode(1)

        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        # A frame must be either 10, 20, or 30 ms in duration for webrtcvad
        FRAME_DURATION = 10
        CHUNK = int(RATE * FRAME_DURATION / 1000)

        device = audio.get_default_input_device_info()
        selected_input_device_id = int(device['index'])

        stream = audio.open(input_device_index=selected_input_device_id,
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = b''
        non_speech = 0
        while True:
            frame = stream.read(CHUNK, exception_on_overflow=False)

            if LiveWav2Vec2.exit_event.is_set():
                break

            if LiveWav2Vec2.pause_event.is_set():
                continue
            cvstate = None
            frame, cvstate = audioop.ratecv(
                frame, 2, 1, 44100,
                16000, cvstate)
            is_speech = vad.is_speech(frame, 16000)
            if is_speech:
                frames += frame
                non_speech = 0
            else:
                non_speech += 1
                if non_speech > 5 and len(frames) > 1:
                    asr_input_queue.put(frames)
                    frames = b''
        stream.stop_stream()
        stream.close()
        audio.terminate()

    @staticmethod
    def asr_process(model_name, in_queue, output_queue):
        wave2vec_asr = Wave2Vec2Inference(model_name, use_lm_if_possible=True, use_gpu=False)

        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break

            float64_buffer = np.frombuffer(
                audio_frames, dtype=np.int16) / 32767
            start = time.perf_counter()
            text, confidence = wave2vec_asr.buffer_to_text(float64_buffer)
            text = text.lower()
            inference_time = time.perf_counter() - start
            sample_length = len(float64_buffer) / 16000  # length in sec
            if len(text) > 3:
                output_queue.put([text, sample_length, inference_time, confidence])

    def get_last_text(self):
        """returns the text, sample length and inference time in seconds."""
        return self.asr_output_queue.get()


if __name__ == "__main__":
    print("Live ASR")

    asr = LiveWav2Vec2("checkpoint-115000")

    asr.start()

    try:
        while True:
            text, sample_length, inference_time, confidence = asr.get_last_text()
            print(f"{sample_length:.3f}s\t{inference_time:.3f}s\t{confidence}\t{text}")

    except KeyboardInterrupt:
        asr.stop()
        exit()
