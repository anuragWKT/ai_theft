from __future__ import annotations

import asyncio
import audioop
import contextlib
import io
import importlib.util
import math
import os
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv
from livekit import api
from livekit import rtc

from src.app_cli import create_runtime, process_turn


@dataclass(frozen=True)
class VoiceRuntimeConfig:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    room_name: str
    identity: str
    stt_api_key: str
    tts_api_key: str
    tts_provider: str
    stt_model: str
    tts_model: str
    cartesia_version: str
    cartesia_voice_id: str

    @classmethod
    def from_env(cls) -> "VoiceRuntimeConfig":
        return cls(
            livekit_url=os.getenv("LIVEKIT_URL", "").strip(),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", "").strip(),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", "").strip(),
            room_name=os.getenv("LIVEKIT_ROOM", "birthday-interrogation").strip(),
            identity=os.getenv("LIVEKIT_IDENTITY", "alex-agent").strip(),
            stt_api_key=os.getenv("VOICE_STT_API_KEY", "").strip(),
            tts_api_key=os.getenv("VOICE_TTS_API_KEY", "").strip(),
            tts_provider=os.getenv("VOICE_TTS_PROVIDER", "deepgram").strip().lower(),
            stt_model=os.getenv("DEEPGRAM_STT_MODEL", "nova-2").strip(),
            tts_model=os.getenv("CARTESIA_TTS_MODEL", os.getenv("DEEPGRAM_TTS_MODEL", "sonic-3")).strip(),
            cartesia_version=os.getenv("CARTESIA_VERSION", "2025-04-16").strip(),
            cartesia_voice_id=os.getenv(
                "CARTESIA_VOICE_ID", "228fca29-3a0a-435c-8728-5cb483251068"
            ).strip(),
        )


def _is_module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def voice_preflight(config: VoiceRuntimeConfig) -> Dict[str, List[str]]:
    missing_env: List[str] = []
    missing_packages: List[str] = []

    required_env = {
        "LIVEKIT_URL": config.livekit_url,
        "LIVEKIT_API_KEY": config.livekit_api_key,
        "LIVEKIT_API_SECRET": config.livekit_api_secret,
        "VOICE_STT_API_KEY": config.stt_api_key,
        "VOICE_TTS_API_KEY": config.tts_api_key,
    }

    for key, value in required_env.items():
        if not value:
            missing_env.append(key)

    for module_name in ["requests", "livekit"]:
        if not _is_module_available(module_name):
            missing_packages.append(module_name)

    return {
        "missing_env": missing_env,
        "missing_packages": missing_packages,
    }


def bootstrap_voice_runtime(project_root: Path | None = None) -> Tuple[object, object]:
    root = project_root or Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    game, dialogue = create_runtime(root)
    return game, dialogue


def handle_transcript(transcript: str, game: object, dialogue: object) -> str:
    outcome = process_turn(transcript, game, dialogue)
    return str(outcome.get("message", ""))


def _deepgram_transcribe(wav_bytes: bytes, api_key: str, model: str) -> str:
    import requests

    url = f"https://api.deepgram.com/v1/listen?model={model}&smart_format=true"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    response = requests.post(url, headers=headers, data=wav_bytes, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return (
        payload.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
        .strip()
    )


def _deepgram_speak_wav(text: str, api_key: str, model: str, sample_rate: int) -> bytes:
    import requests

    url = (
        "https://api.deepgram.com/v1/speak"
        f"?model={model}&encoding=linear16&sample_rate={sample_rate}&container=wav"
    )
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json={"text": text}, timeout=30)
    response.raise_for_status()
    return response.content


def _float32le_to_pcm16(raw_bytes: bytes) -> bytes:
    frame_count = len(raw_bytes) // 4
    output = bytearray(frame_count * 2)
    write_offset = 0
    for idx in range(frame_count):
        sample = struct.unpack_from("<f", raw_bytes, idx * 4)[0]
        if not math.isfinite(sample):
            sample = 0.0
        sample = max(-1.0, min(1.0, sample))
        value = int(sample * 32767.0)
        struct.pack_into("<h", output, write_offset, value)
        write_offset += 2
    return bytes(output)


def _cartesia_speak_pcm16(
    text: str,
    api_key: str,
    model_id: str,
    voice_id: str,
    sample_rate: int,
    version: str,
) -> bytes:
    import requests

    url = "https://api.cartesia.ai/tts/bytes"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Cartesia-Version": version,
        "Content-Type": "application/json",
    }
    payload = {
        "model_id": model_id,
        "transcript": text,
        "voice": {"mode": "id", "id": voice_id},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": sample_rate,
        },
        "language": "en",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return _float32le_to_pcm16(response.content)


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int, channels: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def _wav_to_mono_pcm(wav_bytes: bytes, target_rate: int) -> Tuple[bytes, int]:
    with contextlib.closing(wave.open(io.BytesIO(wav_bytes), "rb")) as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError("Expected 16-bit PCM WAV")

    if channels > 1:
        frames = audioop.tomono(frames, 2, 0.5, 0.5)

    if source_rate != target_rate:
        frames, _ = audioop.ratecv(frames, 2, 1, source_rate, target_rate, None)

    return frames, target_rate


def _build_livekit_token(config: VoiceRuntimeConfig) -> str:
    grants = api.VideoGrants(
        room_join=True,
        room=config.room_name,
        can_publish=True,
        can_subscribe=True,
    )
    return (
        api.AccessToken(config.livekit_api_key, config.livekit_api_secret)
        .with_identity(config.identity)
        .with_name(config.identity)
        .with_grants(grants)
        .to_jwt()
    )


async def run_livekit_room_mode(config: VoiceRuntimeConfig, game: object, dialogue: object) -> None:
    room = rtc.Room()
    stop_event = asyncio.Event()
    active_stream_tasks: Set[asyncio.Task[None]] = set()
    reply_queue: asyncio.Queue[tuple[str, bool]] = asyncio.Queue()
    game_over = False

    tts_sample_rate = 24000
    tts_channels = 1
    tts_frame_ms = 20
    tts_samples_per_frame = int(tts_sample_rate * tts_frame_ms / 1000)
    tts_bytes_per_frame = tts_samples_per_frame * 2 * tts_channels

    speech_rms_threshold = 450
    speech_silence_flush_seconds = 0.9
    speech_min_utterance_seconds = 0.6
    speech_force_flush_seconds = 6.0

    alex_audio_source: rtc.AudioSource | None = None
    reply_worker_task: asyncio.Task[None] | None = None

    async def _speak_alex_in_room(text: str) -> None:
        if alex_audio_source is None:
            return

        speak_text = text.replace("Alex:", "").strip()
        if not speak_text:
            return

        try:
            print("[room] Synthesizing Alex speech...")
            if config.tts_provider == "cartesia":
                pcm_audio = await asyncio.to_thread(
                    _cartesia_speak_pcm16,
                    speak_text,
                    config.tts_api_key,
                    config.tts_model,
                    config.cartesia_voice_id,
                    tts_sample_rate,
                    config.cartesia_version,
                )
            else:
                wav_audio = await asyncio.to_thread(
                    _deepgram_speak_wav,
                    speak_text,
                    config.tts_api_key,
                    config.tts_model,
                    tts_sample_rate,
                )
                pcm_audio, pcm_rate = _wav_to_mono_pcm(wav_audio, tts_sample_rate)
                if pcm_rate != tts_sample_rate:
                    return

            if len(pcm_audio) % 2 != 0:
                pcm_audio = pcm_audio[:-1]

            if not pcm_audio:
                return

            if len(pcm_audio) % tts_bytes_per_frame != 0:
                pad = tts_bytes_per_frame - (len(pcm_audio) % tts_bytes_per_frame)
                pcm_audio += b"\x00" * pad

            for offset in range(0, len(pcm_audio), tts_bytes_per_frame):
                frame_bytes = pcm_audio[offset : offset + tts_bytes_per_frame]
                frame = rtc.AudioFrame(
                    data=frame_bytes,
                    sample_rate=tts_sample_rate,
                    num_channels=tts_channels,
                    samples_per_channel=tts_samples_per_frame,
                )
                await alex_audio_source.capture_frame(frame)

            await alex_audio_source.wait_for_playout()
            print("[room] Alex voice sent to room audio track.")
        except Exception as error:
            print(f"[room] TTS playback failed: {error}")

    async def _queue_response(user_text: str, source_identity: str) -> None:
        nonlocal game_over
        if game_over:
            return

        print(f"[room] {source_identity}: {user_text}")
        turn_outcome = await asyncio.to_thread(process_turn, user_text, game, dialogue)
        response_text = str(turn_outcome.get("message", ""))
        finished = bool(turn_outcome.get("finished", False))
        print(f"[room] Alex: {response_text}")
        print(
            "[room] Status: "
            f"claims={len(game.claims)} | contradictions={game.state.contradiction_count} | "
            f"accusation_made={game.state.accusation_made} | confession_unlocked={game.state.confession_unlocked}"
        )

        if finished:
            game_over = True

        await reply_queue.put((response_text, finished))

    async def _reply_worker() -> None:
        while True:
            response_text, finished = await reply_queue.get()
            try:
                await _speak_alex_in_room(response_text)
                if finished:
                    print("[room] Game over: confession accepted. Ending session.")
                    stop_event.set()
            except Exception as error:
                print(f"[room] Failed to speak Alex reply: {error}")
            finally:
                reply_queue.task_done()

    async def _process_remote_audio(track: rtc.RemoteAudioTrack, identity: str) -> None:
        stream = rtc.AudioStream(track=track)
        utterance_pcm = bytearray()
        speaking = False
        silence_duration = 0.0
        utterance_duration = 0.0
        utterance_sample_rate = 48000

        async def _flush_utterance() -> None:
            nonlocal speaking, silence_duration, utterance_duration
            if game_over:
                utterance_pcm.clear()
                speaking = False
                silence_duration = 0.0
                utterance_duration = 0.0
                return

            if utterance_duration < speech_min_utterance_seconds or not utterance_pcm:
                utterance_pcm.clear()
                speaking = False
                silence_duration = 0.0
                utterance_duration = 0.0
                return

            wav_bytes = _pcm_to_wav_bytes(bytes(utterance_pcm), utterance_sample_rate, 1)
            utterance_pcm.clear()
            speaking = False
            silence_duration = 0.0
            utterance_duration = 0.0

            try:
                transcript = await asyncio.to_thread(
                    _deepgram_transcribe,
                    wav_bytes,
                    config.stt_api_key,
                    config.stt_model,
                )
                transcript = transcript.strip()
                if transcript:
                    await _queue_response(transcript, identity)
            except Exception as error:
                print(f"[room] STT failed for {identity}: {error}")

        try:
            async for event in stream:
                frame = event.frame
                frame_bytes = bytes(frame.data.cast("B"))
                frame_duration = frame.samples_per_channel / frame.sample_rate
                utterance_sample_rate = frame.sample_rate
                frame_rms = audioop.rms(frame_bytes, 2)

                if frame_rms >= speech_rms_threshold:
                    speaking = True
                    silence_duration = 0.0
                    utterance_pcm.extend(frame_bytes)
                    utterance_duration += frame_duration
                elif speaking:
                    silence_duration += frame_duration
                    utterance_pcm.extend(frame_bytes)
                    utterance_duration += frame_duration

                    if silence_duration >= speech_silence_flush_seconds:
                        await _flush_utterance()

                if speaking and utterance_duration >= speech_force_flush_seconds:
                    await _flush_utterance()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"[room] Audio stream error from {identity}: {error}")

    @room.on("connected")
    def _on_connected() -> None:
        print(f"Connected to LiveKit room '{config.room_name}' as '{config.identity}'.")
        print("Listening to remote audio tracks.")

    @room.on("disconnected")
    def _on_disconnected(reason: object) -> None:
        print(f"Disconnected from LiveKit room. Reason: {reason}")
        stop_event.set()

    @room.on("track_subscribed")
    def _on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if participant.identity == config.identity:
            return

        print(f"[room] Subscribed to audio from {participant.identity}.")
        task = asyncio.create_task(_process_remote_audio(track, participant.identity))
        active_stream_tasks.add(task)
        task.add_done_callback(lambda done: active_stream_tasks.discard(done))

    token = _build_livekit_token(config)
    await room.connect(config.livekit_url, token)

    alex_audio_source = rtc.AudioSource(sample_rate=tts_sample_rate, num_channels=tts_channels)
    alex_audio_track = rtc.LocalAudioTrack.create_audio_track("alex-voice", alex_audio_source)
    publish_options = rtc.TrackPublishOptions()
    publish_options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(alex_audio_track, publish_options)
    print("Published Alex audio track.")

    reply_worker_task = asyncio.create_task(_reply_worker())

    try:
        await stop_event.wait()
    finally:
        if reply_worker_task is not None:
            reply_worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reply_worker_task
        for task in list(active_stream_tasks):
            task.cancel()
        if active_stream_tasks:
            await asyncio.gather(*active_stream_tasks, return_exceptions=True)
        await room.disconnect()


def run_voice_agent(project_root: Path | None = None) -> None:
    root = project_root or Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    config = VoiceRuntimeConfig.from_env()
    report = voice_preflight(config)

    if report["missing_env"] or report["missing_packages"]:
        print("Voice runtime preflight failed.")
        if report["missing_env"]:
            print("Missing env:", ", ".join(report["missing_env"]))
        if report["missing_packages"]:
            print("Missing packages:", ", ".join(report["missing_packages"]))
        print("Text mode remains available: python -m src.app_cli")
        return

    game, dialogue = bootstrap_voice_runtime(root)
    print("LiveKit room mode is active.")
    print("Speak from a client publishing a room audio track.")
    print("Alex replies are sent on audio track 'alex-voice'.")
    asyncio.run(run_livekit_room_mode(config, game, dialogue))


if __name__ == "__main__":
    run_voice_agent()
