"""Worker-local compatibility loader for Chatterbox Turbo and Nano.

This module is imported only inside the isolated Chatterbox worker when the
pinned wheel does not expose the current Turbo/Nano local loader. It performs
no model acquisition and expects an already verified local snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import librosa
import perth
import pyloudnorm as ln
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from chatterbox.models.s3gen import S3GEN_SR, S3Gen
from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.voice_encoder import VoiceEncoder


GPT2_SMALL_CONFIG = {
    "activation_function": "gelu_new",
    "architectures": ["GPT2LMHeadModel"],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 8196,
    "n_embd": 768,
    "hidden_size": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 8196,
    "n_special": 0,
    "predict_special_tokens": True,
    "resid_pdrop": 0.1,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {
        "text-generation": {"do_sample": True, "max_length": 50}
    },
    "vocab_size": 50276,
}


def _register_nano_config() -> None:
    # T3 imports this mapping by reference. The worker is one-model-per-process,
    # so adding the Nano-only entry here cannot affect another model profile.
    from chatterbox.models.t3 import llama_configs

    existing = llama_configs.LLAMA_CONFIGS.get("GPT2_small")
    if existing is None:
        llama_configs.LLAMA_CONFIGS["GPT2_small"] = dict(GPT2_SMALL_CONFIG)
    elif dict(existing) != GPT2_SMALL_CONFIG:
        raise RuntimeError("installed Chatterbox GPT2_small config does not match the approved Nano config")


@dataclass
class _Conditionals:
    t3: Any
    gen: dict[str, Any]

    def to(self, device: str) -> "_Conditionals":
        self.t3 = self.t3.to(device=device)
        for key, value in self.gen.items():
            if torch.is_tensor(value):
                self.gen[key] = value.to(device=device)
        return self

    @classmethod
    def load(cls, path: Path, *, map_location: Any) -> "_Conditionals":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict) or "t3" not in payload or "gen" not in payload:
            raise RuntimeError("Turbo/Nano built-in conditionals have an unsupported format")
        return cls(T3Cond(**payload["t3"]), payload["gen"])


def _punc_norm(text: str) -> str:
    if not text:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for old, new in (
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ):
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not any(text.endswith(value) for value in {".", "!", "?", "-", ","}):
        text += "."
    return text


class TurboCompatibilityTTS:
    ENC_COND_LEN = 15 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        voice_encoder: VoiceEncoder,
        tokenizer: Any,
        device: str,
        *,
        conditionals: _Conditionals | None,
        model_label: str,
    ):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = voice_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conditionals
        self.model_label = model_label
        self.watermarker = perth.PerthImplicitWatermarker()

    @staticmethod
    def _norm_loudness(wav: Any, sample_rate: int, target_lufs: float = -27.0) -> Any:
        try:
            meter = ln.Meter(sample_rate)
            loudness = meter.integrated_loudness(wav)
            gain = 10.0 ** ((target_lufs - loudness) / 20.0)
            if math.isfinite(gain) and gain > 0:
                return wav * gain
        except Exception:
            pass
        return wav

    def prepare_conditionals(self, wav_path: str, *, norm_loudness: bool = True) -> None:
        s3_wav, sample_rate = librosa.load(wav_path, sr=S3GEN_SR)
        if len(s3_wav) / float(sample_rate) <= 5.0:
            raise ValueError("Audio prompt must be longer than 5 seconds")
        if norm_loudness:
            s3_wav = self._norm_loudness(s3_wav, sample_rate)
        ref_16k = librosa.resample(s3_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3_ref = self.s3gen.embed_ref(
            s3_wav[: self.DEC_COND_LEN],
            S3GEN_SR,
            device=self.device,
        )
        prompt_len = self.t3.hp.speech_cond_prompt_len
        speech_tokens, _ = self.s3gen.tokenizer.forward(
            [ref_16k[: self.ENC_COND_LEN]],
            max_len=prompt_len,
        )
        speech_tokens = torch.atleast_2d(speech_tokens).to(self.device)
        speaker = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k], sample_rate=S3_SR)
        ).mean(axis=0, keepdim=True).to(self.device)
        t3_cond = T3Cond(
            speaker_emb=speaker,
            cond_prompt_speech_tokens=speech_tokens,
            emotion_adv=torch.zeros(1, 1, 1, device=self.device),
        ).to(device=self.device)
        self.conds = _Conditionals(t3_cond, s3_ref)

    def generate(
        self,
        text: str,
        *,
        audio_prompt_path: str | None = None,
        repetition_penalty: float = 1.2,
        top_p: float = 0.95,
        temperature: float = 0.8,
        top_k: int = 1000,
        **_ignored: Any,
    ) -> Any:
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path)
        if self.conds is None:
            raise RuntimeError("Turbo/Nano requires built-in conditionals or an audio prompt")

        text_tokens = self.tokenizer(
            _punc_norm(text),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids.to(self.device)
        speech_tokens = self.t3.inference_turbo(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        speech_tokens = speech_tokens[speech_tokens < 6561].to(self.device)
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], device=self.device).long()
        speech_tokens = torch.cat([speech_tokens, silence])
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
        watermarked = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked).unsqueeze(0)


def load_turbo_compat(snapshot: Path, *, device: str, nano: bool) -> TurboCompatibilityTTS:
    snapshot = Path(snapshot)
    if nano:
        _register_nano_config()

    map_location = torch.device("cpu") if device in {"cpu", "mps"} else None

    voice_encoder = VoiceEncoder()
    voice_encoder.load_state_dict(load_file(snapshot / "ve.safetensors"))
    voice_encoder.to(device).eval()

    hp = T3Config(text_tokens_dict_size=50276)
    hp.llama_config_name = "GPT2_small" if nano else "GPT2_medium"
    hp.speech_tokens_dict_size = 6563
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = 375
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False

    t3 = T3(hp)
    t3_path = snapshot / ("t3_nano_v1.safetensors" if nano else "t3_turbo_v1.safetensors")
    t3_state = load_file(t3_path)
    if "model" in t3_state:
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)
    del t3.tfmr.wte
    t3.to(device).eval()

    s3gen = S3Gen(meanflow=True)
    s3gen.load_state_dict(load_file(snapshot / "s3gen_meanflow.safetensors"), strict=True)
    s3gen.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer) != 50276:
        raise RuntimeError(f"Turbo/Nano tokenizer length must be 50276, got {len(tokenizer)}")

    conditionals = None
    builtin_voice = snapshot / "conds.pt"
    if builtin_voice.exists():
        conditionals = _Conditionals.load(
            builtin_voice,
            map_location=map_location,
        ).to(device)

    return TurboCompatibilityTTS(
        t3,
        s3gen,
        voice_encoder,
        tokenizer,
        device,
        conditionals=conditionals,
        model_label="Nano" if nano else "Turbo",
    )
