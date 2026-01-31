from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class Config:
    """設定."""

    input_dir: Path
    train_csv: Path
    test_movie_dir: Path
    output_path: Path
    vllm_url: str
    model: str
    max_frames: int
    debug: bool
    prompt_text: str
    allow_missing_ffmpeg: bool
    use_train_style: bool
    style_samples: int
    style_seed: int
    sbert_model: str
    token_limit: int
    token_limit_percentile: int
    num_workers: int
    request_timeout: int
    sbert_device: str
    sbert_batch_size: int
    sbert_max_length: int
    use_video_input: bool
    video_content_type: str
    video_field: str
    video_data_format: str
    video_as_object: bool
    video_payload_mode: str
    video_fallback_to_frames: bool
    frame_size: str


@dataclass
class StyleGuidance:
    """スタイル指針."""

    sentence_target: int
    use_semicolon: bool
    motion_words: List[str]


def parse_args() -> Config:
    """引数を解析する.

    Returns:
        Config: 設定.
    """
    parser = argparse.ArgumentParser(description="Qwen3 VL vLLM ベースライン")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--train-csv", type=Path, default=Path("input/data/train.csv"))
    parser.add_argument("--test-movie-dir", type=Path, default=Path("input/data/test_movie"))
    parser.add_argument("--output-path", type=Path, default=Path("results/submission.csv"))
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--allow-missing-ffmpeg", action="store_true")
    parser.add_argument("--prompt-text", type=str, default="")
    parser.add_argument("--use-train-style", action="store_true", default=True)
    parser.add_argument("--no-use-train-style", action="store_false", dest="use_train_style")
    parser.add_argument("--style-samples", type=int, default=5)
    parser.add_argument("--style-seed", type=int, default=42)
    parser.add_argument("--sbert-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--token-limit", type=int, default=0)
    parser.add_argument("--token-limit-percentile", type=int, default=95)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--sbert-device", type=str, default="cpu")
    parser.add_argument("--sbert-batch-size", type=int, default=8)
    parser.add_argument("--sbert-max-length", type=int, default=0)
    parser.add_argument("--use-video-input", action="store_true", default=False)
    parser.add_argument("--no-use-video-input", action="store_false", dest="use_video_input")
    parser.add_argument("--video-content-type", type=str, default="video_url", choices=["video_url", "video"])
    parser.add_argument("--video-field", type=str, default="url", choices=["url", "data"])
    parser.add_argument("--video-data-format", type=str, default="data_url", choices=["data_url", "base64"])
    parser.add_argument("--video-as-object", action="store_true", default=True)
    parser.add_argument("--no-video-as-object", action="store_false", dest="video_as_object")
    parser.add_argument(
        "--video-payload-mode",
        type=str,
        default="auto",
        choices=["auto", "single"],
    )
    parser.add_argument("--video-fallback-to-frames", action="store_true", default=True)
    parser.add_argument("--no-video-fallback-to-frames", action="store_false", dest="video_fallback_to_frames")
    parser.add_argument("--frame-size", type=str, default="")
    args = parser.parse_args()
    return Config(
        input_dir=args.input_dir,
        train_csv=args.train_csv,
        test_movie_dir=args.test_movie_dir,
        output_path=args.output_path,
        vllm_url=args.vllm_url,
        model=args.model,
        max_frames=args.max_frames,
        debug=args.debug,
        prompt_text=args.prompt_text,
        allow_missing_ffmpeg=args.allow_missing_ffmpeg,
        use_train_style=args.use_train_style,
        style_samples=args.style_samples,
        style_seed=args.style_seed,
        sbert_model=args.sbert_model,
        token_limit=args.token_limit,
        token_limit_percentile=args.token_limit_percentile,
        num_workers=args.num_workers,
        request_timeout=args.request_timeout,
        sbert_device=args.sbert_device,
        sbert_batch_size=args.sbert_batch_size,
        sbert_max_length=args.sbert_max_length,
        use_video_input=args.use_video_input,
        video_content_type=args.video_content_type,
        video_field=args.video_field,
        video_data_format=args.video_data_format,
        video_as_object=args.video_as_object,
        video_payload_mode=args.video_payload_mode,
        video_fallback_to_frames=args.video_fallback_to_frames,
        frame_size=args.frame_size,
    )


def run_command(command: List[str]) -> str:
    """コマンドを実行する.

    Args:
        command: コマンド配列.

    Returns:
        str: 標準出力.

    Raises:
        RuntimeError: 実行失敗時.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg/ffprobe が見つかりません") from exc
    if result.returncode != 0:
        raise RuntimeError(f"コマンドの実行に失敗しました: {result.stderr.strip()}")
    return result.stdout.strip()


def probe_duration(video_path: Path) -> float:
    """動画の秒数を取得する.

    Args:
        video_path: 動画パス.

    Returns:
        float: 秒数.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        output = run_command(command)
        return float(output)
    except Exception:
        return 0.0


def parse_frame_size(frame_size: str) -> Optional[tuple[int, int]]:
    """フレームサイズを解析する.

    Args:
        frame_size: "WIDTHxHEIGHT" 形式の文字列.

    Returns:
        Optional[tuple[int, int]]: 幅と高さ.

    Raises:
        RuntimeError: 形式が不正な場合.
    """
    if not frame_size:
        return None
    parts = frame_size.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise RuntimeError("frame-size が不正です")
    width, height = (int(parts[0]), int(parts[1]))
    if width <= 0 or height <= 0:
        raise RuntimeError("frame-size が不正です")
    return width, height


def extract_frames(video_path: Path, max_frames: int, frame_size: Optional[tuple[int, int]]) -> List[str]:
    """動画からフレームを抽出する.

    Args:
        video_path: 動画パス.
        max_frames: 抽出上限.

    Returns:
        List[str]: data URL一覧.

    Raises:
        RuntimeError: 抽出失敗時.
    """
    if max_frames <= 0:
        raise RuntimeError("フレーム数は1以上にしてください")
    duration = probe_duration(video_path)
    fps = 1.0
    if duration > 0:
        fps = max(1.0 / max(duration / max_frames, 1e-6), 0.1)
    filters = [f"fps={fps}"]
    if frame_size:
        width, height = frame_size
        filters.append(
            "scale="
            f"{width}:{height}"
            ":force_original_aspect_ratio=decrease"
        )
        filters.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
    with tempfile.TemporaryDirectory() as temp_dir:
        out_pattern = Path(temp_dir) / "frame_%03d.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            ",".join(filters),
            "-frames:v",
            str(max_frames),
            str(out_pattern),
        ]
        run_command(command)
        frames = sorted(Path(temp_dir).glob("frame_*.jpg"))
        if not frames:
            raise RuntimeError("フレーム抽出に失敗しました")
        return [encode_image_base64(Path(p)) for p in frames]


def encode_image_base64(image_path: Path) -> str:
    """画像をbase64に変換する.

    Args:
        image_path: 画像パス.

    Returns:
        str: data URL.
    """
    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def encode_video_base64(video_path: Path) -> str:
    """動画をbase64に変換する.

    Args:
        video_path: 動画パス.

    Returns:
        str: data URL.
    """
    data = video_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def encode_video_base64_raw(video_path: Path) -> str:
    """動画をbase64に変換する（プレフィックス無し）.

    Args:
        video_path: 動画パス.

    Returns:
        str: base64文字列.
    """
    data = video_path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def build_video_content(
    video_path: Path,
    content_type: str,
    field: str,
    data_format: str,
    as_object: bool,
) -> dict:
    """動画入力のcontentを作る.

    Args:
        video_path: 動画パス.
        content_type: contentタイプ.
        field: url/dataのどちらを使うか.
        data_format: data_url/base64.
        as_object: フィールドをオブジェクトにするか.

    Returns:
        dict: content要素.
    """
    if data_format == "data_url":
        value = encode_video_base64(video_path)
    else:
        value = encode_video_base64_raw(video_path)
    if as_object:
        payload = {field: value}
    else:
        payload = value
    return {"type": content_type, content_type: payload}


def build_video_candidates(config: Config, video_path: Path) -> List[dict]:
    """動画入力の候補を作る.

    Args:
        config: 設定.
        video_path: 動画パス.

    Returns:
        List[dict]: content候補.
    """
    if config.video_payload_mode == "single":
        return [
            build_video_content(
                video_path,
                config.video_content_type,
                config.video_field,
                config.video_data_format,
                config.video_as_object,
            )
        ]
    return [
        build_video_content(video_path, "video_url", "url", "data_url", True),
        build_video_content(video_path, "video_url", "url", "base64", True),
    ]


def call_vllm_chat(
    vllm_url: str,
    model: str,
    prompt_text: str,
    image_data_urls: Iterable[str],
    video_content: Optional[dict] = None,
    timeout_sec: int = 60,
) -> str:
    """vLLM OpenAI互換APIを呼び出す.

    Args:
        vllm_url: ベースURL.
        model: モデル名.
        prompt_text: 指示文.
        image_data_urls: 画像data URL.
        video_data_url: 動画data URL.
        timeout_sec: タイムアウト.

    Returns:
        str: 生成テキスト.

    Raises:
        RuntimeError: 応答が不正な場合.
    """
    content = [{"type": "text", "text": prompt_text}]
    if video_content:
        content.append(video_content)
    else:
        for data_url in image_data_urls:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": 128,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{vllm_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"vLLM呼び出しに失敗しました: HTTP {exc.code} {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"vLLM呼び出しに失敗しました: {exc}") from exc
    try:
        result = json.loads(body)
        message = result["choices"][0]["message"]["content"]
        if isinstance(message, list):
            return " ".join(part.get("text", "") for part in message).strip()
        return str(message).strip()
    except Exception as exc:
        raise RuntimeError("vLLMの応答形式が不正です") from exc


def build_style_prompt(
    train_prompts: List[str],
    samples: int,
    seed: int,
    token_limit: int,
    guidance: StyleGuidance,
) -> str:
    """学習プロンプトのスタイルに寄せた指示文を作る.

    Args:
        train_prompts: 学習プロンプト.
        samples: サンプル数.
        seed: 乱数シード.

    Returns:
        str: 指示文.
    """
    if not train_prompts:
        return "Describe the video briefly with the main subject, action, and setting."
    cleaned_prompts = [normalize_prompt_text(p) for p in train_prompts]
    cleaned_prompts = [p for p in cleaned_prompts if p]
    rng = np.random.default_rng(seed)
    count = min(samples, len(cleaned_prompts))
    indices = rng.choice(len(cleaned_prompts), size=count, replace=False)
    examples = "\n".join(f"- {cleaned_prompts[i]}" for i in indices)
    limit_text = ""
    if token_limit > 0:
        limit_text = f" Limit to {token_limit} tokens or fewer."
    motion_hint = ""
    if guidance.motion_words:
        motion_hint = " Use words like " + ", ".join(guidance.motion_words) + "."
    semicolon_hint = ""
    if guidance.use_semicolon:
        semicolon_hint = " Semicolons are common."
    return (
        "You are given a short video. Write one English prompt in the same style as the examples. "
        f"Write {guidance.sentence_target} sentences.{semicolon_hint} "
        "Sentence 1: main subject, scene, and action. "
        "Sentence 2: sensory details (light, texture, air, micro-motions). "
        "Sentence 3: explicitly describe continuous or repeating motion."
        " Keep a calm, gentle tone and focus on one subject and one scene."
        " If visible, add camera framing, lens, movement, lighting, and atmosphere."
        f"{motion_hint} Avoid quotes, tags, or bullet points.{limit_text}\n"
        f"Examples:\n{examples}\nNow write the prompt for the video."
    )


def derive_style_guidance(prompts: List[str]) -> StyleGuidance:
    """学習プロンプトからスタイル指針を作る.

    Args:
        prompts: プロンプト一覧.

    Returns:
        StyleGuidance: 指針.
    """
    if not prompts:
        return StyleGuidance(sentence_target=2, use_semicolon=False, motion_words=[])
    cleaned_prompts = [normalize_prompt_text(p) for p in prompts]
    cleaned_prompts = [p for p in cleaned_prompts if p]
    if not cleaned_prompts:
        return StyleGuidance(sentence_target=2, use_semicolon=False, motion_words=[])
    sentence_counts = []
    semicolon_count = 0
    motion_vocab = [
        "continuous",
        "loop",
        "looping",
        "repeat",
        "repeating",
        "steady",
        "gentle",
        "seamless",
        "rhythm",
        "cadence",
        "drift",
        "glide",
        "sway",
        "slow",
    ]
    motion_hits = {word: 0 for word in motion_vocab}
    for prompt in cleaned_prompts:
        parts = [p for p in re.split(r"[.!?]+", prompt.strip()) if p.strip()]
        sentence_counts.append(len(parts))
        if ";" in prompt:
            semicolon_count += 1
        lower = prompt.lower()
        for word in motion_vocab:
            if word in lower:
                motion_hits[word] += 1
    sentence_target = int(np.median(sentence_counts)) if sentence_counts else 2
    sentence_target = max(2, min(3, sentence_target))
    use_semicolon = semicolon_count / max(1, len(cleaned_prompts)) >= 0.3
    motion_sorted = sorted(motion_hits.items(), key=lambda x: x[1], reverse=True)
    motion_words = [w for w, c in motion_sorted if c > 0][:6]
    return StyleGuidance(
        sentence_target=sentence_target,
        use_semicolon=use_semicolon,
        motion_words=motion_words,
    )


def count_tokens(text: str) -> int:
    """簡易的にトークン数を数える.

    Args:
        text: 文字列.

    Returns:
        int: トークン数.
    """
    if not text:
        return 0
    return len(text.strip().split())


def truncate_tokens(text: str, limit: int) -> str:
    """トークン数を上限で切り詰める.

    Args:
        text: 文字列.
        limit: 上限.

    Returns:
        str: 切り詰め後の文字列.
    """
    if limit <= 0:
        return text
    tokens = text.strip().split()
    if len(tokens) <= limit:
        return text
    return " ".join(tokens[:limit])


def estimate_token_limit(prompts: List[str], percentile: int) -> int:
    """学習プロンプトからトークン数上限を推定する.

    Args:
        prompts: プロンプト一覧.
        percentile: パーセンタイル.

    Returns:
        int: 上限値.
    """
    if not prompts:
        return 0
    lengths = [count_tokens(prompt) for prompt in prompts]
    if not lengths:
        return 0
    percentile = max(1, min(100, percentile))
    value = int(np.percentile(lengths, percentile))
    return max(1, value)


def load_sbert_model(model_name: str, device: str, max_length: int):
    """SentenceTransformerを読み込む.

    Args:
        model_name: モデル名.
        device: デバイス.
        max_length: 最大長.

    Returns:
        SentenceTransformer: モデル.

    Raises:
        RuntimeError: ロード失敗時.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("sentence-transformers が見つかりません") from exc
    model = SentenceTransformer(model_name, device=device)
    if max_length > 0:
        model.max_seq_length = max_length
    return model


def embed_captions_batch(
    captions: List[str], model, batch_size: int
) -> np.ndarray:
    """キャプションをまとめて埋め込む.

    Args:
        captions: キャプション一覧.
        model: SentenceTransformer.
        batch_size: バッチサイズ.

    Returns:
        np.ndarray: 埋め込み配列.
    """
    if not captions:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = model.encode(
        captions,
        normalize_embeddings=True,
        batch_size=max(1, batch_size),
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2正規化する.

    Args:
        vec: ベクトル.

    Returns:
        np.ndarray: 正規化ベクトル.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def dummy_image_data_url() -> str:
    """ダミー画像のdata URLを返す.

    Returns:
        str: data URL.
    """
    data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO9y2hYAAAAASUVORK5CYII="
    )
    return f"data:image/png;base64,{data}"


def save_submission(output_path: Path, video_ids: List[int], embeddings: np.ndarray) -> None:
    """提出ファイルを保存する.

    Args:
        output_path: 出力先.
        video_ids: 動画ID.
        embeddings: 埋め込み配列.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for vid, emb in zip(video_ids, embeddings, strict=True):
        for idx, value in enumerate(emb):
            rows.append({"emb_id": f"{vid}-{idx}", "value": float(value)})
    df = pd.DataFrame(rows, columns=["emb_id", "value"])
    df.to_csv(output_path, index=False)


def load_train_prompts(train_csv: Path) -> List[str]:
    """学習プロンプトを読み込む.

    Args:
        train_csv: 学習CSVパス.

    Returns:
        List[str]: プロンプト一覧.
    """
    df = pd.read_csv(train_csv)
    return df["prompt_en"].astype(str).tolist()


def normalize_prompt_text(text: str) -> str:
    """プロンプト文字列を正規化する.

    Args:
        text: 文字列.

    Returns:
        str: 正規化後の文字列.
    """
    if not text:
        return ""
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"[^\x00-\x7F]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def list_test_videos(test_movie_dir: Path) -> List[Path]:
    """テスト動画一覧を取得する.

    Args:
        test_movie_dir: テスト動画ディレクトリ.

    Returns:
        List[Path]: 動画パス一覧.
    """
    if not test_movie_dir.exists():
        raise RuntimeError("テスト動画ディレクトリが存在しません")
    return sorted(test_movie_dir.glob("*.mp4"), key=lambda p: int(p.stem))


def main() -> None:
    """エントリポイント."""
    config = parse_args()
    if config.debug:
        print("デバッグモード: 有効")
    ffmpeg_missing = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None
    if config.use_video_input and config.video_fallback_to_frames and ffmpeg_missing:
        print("警告: 動画入力が失敗した場合にフレームへフォールバックできません。")
    if not config.use_video_input:
        if ffmpeg_missing and not config.allow_missing_ffmpeg:
            raise RuntimeError("ffmpeg/ffprobe が見つかりません")
        if ffmpeg_missing and config.allow_missing_ffmpeg:
            print("警告: ffmpeg/ffprobe が見つかりません。ダミー画像で続行します。")
    if not config.train_csv.exists():
        raise RuntimeError("学習CSVが見つかりません")
    prompts = load_train_prompts(config.train_csv)
    frame_size = parse_frame_size(config.frame_size)
    token_limit = config.token_limit
    if token_limit <= 0:
        token_limit = estimate_token_limit(prompts, config.token_limit_percentile)
    if config.debug:
        print(f"token_limit: {token_limit} (percentile={config.token_limit_percentile})")
    num_workers = config.num_workers
    if num_workers <= 0:
        num_workers = min(4, max(1, os.cpu_count() or 1))
    style_guidance = derive_style_guidance(prompts)
    if config.prompt_text:
        prompt_text = config.prompt_text
    elif config.use_train_style:
        prompt_text = build_style_prompt(
            prompts,
            config.style_samples,
            config.style_seed,
            token_limit,
            style_guidance,
        )
    else:
        limit_text = ""
        if token_limit > 0:
            limit_text = f" Limit to {token_limit} tokens or fewer."
        motion_hint = ""
        if style_guidance.motion_words:
            motion_hint = " Use words like " + ", ".join(style_guidance.motion_words) + "."
        prompt_text = (
            "Describe the video with subject, action, setting, and motion."
            f"{motion_hint}{limit_text}"
        )
    test_videos = list_test_videos(config.test_movie_dir)

    sbert_model = load_sbert_model(
        config.sbert_model, config.sbert_device, config.sbert_max_length
    )
    debug_rows = []
    captions = {}
    def process_video(video_path: Path) -> dict:
        start_time = time.time()
        if config.use_video_input:
            caption = ""
            payload_used = ""
            candidates = build_video_candidates(config, video_path)
            last_error: Optional[Exception] = None
            for idx, candidate in enumerate(candidates):
                try:
                    caption = call_vllm_chat(
                        config.vllm_url,
                        config.model,
                        prompt_text,
                        [],
                        video_content=candidate,
                        timeout_sec=config.request_timeout,
                    )
                    payload_used = f"{candidate.get('type','unknown')}_{idx}"
                    break
                except Exception as exc:
                    last_error = exc
                    if config.video_payload_mode == "single":
                        break
            if not caption and last_error:
                error_text = str(last_error)
                if config.video_fallback_to_frames:
                    if ffmpeg_missing:
                        raise RuntimeError("動画入力に失敗しました。ffmpeg/ffprobe が必要です") from last_error
                    frames = extract_frames(video_path, config.max_frames, frame_size)
                    caption = call_vllm_chat(
                        config.vllm_url,
                        config.model,
                        prompt_text,
                        frames,
                        timeout_sec=config.request_timeout,
                    )
                    payload_used = "fallback_images"
                else:
                    raise RuntimeError(f"動画入力に失敗しました: {error_text}") from last_error
        else:
            if ffmpeg_missing:
                frames = [dummy_image_data_url()]
            else:
                frames = extract_frames(video_path, config.max_frames, frame_size)
            caption = call_vllm_chat(
                config.vllm_url,
                config.model,
                prompt_text,
                frames,
                timeout_sec=config.request_timeout,
            )
            payload_used = "images"
        return {
            "video_id": int(video_path.stem),
            "caption": caption,
            "elapsed_sec": round(time.time() - start_time, 3),
            "payload": payload_used,
        }

    if num_workers == 1:
        for video_path in tqdm(test_videos, desc="videos"):
            result = process_video(video_path)
            caption = truncate_tokens(result["caption"], token_limit)
            captions[result["video_id"]] = caption
            if config.debug:
                debug_rows.append(
                    {
                        "video_id": result["video_id"],
                        "caption": caption,
                        "elapsed_sec": result["elapsed_sec"],
                        "payload": result["payload"],
                    }
                )
                print(f'{result["video_id"]}.mp4: {caption}')
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_video, vp): vp for vp in test_videos}
            for future in tqdm(as_completed(futures), total=len(futures), desc="videos"):
                result = future.result()
                caption = truncate_tokens(result["caption"], token_limit)
                captions[result["video_id"]] = caption
                if config.debug:
                    debug_rows.append(
                        {
                            "video_id": result["video_id"],
                            "caption": caption,
                            "elapsed_sec": result["elapsed_sec"],
                            "payload": result["payload"],
                        }
                    )
                    tqdm.write(f'{result["video_id"]}.mp4: {caption}')

    video_ids = sorted(captions.keys())
    caption_list = [captions[vid] for vid in video_ids]
    embeddings = embed_captions_batch(
        caption_list, sbert_model, config.sbert_batch_size
    )
    embeddings = np.vstack([l2_normalize(vec) for vec in embeddings])
    save_submission(config.output_path, video_ids, embeddings)

    if config.debug and debug_rows:
        debug_path = config.output_path.parent / "debug.csv"
        pd.DataFrame(debug_rows).to_csv(debug_path, index=False)
        print(f"debug: {debug_path}")


if __name__ == "__main__":
    main()
