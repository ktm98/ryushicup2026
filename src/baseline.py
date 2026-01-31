from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import shutil

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    top_k: int
    debug: bool
    prompt_text: str
    allow_missing_ffmpeg: bool


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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--allow-missing-ffmpeg", action="store_true")
    parser.add_argument(
        "--prompt-text",
        type=str,
        default=(
            "Describe the video briefly with the main subject, action, and setting."
        ),
    )
    args = parser.parse_args()
    return Config(
        input_dir=args.input_dir,
        train_csv=args.train_csv,
        test_movie_dir=args.test_movie_dir,
        output_path=args.output_path,
        vllm_url=args.vllm_url,
        model=args.model,
        max_frames=args.max_frames,
        top_k=args.top_k,
        debug=args.debug,
        prompt_text=args.prompt_text,
        allow_missing_ffmpeg=args.allow_missing_ffmpeg,
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


def extract_frames(video_path: Path, max_frames: int) -> List[str]:
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
    with tempfile.TemporaryDirectory() as temp_dir:
        out_pattern = Path(temp_dir) / "frame_%03d.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
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


def call_vllm_chat(
    vllm_url: str,
    model: str,
    prompt_text: str,
    image_data_urls: Iterable[str],
    timeout_sec: int = 60,
) -> str:
    """vLLM OpenAI互換APIを呼び出す.

    Args:
        vllm_url: ベースURL.
        model: モデル名.
        prompt_text: 指示文.
        image_paths: 画像パス.
        timeout_sec: タイムアウト.

    Returns:
        str: 生成テキスト.

    Raises:
        RuntimeError: 応答が不正な場合.
    """
    content = [{"type": "text", "text": prompt_text}]
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


def build_tfidf(train_prompts: List[str]) -> TfidfVectorizer:
    """TF-IDFを構築する.

    Args:
        train_prompts: 学習プロンプト.

    Returns:
        TfidfVectorizer: 学習済みベクトライザ.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(train_prompts)
    return vectorizer


def predict_embedding(
    caption: str,
    vectorizer: TfidfVectorizer,
    train_matrix: np.ndarray,
    train_embeddings: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """キャプションから埋め込みを推定する.

    Args:
        caption: 生成キャプション.
        vectorizer: TF-IDF.
        train_matrix: 学習文書行列.
        train_embeddings: 学習埋め込み.
        top_k: 上位数.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, float]]]: 予測埋め込みと近傍情報.
    """
    query_vec = vectorizer.transform([caption])
    sims = cosine_similarity(query_vec, train_matrix)[0]
    if np.allclose(sims, 0):
        pred = train_embeddings.mean(axis=0)
        return pred, []
    top_k = max(1, min(top_k, len(sims)))
    top_idx = np.argsort(sims)[-top_k:][::-1]
    weights = sims[top_idx]
    weights = weights / (weights.sum() + 1e-8)
    pred = np.average(train_embeddings[top_idx], axis=0, weights=weights)
    neighbors = [(int(i), float(sims[i])) for i in top_idx]
    return pred, neighbors


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


def load_train_data(train_csv: Path) -> Tuple[List[str], np.ndarray]:
    """学習データを読み込む.

    Args:
        train_csv: 学習CSVパス.

    Returns:
        Tuple[List[str], np.ndarray]: プロンプトと埋め込み.
    """
    df = pd.read_csv(train_csv)
    prompt_list = df["prompt_en"].astype(str).tolist()
    emb_cols = [col for col in df.columns if col.startswith("emb_")]
    embeddings = df[emb_cols].to_numpy(dtype=np.float32)
    return prompt_list, embeddings


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
    if ffmpeg_missing and not config.allow_missing_ffmpeg:
        raise RuntimeError("ffmpeg/ffprobe が見つかりません")
    if ffmpeg_missing and config.allow_missing_ffmpeg:
        print("警告: ffmpeg/ffprobe が見つかりません。ダミー画像で続行します。")
    if not config.train_csv.exists():
        raise RuntimeError("学習CSVが見つかりません")
    prompts, train_embeddings = load_train_data(config.train_csv)
    vectorizer = build_tfidf(prompts)
    train_matrix = vectorizer.transform(prompts)
    test_videos = list_test_videos(config.test_movie_dir)

    predictions = []
    debug_rows = []
    for video_path in test_videos:
        start_time = time.time()
        if ffmpeg_missing:
            frames = [dummy_image_data_url()]
        else:
            frames = extract_frames(video_path, config.max_frames)
        caption = call_vllm_chat(
            config.vllm_url,
            config.model,
            config.prompt_text,
            frames,
        )
        pred, neighbors = predict_embedding(
            caption, vectorizer, train_matrix, train_embeddings, config.top_k
        )
        pred = l2_normalize(pred)
        predictions.append(pred)
        if config.debug:
            debug_rows.append(
                {
                    "video_id": int(video_path.stem),
                    "caption": caption,
                    "neighbors": json.dumps(neighbors, ensure_ascii=True),
                    "elapsed_sec": round(time.time() - start_time, 3),
                }
            )
            print(f"{video_path.name}: {caption}")

    video_ids = [int(p.stem) for p in test_videos]
    save_submission(config.output_path, video_ids, np.array(predictions))

    if config.debug and debug_rows:
        debug_path = config.output_path.parent / "debug.csv"
        pd.DataFrame(debug_rows).to_csv(debug_path, index=False)
        print(f"debug: {debug_path}")


if __name__ == "__main__":
    main()
