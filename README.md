ryushicup2026 baseline

概要
- vLLM(OpenAI互換)のQwen3 VLで動画キャプションを生成
- SentenceTransformer(all-MiniLM-L6-v2)でキャプションを埋め込み
- 予測埋め込みをsubmission.csvで出力
- train.csvのprompt_enからトークン上限を推定して出力長を制限

前提
- ffmpeg/ffprobe が PATH にあること
- vLLM サーバーが localhost で起動していること
- SentenceTransformer を使う場合は初回にモデルがダウンロードされます
- vLLM への同時リクエスト数は --num-workers で調整できます（OOM対策で既定は1）
- 動画をそのまま渡す場合は --use-video-input を指定してください
- vLLM側で `Unknown part type: video` が出る場合は、動画入力は未対応のためフレーム入力へ切り替えてください
- フレームのサイズを固定する場合は --frame-size を使います（例: 512x512、縦横比は保持してパディング）

モックサーバー起動
```
python src/mock_vllm_server.py --host 127.0.0.1 --port 8000 --debug
```

ベースライン実行
```
python src/baseline.py \
  --train-csv input/data/train.csv \
  --test-movie-dir input/data/test_movie \
  --output-path results/submission.csv \
  --vllm-url http://localhost:8000 \
  --model Qwen/Qwen3-VL \
  --max-frames 8 \
  --use-train-style \
  --token-limit-percentile 95 \
  --num-workers 1 \
  --sbert-device cpu \
  --sbert-batch-size 8 \
  --frame-size 512x512 \
  --debug
```

動画をそのまま入力（ffmpeg不要）
```
python src/baseline.py --train-csv input/data/train.csv --test-movie-dir input/data/test_movie --output-path results/submission.csv --vllm-url http://localhost:8000 --model Qwen/Qwen3-VL --use-video-input --use-train-style --token-limit-percentile 95 --num-workers 1 --sbert-device cpu --sbert-batch-size 8 --debug
```

video入力でBad Requestが出る場合
- `--video-payload-mode single` と `--video-content-type` / `--video-field` / `--video-data-format` / `--video-as-object` を調整してください
- 例: `--video-payload-mode single --video-content-type video --video-field data --video-data-format base64 --video-as-object`
 - `Unknown part type: video` が出る場合は `video_url` が正です（`--video-content-type video_url` を使う）

ffmpeg/ffprobe が無い場合（デバッグ用）
```
python src/baseline.py \
  --train-csv input/data/train.csv \
  --test-movie-dir input/data/test_movie \
  --output-path results/submission.csv \
  --vllm-url http://localhost:8000 \
  --model Qwen/Qwen3-VL \
  --max-frames 8 \
  --debug \
  --allow-missing-ffmpeg
```
