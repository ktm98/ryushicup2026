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
  --debug
```

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
