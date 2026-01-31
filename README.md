ryushicup2026 baseline

概要
- vLLM(OpenAI互換)のQwen3 VLで動画キャプションを生成
- train.csvのprompt_enとTF-IDF類似度で近傍埋め込みを推定
- 予測埋め込みをsubmission.csvで出力

前提
- ffmpeg/ffprobe が PATH にあること
- vLLM サーバーが localhost で起動していること

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
  --top-k 5 \
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
  --top-k 5 \
  --debug \
  --allow-missing-ffmpeg
```
