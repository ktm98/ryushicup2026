## Baseline

前立腺上皮セグメンテーションのベースラインは `src/baseline.py` です。入力は `input/`、出力は `results/` がデフォルトです。

### 例

`uv run python src/baseline.py`

### 主な引数

- `--input-dir`: 入力ディレクトリ（デフォルト: `input`）
- `--output-dir`: 出力ディレクトリ（デフォルト: `results`）
- `--epochs`: 学習エポック数
- `--batch-size`: バッチサイズ
- `--lr`: 学習率
- `--num-workers`: DataLoaderのワーカー数
- `--device`: 使用デバイス
- `--debug`: デバッグモード（小さなデータで実行）

`--help` で詳細を確認できます。
