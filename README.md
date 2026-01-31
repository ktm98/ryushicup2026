## Baseline

前立腺上皮セグメンテーションのベースラインは `src/baseline.py` です。入力は `input/`、出力は `results/` がデフォルトです。`--context-slices` のデフォルトは 1 で、2.5D (前後1枚) です。

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
- `--arch`: アーキテクチャ（`unetplusplus` など）
- `--encoder`: エンコーダ名（デフォルト: `timm-convnext_tiny`）。未対応の場合は `resnet34` に自動フォールバックします。
- `--loss`: 損失関数（`dice_bce` など）
- `--tta`: 推論時TTA
- `--min-area`: 小さい予測マスクの除外
- `--amp`: AMPの有効化
- `--context-slices`: 2.5D用の前後スライス数（0で2D）
- `--auto-threshold`: 検証でしきい値を探索
- `--thresholds`: しきい値候補の明示指定
- `--sampler`: 学習サンプラー（`weighted` など）
- `--pos-boost`: 正例サンプルの重み強化

`--help` で詳細を確認できます。
