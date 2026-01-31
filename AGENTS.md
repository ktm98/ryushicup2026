

Pythonコード規約

ASCIIを基本とし、必要時のみ非ASCIIを使う
docstringはGoogle形式、日本語で記述する
例外メッセージは日本語、要点を短く

実行とI/O

デフォルトの入出力先は input/ と results/
設定は Config dataclass と argparse で上書きできるようにする
モジュールimport時に重い処理を実行しない
チェック

実行手順や引数の変更があれば README.md を更新する

ハイパラはargparseで指定する。
--helpでhelpを見れるようにする

共通化できそうなものがあれば、agent skillsを更新する

typing annotationを行う。

デバッグモードが実装する

uvでデバッグしてください。


コマンドは改行しないでください。



# competition task

Overview
Prostate Epithelium Segmentation Challenge
Important Note
本コンペティションは所属組織とは関係なく個人としての活動です。

Overview
前立腺生検サンプルの3D蛍光顕微鏡画像から、上皮組織（Epithelium）を正確に3Dセグメンテーションするモデルを構築してください。

Background
前立腺がんの診断において、上皮組織の形態学的評価は重要な役割を果たします。本コンペティションでは、TCIA (The Cancer Imaging Archive) の PCa_Bx_3Dpathology データセットから生成された画像を使用します。

元データは二重チャンネル蛍光顕微鏡画像（核染色 + 細胞質染色）で、H&E染色に類似した False Color 画像に変換されています。

Task
入力: 320×320 RGB画像（H&E-like False Color） 出力: 320×320 二値マスク（Epithelium = 1, Background = 0）

Timeline
開始: コンペ開始時刻
終了: 6時間後
最終提出締切: 終了時刻
Evaluation
Dice Coefficient で評価します。


X: 予測マスク
Y: Ground Truth マスク
各画像のDiceスコアの平均値がリーダーボードスコアとなります
Prizes
1位: 🥇
2位: 🥈
3位: 🥉
Rules
事前学習済みモデル（ImageNet等）の使用は許可
外部データの利用は許可。ただし、元データを利用したデータ水増しは不可
テストデータに対する疑似ラベル等は不可
Acknowledgments
The Competition Data is derived from the TCIA PCa_Bx_3Dpathology dataset, and the applicable data license is inherited from (and therefore tied to) the original source dataset. Accordingly, use of the Competition Data is subject to the CC BY 4.0 license. Citation: Liechti, R., et al. (2024). Three-dimensional imaging mass cytometry of human prostate cancer biopsies [Data set]. The Cancer Imaging Archive.

Data Source: TCIA PCa_Bx_3Dpathology
License: CC BY 4.0
Citation
Xie, W., Reder, N. P., Koyuncu, C. F., Leo, P., Hawley, S., Huang, H., Mao, C., POSTUPNA, N. A. D. I. A., kang, soyoung, Serafin, R., Gao, G., Han, Q., Bishop, K., Barner, L., Fu, P., Wright, J., Keene, C., Vaughan, J., Janowczyk, A., … Liu, J. (2023). 3D pathology of prostate biopsies with biochemical recurrence outcomes: raw H&E-analog datasets and image translation-assisted segmentation in 3D (ITAS3D) datasets (PCa_Bx_3Dpathology) (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/44MA-GX21
Start

19 minutes ago
Close
6 hours to go
Evaluation
Metric
このコンペティションは Dice Coefficient (Dice Score) で評価されます。

Formula

Where:

$X$ = 予測マスク（二値）
$Y$ = Ground Truth マスク（二値）
$|X \cap Y|$ = True Positive ピクセル数
$|X|$ = 予測でPositiveとしたピクセル数
$|Y|$ = Ground TruthでPositiveなピクセル数
Score Calculation
各テスト画像についてDice Scoreを計算
全画像のDice Scoreの平均値が最終スコア
Edge Cases
予測もGTも空の場合: Dice = 1.0
予測のみ空の場合: Dice = 0.0
GTのみ空の場合: Dice = 0.0
Python Implementation
def dice_score(pred, target):
    """
    Calculate Dice score.

    Args:
        pred: Binary prediction mask (H, W), values 0 or 1
        target: Binary ground truth mask (H, W), values 0 or 1

    Returns:
        Dice coefficient (0.0 to 1.0)
    """
    pred = pred.flatten()
    target = target.flatten()

    # Handle edge case: both empty
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0

    intersection = (pred * target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)
Submission Format
提出ファイルは CSV 形式で、以下のカラムを含む必要があります:

Column	Description
Id	テスト画像の識別子
Expected	RLE形式でエンコードされた予測マスク
RLE Encoding
def rle_encode(mask):
    """
    Run-length encode a binary mask.

    Args:
        mask: Binary mask (H, W), values 0 or 1

    Returns:
        RLE string
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(rle_string, shape):
    """
    Decode RLE string to binary mask.

    Args:
        rle_string: RLE encoded string
        shape: (height, width) of output mask

    Returns:
        Binary mask (H, W)
    """
    if not rle_string or rle_string == '':
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, rle_string.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1  # 1-indexed to 0-indexed
    ends = starts + np.array(lengths)

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1

    return mask.reshape(shape)




## dataset description
Dataset Description
Files
data/
├── train/
│   ├── images/          # 学習用画像 (JPG)
│   ├── labels/          # 学習用ラベル (PNG, multi-class)
│   └── train.csv        # メタデータ
├── test/
│   ├── images/          # テスト用画像 (JPG)
│   └── test.csv         # メタデータ
└── sample_submission.csv
Image Format
形式: JPEG (images), PNG (labels)
サイズ: 320×320 pixels
チャンネル: RGB (3チャンネル)
Train Data
train/images/: 学習用H&E-like画像
train/labels/: 対応するセグメンテーションラベル（マルチクラス）
0: Background
2: Epithelium（評価対象）
3: Lumens
4: Biopsy Region
train.csv:
image_id: 画像識別子
crop_id: 元の3Dクロップ識別子
slice_id: スライス番号 (0-63)
Test Data
test/images/: テスト用H&E-like画像
test.csv:
image_id: 画像識別子
crop_id: 元の3Dクロップ識別子
slice_id: スライス番号 (0-63)
Sample Submission
sample_submission.csv のフォーマット:

Id	Expected
test_0000	1 10 50 20 …
test_0001	100 5 200 15 …
RLE (Run-Length Encoding) 形式:

start1 length1 start2 length2 ...
ピクセルは左上から右下へ、行優先で番号付け
1-indexed (最初のピクセルは1)
空のマスクは空文字列 ""
RLE Example
マスク (3x3):
0 0 1
1 1 1
0 1 0

RLE: "3 1 4 3 8 1"
説明: 位置3から1ピクセル、位置4から3ピクセル、位置8から1ピクセル
Data Statistics
Split	Images	Crops	Note
Train	3,200	50	10 samples
Test	2,560	40	Public + Private
Important Notes
Patient-level split: 同一患者のデータがtrain/testで混在しないよう分割済み
Multi-class labels: 学習データには複数クラスのラベルが含まれるが、評価はEpithelium（クラス2）のみ
3D context: 同じcrop_idの画像は連続するスライス（隣接画像として利用可能）
ラベルの作り方*: 公開元の配布データのラベルがめちゃめちゃズレていたのでシルエット⇒FFTベースの位置合わせで補正しています。そのため若干のずれはあるかもしれないです。