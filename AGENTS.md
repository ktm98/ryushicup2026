
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
本コンペは、テキスト生成動画モデル Wan 2.2 によって生成された動画を対象に、「どのプロンプトを元に生成された動画なのか」を推定するタスクです。

参加者には、プロンプトIDが付与された学習用動画と、プロンプトIDが伏せられたテスト動画が提供されます。各テスト動画に対して、最も可能性の高いプロンプトIDを上位3件まで予測し、提出していただきます。

評価は Mean Average Cosine Similarity（平均コサイン類似度） により行われます。 これにより、完全一致だけでなく「内容として近いプロンプト」をより高く評価できる設計になっています。

Start

6 minutes ago
Close
a day to go
Description
生成AIによる動画は、研究・教育・広告・エンタメなどあらゆる領域に急速に浸透し、オープンな動画生成モデル（例：Wan2.2）の登場によって、個人や小規模チームでも高品質な映像を扱える時代になりました。こうした環境は創作の可能性を大きく広げる一方で、「その映像は何を意図して作られたのか」「どんな指示（プロンプト）に基づくのか」「出所や編集履歴をどう検証するのか」といった、透明性・説明可能性・真正性の課題を社会に突きつけています。実際に、コンテンツの来歴（provenance）を扱う標準仕様（例：C2PA / Content Credentials）や、合成コンテンツの表示・ラベリングに関する制度設計・ガイダンスの議論が国際的に進んでいます。

本コンペは、その核心にある問い――「生成物から生成意図（プロンプト）を読み解けるか」――に、データサイエンスとして挑む競技です。動画を“それっぽく説明する”だけでなく、映像の主題・動作・構図・質感・時間変化を手がかりに、生成モデルが反応しやすい言語表現としてのプロンプトへ落とし込み、最も整合する答えを推定してください。これは、生成AI時代のリテラシー向上に直結するだけでなく、監査や検証の補助技術（来歴情報が欠落した場合の推定、意図の説明、解析可能性の向上）としても重要な基礎能力になります。

Background
合成（synthetic）コンテンツのリスク低減には、来歴（provenance）の記録・認証、透かし・メタデータ、検出技術など多層の対策が必要と整理されています

“出所の証明”を支える標準化が進行しており、C2PA / Content Credentials のように、作成者・編集履歴などの主張を暗号学的に束ねて扱う枠組みが普及しつつあります

制度・ガイドラインの面でも、合成であることの明示（ラベリング）や検知可能性を重視する流れが強まっています 研究面でも、生成物から元プロンプトを復元する Prompt Inversion / Prompt Recovery は活発なテーマです

Task
入力：短尺の生成動画（Wan2.2 により生成）
出力：対応するプロンプトのembedding
この問題は、いわゆるキャプション生成に近い側面を持ちながらも、目的は「説明文」ではなく「生成意図としてのプロンプト」を当てる点にあります。映像の細部と時間方向の変化を読み取り、モデルがその映像を出力しやすい“指示の言語”として再構成することが鍵になります。

Evaluation
評価は Mean of the Cosine Similarity within Subgroups（サブグループ内コサイン類似度の平均） により行われます。 あらかじめ定義されたサブグループ（例：カテゴリや難易度、生成条件など）ごとに、予測結果と正解（参照）との コサイン類似度 を算出し、そのサブグループ内で平均を取ります。最終スコアは、各サブグループで得られた平均値を統合して算出され、値が高いほど高評価となります。

この指標は、全体平均だけでは見えにくい「特定条件下での性能差」を反映しやすく、データの偏りに引っ張られにくい形でモデルの汎化性能を評価できるよう設計されています。

## dataset discription
Datasetの説明
概要
本データセットは、英語プロンプト（prompt_en）と対応する動画（mp4）のペアを含みます。
学習用（train）には「プロンプト＋動画」が、テスト用（test）には「プロンプト」が用意されています。提出形式は sample_submission.csv を参照してください。

データ構成
2つの動画が入ったディレクトリと、3つのcsvファイルから構成されています。 ディレクトリ

train_movie/
数字.mp4 形式の動画が 30件
test_movie/
数字.mp4 形式の動画が 70件
ファイル

train.csv

学習用メタデータ（row_id, prompt_en, video_id）
video_id
train_movie/ 内の mp4 ファイル名（拡張子を除く数字）に対応
prompt_en
動画生成に用いた英語プロンプト
emb_{i}
embeddingの次元を表す。iは0~383までの値をとる。
sample_submission.csv

提出ファイルのサンプル（この形式に合わせて提出してください）
こちらの"emb_id"カラムは、video_idとembeddingの次元を"_"で繋いだものです
例えば、"31-0"ならvideo_id=31の0次元目です。
フォルダ構造
train_movie/{video_id}.mp4
test_movie/{video_id}.mp4
※ {video_id} は 半角数字 を想定しています（例: 0.mp4, 12.mp4 など）

動画ファイルについて
動画ファイル名は 数字.mp4 形式です
video_id は「mp4のファイル名（拡張子除く）」として扱ってください

embedding作成コード
```python
import pandas as pd
from sentence_transformers import SentenceTransformer

train_df = pd.read_csv('/kaggle/input/dummy-123455/train.csv')

test_prompt = "happy new year! 2026"

model = SentenceTransformer('all-MiniLM-L6-v2')

# embedding
embeddings = model.encode([test_prompt])

# 列名を作る
embedding_cols = [f"emb{i}" for i in range(embeddings.shape[1])]

# DataFrame化（1行）
embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)

embedding_df.head()

```

