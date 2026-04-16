# DeepSeek → Hugging Face 知識蒸留

DeepSeek APIを**教師モデル**として使い、小さなHugging Faceモデルを知識蒸留で訓練するパイプラインです。

## ファイル構成

```
distillation/
├── config.yaml               # 設定ファイル
├── distillation_dataset.py   # Step 1: DeepSeekから教師出力を収集
├── train_student.py          # Step 3: 学生モデルの訓練
├── requirements.txt
├── makedate.py           # Step 2: プロンプトの作成
└── data/
    ├── prompts.jsonl         # 入力プロンプト
    └── distillation_data.jsonl  # 収集した教師出力（自動生成）
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

### Step 0: APIキーの設定

`config.yaml` を編集するか、環境変数で設定：
```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

### Step 1: サンプルプロンプトを生成（初回のみ）

```bash
python distillation_dataset.py --config config.yaml --create-sample
```

または独自の `data/prompts.jsonl` を用意：
```jsonl
{"prompt": "機械学習とは何ですか？"}
{"prompt": "Pythonの特徴を教えてください。", "system": "あなたは親切なアシスタントです。"}
```

### Step 2: DeepSeekから教師出力を収集

```bash
python distillation_dataset.py --config config.yaml
```

- 途中で中断しても**自動再開**（resume）します
- 出力は `data/distillation_data.jsonl` に保存

### Step 3: 学生モデルを訓練

```bash
python train_student.py --config config.yaml
```

- ベストモデルは `student_model/best_model/` に保存
- チェックポイントは `student_model/checkpoint-{step}/` に保存

## 損失関数

```
L = α × KLDiv(student ‖ teacher) + (1-α) × CrossEntropy(labels, student)
```

| パラメータ | 説明 | デフォルト |
|-----------|------|---------|
| `temperature` | 蒸留温度 T（大きいほどソフトなラベル） | 4.0 |
| `alpha` | KLDivLoss の重み | 0.7 |

## 推奨学生モデル（日本語）

| モデル | パラメータ数 | 特徴 |
|--------|------------|------|
| `rinna/japanese-gpt2-medium` | 336M | 日本語GPT-2 |
| `cyberagent/open-calm-small` | 160M | 軽量日本語モデル |
| `line-corporation/japanese-large-lm-1.7b` | 1.7B | 高性能 |

英語の場合は `distilgpt2`、`facebook/opt-125m` なども利用可能。

## 注意事項

- DeepSeek APIの **利用規約**を確認し、生成データの商用利用が許可されているか確認してください
- `logprobs=True` は DeepSeek の対応状況によって変わる場合があります
- GPU がない場合は `torch.float32` で動作しますが、訓練時間が大幅に増加します
