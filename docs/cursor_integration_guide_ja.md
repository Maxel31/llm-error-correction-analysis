# Cursorとの連携ガイド

このガイドでは、LLM Error Correction Analysisプロジェクトを[Cursor](https://cursor.sh/)エディタで開き、編集する方法と、GPU環境でLlama-3-7Bモデルを実行する方法について説明します。

## 1. GitHubリポジトリの作成とプッシュ

### 1.1 GitHubリポジトリの作成

1. [GitHub](https://github.com/)にログインします
2. 右上の「+」ボタンをクリックし、「New repository」を選択します
3. リポジトリ名を「llm-error-correction-analysis」と入力します
4. 説明（Description）に「Investigating error correction in language models by analyzing activation differences」と入力します
5. リポジトリを「Public」に設定します
6. 「Create repository」ボタンをクリックします

### 1.2 ローカルプロジェクトのプッシュ

ターミナルで以下のコマンドを実行します：

```bash
# プロジェクトディレクトリに移動
cd /path/to/llm-error-correction

# Gitリポジトリを初期化（すでに初期化されている場合はスキップ）
git init

# すべてのファイルをステージング
git add .

# 初期コミットを作成
git commit -m "Initial commit"

# リモートリポジトリを追加
git remote add origin https://github.com/Maxel31/llm-error-correction-analysis.git

# メインブランチをプッシュ
git push -u origin main
```

## 2. CursorでGitHubリポジトリをクローン

### 2.1 Cursorのインストール

1. [Cursor公式サイト](https://cursor.sh/)からCursorをダウンロードしてインストールします
2. Cursorを起動します

### 2.2 GitHubリポジトリのクローン

1. Cursorを起動します
2. 左側のサイドバーで「Source Control」アイコン（分岐アイコン）をクリックします
3. 「Clone Repository」ボタンをクリックします
4. 「https://github.com/Maxel31/llm-error-correction-analysis.git」と入力します
5. リポジトリを保存する場所を選択します
6. 「Clone」ボタンをクリックします

これで、プロジェクトがCursorで開かれ、編集できるようになります。

## 3. GPU環境のセットアップ

Llama-3-7Bのような大規模言語モデルを実行するには、十分なGPUリソースが必要です。以下のオプションがあります：

### 3.1 Google Colabの利用

1. [Google Colab](https://colab.research.google.com/)にアクセスします
2. 新しいノートブックを作成します
3. GPUランタイムを有効にします：
   - 「ランタイム」→「ランタイムのタイプを変更」→「ハードウェアアクセラレータ」→「GPU」を選択
4. GitHubからリポジトリをクローンします：

```python
!git clone https://github.com/Maxel31/llm-error-correction-analysis.git
%cd llm-error-correction-analysis
!pip install -r requirements.txt
```

5. 必要なコードを実行します：

```python
# データセットの生成
!python -m src.data_generation.generate_dataset

# アクティベーション分析
!python -m src.model_analysis.analyze_activations --dataset_path data/sentence_pairs_YYYYMMDD_HHMMSS.json
```

### 3.2 リモートGPUサーバーの利用

SSH経由でリモートGPUサーバーに接続する場合：

1. SSHでサーバーに接続します：

```bash
ssh username@your-gpu-server.com
```

2. リポジトリをクローンします：

```bash
git clone https://github.com/Maxel31/llm-error-correction-analysis.git
cd llm-error-correction-analysis
pip install -r requirements.txt
```

3. CursorからSSH経由でリモートサーバーに接続します：
   - Cursorを起動します
   - 「File」→「Open Remote Repository」を選択します
   - SSHの接続情報を入力します（例：`username@your-gpu-server.com:/path/to/llm-error-correction-analysis`）

### 3.3 Hugging Faceインフェレンスエンドポイントの利用

Hugging Faceのインフェレンスエンドポイントを使用すると、GPUサーバーを自分で管理する必要がなくなります：

1. [Hugging Face](https://huggingface.co/)にアクセスし、アカウントを作成します
2. Llama-3-7Bモデルのインフェレンスエンドポイントを設定します
3. APIキーを取得します
4. コードを以下のように修正して、Hugging FaceのAPIを使用します：

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="meta-llama/Llama-3-7b-hf", token="YOUR_HF_TOKEN")

# モデルの出力を取得
response = client.text_generation(prompt="Your prompt here", max_new_tokens=100)
```

## 4. プロジェクトの実行

### 4.1 データセットの生成

```bash
# .envファイルにOpenAI APIキーが設定されていることを確認
python -m src.data_generation.generate_dataset
```

### 4.2 アクティベーション分析

```bash
python -m src.model_analysis.analyze_activations --dataset_path data/sentence_pairs_YYYYMMDD_HHMMSS.json
```

## 5. トラブルシューティング

### 5.1 CUDA関連のエラー

```bash
# CUDAバージョンの確認
python -c "import torch; print(torch.version.cuda)"

# GPUメモリの確認
nvidia-smi
```

### 5.2 メモリ不足エラー

Llama-3-7Bモデルは大量のメモリを必要とします。メモリ不足エラーが発生した場合：

1. `--device cpu` オプションを使用してCPUで実行する（非常に遅くなります）
2. より小さなモデルを使用する
3. バッチサイズを小さくする
4. 8ビット量子化を使用する：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True
)
```

## 6. 参考リソース

- [Cursor エディタ ドキュメント](https://cursor.sh/docs)
- [Hugging Face Transformers ドキュメント](https://huggingface.co/docs/transformers/index)
- [PyTorch CUDA ドキュメント](https://pytorch.org/docs/stable/cuda.html)
- [Google Colab ドキュメント](https://colab.research.google.com/)
