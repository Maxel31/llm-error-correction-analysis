# GPU環境のセットアップとCursorとの連携ガイド

このドキュメントでは、LLM Error Correction Analysisプロジェクトを実行するためのGPU環境のセットアップ方法と、Cursorエディタとの連携方法について説明します。

## 1. GPU環境のセットアップ

Llama-3-7Bのような大規模言語モデルを実行するには、十分なGPUリソースが必要です。以下のオプションがあります：

### オプション1: ローカルGPU環境

ローカルマシンに適切なGPUがある場合：

```bash
# 必要なパッケージのインストール
pip install -r requirements.txt

# CUDAがインストールされていることを確認
python -c "import torch; print(torch.cuda.is_available())"
```

### オプション2: クラウドGPU環境（Google Colab）

Google Colabを使用する場合：

1. プロジェクトをGitHubにプッシュします
2. Google Colabで新しいノートブックを作成します
3. 以下のコードを実行してリポジトリをクローンします：

```python
!git clone https://github.com/yourusername/llm-error-correction.git
%cd llm-error-correction
!pip install -r requirements.txt
```

4. GPUランタイムを有効にします（「ランタイム」→「ランタイムのタイプを変更」→「ハードウェアアクセラレータ」→「GPU」）

### オプション3: リモートGPUサーバー

SSH経由でリモートGPUサーバーに接続する場合：

```bash
# リモートサーバーに接続
ssh username@your-gpu-server.com

# リポジトリをクローン
git clone https://github.com/yourusername/llm-error-correction.git
cd llm-error-correction

# 環境のセットアップ
pip install -r requirements.txt
```

## 2. Cursorエディタとの連携

[Cursor](https://cursor.sh/)はAI機能を備えたコードエディタです。以下の方法でプロジェクトをCursorで開くことができます：

### ローカル環境でCursorを使用する場合

1. Cursorをダウンロードしてインストールします（[https://cursor.sh/](https://cursor.sh/)）
2. Cursorを起動し、「File」→「Open Folder」からプロジェクトフォルダを選択します
3. プロジェクトのコードを編集できるようになります

### リモートサーバーとCursorを連携する場合

Cursorは、SSH経由でリモートサーバー上のコードを編集する機能をサポートしています：

1. Cursorを起動します
2. 「File」→「Open Remote Repository」を選択します
3. SSHの接続情報を入力します（例：`username@your-gpu-server.com:/path/to/llm-error-correction`）
4. 接続が確立されると、リモートサーバー上のコードを直接編集できます

## 3. プロジェクトの実行

GPU環境とCursorの設定が完了したら、以下の手順でプロジェクトを実行できます：

1. データセットの生成：

```bash
python -m src.data_generation.generate_dataset
```

2. アクティベーション分析：

```bash
python -m src.model_analysis.analyze_activations --dataset_path data/sentence_pairs_YYYYMMDD_HHMMSS.json
```

## 4. トラブルシューティング

### CUDA関連のエラー

```bash
# CUDAバージョンの確認
python -c "import torch; print(torch.version.cuda)"

# GPUメモリの確認
nvidia-smi
```

### メモリ不足エラー

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

## 5. 参考リソース

- [Hugging Face Transformers ドキュメント](https://huggingface.co/docs/transformers/index)
- [PyTorch CUDA ドキュメント](https://pytorch.org/docs/stable/cuda.html)
- [Cursor エディタ ドキュメント](https://cursor.sh/docs)
