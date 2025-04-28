# LLM エラー修正分析

このプロジェクトは、1つのトークンだけ異なる文のペアを言語モデル（LLM）に入力した際の活性化の差異を分析することで、エラー修正メカニズムを調査します。仮説として、複数の異なる文ペアにおいて特定の次元（またはレイヤー）が一貫して小さな活性化変化を示す場合、それらの次元（またはレイヤー）がエラー修正に寄与している可能性があります。

## プロジェクト概要

このプロジェクトは主に3つのコンポーネントで構成されています：

1. **データセット生成**: ChatGPT APIを使用して、1つのトークンだけ異なる文のペアを作成します。
2. **活性化分析**: 各文ペアに対してLlama-3-7Bモデルから活性化を抽出し、比較します。
3. **結果の保存と分析**: 活性化の差異を保存し、パターンを特定するために分析します。

## インストール方法

### 前提条件

- Linux環境（GPUセットアップ済み）
- Python 3.8以上
- Git
- rye（Pythonパッケージマネージャー）

### 基本的なセットアップ

このプロジェクトでは、ryeを使用してPythonモジュールのインストールを行います。

```bash
# リポジトリをクローン
git clone https://github.com/Maxel31/llm-layer-exchange-analysis.git
cd llm-layer-exchange-analysis

# ryeを使用して依存関係をインストール
rye sync
```

### GPUセットアップ

このプロジェクトでは、以下のコードを使用してGPUデバイスを設定します：

```python
def setup_device(gpu_id: str = "0") -> torch.device:
    """GPUが利用可能かどうかを確認し、適切なデバイスを返す"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス: {device} (GPU ID: {gpu_id})")
    return device
```

Cursorとの連携については、[Cursor連携ガイド](docs/cursor_integration_guide_ja.md)を参照してください。

## 使用方法

1. OpenAI APIキーを`.env`ファイルに設定します：
```
OPENAI_API_KEY=あなたのAPIキーをここに入力
```

2. 文ペアのデータセットを生成します：
```bash
python -m src.data_generation.generate_dataset --num_pairs 100 --output_dir data
```

3. 活性化を分析し、結果を保存します：
```bash
python -m src.model_analysis.analyze_activations --dataset_path data/sentence_pairs_YYYYMMDD_HHMMSS.json --output_dir results
```

4. 結果を可視化します：
```bash
python -m src.visualization.visualize_activations --results_path results/activation_results_YYYYMMDD_HHMMSS.json
```

5. Web UIで結果を表示します：
```bash
cd src/web_ui
python run.py --results_dir ../../results/results
```
その後、ブラウザで http://localhost:5000 にアクセスします。

## プロジェクト構造

```
llm-error-correction/
├── LICENSE                # MITライセンス
├── README.md              # このファイル
├── requirements.txt       # 依存関係リスト
├── .gitignore             # Gitが無視するファイルリスト
├── .env                   # 環境変数（APIキーなど）
├── data/                  # 生成されたデータセットの保存ディレクトリ
├── results/               # 分析結果の保存ディレクトリ
├── docs/                  # ドキュメント
│   ├── cursor_integration_guide_ja.md  # Cursor連携ガイド（日本語）
│   ├── gpu_setup_and_cursor_integration.md  # GPU設定ガイド
│   └── testing_instructions_ja.md      # テスト手順書（日本語）
├── src/                   # ソースコード
│   ├── __init__.py
│   ├── data_generation/   # データセット生成スクリプト
│   │   ├── __init__.py
│   │   ├── generate_dataset.py  # データセット生成メイン
│   │   └── utils.py      # ユーティリティ関数
│   ├── model_analysis/    # モデル活性化分析スクリプト
│   │   ├── __init__.py
│   │   ├── analyze_activations.py  # 活性化分析メイン
│   │   └── model_utils.py  # モデル関連ユーティリティ
│   ├── visualization/     # 結果可視化スクリプト
│   │   ├── __init__.py
│   │   └── visualize_activations.py  # 可視化メイン
│   ├── web_ui/           # Web UI
│   │   ├── app.py        # Flaskアプリケーション
│   │   ├── run.py        # 実行スクリプト
│   │   ├── requirements.txt  # Web UI依存関係
│   │   ├── templates/    # HTMLテンプレート
│   │   └── static/       # 静的ファイル（JS、CSS）
│   └── test_activation_analysis.py  # テスト用スクリプト
```

## 研究手法

研究手法は以下の手順で進められます：

1. **文ペアの作成**: 1つのトークンだけ異なる文のペアを生成し、タイプ別（意味の違い、文法エラーなど）に分類します。
2. **活性化の抽出**: これらの文ペアをLlama-3-7Bモデルに入力し、各レイヤーと次元の活性化を抽出します。
3. **差異の分析**: 活性化の差異が最小となる次元を特定し、エラー修正行動を示す可能性があるかを分析します。

## テスト方法

詳細なテスト手順については、[テスト手順書](docs/testing_instructions_ja.md)を参照してください。

## トラブルシューティング

- **メモリエラー**: Llama-3-7Bモデルは大量のメモリを必要とします。メモリ不足エラーが発生した場合は、`--load_in_8bit`または`--load_in_4bit`オプションを使用して量子化を有効にしてください。
- **APIエラー**: OpenAI APIでエラーが発生した場合は、APIキーが正しく設定されていることを確認し、レート制限に達していないか確認してください。
- **GPUエラー**: CUDA関連のエラーが発生した場合は、適切なCUDAバージョンがインストールされていることを確認してください。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています - 詳細は[LICENSE](LICENSE)ファイルを参照してください。
