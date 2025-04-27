# LLM層操作による次単語予測性能変化の分析

大規模言語モデル（LLM）の内部表現と性能の関係を理解するための研究プロジェクト。このリポジトリでは、LLMの各層を削除または交換したときの次単語予測性能の変化を分析するためのツールとコードを提供しています。

## 研究の背景と目的

大規模言語モデルの内部表現がどのように構成されているかを理解することは、モデルの動作原理を解明し、より効率的で解釈可能なモデルの開発につながります。本研究では特に：

- 各層の削除または隣接層の交換が次単語予測性能に与える影響
- 品詞タグごとの性能変化の分析
- 複数のモデルアーキテクチャ間での比較

を通じて、LLMの内部表現の特性を明らかにすることを目指しています。

## 機能と分析ステップ

このツールは以下の3つのステップで分析を行います：

1. **レイヤー操作と性能評価**: 各層の削除または隣接層の交換を行い、次単語予測性能を評価
2. **品詞タグ分析**: レイヤー操作の影響を品詞タグごとに分析
3. **モデル比較**: 複数モデルの分析結果を比較し、可視化

## ファイル構成

- `src/experiments_layer_removal_and_exchange.py` - **Step1**: 隣接層の交換/層の除去による次単語予測性能の評価、jsonファイルへの結果保存 
- `src/analysis_performance_by_pos_tag.py` - **Step2**: Step1で生成されたjsonファイルを分析し、品詞タグごとの性能変化を分析
- `src/comparison_figures.py` - **Step3**: 複数モデルの品詞タグ分析結果を比較するスクリプト、各モデルのPOS解析結果を1つのプロットに重ねて表示

## 使い方

### 環境構築

```bash
# 仮想環境の作成と依存関係のインストール
python -m venv .venv
source .venv/bin/activate  # Linuxの場合
# または
.venv\Scripts\activate  # Windowsの場合

pip install -r requirements.txt
```

### Step1: レイヤー操作と性能評価

```bash
python src/experiments_layer_removal_and_exchange.py --model_name meta-llama/Meta-Llama-3-8B \
                            --dataset_type wiki-text-2 \
                            --max_samples 1000 \
                            --experiment removal \
                            --output_dir results \
                            --gpu_id 0
```

オプション:
- `--model_name`: 分析するモデル名
- `--dataset_type`: 使用するデータセット（wiki-text-2, gpt2-output-dataset, wiki-text-103, bookcorpus）
- `--max_samples`: 評価するサンプル数
- `--experiment`: 実験タイプ（removal: 層削除, exchange: 層交換）
- `--output_dir`: 結果を保存するディレクトリ
- `--gpu_id`: 使用するGPUのID

### Step2: 品詞タグ分析

```bash
python src/analysis_performance_by_pos_tag.py --dataset_type wiki-text-2 \
                                     --experiment removal \
                                     --sample_size 1000 \
                                     --model_name Meta-Llama-3-8B
```

### Step3: モデル比較

```bash
python src/comparison_figures.py --json_files results/removal/meta_llama_Meta_Llama_3_8B_wiki-text-2_n1000/ppl.json \
                                results/removal/meta_llama_Meta_Llama_3_8B_Instruct_wiki-text-2_n1000/ppl.json \
                                --experiment removal \
                                --output_dir figures/comparisons
```

## 実験結果の例

各実験によって生成される結果は、指定した出力ディレクトリに保存されます。主な出力ファイルは以下の通りです：

- 層操作実験の結果JSONファイル（例: `results/removal/meta_llama_Meta_Llama_3_8B_wiki-text-2_n1000/ppl.json`）
- 品詞タグ分析の結果と図表
- モデル比較グラフ

## 対応モデル

このツールは以下のモデルに対応しています：
- Meta-Llama-3シリーズ
- LLaMA 2シリーズ
- Mistral
- Mixtral
- 他のTransformersライブラリでサポートされているデコーダーモデル

## ライセンス

このプロジェクトは[MITライセンス](LICENSE)の下で公開されています。

## 引用

このプロジェクトを研究で使用される場合は、以下の形式で引用してください：

```
@misc{LLMLayerAnalysis2025,
  author = {Authors},
  title = {Characterization of the Availability of Layer Exchange in Large Language Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/characterization-of-the-avalilablity-of-layer-exchange}
}
```