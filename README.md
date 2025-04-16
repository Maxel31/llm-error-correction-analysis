# 隣接層の入れ替え/層の削除による次単語予測性能の変化の分析

LLMの各層を削除または交換したときの性能変化を分析するためのツール(For EACL2025)

## 概要

このツールは以下の3つのステップで分析を行います：

1. **レイヤー操作と性能評価**: 各層の削除または隣接層の交換を行い、次単語予測性能を評価
2. **品詞タグ分析**: レイヤー操作の影響を品詞タグごとに分析
3. **モデル比較**: 複数モデルの分析結果を比較し、可視化

## ファイル構成

- `src/experiments_layer_removal_and_exchange.py` - **Step1**: 隣接層の交換/層の除去による次単語予測性能の評価、jsonファイルへの結果保存 
- `src/analysis_performance_by_pos_tag.py` - **Step2**: Step1で生成されたjsonファイルを分析し、品詞タグごとの性能変化を分析
- `src/comparison_figures.py` - **Step3**: 複数モデルの品詞タグ分析結果を比較するスクリプト、各モデルのPOS解析結果を1つのプロットに重ねて表示

## 使い方

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

オプション:
- `--json_files`: 比較するモデルの分析結果JSONファイル（複数指定可能）
- `--experiment`: 分析実験の種類 (removal または exchange)
- `--output_dir`: 出力ディレクトリのパス