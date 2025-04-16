# レイヤー削除分析ツール

Llama 3モデルの各層を削除したときの性能変化を分析するためのツールです。

## 修正とリファクタリングについて

元のノートブック実装にあった以下の問題を修正しました：

1. データを正しく参照できていない問題
2. 一部ケースでPPLやエラー率がNaNになる問題
3. コードの構造や可読性の改善

## ファイル構成

- `src/layer_removal_analysis.py` - メインの実装（Pythonモジュールとして使用可能）
- `src/test_layer_removal.py` - テスト用スクリプト
- `src/notebook/layer_removal_analysis.ipynb` - 元のノートブック
- `src/notebook/results/` - 結果出力ディレクトリ

## 使い方

### コマンドラインからの実行

```bash
# 1. 既存の結果ファイルを分析する
python src/test_layer_removal.py --mode result --result_path src/notebook/results/layer_removal_token_prediction_wiki-text-2_n1.json

# 2. 単一テキストでの評価をテストする
python src/test_layer_removal.py --mode text --text "The quick brown fox jumps over the lazy dog."

# 3. メインスクリプトを実行してレイヤー削除分析を行う
python src/layer_removal_analysis.py
```

### Pythonモジュールとして使用する

```python
from layer_removal_analysis import (
    setup_device, 
    load_model_and_tokenizer, 
    analyze_layer_removal,
    visualize_layer_removal_results
)

# デバイス設定
device = setup_device()

# モデルとトークナイザーを読み込む
model_name = 'meta-llama/Meta-Llama-3-8B'
model, tokenizer = load_model_and_tokenizer(model_name, device)

# 分析実行
results = analyze_layer_removal(
    model=model,
    tokenizer=tokenizer,
    max_samples=1,  # サンプル数
    output_dir="results",
    dataset_type="wiki-text-2"  # または "gpt2-output-dataset"
)

# 結果の可視化
visualize_layer_removal_results(results, "wiki-text-2", 1, language="ja")
visualize_layer_removal_results(results, "wiki-text-2", 1, language="en")
```

## 主な修正点

1. トークナイザーの正しい渡し方を実装
2. NaNやInfの値の適切な処理
3. リストアクセス前のインデックスチェック
4. エラーハンドリングの強化
5. ロギング機能の追加
6. 型ヒントと関数ドキュメントの追加
7. 変数名とコードフローの改善

## 要件

- Python 3.8以上
- PyTorch
- Transformers
- NumPy
- Matplotlib
- tqdm
- datasets

## 注意事項

- GPUメモリを大量に使用します
- モデル読み込みに時間がかかります
- 大きなサンプル数での実行は長時間かかります
