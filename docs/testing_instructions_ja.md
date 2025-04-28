# テスト手順書

このドキュメントでは、LLMエラー修正プロジェクトのテスト方法について説明します。

## 目次

1. [環境設定](#環境設定)
2. [データセット生成のテスト](#データセット生成のテスト)
3. [活性化分析のテスト](#活性化分析のテスト)
4. [結果の保存と検査](#結果の保存と検査)
5. [トラブルシューティング](#トラブルシューティング)

## 環境設定

テストを実行する前に、以下の環境設定を行ってください：

1. リポジトリをクローンし、必要なパッケージをインストールします：

```bash
git clone https://github.com/Maxel31/llm-error-correction-analysis.git
cd llm-error-correction-analysis
pip install -r requirements.txt
```

2. OpenAI APIキーを設定します：

```bash
# .envファイルを作成
echo "OPENAI_API_KEY=あなたのOpenAIキー" > .env
```

3. GPUの設定（Llama-3-7Bモデルを実行するため）：
   - ローカルGPUがある場合：特に設定は不要です
   - Google Colabを使用する場合：`docs/gpu_setup_and_cursor_integration.md`を参照してください
   - リモートサーバーを使用する場合：`docs/gpu_setup_and_cursor_integration.md`を参照してください

## データセット生成のテスト

データセット生成をテストするには、以下のコマンドを実行します：

```bash
# 少数のペアでテスト（APIの過剰使用を避けるため）
python -m src.data_generation.generate_dataset --num_pairs 5 --output_dir data

# 完全なデータセットの生成（100ペア程度）
python -m src.data_generation.generate_dataset --num_pairs 17 --output_dir data
```

### 検証ポイント

1. 生成されたデータセットが`data`ディレクトリに保存されていることを確認します
2. JSONファイルを開き、以下の構造になっていることを確認します：
   - `metadata`：生成時間、使用モデル、トークナイザー情報
   - `diff_token_type`：文の種類（意味の違い、文法エラーなど）
   - 各タイプに複数のサブタイプが含まれていること
   - 各サブタイプに複数の文ペアが含まれていること

3. 文ペアが実際に1トークンだけ異なることを確認します：

```bash
# データセットの検証
python -c "
import json
from transformers import AutoTokenizer
from src.data_generation.utils import verify_token_difference

# データセットを読み込む
with open('data/sentence_pairs_YYYYMMDD_HHMMSS.json', 'r') as f:
    dataset = json.load(f)

# トークナイザーを読み込む
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3-7b-hf')

# 各ペアを検証
valid_count = 0
total_count = 0

for pair_type, subtypes in dataset['diff_token_type'].items():
    for subtype, data in subtypes.items():
        for pair in data['sentence_pairs']:
            is_valid, diff_count = verify_token_difference(
                pair['sentence1'], 
                pair['sentence2'], 
                tokenizer
            )
            total_count += 1
            if is_valid:
                valid_count += 1
            else:
                print(f'Invalid pair: {pair}')
                print(f'Differs by {diff_count} tokens')

print(f'Valid pairs: {valid_count}/{total_count} ({valid_count/total_count*100:.2f}%)')
"
```

## 活性化分析のテスト

活性化分析をテストするには、以下のコマンドを実行します：

```bash
# テスト用スクリプトを使用（少数のレイヤーと少数のペアでテスト）
python -m src.test_activation_analysis \
  --model_name meta-llama/Llama-3-7b-hf \
  --dataset_path data/sentence_pairs_YYYYMMDD_HHMMSS.json \
  --layer_indices 0,1,2 \
  --load_in_8bit \
  --visualize \
  --output_dir results

# メモリが少ない場合は4ビット量子化を使用
python -m src.test_activation_analysis \
  --model_name meta-llama/Llama-3-7b-hf \
  --dataset_path data/sentence_pairs_YYYYMMDD_HHMMSS.json \
  --layer_indices 0,1,2 \
  --load_in_4bit \
  --visualize \
  --output_dir results
```

### 検証ポイント

1. モデルが正常に読み込まれることを確認します
2. 活性化分析が正常に実行されることを確認します
3. 結果が`results/results`ディレクトリに保存されることを確認します
4. 可視化が`results/figures`ディレクトリに生成されることを確認します

## 結果の保存と検査

結果の保存と検査をテストするには、以下の手順を実行します：

1. 生成された結果ファイルを確認します：

```bash
# 結果ファイルを表示
python -c "
import json
import pprint

# 結果ファイルを読み込む
with open('results/results/activation_results_YYYYMMDD_HHMMSS.json', 'r') as f:
    results = json.load(f)

# メタデータを表示
print('Metadata:')
pprint.pprint(results['metadata'])

# 最初のペアの結果を表示
pair_type = list(results['diff_token_type'].keys())[0]
subtype = list(results['diff_token_type'][pair_type].keys())[0]
pair_id = list(results['diff_token_type'][pair_type][subtype].keys())[0]
pair_data = results['diff_token_type'][pair_type][subtype][pair_id]

print(f'\nSample pair ({pair_type}/{subtype}/{pair_id}):')
print(f'Sentence 1: {pair_data[\"sentence_pair\"][\"sentence1\"]}')
print(f'Sentence 2: {pair_data[\"sentence_pair\"][\"sentence2\"]}')

# 最初のレイヤーの最初の次元の活性化差を表示
layer = list(pair_data['layers'].keys())[0]
dim = list(pair_data['layers'][layer]['dimensions'].keys())[0]
dim_data = pair_data['layers'][layer]['dimensions'][dim]

print(f'\nActivation difference for {layer}, {dim}:')
print(f'Sentence 1 activation: {dim_data[\"sentence1_activation\"]}')
print(f'Sentence 2 activation: {dim_data[\"sentence2_activation\"]}')
print(f'Activation difference: {dim_data[\"activation_diff\"]}')
"
```

2. 最小活性化差を持つ次元を検査します：

```bash
# 最小活性化差を持つ次元を表示
python -c "
import json
import numpy as np

# 結果ファイルを読み込む
with open('results/results/activation_results_YYYYMMDD_HHMMSS.json', 'r') as f:
    results = json.load(f)

# 各レイヤーの最小活性化差を持つ次元を見つける
for pair_type, subtypes in results['diff_token_type'].items():
    for subtype, pairs in subtypes.items():
        for pair_id, pair_data in pairs.items():
            print(f'Pair: {pair_type}/{subtype}/{pair_id}')
            print(f'Sentence 1: {pair_data[\"sentence_pair\"][\"sentence1\"]}')
            print(f'Sentence 2: {pair_data[\"sentence_pair\"][\"sentence2\"]}')
            
            for layer, layer_data in pair_data['layers'].items():
                # 各次元の活性化差を取得
                diffs = [(dim, data['activation_diff']) for dim, data in layer_data['dimensions'].items()]
                
                # 活性化差でソート
                diffs.sort(key=lambda x: x[1])
                
                # 最小の5つを表示
                print(f'\\n{layer} - Top 5 dimensions with smallest activation differences:')
                for dim, diff in diffs[:5]:
                    print(f'{dim}: {diff}')
                
                print('\\n' + '-'*50)
            
            # 1つのペアだけ表示して終了
            break
        break
    break
"
```

3. 可視化結果を確認します：
   - `results/figures`ディレクトリに生成された画像ファイルを開きます
   - ヒートマップ、棒グラフ、散布図などが含まれていることを確認します

## トラブルシューティング

### APIエラー

OpenAI APIでエラーが発生した場合：
- APIキーが正しく設定されていることを確認してください
- レート制限に達していないか確認してください
- 少し待ってから再試行してください

### メモリエラー

Llama-3-7Bモデルの読み込み中にメモリエラーが発生した場合：
- `--load_in_8bit`または`--load_in_4bit`オプションを使用して量子化を有効にしてください
- より少ないレイヤーを分析するために`--layer_indices`オプションを使用してください
- Google ColabやHugging Face Spacesなどのクラウドリソースの使用を検討してください

### トークン化の問題

文ペアが1トークンだけ異なることを確認できない場合：
- 生成プロンプトを調整して、より明確な指示を与えてください
- 異なるモデル（例：GPT-4）を試してみてください
- 手動で文ペアを編集して、1トークンの違いを確保してください
