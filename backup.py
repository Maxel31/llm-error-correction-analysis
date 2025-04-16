"""
レイヤー除去分析のための可視化と分析スクリプト
JSON形式の結果ファイルから、トークンごとの予測精度、PPL値などを分析します
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib_fontja
import warnings
import os
import glob
import argparse
from pathlib import Path
warnings.filterwarnings('ignore')

# 設定
RESULTS_DIR = Path('results')
OUTPUT_DIR = Path('figures/')
# フォルダが存在しなければ作成
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 言語設定とラベル
LABELS = {
    'ja': {
        'title_overall_ppl': 'レイヤー除去後の平均パープレキシティ',
        'title_overall_error': 'レイヤー除去後のエラー率',
        'title_pos_analysis': '品詞タグごとのPPL分析',
        'title_pos_transition': '品詞間遷移のPPL分析',
        'title_all_tokens': '全トークン平均PPL',
        'xlabel_layer': '削除したレイヤー',
        'ylabel_ppl': '平均パープレキシティ (PPL)',
        'ylabel_error': 'エラー率 (%)',
        'legend_original': '元モデル',
        'legend_layers': '各レイヤー削除時',
        'title_pos_tag': '品詞タグ: {}',
        'current_pos': '現在トークンの品詞',
        'next_pos': '次トークンの品詞',
    },
    'en': {
        'title_overall_ppl': 'Average Perplexity after Layer Removal',
        'title_overall_error': 'Error Rate after Layer Removal',
        'title_pos_analysis': 'PPL Analysis by POS Tags',
        'title_pos_transition': 'PPL Analysis by POS Transition',
        'title_all_tokens': 'All Tokens Average PPL',
        'xlabel_layer': 'Removed Layer',
        'ylabel_ppl': 'Average Perplexity (PPL)',
        'ylabel_error': 'Error Rate (%)',
        'legend_original': 'Original',
        'legend_layers': 'Removed Each Layer',
        'title_pos_tag': 'POS Tag: {}',
        'current_pos': 'Current Token POS',
        'next_pos': 'Next Token POS',
    }
}

def load_json_data(json_path):
    """JSONファイルからデータを読み込む"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"JSONファイルの読み込みエラー: {e}")
        return None

def token_results_to_df(token_results):
    """トークンレベルのデータをDataFrame形式に変換する関数"""
    if not token_results:
        return pd.DataFrame()
        
    rows = []
    
    for text_id, tokens_data in token_results.items():
        if not isinstance(tokens_data, dict):
            continue
            
        for token_idx, token_info in tokens_data.items():
            if not isinstance(token_info, dict):
                continue
                
            # 辞書形式のデータを使用
            info_dict = token_info.copy()
            info_dict['text_id'] = text_id
            info_dict['token_idx'] = token_idx
            rows.append(info_dict)
    
    if not rows:
        return pd.DataFrame()
        
    # DataFrameに変換し、pplカラムが確実に数値型になるように処理
    df = pd.DataFrame(rows)
    if 'ppl' in df.columns:
        df['ppl'] = pd.to_numeric(df['ppl'], errors='coerce')
        
        # 無限大や大きすぎる値をクリーニング (例えば10000は極端な値のため制限)
        mask = (df['ppl'] > 10000)
        df.loc[mask, 'ppl'] = 10000.0
        
        # nanやinfをフィルタリング (計算時に除外するが、データ自体は保持)
        #df = df[np.isfinite(df['ppl'])]
    
    # is_correctが文字列の場合、bool型に変換
    if 'is_correct' in df.columns and df['is_correct'].dtype == 'object':
        df['is_correct'] = df['is_correct'].map({'true': True, 'false': False, 'True': True, 'False': False})
        
    return df

def get_pos_tag(token):
    """トークンの品詞タグを取得（タグ付けできない場合はOTHERSを返す）"""
    try:
        # トークンが空文字やNoneでないことを確認
        if token and isinstance(token, str):
            # 特殊文字や記号を処理
            if token in [' ', '\n'] or all(c in ',.;:!?()[]{}"\'`~@#$%^&*_+-=<>|\\/' for c in token):
                return 'PUNCT'  # 記号・句読点
            
            # 数字だけの場合
            if token.isdigit():
                return 'NUM'  # 数値
            
            # 通常の品詞タグ付け
            tags = pos_tag([token])
            if tags and len(tags) > 0:
                return tags[0][1]  # 品詞タグを返す
            
        return 'OTHERS'  # タグ付けできない場合
    except Exception as e:
        print(f"品詞タグ付けエラー: {e} (トークン: {token})")
        return 'OTHERS'

def analyze_errors_by_pos(error_df):
    """品詞ごとのエラーを分析"""
    if error_df.empty:
        return {
            'total_errors': 0,
            'pos_errors': {},
            'pos_error_ratios': {},
            'pos_ppl_avg': {},
            'pos_examples': {},
            'avg_ppl': np.nan
        }
    
    total_errors = len(error_df)
    
    # 各トークンの品詞を取得
    error_df = error_df.copy()
    error_df['next_token_pos'] = error_df['actual_next_token'].apply(get_pos_tag)
    error_df['current_token_pos'] = error_df['current_token'].apply(get_pos_tag)
    
    # pplの値がNaNやInfを含む可能性があるためフィルタリング
    valid_ppl_df = error_df.copy()
    if 'ppl' in valid_ppl_df.columns:
        valid_ppl_df['ppl'] = pd.to_numeric(valid_ppl_df['ppl'], errors='coerce')
        valid_ppl_df = valid_ppl_df[np.isfinite(valid_ppl_df['ppl'])]
    
    # 全体の平均PPLを計算 (有効なPPL値のみ)
    avg_ppl = valid_ppl_df['ppl'].mean() if 'ppl' in valid_ppl_df.columns and not valid_ppl_df.empty else np.nan
    
    # 品詞ごとのエラー数をカウント（next_token_posを使用）
    pos_errors = error_df['next_token_pos'].value_counts().to_dict()
    
    # 品詞ごとのエラー割合を計算
    pos_error_ratios = {pos: count/total_errors for pos, count in pos_errors.items()}
    
    # 品詞ごとの平均PPLを計算
    pos_ppl_avg = {}
    for pos in pos_errors.keys():
        pos_df = valid_ppl_df[valid_ppl_df['next_token_pos'] == pos]
        if 'ppl' in pos_df.columns and not pos_df.empty:
            pos_ppl_avg[pos] = pos_df['ppl'].mean()
        else:
            pos_ppl_avg[pos] = np.nan
    
    # 品詞ごとのエラー例を収集
    pos_examples = {}
    for pos in pos_errors.keys():
        pos_df = error_df[error_df['next_token_pos'] == pos]
        if not pos_df.empty:
            pos_examples[pos] = pos_df.sample(min(3, len(pos_df))).to_dict('records')
        else:
            pos_examples[pos] = []
    
    return {
        'total_errors': total_errors,
        'pos_errors': pos_errors,
        'pos_error_ratios': pos_error_ratios,
        'pos_ppl_avg': pos_ppl_avg,
        'pos_examples': pos_examples,
        'avg_ppl': avg_ppl
    }

def analyze_all_data(df):
    """データセット全体のPPLとエラー率を分析"""
    if df.empty:
        return {
            'total_tokens': 0,
            'error_count': 0,
            'error_rate': np.nan,
            'avg_ppl': np.nan
        }
    
    # pplの値を適切に処理
    valid_ppl_df = df.copy()
    if 'ppl' in valid_ppl_df.columns:
        valid_ppl_df['ppl'] = pd.to_numeric(valid_ppl_df['ppl'], errors='coerce')
        # 有効なPPL値のみを考慮（計算時）
        valid_ppl_df_for_calc = valid_ppl_df[np.isfinite(valid_ppl_df['ppl'])]
    else:
        valid_ppl_df_for_calc = valid_ppl_df
        
    total_tokens = len(df)
    error_count = df['is_correct'].value_counts().get(False, 0)
    error_rate = error_count / total_tokens if total_tokens > 0 else np.nan
    
    # 有効なPPL値のみの平均を計算
    avg_ppl = valid_ppl_df_for_calc['ppl'].mean() if 'ppl' in valid_ppl_df_for_calc.columns and not valid_ppl_df_for_calc.empty else np.nan
    
    return {
        'total_tokens': total_tokens,
        'error_count': error_count,
        'error_rate': error_rate,
        'avg_ppl': avg_ppl
    }

def identify_new_errors(original_df, layer_df):
    """オリジナルで正解だがレイヤー削除後に誤りとなったトークンを特定"""
    if original_df.empty or layer_df.empty:
        return pd.DataFrame()
        
    # 共通のテキストIDとトークンIDをマージのためのキーとして使用
    original_df = original_df.copy()
    layer_df = layer_df.copy()
    
    # マージのためのIDを作成
    original_df['merge_id'] = original_df['text_id'] + '_' + original_df['token_idx'].astype(str)
    layer_df['merge_id'] = layer_df['text_id'] + '_' + layer_df['token_idx'].astype(str)
    
    # オリジナルで正解だったもののみ抽出
    correct_in_original = original_df[original_df['is_correct'] == True]
    
    # レイヤー削除後のデータと結合
    try:
        cols_to_merge = ['merge_id', 'is_correct', 'predicted_next_token', 'current_token', 'actual_next_token']
        if 'ppl' in layer_df.columns:
            cols_to_merge.append('ppl')
            
        merged = pd.merge(
            correct_in_original,
            layer_df[cols_to_merge],
            on='merge_id',
            suffixes=('_original', '')
        )
    except Exception as e:
        print(f"データ結合エラー: {e}")
        return pd.DataFrame()
    
    # レイヤー削除後に誤りとなったものを抽出
    new_errors = merged[merged['is_correct'] == False]
    
    return new_errors

def plot_overall_metrics(layer_analysis, original_analysis, file_prefix, output_dir=OUTPUT_DIR):
    """レイヤーごとの全体的なPPLとエラー率をプロット(日英両言語、対数スケール)"""
    for lang in ['ja', 'en']:
        labels = LABELS[lang]
        
        # レイヤーごとのPPLとエラー率を抽出
        layers = list(layer_analysis.keys())
        ppls = [layer_analysis[layer]['avg_ppl'] for layer in layers]
        error_rates = [layer_analysis[layer]['error_rate'] * 100 for layer in layers]  # パーセンテージに変換
        
        # レイヤー番号を抽出（例："layer_1" -> 1）
        layer_nums = [int(layer.split('_')[1]) for layer in layers]
        
        # オリジナルのPPLとエラー率
        original_ppl = original_analysis['avg_ppl'] 
        original_error_rate = original_analysis['error_rate'] * 100  # パーセンテージに変換
        
        # プロット設定
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 学術的なスタイルを設定
        # plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['axes.edgecolor'] = 'grey'
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        
        # PPLのプロット (対数スケール)
        ax1.plot(layer_nums, ppls, marker='o', linestyle='-', color='#1f77b4', linewidth=2, 
                markersize=6, markeredgecolor='white', markeredgewidth=1, label=labels['legend_layers'])
        
        # 凡例の文字列を言語設定に応じて設定（日本語文字化け対策）
        orig_label = f"{labels['legend_original']} ({original_ppl:.2f})"
        
        ax1.axhline(y=original_ppl, color='#d62728', linestyle='--', linewidth=2.5, 
                   label=orig_label)
        ax1.set_ylabel(labels['ylabel_ppl'], fontsize=12, fontweight='bold')
        ax1.set_title(labels['title_overall_ppl'], fontsize=14, fontweight='bold')
        ax1.set_yscale('log')  # 対数スケールに設定
        ax1.legend(frameon=True, framealpha=0.9, edgecolor='grey')
        ax1.grid(True, which="both", alpha=0.7)  # メジャーとマイナーの両方のグリッドを表示
        ax1.spines['top'].set_visible(False)  # 上部の枠線を非表示
        ax1.spines['right'].set_visible(False)  # 右側の枠線を非表示
        
        # エラー率のプロット
        ax2.plot(layer_nums, error_rates, marker='o', linestyle='-', color='#2ca02c', linewidth=2, 
                markersize=6, markeredgecolor='white', markeredgewidth=1, label=labels['legend_layers'])
        
        # 凡例の文字列を言語設定に応じて設定（日本語文字化け対策）
        orig_error_label = f"{labels['legend_original']} ({original_error_rate:.2f}%)"
        
        ax2.axhline(y=original_error_rate, color='#d62728', linestyle='--', linewidth=2.5, 
                   label=orig_error_label)
        ax2.set_xlabel(labels['xlabel_layer'], fontsize=12, fontweight='bold')
        ax2.set_ylabel(labels['ylabel_error'], fontsize=12, fontweight='bold')
        ax2.set_title(labels['title_overall_error'], fontsize=14, fontweight='bold')
        ax2.legend(frameon=True, framealpha=0.9, edgecolor='grey')
        ax2.grid(True, alpha=0.7)
        ax2.spines['top'].set_visible(False)  # 上部の枠線を非表示
        ax2.spines['right'].set_visible(False)  # 右側の枠線を非表示
        
        # X軸の範囲を設定
        min_layer = min(layer_nums)
        max_layer = max(layer_nums)
        ax2.set_xlim(min_layer - 0.5, max_layer + 0.5)
        ax2.set_xticks(layer_nums)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'overall_metrics_{lang}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_new_errors_by_layer(layer_keys, new_error_counts, file_prefix, output_dir=OUTPUT_DIR):
    """レイヤーごとの新たに発生したエラー数をプロット"""
    # レイヤー番号を抽出（例："layer_1" -> 1）
    layer_nums = [int(layer.split('_')[1]) for layer in layer_keys]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(layer_nums, new_error_counts, color='orange')
    plt.xlabel('削除したレイヤー')
    plt.ylabel('新たに発生したエラー数')
    plt.title('レイヤー削除後に新たに発生したエラー数')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # X軸の範囲とラベルを設定
    min_layer = min(layer_nums)
    max_layer = max(layer_nums)
    plt.xlim(min_layer - 0.5, max_layer + 0.5)
    plt.xticks(layer_nums)
    
    # バーの上に数値を表示
    for bar, count in zip(bars, new_error_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                 str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'new_errors_by_layer.png', dpi=300)
    plt.close()

def plot_pos_error_distribution(pos_errors, title, output_path):
    """品詞ごとのエラー分布をプロット"""
    if not pos_errors:
        return
        
    # 品詞とエラー数を抽出
    pos_tags = list(pos_errors.keys())
    error_counts = list(pos_errors.values())
    
    # 降順にソート
    sorted_indices = np.argsort(error_counts)[::-1]
    pos_tags = [pos_tags[i] for i in sorted_indices]
    error_counts = [error_counts[i] for i in sorted_indices]
    
    # 上位10件のみプロット
    if len(pos_tags) > 10:
        pos_tags = pos_tags[:10]
        error_counts = error_counts[:10]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(pos_tags, error_counts, color='skyblue')
    plt.xlabel('品詞タグ')
    plt.ylabel('エラー数')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # バーの上に数値を表示
    for bar, count in zip(bars, error_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_ppl_by_pos_tags(original_df, layer_dfs, layer_analysis, original_analysis, file_prefix, output_dir=OUTPUT_DIR):
    """
    品詞タグごとに、各レイヤー削除時のPPL値の変化を折線グラフとして表示する
    すべての品詞タグのサブプロットを1つの大きな図にまとめる
    各品詞タグの横にデータ数を表示する
    (日英両言語対応、対数スケール)
    """
    if not layer_dfs:
        print("レイヤーデータが見つかりません。")
        return
    
    # 各品詞タグごとのPPLデータを収集
    layer_keys = list(layer_dfs.keys())
    layer_nums = [int(layer.split('_')[1]) for layer in layer_keys]
    
    # 品詞タグを取得（オリジナルとすべてのレイヤーから）
    all_tokens = pd.concat([original_df] + list(layer_dfs.values()))
    all_tokens['next_token_pos'] = all_tokens['actual_next_token'].apply(get_pos_tag)
    pos_tags = all_tokens['next_token_pos'].value_counts()
    
    # 出現頻度が10回以上の品詞タグのみを使用
    frequent_pos_tags = pos_tags[pos_tags >= 10].index.tolist()
    print(f"分析対象の品詞タグ（出現頻度10回以上）: {frequent_pos_tags}")
    
    # 各品詞タグのデータ数を取得
    pos_counts = pos_tags[frequent_pos_tags].to_dict()
    print(f"各品詞タグのデータ数: {pos_counts}")
    
    # すべての品詞のデータ数の合計を計算（正確な総トークン数）
    total_pos_tokens = sum(pos_counts.values())
    print(f"全品詞の合計データ数: {total_pos_tokens}")
    
    if not frequent_pos_tags:
        print("頻出する品詞タグが見つかりません。")
        return
    
    for lang in ['ja', 'en']:
        labels = LABELS[lang]
        
        # グラフのグリッドサイズを計算
        n_tags = len(frequent_pos_tags) + 1  # すべての品詞タグ + 全体の1つ
        n_cols = min(3, n_tags)  # 列数は最大3
        n_rows = (n_tags + n_cols - 1) // n_cols  # 必要な行数
        
        # 学術的なスタイルを設定
        # plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['axes.edgecolor'] = 'grey'
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        
        fig = plt.figure(figsize=(7 * n_cols, 5 * n_rows))
        
        # 全体プロットを最初に作成
        ax_all = fig.add_subplot(n_rows, n_cols, 1)
        ppls = [layer_analysis[layer]['avg_ppl'] for layer in layer_keys]
        
        # 改良されたプロットスタイル
        ax_all.plot(layer_nums, ppls, marker='o', linestyle='-', color='#1f77b4', linewidth=2, 
                   markersize=6, markeredgecolor='white', markeredgewidth=1, label=labels['legend_layers'])
        
        # オリジナルモデルのPPL
        original_ppl = original_analysis['avg_ppl']
        ax_all.axhline(y=original_ppl, color='#d62728', linestyle='--', linewidth=2.5, 
                      label=f"{labels['legend_original']} ({original_ppl:.2f})")
        
        # 総トークン数を表示（すべての品詞の合計データ数）
        ax_all.set_title(f"{labels['title_all_tokens']} (n={total_pos_tokens})", fontsize=14, fontweight='bold')
        ax_all.set_xlabel(labels['xlabel_layer'], fontsize=12)
        ax_all.set_ylabel(labels['ylabel_ppl'], fontsize=12)
        ax_all.set_yscale('log')  # 対数スケールに設定
        ax_all.legend(frameon=True, framealpha=0.9, edgecolor='grey')
        
        # グリッドの調整
        ax_all.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7)
        ax_all.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
        
        # 軸ラベルと目盛りのスタイル
        ax_all.tick_params(axis='both', which='major', labelsize=10)
        
        # 品詞タグごとのプロットを作成
        for i, pos_tag in enumerate(frequent_pos_tags, start=2):  # 2からスタート（1は全体用）
            ax = fig.add_subplot(n_rows, n_cols, i)
            
            # データ数を取得
            pos_count = pos_counts.get(pos_tag, 0)
            
            # 各レイヤーの当該品詞タグのPPL平均を計算
            pos_ppls = []
            for layer_key in layer_keys:
                layer_df = layer_dfs[layer_key]
                # 特定の品詞タグを持つトークンを抽出
                layer_df['next_token_pos'] = layer_df['actual_next_token'].apply(get_pos_tag)
                pos_df = layer_df[layer_df['next_token_pos'] == pos_tag]
                
                # 有効なPPL値のみ使用して平均を計算
                if not pos_df.empty and 'ppl' in pos_df.columns:
                    valid_ppl = pd.to_numeric(pos_df['ppl'], errors='coerce')
                    valid_ppl = valid_ppl[np.isfinite(valid_ppl)]
                    if not valid_ppl.empty:
                        pos_ppls.append(valid_ppl.mean())
                    else:
                        pos_ppls.append(np.nan)
                else:
                    pos_ppls.append(np.nan)
            
            # 欠損値（NaN）を処理
            valid_pairs = [(x, y) for x, y in zip(layer_nums, pos_ppls) if not np.isnan(y)]
            if valid_pairs:
                valid_layers, valid_ppls = zip(*valid_pairs)
                ax.plot(valid_layers, valid_ppls, marker='o', linestyle='-', color='#1f77b4', linewidth=2, 
                       markersize=6, markeredgecolor='white', markeredgewidth=1, label=labels['legend_layers'])
            
            # オリジナルモデルの当該品詞タグのPPL平均
            original_df['next_token_pos'] = original_df['actual_next_token'].apply(get_pos_tag)
            orig_pos_df = original_df[original_df['next_token_pos'] == pos_tag]
            if not orig_pos_df.empty and 'ppl' in orig_pos_df.columns:
                valid_orig_ppl = pd.to_numeric(orig_pos_df['ppl'], errors='coerce')
                valid_orig_ppl = valid_orig_ppl[np.isfinite(valid_orig_ppl)]
                if not valid_orig_ppl.empty:
                    orig_pos_ppl = valid_orig_ppl.mean()
                    ax.axhline(y=orig_pos_ppl, color='#d62728', linestyle='--', linewidth=2.5, 
                              label=f"{labels['legend_original']} ({orig_pos_ppl:.2f})")
            
            # 品詞タグとデータ数を表示
            ax.set_title(f"{labels['title_pos_tag'].format(pos_tag)} (n={pos_count})", fontsize=14, fontweight='bold')
            ax.set_xlabel(labels['xlabel_layer'], fontsize=12)
            ax.set_ylabel(labels['ylabel_ppl'], fontsize=12)
            ax.set_yscale('log')  # 対数スケールに設定
            ax.legend(frameon=True, framealpha=0.9, edgecolor='grey')
            
            # グリッドの調整
            ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
            
            # 軸ラベルと目盛りのスタイル
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'pos_tag_ppl_analysis_{file_prefix}_{lang}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_pos_transition_heatmap(original_df, layer_dfs, file_prefix, output_dir=OUTPUT_DIR):
    """
    品詞間遷移のPPL値をヒートマップとして表示する
    各レイヤーごとにサブプロットを作成
    (日英両言語対応)
    """
    if not layer_dfs:
        print("レイヤーデータが見つかりません。")
        return
    
    # レイヤーキー取得とソート
    layer_keys = list(layer_dfs.keys())
    sorted_layer_keys = sorted(layer_keys, key=lambda x: int(x.split('_')[1]))
    
    # すべてのレイヤーを表示対象とする（originalも含む）
    selected_layers = ['original'] + sorted_layer_keys
    print(f"品詞遷移ヒートマップ: {len(selected_layers)}個のモデル（オリジナル+{len(sorted_layer_keys)}レイヤー）を描画します")
    
    # 品詞タグを取得（オリジナルとすべてのレイヤーから）
    all_df = pd.concat([original_df] + list(layer_dfs.values()))
    all_df['current_token_pos'] = all_df['current_token'].apply(get_pos_tag)
    all_df['next_token_pos'] = all_df['actual_next_token'].apply(get_pos_tag)
    
    # 頻出する品詞タグのみを使用
    current_pos_counts = all_df['current_token_pos'].value_counts()
    next_pos_counts = all_df['next_token_pos'].value_counts()
    
    frequent_current_pos = current_pos_counts[current_pos_counts >= 10].index.tolist()
    frequent_next_pos = next_pos_counts[next_pos_counts >= 10].index.tolist()
    
    if not frequent_current_pos or not frequent_next_pos:
        print("頻出する品詞タグが見つかりません。")
        return
    
    print(f"分析対象の現在トークン品詞タグ（出現頻度10回以上）: {frequent_current_pos}")
    print(f"分析対象の次トークン品詞タグ（出現頻度10回以上）: {frequent_next_pos}")
    
    for lang in ['ja', 'en']:
        labels = LABELS[lang]
        
        # グラフのグリッドサイズを計算
        n_layers = len(selected_layers)
        n_cols = 3  # 列数は固定で3列（見やすさのため）
        n_rows = (n_layers + n_cols - 1) // n_cols  # 必要な行数
        
        # 学術的なスタイルを設定
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['axes.edgecolor'] = 'black'
        
        # 大きなfigureサイズを設定（多数のサブプロットに対応）
        fig = plt.figure(figsize=(8 * n_cols, 5 * n_rows))
        
        # 各レイヤーのデータを処理
        for i, layer_key in enumerate(selected_layers):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            if layer_key == 'original':
                df = original_df
                title = labels['legend_original']
            else:
                df = layer_dfs[layer_key]
                layer_num = int(layer_key.split('_')[1])
                title = f"{labels['xlabel_layer']} {layer_num}"
            
            # 品詞間遷移のPPL平均値を計算
            transition_matrix = pd.DataFrame(
                index=frequent_current_pos,
                columns=frequent_next_pos
            )
            
            # サンプル数を保存する行列
            count_matrix = pd.DataFrame(
                0,
                index=frequent_current_pos,
                columns=frequent_next_pos
            )
            
            # 各品詞間遷移ごとのPPL平均値を計算
            df = df.copy()  # コピーを作成して元のデータを変更しないようにする
            df['current_token_pos'] = df['current_token'].apply(get_pos_tag)
            df['next_token_pos'] = df['actual_next_token'].apply(get_pos_tag)
            
            for curr_pos in frequent_current_pos:
                for next_pos in frequent_next_pos:
                    # 特定の品詞遷移を持つトークンを抽出
                    trans_df = df[(df['current_token_pos'] == curr_pos) & (df['next_token_pos'] == next_pos)]
                    count_matrix.loc[curr_pos, next_pos] = len(trans_df)
                    
                    # 有効なPPL値のみを使用して平均を計算
                    if not trans_df.empty and 'ppl' in trans_df.columns:
                        valid_ppl = pd.to_numeric(trans_df['ppl'], errors='coerce')
                        valid_ppl = valid_ppl[np.isfinite(valid_ppl)]
                        if not valid_ppl.empty:
                            transition_matrix.loc[curr_pos, next_pos] = valid_ppl.mean()
            
            # NaN値を0に変換（ヒートマップ描画のため）
            transition_matrix = transition_matrix.fillna(0)
            
            # カラーマップの設定
            cmap = plt.cm.viridis
            
            # マスクの作成
            mask = transition_matrix == 0
            
            # カラーマップのスケールを最適化（極端に高い値が一部にある場合への対応）
            non_zero_vals = transition_matrix.values.flatten()[transition_matrix.values.flatten() > 0]
            if len(non_zero_vals) > 0:
                vmax = np.percentile(non_zero_vals, 95)
            else:
                vmax = 1.0  # デフォルト値
            
            # アノテーションテキストの作成 (PPL値とサンプル数)
            annot_texts = np.empty_like(transition_matrix, dtype=object)
            for ii, curr_pos in enumerate(frequent_current_pos):
                for jj, next_pos in enumerate(frequent_next_pos):
                    val = transition_matrix.iloc[ii, jj]
                    count = count_matrix.iloc[ii, jj]
                    if val > 0:
                        annot_texts[ii, jj] = f"{val:.1f}\n(n={int(count)})"
                    else:
                        annot_texts[ii, jj] = ""
            
            # ヒートマップを描画
            sns.heatmap(
                transition_matrix,
                annot=annot_texts,
                fmt="",
                cmap=cmap,
                mask=mask,
                ax=ax,
                vmin=0,
                vmax=vmax,
                cbar_kws={"label": "平均PPL値" if lang == "ja" else "Average PPL", "shrink": 0.8}
            )
            
            # 背景色の設定
            ax.set_facecolor('whitesmoke')
            
            # タイトルと軸ラベルの設定
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(labels['next_pos'], fontsize=12, fontweight='bold')
            ax.set_ylabel(labels['current_pos'], fontsize=12, fontweight='bold')
            
            # 軸目盛りのフォントサイズ調整
            ax.tick_params(axis='both', labelsize=10)
        
        plt.tight_layout(pad=3.0)
        plt.suptitle(labels['title_pos_transition'], fontsize=16, fontweight='bold', y=1.02)
        plt.savefig(output_dir / f'pos_transition_heatmap_{file_prefix}_{lang}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_detailed_pos_transition_heatmap(original_df, layer_dfs, file_prefix, output_dir=OUTPUT_DIR):
    """
    品詞間遷移のPPL値をヒートマップとして詳細に表示する
    縦軸にcurrent_tokenの品詞、横軸にactual_next_tokenの品詞を配置
    (日英両言語対応)
    """
    if not layer_dfs:
        print("レイヤーデータが見つかりません。")
        return
    
    # 品詞タグを取得（オリジナルとすべてのレイヤーから）
    all_df = pd.concat([original_df] + list(layer_dfs.values()))
    all_df['current_token_pos'] = all_df['current_token'].apply(get_pos_tag)
    all_df['next_token_pos'] = all_df['actual_next_token'].apply(get_pos_tag)
    
    # 頻出する品詞タグのみを使用
    current_pos_counts = all_df['current_token_pos'].value_counts()
    next_pos_counts = all_df['next_token_pos'].value_counts()
    
    frequent_current_pos = current_pos_counts[current_pos_counts >= 10].index.tolist()
    frequent_next_pos = next_pos_counts[next_pos_counts >= 10].index.tolist()
    
    if not frequent_current_pos or not frequent_next_pos:
        print("頻出する品詞タグが見つかりません。")
        return
    
    print(f"詳細品詞遷移分析 - 現在トークン品詞タグ（出現頻度10回以上）: {frequent_current_pos}")
    print(f"詳細品詞遷移分析 - 次トークン品詞タグ（出現頻度10回以上）: {frequent_next_pos}")
    
    # オリジナルモデルのみを処理
    for lang in ['ja', 'en']:
        labels = LABELS[lang]
        
        # matplotlib設定を初期化
        plt.rcParams['axes.unicode_minus'] = False
        
        # オリジナルデータのみを処理
        df = original_df
        title = f"{labels['legend_original']} - {labels['title_pos_transition']}"
        filename = f'detailed_pos_transition_original_{file_prefix}_{lang}.png'
        
        # 品詞タグを付与
        df = df.copy()
        df['current_token_pos'] = df['current_token'].apply(get_pos_tag)
        df['next_token_pos'] = df['actual_next_token'].apply(get_pos_tag)
        
        # 品詞間遷移のPPL平均値を計算する行列を作成
        transition_matrix = pd.DataFrame(
            index=frequent_current_pos,
            columns=frequent_next_pos
        )
        
        # サンプル数をカウントする行列を作成
        count_matrix = pd.DataFrame(
            0,
            index=frequent_current_pos,
            columns=frequent_next_pos
        )
        
        # 各品詞間遷移ごとのPPL平均値とカウントを計算
        for curr_pos in frequent_current_pos:
            for next_pos in frequent_next_pos:
                # 特定の品詞遷移を持つトークンを抽出
                trans_df = df[(df['current_token_pos'] == curr_pos) & (df['next_token_pos'] == next_pos)]
                count_matrix.loc[curr_pos, next_pos] = len(trans_df)
                
                # 有効なPPL値のみを使用して平均を計算
                if not trans_df.empty and 'ppl' in trans_df.columns:
                    valid_ppl = pd.to_numeric(trans_df['ppl'], errors='coerce')
                    valid_ppl = valid_ppl[np.isfinite(valid_ppl)]
                    if not valid_ppl.empty:
                        transition_matrix.loc[curr_pos, next_pos] = valid_ppl.mean()
        
        # NaN値を0に変換（ヒートマップ描画のため）
        transition_matrix = transition_matrix.fillna(0)
        
        # 学術的なスタイルでヒートマップ描画
        plt.figure(figsize=(48, 36))
        ax = plt.gca()
        
        # カラーマップのスケールを最適化（極端に高い値が一部にある場合への対応）
        non_zero_vals = transition_matrix.values.flatten()[transition_matrix.values.flatten() > 0]
        if len(non_zero_vals) > 0:
            vmax = np.percentile(non_zero_vals, 95)
        else:
            vmax = 1.0  # デフォルト値
        
        # 新しい学術的なスタイルでヒートマップを描画
        cmap = plt.cm.viridis  # よりはっきりしたカラーマップを使用
        mask = transition_matrix == 0  # 0の値はマスク（白抜き）
        
        # アノテーションのカスタマイズ（PPL値とサンプル数を表示）
        def annotate_func(val, count):
            if val == 0:
                return ""
            count_val = int(count)
            return f"{val:.1f}\n(n={count_val})" if count_val > 0 else ""
        
        # アノテーションテキストを作成
        annot_texts = np.empty_like(transition_matrix, dtype=object)
        for i, curr_pos in enumerate(frequent_current_pos):
            for j, next_pos in enumerate(frequent_next_pos):
                val = transition_matrix.iloc[i, j]
                count = count_matrix.iloc[i, j]
                annot_texts[i, j] = annotate_func(val, count)
        
        # ヒートマップを描画
        sns.heatmap(
            transition_matrix,
            annot=annot_texts,
            fmt="",
            cmap=cmap,
            mask=mask,
            ax=ax,
            vmin=0,
            vmax=vmax,
            cbar_kws={"label": "平均PPL値" if lang == "ja" else "Average PPL"}
        )
        
        # グリッドの設定
        ax.set_facecolor('whitesmoke')  # 背景色を設定
        
        # タイトルと軸ラベルのスタイル設定
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(labels['next_pos'], fontsize=14, fontweight='bold')
        plt.ylabel(labels['current_pos'], fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

def main(dataset_type="wiki-text-2", experiment_type="removal", sample_size=10000):
    """メイン関数"""
    
    # データセットタイプとexperiment_typeに基づいてファイルパスを決定
    if experiment_type == "removal":
        exp_prefix = "layer_removal"
    elif experiment_type == "exchange":
        exp_prefix = "layer_exchange"
    else:
        raise ValueError(f"サポートされていない実験タイプ: {experiment_type}")
    
    json_path = Path(f'results/{exp_prefix}_token_prediction_{dataset_type}_n{sample_size}.json')
    
    print(f"JSONファイルを読み込んでいます: {json_path}")
    data = load_json_data(json_path)
    
    if not data:
        print("データを読み込めませんでした。終了します。")
        return
    
    print("データを読み込みました。分析を開始します...")
    print(f"分析対象データセット: {dataset_type}")
    print(f"分析対象実験タイプ: {experiment_type}")
    
    # オリジナルデータを取得
    original_data = data.get('original', {})
    
    # 実験タイプに応じてデータを取得
    if experiment_type == "removal":
        experiment_data = data.get('removed', {})
    elif experiment_type == "exchange":
        experiment_data = data.get('exchanged', {})
    else:
        experiment_data = {}
    
    if not original_data or not experiment_data:
        print("必要なデータが見つかりません。終了します。")
        return
    
    # 各レイヤーキーを取得
    layer_keys = list(experiment_data.keys())
    print(f"検出されたレイヤーキー: {layer_keys}")
    
    # オリジナルデータをDataFrameに変換
    if 'token_level_results' in original_data:
        original_df = token_results_to_df(original_data['token_level_results'])
        print(f"オリジナルデータをDataFrameに変換しました: {len(original_df)}行")
    else:
        original_df = pd.DataFrame()
        print("オリジナルデータにtoken_level_resultsが見つかりません")
    
    # 各レイヤーのデータをDataFrameに変換
    layer_dfs = {}
    for layer_key in layer_keys:
        layer_data = experiment_data[layer_key]
        if 'token_level_results' in layer_data:
            layer_dfs[layer_key] = token_results_to_df(layer_data['token_level_results'])
            print(f"{layer_key}のデータをDataFrameに変換しました: {len(layer_dfs[layer_key])}行")
        else:
            print(f"{layer_key}にtoken_level_resultsが見つかりません")
    
    # データ検証: NaNやInfの値がないか確認
    def check_data_validity(df, name):
        if 'ppl' in df.columns:
            ppl_values = pd.to_numeric(df['ppl'], errors='coerce')
            nan_count = ppl_values.isna().sum()
            inf_count = np.isinf(ppl_values).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"警告: {name}のデータに{nan_count}個のNaNと{inf_count}個のInf値が含まれています")
                # NaNやInfをフィルタリングせずに警告のみ表示
    
    check_data_validity(original_df, "オリジナル")
    for layer_key, layer_df in layer_dfs.items():
        check_data_validity(layer_df, layer_key)
    
    # オリジナルモデルの全体分析
    original_analysis = analyze_all_data(original_df)
    print(f"オリジナルモデルの全体平均PPL: {original_analysis['avg_ppl']:.2f}")
    print(f"オリジナルモデルのエラー率: {original_analysis['error_rate']:.2%}")
    
    # 各レイヤーの全体分析
    layer_analysis = {}
    for layer_key, layer_df in layer_dfs.items():
        layer_analysis[layer_key] = analyze_all_data(layer_df)
        print(f"\n{layer_key}の全体平均PPL: {layer_analysis[layer_key]['avg_ppl']:.2f}")
        print(f"{layer_key}のエラー率: {layer_analysis[layer_key]['error_rate']:.2%}")
        
        # データ整合性検証: レイヤー削除時のPPLがオリジナルより大幅に低い場合は警告
        if layer_analysis[layer_key]['avg_ppl'] < original_analysis['avg_ppl'] * 0.5:
            print(f"警告: {layer_key}のPPL ({layer_analysis[layer_key]['avg_ppl']:.2f}) が"
                  f"オリジナル ({original_analysis['avg_ppl']:.2f}) より大幅に低いです")
    
    # 実験タイプと対象データセットを含むファイル名の接頭辞を作成
    file_prefix = f"{exp_prefix}_{dataset_type}_n{sample_size}"
    
    # ファイル保存用のディレクトリ
    output_dir = OUTPUT_DIR / file_prefix
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # オリジナルでの誤り（is_correctがFalse）を抽出
    original_errors = original_df[original_df['is_correct'] == False]
    original_pos_analysis = analyze_errors_by_pos(original_errors)
    
    # 全体的なメトリクスをプロット (日英両言語)
    plot_overall_metrics(layer_analysis, original_analysis, file_prefix, output_dir=output_dir)
    
    # 品詞タグ別のPPL分析プロット (日英両言語、対数スケール)
    plot_ppl_by_pos_tags(original_df, layer_dfs, layer_analysis, original_analysis, file_prefix, output_dir=output_dir)
    
    # 品詞間遷移のヒートマップを作成 (日英両言語)
    plot_pos_transition_heatmap(original_df, layer_dfs, file_prefix, output_dir=output_dir)
    
    # レイヤーごとに新たに発生したエラーを分析
    new_error_analyses = {}
    new_error_counts = []
    
    for layer_key, layer_df in layer_dfs.items():
        # オリジナルで正解だが、レイヤー削除後に誤りとなったトークンを特定
        new_errors = identify_new_errors(original_df, layer_df)
        
        # 新たなエラーの分析
        new_error_analysis = analyze_errors_by_pos(new_errors)
        new_error_analyses[layer_key] = new_error_analysis
        new_error_counts.append(new_error_analysis['total_errors'])
        
        print(f"\n{layer_key}で新たに発生した誤り数: {new_error_analysis['total_errors']}")
        
        # 品詞ごとのエラー分析を表示
        if new_error_analysis['total_errors'] > 0:
            print("\n品詞ごとのエラー分析:\n")
            print("| 品詞タグ | エラー数 | エラー割合 | 平均PPL |")
            print("|----------|----------|------------|---------|")
            
            # 品詞ごとのエラー数を降順にソート
            sorted_pos = sorted(new_error_analysis['pos_errors'].items(), 
                               key=lambda x: x[1], reverse=True)
            
            for pos, count in sorted_pos:
                ratio = new_error_analysis['pos_error_ratios'][pos]
                ppl = new_error_analysis['pos_ppl_avg'].get(pos, np.nan)
                ppl_str = f"{ppl:.2f}" if not np.isnan(ppl) else "N/A"
                print(f"| {pos} | {count} | {ratio:.2%} | {ppl_str} |")
    
    # 新たに発生したエラー数をプロット
    plot_new_errors_by_layer(layer_keys, new_error_counts, file_prefix, output_dir=output_dir)
    
    # 詳細品詞遷移のヒートマップを作成 (日英両言語)
    plot_detailed_pos_transition_heatmap(original_df, layer_dfs, file_prefix, output_dir=output_dir)
    
    print(f"\n分析が完了しました。結果は'{output_dir}'ディレクトリに保存されています。")
    print(f"データセット: {dataset_type}, 実験タイプ: {experiment_type}, サンプルサイズ: {sample_size}")

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='レイヤー除去/交換分析スクリプト')
    
    parser.add_argument('--dataset', type=str, default='wiki-text-2',
                        choices=['wiki-text-2', 'gpt2-output-dataset'],
                        help='分析するデータセットの種類 (wiki-text-2 または gpt2-output-dataset)')
    
    parser.add_argument('--experiment', type=str, default='removal',
                        choices=['removal', 'exchange'],
                        help='分析する実験の種類 (removal または exchange)')
    
    parser.add_argument('--samples', type=int, default=10000,
                        help='サンプルサイズ')
    
    args = parser.parse_args()
    
    # 引数からパラメータを取得
    dataset_type = args.dataset
    experiment_type = args.experiment
    sample_size = args.samples
    
    print(f"実行中: dataset={dataset_type}, experiment={experiment_type}, samples={sample_size}")
    
    # メイン関数を実行
    main(dataset_type=dataset_type, experiment_type=experiment_type, sample_size=sample_size)