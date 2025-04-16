#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step.3:
複数モデルの品詞タグ分析結果を比較するスクリプト
各モデルのPOS解析結果を1つのプロットに重ねて表示する
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import glob
import re
import os
import matplotlib_fontja
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# 設定
OUTPUT_DIR = Path('figures/comparisons/')
# フォルダが存在しなければ作成
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# カラーパレット（複数モデルを区別するため）
MODEL_COLORS = [
    '#1f77b4',  # 青
    '#ff7f0e',  # オレンジ
    '#2ca02c',  # 緑
    '#d62728',  # 赤
    '#9467bd',  # 紫
    '#8c564b',  # 茶色
    '#e377c2',  # ピンク
    '#7f7f7f',  # グレー
    '#bcbd22',  # 黄緑
    '#17becf',  # 水色
]

# 言語設定とラベル
LABELS = {
    'ja': {
        'title_overall_ppl': 'モデル・レイヤー間比較: 平均パープレキシティ',
        'title_pos_analysis': '品詞タグごとのPPLモデル比較',
        'title_all_tokens': '全トークン平均PPL',
        'xlabel_layer': 'レイヤー',
        'ylabel_ppl': '平均パープレキシティ (PPL)',
        'legend_original': '元モデル',
        'title_pos_tag': '品詞タグ: {}',
        'sig_improved': '改善 ↑',
        'sig_degraded': '悪化 ↓',
        'largely_improved': '改善 ↑↑',
        'largely_degraded': '悪化 ↓↓'
    },
    'en': {
        'title_overall_ppl': 'Model & Layer Comparison: Average Perplexity',
        'title_pos_analysis': 'POS Tag PPL Model Comparison',
        'title_all_tokens': 'All Tokens Average PPL',
        'xlabel_layer': 'Layer',
        'ylabel_ppl': 'Average Perplexity (PPL)',
        'legend_original': 'Original',
        'title_pos_tag': 'POS Tag: {}',
        'sig_improved': 'Improved ↑',
        'sig_degraded': 'Degraded ↓',
        'largely_improved': 'Improved ↑↑',
        'largely_degraded': 'Degraded ↓↓'
    }
}

def extract_model_name(filename):
    """ファイル名からモデル名を抽出する関数"""
    # ファイル名からモデル名を抽出するパターン
    patterns = [
        r'layer_removal_(.+?)_wiki-text',  # wiki-text
        r'layer_removal_(.+?)_gpt2',       # gpt2
        r'layer_exchange_(.+?)_wiki-text',  # exchange wiki-text
        r'layer_exchange_(.+?)_gpt2'        # exchange gpt2
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            model_name = match.group(1)
            # モデル名を短い表記に置換
            if model_name == "meta_llama_Meta_Llama_3_8B":
                return "Llama-3-8B"
            elif model_name == "meta_llama_Meta_Llama_3_8B_Instruct":
                return "Llama-3-8B-Instruct"
            return model_name
    
    # パターンに一致しない場合はファイル名自体を返す
    return Path(filename).stem

def load_json_data(json_path):
    """JSONファイルからデータを読み込む"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"JSONファイルの読み込みエラー: {e}")
        return None

def extract_layer_info(layer_key, experiment_type="removal"):
    """レイヤーキーから情報を抽出する関数"""
    if experiment_type == "removal":
        # 例: "layer_1" -> 1
        parts = layer_key.split('_')
        if len(parts) >= 2 and parts[0] == "layer":
            try:
                return int(parts[1])
            except ValueError:
                pass
    elif experiment_type == "exchange":
        # 例: "layer_1_with_2" -> 1
        parts = layer_key.split('_')
        if len(parts) >= 2 and parts[0] == "layer":
            try:
                return int(parts[1])
            except ValueError:
                pass
    return 0

def get_exchange_layer_pair(layer_key):
    """exchangeモードの場合にレイヤーペアを抽出する関数"""
    # 例: "layer_1_with_2" -> "1-2"
    parts = layer_key.split('_')
    if len(parts) >= 4 and parts[0] == "layer" and parts[2] == "with":
        return f"{parts[1]}-{parts[3]}"
    return layer_key  # 形式が違う場合は元のキーを返す

def plot_pos_tags_comparison(model_data_list, experiment_type="removal", output_dir=OUTPUT_DIR, significance_threshold=0.1, larger_threshold=0.5):
    """
    複数モデルの品詞タグごとのPPL値を比較する折線グラフを作成
    """
    if not model_data_list:
        print("モデルデータが見つかりません。")
        return
    
    # モデル名リスト
    model_names = [model_info["model_name"] for model_info in model_data_list]
    print(f"比較対象モデル: {model_names}")
    
    # 各モデルのPOS辞書からすべての品詞タグを収集
    all_pos_tags = set()
    for model_info in model_data_list:
        data = model_info["data"]
        pos_tags = data["original"]["pos_tags"].keys()
        all_pos_tags.update(pos_tags)
    
    # 指定された順序で品詞タグを並べる
    desired_order = ['NN', 'PUNCT', 'NNS', 'NUM', 'VBN', 'JJ', 'RB', 'VBG', 'VB', 'IN', 'JJS', 'DT', 'PRP', 'VBD', 'VBZ', 'CC', 'VBP', 'PRP$', 'NNP', 'RBR', 'CD', 'MD', 'TO', 'SYM', 'WRB']
    # 指定された順序に従って並び替え（存在するタグのみ）
    frequent_pos_tags = [tag for tag in desired_order if tag in all_pos_tags]
    # 指定されていないタグがあれば最後に追加
    for tag in sorted(all_pos_tags):
        if tag not in desired_order:
            frequent_pos_tags.append(tag)
    
    print(f"分析対象の品詞タグ: {frequent_pos_tags}")
    
    if not frequent_pos_tags:
        print("分析対象の品詞タグが見つかりません。")
        return
    
    # レイヤー情報を取得（最初のモデルから）
    first_model_data = model_data_list[0]["data"]
    layer_keys = list(first_model_data["layers"].keys())
    layer_nums = [extract_layer_info(layer, experiment_type) for layer in layer_keys]
    
    # X軸のラベルを設定
    if experiment_type == "exchange":
        x_labels = [get_exchange_layer_pair(layer) for layer in layer_keys]
    else:
        x_labels = layer_nums
    
    # 5個飛ばしのインデックスを選択（表示用）
    displayed_indices = []
    for i in range(0, len(layer_keys), 5):
        if i < len(layer_keys):
            displayed_indices.append(i)
    
    # インデックス表示用のラベルを作成
    tick_positions = [layer_nums[i] for i in displayed_indices]
    tick_labels = [x_labels[i] for i in displayed_indices]
    
    print(f"5個飛ばしで表示するインデックス: {tick_labels}")
    
    # 言語ごとにプロットを生成
    for lang_idx, lang in enumerate(['ja', 'en']):
        print(f"{lang}言語版のグラフを作成中...")
        labels = LABELS[lang]
        
        # 言語に応じてラベルを設定
        sig_improved_label = labels['sig_improved']
        sig_degraded_label = labels['sig_degraded']
        largely_improved_label = labels['largely_improved']
        largely_degraded_label = labels['largely_degraded']
        
        # X軸ラベルを言語に合わせて設定
        if experiment_type == "exchange":
            xlabel = "入れ替えたレイヤーペア" if lang == "ja" else "Exchanged Layer Pair"
        else:
            xlabel = "削除した層" if lang == "ja" else "Removed Layer"
        
        # グラフのグリッドサイズを計算
        n_tags = len(frequent_pos_tags) + 1  # すべての品詞タグ + 全体の1つ
        n_cols = min(3, n_tags)  # 列数は最大3
        n_rows = (n_tags + n_cols - 1) // n_cols  # 必要な行数
        
        # 学術的なスタイルを設定
        plt.rcParams['axes.edgecolor'] = 'grey'
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        
        fig = plt.figure(figsize=(7 * n_cols, 5 * n_rows))
        
        # 全体プロットを最初に作成
        ax_all = fig.add_subplot(n_rows, n_cols, 1)
        
        # 各モデルのデータをプロット
        for model_idx, model_info in enumerate(model_data_list):
            model_name = model_info["model_name"]
            model_data = model_info["data"]
            model_color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]
            
            # オリジナルの全体PPL値
            original_ppl = model_data["original"]["total"]
            
            # レイヤーごとの全体PPL値
            model_ppls = []
            for layer_key in layer_keys:
                if layer_key in model_data["layers"]:
                    model_ppls.append(model_data["layers"][layer_key]["total"])
                else:
                    model_ppls.append(np.nan)
            
            # メインの折れ線プロット
            line, = ax_all.plot(layer_nums, model_ppls, marker='o', linestyle='-', 
                       color=model_color, linewidth=2, markersize=6, 
                       markeredgecolor='white', markeredgewidth=1, label=model_name)
            
            # オリジナルモデルのPPLを水平線で表示
            ax_all.axhline(y=original_ppl, color=model_color, linestyle='--', linewidth=1.5, 
                          label=f"{labels['legend_original']} {model_name} ({original_ppl:.2f})")
            
            # 変化したデータポイントを強調表示するための配列を初期化
            significant_improved_indices = []
            significant_degraded_indices = []
            largely_improved_indices = []
            largely_degraded_indices = []
            
            # 全体性能の相対的な変化を計算
            for i, layer_key in enumerate(layer_keys):
                if layer_key in model_data["layers"]:
                    layer_total_ppl = model_data["layers"][layer_key]["total"]
                    relative_change = (layer_total_ppl - original_ppl) / original_ppl
                    
                    # 大幅な変化
                    if relative_change < -larger_threshold:
                        largely_improved_indices.append(i)
                    elif relative_change > larger_threshold:
                        largely_degraded_indices.append(i)
                    # 有意な変化（大幅な変化ではない場合のみ）
                    elif relative_change < -significance_threshold:
                        significant_improved_indices.append(i)
                    elif relative_change > significance_threshold:
                        significant_degraded_indices.append(i)
            
            # 有意に改善したポイントを緑色の輪郭で強調
            if significant_improved_indices:
                improved_x = [layer_nums[i] for i in significant_improved_indices]
                improved_y = [model_ppls[i] for i in significant_improved_indices]
                ax_all.plot(improved_x, improved_y, 'o', markersize=10, markerfacecolor='none', 
                          markeredgecolor='green', markeredgewidth=2, 
                          label=f"{sig_improved_label}" if model_idx == 0 else "_nolegend_")
            
            # 有意に悪化したポイントを赤色の輪郭で強調
            if significant_degraded_indices:
                degraded_x = [layer_nums[i] for i in significant_degraded_indices]
                degraded_y = [model_ppls[i] for i in significant_degraded_indices]
                ax_all.plot(degraded_x, degraded_y, 'o', markersize=10, markerfacecolor='none', 
                          markeredgecolor='red', markeredgewidth=2, 
                          label=f"{sig_degraded_label}" if model_idx == 0 else "_nolegend_")
            
            # 大幅に改善したポイントを塗りつぶした緑色の丸で強調
            if largely_improved_indices:
                largely_improved_x = [layer_nums[i] for i in largely_improved_indices]
                largely_improved_y = [model_ppls[i] for i in largely_improved_indices]
                ax_all.plot(largely_improved_x, largely_improved_y, 'o', markersize=10, markerfacecolor='green', 
                          markeredgecolor='green', markeredgewidth=2, 
                          label=f"{largely_improved_label}" if model_idx == 0 else "_nolegend_")
            
            # 大幅に悪化したポイントを塗りつぶした赤色の丸で強調
            if largely_degraded_indices:
                largely_degraded_x = [layer_nums[i] for i in largely_degraded_indices]
                largely_degraded_y = [model_ppls[i] for i in largely_degraded_indices]
                ax_all.plot(largely_degraded_x, largely_degraded_y, 'o', markersize=10, markerfacecolor='red', 
                          markeredgecolor='red', markeredgewidth=2, 
                          label=f"{largely_degraded_label}" if model_idx == 0 else "_nolegend_")
        
        # 総合的なタイトルを追加
        ax_all.set_title(f"{labels['title_all_tokens']}", fontsize=14, fontweight='bold')
        ax_all.set_xlabel(xlabel, fontsize=12)
        ax_all.set_ylabel(labels['ylabel_ppl'], fontsize=12)
        ax_all.set_yscale('log')  # 対数スケールに設定
        ax_all.legend(frameon=True, framealpha=0.9, edgecolor='grey')
        
        # グリッドの調整
        ax_all.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7)
        ax_all.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
        
        # X軸の設定（5個飛ばしのみ表示）
        ax_all.set_xticks(tick_positions)
        if experiment_type == "exchange":
            ax_all.set_xticklabels(tick_labels, rotation=45, ha='right')  # 斜めに表示
        else:
            ax_all.set_xticklabels(tick_labels)
        
        # 軸ラベルと目盛りのスタイル
        ax_all.tick_params(axis='both', which='major', labelsize=10)
        
        # 各品詞タグごとのサブプロットを作成
        for pos_idx, pos_tag in enumerate(frequent_pos_tags, start=2):  # 2からスタート（1は全体用）
            if pos_idx > n_rows * n_cols:
                print(f"警告: {pos_tag}のプロットをスキップします（プロット数の上限に達しました）")
                continue
                
            ax = fig.add_subplot(n_rows, n_cols, pos_idx)
            
            # 各モデルのデータをプロット
            for model_idx, model_info in enumerate(model_data_list):
                model_name = model_info["model_name"]
                model_data = model_info["data"]
                model_color = MODEL_COLORS[model_idx % len(MODEL_COLORS)]
                
                # オリジナルの品詞タグPPL
                original_pos_ppl = None
                if pos_tag in model_data["original"]["pos_tags"]:
                    original_pos_ppl = model_data["original"]["pos_tags"][pos_tag]
                
                # レイヤーごとの品詞タグPPL
                pos_ppls = []
                for layer_key in layer_keys:
                    if (layer_key in model_data["layers"] and 
                        pos_tag in model_data["layers"][layer_key]["pos_tags"]):
                        pos_ppls.append(model_data["layers"][layer_key]["pos_tags"][pos_tag])
                    else:
                        pos_ppls.append(np.nan)
                
                # 有効な値のみでプロット
                valid_pairs = [(x, y) for x, y in zip(layer_nums, pos_ppls) if not np.isnan(y)]
                
                if valid_pairs:
                    valid_layers, valid_ppls = zip(*valid_pairs)
                    line, = ax.plot(valid_layers, valid_ppls, marker='o', linestyle='-', 
                           color=model_color, linewidth=2, markersize=6, 
                           markeredgecolor='white', markeredgewidth=1, label=model_name)
                    
                    # オリジナルモデルのPPLを水平線で表示
                    if original_pos_ppl is not None:
                        ax.axhline(y=original_pos_ppl, color=model_color, linestyle='--', linewidth=1.5, 
                                  label=f"{labels['legend_original']} {model_name} ({original_pos_ppl:.2f})")
                    
                    # 有意な変化/大幅な変化があるポイントを強調表示
                    if original_pos_ppl is not None:
                        # 変化したデータポイントを特定
                        improved_indices = []
                        degraded_indices = []
                        largely_improved_indices = []
                        largely_degraded_indices = []
                        
                        for idx, (layer_idx, layer_ppl) in enumerate(zip(valid_layers, valid_ppls)):
                            relative_change = (layer_ppl - original_pos_ppl) / original_pos_ppl
                            
                            # 大幅な変化
                            if relative_change < -larger_threshold:
                                largely_improved_indices.append(idx)
                            elif relative_change > larger_threshold:
                                largely_degraded_indices.append(idx)
                            # 有意な変化（大幅な変化ではない場合のみ）
                            elif relative_change < -significance_threshold:
                                improved_indices.append(idx)
                            elif relative_change > significance_threshold:
                                degraded_indices.append(idx)
                        
                        # 有意に改善したポイントを緑色の輪郭で強調
                        if improved_indices:
                            improved_x = [valid_layers[i] for i in improved_indices]
                            improved_y = [valid_ppls[i] for i in improved_indices]
                            ax.plot(improved_x, improved_y, 'o', markersize=10, markerfacecolor='none', 
                                  markeredgecolor='green', markeredgewidth=2, 
                                  label=f"{sig_improved_label}" if model_idx == 0 else "_nolegend_")
                        
                        # 有意に悪化したポイントを赤色の輪郭で強調
                        if degraded_indices:
                            degraded_x = [valid_layers[i] for i in degraded_indices]
                            degraded_y = [valid_ppls[i] for i in degraded_indices]
                            ax.plot(degraded_x, degraded_y, 'o', markersize=10, markerfacecolor='none', 
                                  markeredgecolor='red', markeredgewidth=2, 
                                  label=f"{sig_degraded_label}" if model_idx == 0 else "_nolegend_")
                        
                        # 大幅に改善したポイントを塗りつぶした緑色の丸で強調
                        if largely_improved_indices:
                            largely_improved_x = [valid_layers[i] for i in largely_improved_indices]
                            largely_improved_y = [valid_ppls[i] for i in largely_improved_indices]
                            ax.plot(largely_improved_x, largely_improved_y, 'o', markersize=10, markerfacecolor='green', 
                                  markeredgecolor='green', markeredgewidth=2, 
                                  label=f"{largely_improved_label}" if model_idx == 0 else "_nolegend_")
                        
                        # 大幅に悪化したポイントを塗りつぶした赤色の丸で強調
                        if largely_degraded_indices:
                            largely_degraded_x = [valid_layers[i] for i in largely_degraded_indices]
                            largely_degraded_y = [valid_ppls[i] for i in largely_degraded_indices]
                            ax.plot(largely_degraded_x, largely_degraded_y, 'o', markersize=10, markerfacecolor='red', 
                                  markeredgecolor='red', markeredgewidth=2, 
                                  label=f"{largely_degraded_label}" if model_idx == 0 else "_nolegend_")
            
            # 品詞タグを表示（データ数は表示しない）
            ax.set_title(f"{labels['title_pos_tag'].format(pos_tag)}", fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(labels['ylabel_ppl'], fontsize=12)
            ax.set_yscale('log')  # 対数スケールに設定
            ax.legend(frameon=True, framealpha=0.9, edgecolor='grey')
            
            # X軸の設定（5個飛ばしのみ表示）
            ax.set_xticks(tick_positions)
            if experiment_type == "exchange":
                ax.set_xticklabels(tick_labels, rotation=45, ha='right')  # 斜めに表示
            else:
                ax.set_xticklabels(tick_labels)
            
            # グリッドの調整
            ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
            
            # 軸ラベルと目盛りのスタイル
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        
        # ファイル名を生成（モデル名を連結）
        model_suffix = "_".join([model_info["model_name"] for model_info in model_data_list])
        # 長すぎる場合は短縮
        if len(model_suffix) > 100:
            model_suffix = model_suffix[:50] + "..."
        
        # 通常のグラフを保存
        output_file = output_dir / f'pos_tag_comparison_{experiment_type}_{model_suffix}_{lang}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{lang}言語版の比較グラフを保存しました: {output_file}")

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='複数モデルの品詞タグ分析を比較するスクリプト')
    
    parser.add_argument('--json_files', type=str, nargs='+', required=True,
                        help='比較対象のJSONファイルパス（複数指定可能）')
    
    parser.add_argument('--experiment', type=str, default='removal',
                        choices=['removal', 'exchange'],
                        help='分析する実験の種類 (removal または exchange)')
    
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR),
                        help='出力ディレクトリのパス')
    
    args = parser.parse_args()
    
    # 出力ディレクトリの設定
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # JSONファイルの読み込みと処理
    model_data_list = []
    
    for json_file in args.json_files:
        print(f"JSONファイルを読み込んでいます: {json_file}")
        data = load_json_data(json_file)
        
        if data:
            model_name = extract_model_name(json_file)
            model_data_list.append({
                "model_name": model_name,
                "data": data
            })
            print(f"モデル {model_name} のデータを読み込みました")
        else:
            print(f"ファイル {json_file} の読み込みに失敗しました")
    
    if not model_data_list:
        print("有効なモデルデータが見つかりません。終了します。")
        return
    
    # 品詞タグごとの比較プロットを作成
    plot_pos_tags_comparison(model_data_list, args.experiment, output_dir)
    
    print(f"\n比較分析が完了しました。結果は'{output_dir}'ディレクトリに保存されています。")

if __name__ == "__main__":
    main() 