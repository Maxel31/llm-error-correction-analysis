#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step1"
隣接層の交換/層の除去による次単語予測性能の評価、jsonファイルへの結果保存
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import argparse
from typing import List, Dict, Tuple, Optional, Any, Union
import traceback
import random
from datasets import load_dataset, disable_caching
import matplotlib_fontja
from pathlib import Path
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_device(gpu_id: str = "0") -> torch.device:
    """GPUが利用可能かどうかを確認し、適切なデバイスを返す"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス: {device} (GPU ID: {gpu_id})")
    return device

def load_model_and_tokenizer(model_name: str, device: torch.device) -> Tuple[nn.Module, Any]:
    """モデルとトークナイザーを読み込む"""
    logger.info(f"モデル読み込み中: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 量子化設定
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 量子化したモデルを読み込み
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    model.eval()
    
    # モデルをGPUに移動
    model = model.to(device)
    
    return model, tokenizer

def block_remove_llama(model: nn.Module, kill_list: List[int]) -> nn.Module:
    """指定された層を削除する関数"""
    # モデルの現在のデバイスを保存
    device = next(model.parameters()).device
    # CPUにモデルを移動してからコピーを作成
    model = model.to('cpu')
    model_copy = copy.deepcopy(model)
    # 元のモデルを元のデバイスに戻す
    model = model.to(device)
    
    # リストをソートして順番に処理
    kill_list = sorted(kill_list.copy())  # 元のリストを変更しないようにコピー
    
    while len(kill_list) > 0:
        # 最初の要素を削除
        del model_copy.model.layers[kill_list[0]]
        del kill_list[0]
        
        # 残りのインデックスを調整
        for i in range(len(kill_list)):
            kill_list[i] -= 1
    
    # レイヤーのインデックスを更新
    for i in range(len(model_copy.model.layers)):
        model_copy.model.layers[i].self_attn.layer_idx = i
    
    # 処理後のモデルを元のデバイスに戻す
    model_copy = model_copy.to(device)
    
    return model_copy

def exchange_layers_llama(model: nn.Module, layer_idx: int) -> nn.Module:
    """隣接する層を交換"""
    temp = model.model.layers[layer_idx]
    model.model.layers[layer_idx] = model.model.layers[layer_idx + 1]
    model.model.layers[layer_idx + 1] = temp
    return model

def load_dataset_gpt2_output_dataset(file_path: str) -> List[str]:
    """GPT-2のアウトプットデータセットを読み込む"""
    logger.info(f"GPT-2 output dataset読み込み中: {file_path}")
    texts = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                texts.append(data['text'])
            except json.JSONDecodeError:
                continue  # JSONとして解析できない行はスキップ
            except KeyError:
                continue  # 'text'キーがない行はスキップ
    return texts

def evaluate_token_prediction(model: nn.Module, text: str, tokenizer) -> Tuple[float, List[str], List[str], List[float], int, int]:
    """単一テキストに対するトークン予測の評価"""
    try:
        # トークン化
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids[0]
        
        # 短すぎるテキストは処理しない
        if len(input_ids) <= 1:
            return 1e4, [], [], [], 0, 0
        
        # 元のトークンをデコード
        original_tokens = [tokenizer.decode([token_id]) for token_id in input_ids]
        
        # 一度だけ推論を実行（全トークンの予測を一度に取得）
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))
            logits = outputs.logits[0, :-1, :]  # 最後のトークンを除く全トークンの予測
        
        # 予測トークンとpplのリスト
        predicted_tokens = []
        token_ppls = []
        
        # 対数確率の合計と正解数
        total_log_prob = 0.0
        correct_count = 0
        total_count = 0
        
        # 各位置で次トークンの予測を評価
        for i in range(len(input_ids) - 1):
            # 予測対象のトークン
            target_token_id = input_ids[i+1].item()
            
            # 予測された次トークンのlogits
            next_token_logits = logits[i]
            
            # ソフトマックスで確率を計算
            probs = torch.nn.functional.softmax(next_token_logits, dim=0)
            target_prob = probs[target_token_id].item()
            
            # 確率が非常に小さい場合のバックオフ
            if target_prob < 1e-8:
                target_prob = 1e-8
            
            # 対数確率を累積
            log_prob = torch.log(torch.tensor(target_prob)).item()
            total_log_prob += log_prob
            
            # 最も確率の高いトークンを取得
            pred_token_id = torch.argmax(next_token_logits).item()
            pred_token = tokenizer.decode([pred_token_id])
            
            # 予測結果を記録
            predicted_tokens.append(pred_token)
            
            # 予測が合っているかチェック
            if pred_token_id == target_token_id:
                correct_count += 1
            total_count += 1
            
            # トークンごとのパープレキシティ
            token_ppl = 1.0 / target_prob
            token_ppl = min(token_ppl, 1e4)  # クリッピング
            
            token_ppls.append(token_ppl)
        
        # 平均パープレキシティの計算
        if total_count > 0:
            avg_log_prob = total_log_prob / total_count
            avg_ppl = torch.exp(-torch.tensor(avg_log_prob)).item()
            
            if torch.isnan(torch.tensor(avg_ppl)) or torch.isinf(torch.tensor(avg_ppl)) or avg_ppl > 1e4:
                avg_ppl = 1e4
        else:
            avg_ppl = 1e4
        
        return avg_ppl, original_tokens, predicted_tokens, token_ppls, correct_count, total_count
    
    except Exception as e:
        logger.error(f"トークン予測評価中にエラー: {e}")
        traceback.print_exc()
        return 1e4, [], [], [], 0, 0

def evaluate_token_predictions_for_texts(model: nn.Module, texts, dataset_type: str, tokenizer) -> Tuple[float, Dict, Dict]:
    """複数テキストに対するトークン予測の評価"""
    logger.info(f"テキスト評価中: {dataset_type}")
    
    if dataset_type == "gpt2-output-dataset":
        samples = texts
    elif dataset_type == "wiki-text-2":
        # datasetsのDatasetオブジェクトからテキストを取得
        samples = [item['text'] for item in texts]
    elif dataset_type == "wiki-text-103":
        samples = [item['text'] for item in texts]
    elif dataset_type == "bookcorpus":
        samples = texts
    else:
        raise ValueError(f"サポートされていないデータセットタイプ: {dataset_type}")
    
    # 評価結果を格納
    ppls = []
    text_level_results = {}
    token_level_results = {}
    
    # 各テキストを個別に評価
    for idx, text in enumerate(tqdm(samples, desc="テキスト評価中")):
        # 空でないテキストのみ処理
        if not text or len(text) < 4:
            continue
            
        avg_ppl, original_tokens, predicted_tokens, token_ppls, correct_count, total_count = evaluate_token_prediction(model, text, tokenizer)
        
        # 有効なpplのみ追加
        if 0 < avg_ppl < 1e4:
            ppls.append(avg_ppl)
        
        # 文章単位の結果を保存
        text_level_results[f"sample_{idx}"] = {
            "original_text": text,
            "avg_ppl": float(avg_ppl),
            "accuracy": float(correct_count) / float(total_count) if total_count > 0 else 0.0,
            "correct_predictions": correct_count,
            "total_predictions": total_count
        }
        
        # トークンごとの予測結果を格納
        tokens_result = {}
        for k in range(len(original_tokens) - 1):
            current_token = original_tokens[k]
            next_token = original_tokens[k+1]
            pred_token = predicted_tokens[k] if k < len(predicted_tokens) else "N/A"
            is_correct = (pred_token == next_token)
            
            tokens_result[f"token_{k+1}"] = {
                "current_token": current_token,
                "actual_next_token": next_token,
                "predicted_next_token": pred_token,
                "is_correct": is_correct,
                "ppl": float(token_ppls[k]) if k < len(token_ppls) else 0.0
            }
        
        if len(original_tokens) > 0:
            last_idx = len(original_tokens)
            tokens_result[f"token_{last_idx}"] = {
                "current_token": original_tokens[-1],
                "actual_next_token": "N/A",
                "predicted_next_token": "N/A",
                "is_correct": False,
                "ppl": 0.0
            }
        
        token_level_results[f"sample_{idx}"] = tokens_result
    
    # 平均パープレキシティの計算
    avg_ppl = sum(ppls) / len(ppls) if ppls else 1e4
    
    return avg_ppl, text_level_results, token_level_results 

def analyze_layer_removal(model: nn.Module, tokenizer, model_name: str, max_samples: int = 1000, 
                         output_dir: str = "results", dataset_type: str = "wiki-text-2"):
    """各層を削除した場合のトークン予測性能を分析"""
    random.seed(42)  # シードを固定
    
    num_layers = len(model.model.layers)
    logger.info(f"モデルの総層数: {num_layers}")
    
    # モデル名からファイル名用の文字列を生成
    model_name_for_file = model_name.replace('/', '_').replace('-', '_')
    
    # データセットの読み込み
    if dataset_type == "wiki-text-2":
        # キャッシュを無効化
        disable_caching()
        valid_texts = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        logger.info("Wiki-Text-2 を読み込みました")
        sample_indices = random.sample(range(len(valid_texts)), min(max_samples, len(valid_texts)))
        samples = [valid_texts[i] for i in sample_indices]
    elif dataset_type == "wiki-text-103":
        valid_texts = load_dataset('wikitext', 'wikitext-103-v1', split='train')
        logger.info("Wiki-Text-103 を読み込みました")
        sample_indices = random.sample(range(len(valid_texts)), min(max_samples, len(valid_texts)))
        samples = [valid_texts[i] for i in sample_indices]
    elif dataset_type == "bookcorpus":
        valid_texts = load_dataset('SamuelYang/bookcorpus', split='train')
        logger.info("BookCorpus を読み込みました")
        sample_indices = random.sample(range(len(valid_texts)), min(max_samples, len(valid_texts)))
        samples = [valid_texts[i] for i in sample_indices]
    elif dataset_type == "gpt2-output-dataset":
        valid_texts = load_dataset_gpt2_output_dataset("../../data/webtext.train.jsonl")
        logger.info("GPT-2 Output Dataset を読み込みました")
        sample_indices = random.sample(range(len(valid_texts)), min(max_samples, len(valid_texts)))
        samples = [valid_texts[i] for i in sample_indices]
    else:
        raise ValueError(f"サポートされていないデータセットタイプ: {dataset_type}")
    
    # 結果を格納する辞書
    results = {
        "model_name": model_name,
        "original": {
            "avg_ppl": 0.0,
            "text_level_results": {},
            "token_level_results": {}
        },
        "removed": {}
    }
    
    try:
        # 元のモデルの評価
        logger.info("元のモデルのトークン予測性能を評価中...")
        original_avg_ppl, original_text_results, original_token_results = evaluate_token_predictions_for_texts(model, samples, dataset_type, tokenizer)
        results["original"]["avg_ppl"] = float(original_avg_ppl)
        results["original"]["text_level_results"] = original_text_results
        results["original"]["token_level_results"] = original_token_results
        logger.info(f"元のモデルの平均ppl: {original_avg_ppl:.6f}")
        
        # 元のモデルをディープコピー
        original_model = copy.deepcopy(model)
        
        # 各層を削除して評価
        for i in range(num_layers):
            logger.info(f"層 {i} を削除して評価中...")
            
            # 削除する層のリスト（単一層）
            kill_list = [i]
            
            # 層を削除したモデルを作成
            try:
                model_removed = block_remove_llama(original_model, kill_list)
                
                # 評価を実行
                avg_ppl, text_results, token_results = evaluate_token_predictions_for_texts(model_removed, samples, dataset_type, tokenizer)
                results["removed"][f"layer_{i}"] = {
                    "avg_ppl": float(avg_ppl),
                    "text_level_results": text_results,
                    "token_level_results": token_results
                }
                logger.info(f"層 {i} 削除時の平均ppl: {avg_ppl:.6f}")
                
                # メモリ解放
                del model_removed
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"層 {i} 削除の評価中にエラー: {e}")
                results["removed"][f"layer_{i}"] = {
                    "avg_ppl": 1e4,
                    "text_level_results": {},
                    "token_level_results": {}
                }
        
        # 出力ディレクトリを作成
        output_path = Path(output_dir) / "removal" / f"{model_name_for_file}_{dataset_type}_n{max_samples}" / "ppl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 結果をJSONファイルに保存
        with open(f'{output_path}.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"結果をJSONファイルに保存しました: {output_path}.json")
        
    except Exception as e:
        logger.error(f"層削除実験中にエラーが発生しました: {e}")
        traceback.print_exc()
    
    return results

def analyze_layer_exchange(model: nn.Module, tokenizer, model_name: str, max_samples: int = 1000, 
                         output_dir: str = "results", dataset_type: str = "wiki-text-2"):
    """各層を隣接層と交換した場合のトークン予測性能を分析"""
    random.seed(42)  # シードを固定
    
    num_layers = len(model.model.layers)
    logger.info(f"モデルの総層数: {num_layers}")
    
    # モデル名からファイル名用の文字列を生成
    model_name_for_file = model_name.replace('/', '_').replace('-', '_')
    
    # データセットの読み込み
    if dataset_type == "wiki-text-2":
        # キャッシュを無効化
        disable_caching()
        valid_texts = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        logger.info("Wiki-Text-2 を読み込みました")
        sample_indices = random.sample(range(len(valid_texts)), min(max_samples, len(valid_texts)))
        samples = [valid_texts[i] for i in sample_indices]
    elif dataset_type == "wiki-text-103":
        valid_texts = load_dataset('wikitext', 'wikitext-103-v1', split='train')
        logger.info("Wiki-Text-103 を読み込みました")
        sample_indices = random.sample(range(len(valid_texts)), min(max_samples, len(valid_texts)))
        samples = [valid_texts[i] for i in sample_indices]
    elif dataset_type == "bookcorpus":
        valid_texts = load_dataset('SamuelYang/bookcorpus', split='train')
        logger.info("BookCorpus を読み込みました")
        sample_indices = random.sample(range(len(valid_texts)), min(max_samples, len(valid_texts)))
        samples = [valid_texts[i] for i in sample_indices]
    elif dataset_type == "gpt2-output-dataset":
        valid_texts = load_dataset_gpt2_output_dataset("../../data/webtext.train.jsonl")
        logger.info("GPT-2 Output Dataset を読み込みました")
        sample_indices = random.sample(range(len(valid_texts)), min(max_samples, len(valid_texts)))
        samples = [valid_texts[i] for i in sample_indices]
    else:
        raise ValueError(f"サポートされていないデータセットタイプ: {dataset_type}")
    
    # 結果を格納する辞書
    results = {
        "model_name": model_name,
        "original": {
            "avg_ppl": 0.0,
            "text_level_results": {},
            "token_level_results": {}
        },
        "exchanged": {}
    }
    
    try:
        # 元のモデルの評価
        logger.info("元のモデルのトークン予測性能を評価中...")
        original_avg_ppl, original_text_results, original_token_results = evaluate_token_predictions_for_texts(model, samples, dataset_type, tokenizer)
        results["original"]["avg_ppl"] = float(original_avg_ppl)
        results["original"]["text_level_results"] = original_text_results
        results["original"]["token_level_results"] = original_token_results
        logger.info(f"元のモデルの平均ppl: {original_avg_ppl:.6f}")
        
        # 元のモデルをディープコピー
        original_model = copy.deepcopy(model)
        
        # 各層を交換して評価（最後の層を除く）
        for i in range(num_layers - 1):
            logger.info(f"層 {i} と層 {i+1} を交換中...")
            
            # 元のモデルからコピーを作成して層を交換
            try:
                model_exchanged = copy.deepcopy(original_model)
                model_exchanged = exchange_layers_llama(model_exchanged, i)
                
                # 評価を実行
                avg_ppl, text_results, token_results = evaluate_token_predictions_for_texts(model_exchanged, samples, dataset_type, tokenizer)
                results["exchanged"][f"layer_{i}_with_{i+1}"] = {
                    "avg_ppl": float(avg_ppl),
                    "text_level_results": text_results,
                    "token_level_results": token_results
                }
                logger.info(f"層 {i} と層 {i+1} 交換時の平均ppl: {avg_ppl:.6f}")
                
                # メモリ解放
                del model_exchanged
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"層 {i} と層 {i+1} の交換評価中にエラー: {e}")
                results["exchanged"][f"layer_{i}_with_{i+1}"] = {
                    "avg_ppl": 1e4,
                    "text_level_results": {},
                    "token_level_results": {}
                }
        
        # 出力ディレクトリを作成
        output_path = Path(output_dir) / "exchange" / f"{model_name_for_file}_{dataset_type}_n{max_samples}" / "ppl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 結果をJSONファイルに保存
        with open(f'{output_path}.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"結果をJSONファイルに保存しました: {output_path}.json")
        
    except Exception as e:
        logger.error(f"層交換実験中にエラーが発生しました: {e}")
        traceback.print_exc()
        
    return results 

def visualize_results(results: Dict, dataset_type: str, max_samples: int, experiment_type: str, language: str = "ja", output_dir: str = "results"):
    """実験結果を可視化する"""
    try:
        plt.figure(figsize=(14, 6))
        
        # モデル名を取得
        model_name = results.get("model_name", "unknown_model")
        model_name_for_file = model_name.replace('/', '_').replace('-', '_')
        
        # 元のモデルの精度を水平線で表示
        original_ppl = results["original"]["avg_ppl"]
        
        # 実験タイプに応じたデータと設定
        if experiment_type == "removal":
            result_key = "removed"
            if language == "ja":
                plt.axhline(y=original_ppl, color='r', linestyle='--', label='元のモデル')
                plt.xlabel('削除した層のインデックス')
                plt.ylabel('パープレキシティ (対数スケール)')
                plt.title(f'層の削除による予測精度の変化 ({model_name})')
            else:  # 英語
                plt.axhline(y=original_ppl, color='r', linestyle='--', label='Original Model')
                plt.xlabel('Removed Layer Index')
                plt.ylabel('ppl')
                plt.title(f'Effect of Layer Removal on Prediction Performance ({model_name})')
            
            # 層削除時の精度をプロット
            layers = sorted([int(key.split('_')[1]) for key in results[result_key].keys()])
            avg_ppls = [results[result_key][f"layer_{i}"]["avg_ppl"] for i in layers]
            
            if layers and avg_ppls:
                if language == "ja":
                    plt.plot(layers, avg_ppls, 'b-o', label='各層削除時')
                else:
                    plt.plot(layers, avg_ppls, 'b-o', label='After Layer Removal')
        
        elif experiment_type == "exchange":
            result_key = "exchanged"
            if language == "ja":
                plt.axhline(y=original_ppl, color='r', linestyle='--', label='元のモデル')
                plt.xlabel('交換した層のインデックスペア')
                plt.ylabel('パープレキシティ (対数スケール)')
                plt.title(f'層の交換による予測精度の変化 ({model_name})')
            else:  # 英語
                plt.axhline(y=original_ppl, color='r', linestyle='--', label='Original Model')
                plt.xlabel('Exchanged Layer Index Pairs')
                plt.ylabel('ppl')
                plt.title(f'Effect of Layer Exchange on Prediction Performance ({model_name})')
            
            # 層交換時の精度をプロット
            layers = sorted([int(key.split('_')[1]) for key in results[result_key].keys()])
            avg_ppls = [results[result_key][f"layer_{i}_with_{i+1}"]["avg_ppl"] for i in layers]
            
            if layers and avg_ppls:
                # 横軸のラベルをペアで表示
                layer_pair_labels = [f"{i}-{i+1}" for i in layers]
                plt.xticks(layers, layer_pair_labels)
                
                if language == "ja":
                    plt.plot(layers, avg_ppls, 'b-o', label='層交換時')
                else:
                    plt.plot(layers, avg_ppls, 'b-o', label='After Layer Exchange')
        
        # ラベルを斜めに表示する設定
        plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right')
        
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # 縦軸を対数スケールに設定
        
        # グラフ下部にラベルが収まるように余白を調整
        plt.tight_layout()
        
        # 出力ディレクトリを作成
        output_dir_path = Path(output_dir) / experiment_type / f"{model_name_for_file}_{dataset_type}_n{max_samples}"
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # プロットを保存
        output_path = output_dir_path / f"{experiment_type}_{dataset_type}_n{max_samples}_{language}.png"
        plt.savefig(output_path)
        logger.info(f"プロットを保存しました: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"可視化中にエラーが発生しました: {e}")
        traceback.print_exc()

def main():
    """メイン関数"""
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='LLMのレイヤー操作実験')
    
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B',
                      help='実験に使用するモデル名')
    
    parser.add_argument('--dataset_type', type=str, default='wiki-text-2',
                      choices=['wiki-text-2', 'gpt2-output-dataset', 'wiki-text-103', 'bookcorpus'],
                      help='使用するデータセットの種類')
    
    parser.add_argument('--max_samples', type=int, default=1000,
                      help='評価するサンプル数')
    
    parser.add_argument('--experiment', type=str, default='removal',
                      choices=['removal', 'exchange'],
                      help='実験の種類（removal: 層削除, exchange: 層交換）')
    
    parser.add_argument('--output_dir', type=str, default='results',
                      help='結果を保存するディレクトリ')
    
    parser.add_argument('--gpu_id', type=str, default='0',
                      help='使用するGPUのID (例: "0", "1", "0,1")')
    
    args = parser.parse_args()
    
    # デバイスのセットアップ
    device = setup_device(args.gpu_id)
    
    # モデルとトークナイザーの読み込み
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)
    
    # 簡潔なモデル名を抽出
    model_name_short = args.model_name.split('/')[-1]
    
    # 実験の実行
    if args.experiment == 'removal':
        logger.info(f"レイヤー削除実験を開始します: {model_name_short}")
        results = analyze_layer_removal(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name_short,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            dataset_type=args.dataset_type
        )
    elif args.experiment == 'exchange':
        logger.info(f"レイヤー交換実験を開始します: {model_name_short}")
        results = analyze_layer_exchange(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name_short,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            dataset_type=args.dataset_type
        )
    
    # 結果の可視化（日英両方）
    visualize_results(results, args.dataset_type, args.max_samples, args.experiment, language="ja", output_dir=args.output_dir)
    visualize_results(results, args.dataset_type, args.max_samples, args.experiment, language="en", output_dir=args.output_dir)
    
    logger.info("実験が完了しました")

if __name__ == "__main__":
    main() 