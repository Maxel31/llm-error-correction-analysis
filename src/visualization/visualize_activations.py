"""
Visualize activation differences between sentence pairs.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def plot_activation_differences(
    results: Dict[str, Any],
    output_dir: str,
    top_n: int = 10,
    threshold: float = 0.01
) -> None:
    """
    Plot activation differences for each pair type and subtype.
    
    Args:
        results: Dictionary of activation differences
        output_dir: Directory to save the plots
        top_n: Number of dimensions with smallest differences to highlight
        threshold: Threshold for considering a difference minimal
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for pair_type, subtypes in results["diff_token_type"].items():
        for subtype, pairs in subtypes.items():
            subtype_dir = os.path.join(output_dir, pair_type, subtype)
            os.makedirs(subtype_dir, exist_ok=True)
            
            for pair_id, pair_data in pairs.items():
                sentence1 = pair_data["sentence_pair"]["sentence1"]
                sentence2 = pair_data["sentence_pair"]["sentence2"]
                
                plot_activation_heatmap(
                    pair_data=pair_data,
                    pair_id=pair_id,
                    sentence1=sentence1,
                    sentence2=sentence2,
                    output_dir=subtype_dir,
                    threshold=threshold
                )
                
                plot_top_minimal_dimensions(
                    pair_data=pair_data,
                    pair_id=pair_id,
                    sentence1=sentence1,
                    sentence2=sentence2,
                    output_dir=subtype_dir,
                    top_n=top_n,
                    threshold=threshold
                )

def plot_activation_heatmap(
    pair_data: Dict[str, Any],
    pair_id: str,
    sentence1: str,
    sentence2: str,
    output_dir: str,
    threshold: float = 0.01
) -> None:
    """
    Plot heatmap of activation differences across layers and dimensions.
    
    Args:
        pair_data: Dictionary of activation differences for a pair
        pair_id: ID of the pair
        sentence1: First sentence
        sentence2: Second sentence
        output_dir: Directory to save the plot
        threshold: Threshold for considering a difference minimal
    """
    layers = pair_data["layers"]
    
    sample_size = 100
    
    layer_names = []
    dim_names = []
    diff_values = []
    
    for layer_name, layer_data in layers.items():
        dimensions = layer_data["dimensions"]
        
        if len(dimensions) > sample_size:
            sampled_dims = np.random.choice(list(dimensions.keys()), sample_size, replace=False)
            dimensions = {dim: dimensions[dim] for dim in sampled_dims}
        
        for dim_name, dim_data in dimensions.items():
            layer_names.append(layer_name)
            dim_names.append(dim_name)
            diff_values.append(dim_data["activation_diff"])
    
    df = pd.DataFrame({
        "layer": layer_names,
        "dimension": dim_names,
        "difference": diff_values
    })
    
    heatmap_data = df.pivot(index="layer", columns="dimension", values="difference")
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        heatmap_data,
        cmap="viridis_r",  # Reversed viridis (darker = smaller difference)
        vmin=0,
        vmax=max(threshold * 2, heatmap_data.values.max()),  # Cap at 2x threshold or max value
        cbar_kws={"label": "Activation Difference"}
    )
    
    plt.title(f"Activation Differences: '{sentence1}' vs '{sentence2}'")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{pair_id}_heatmap.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_top_minimal_dimensions(
    pair_data: Dict[str, Any],
    pair_id: str,
    sentence1: str,
    sentence2: str,
    output_dir: str,
    top_n: int = 10,
    threshold: float = 0.01
) -> None:
    """
    Plot top dimensions with smallest activation differences.
    
    Args:
        pair_data: Dictionary of activation differences for a pair
        pair_id: ID of the pair
        sentence1: First sentence
        sentence2: Second sentence
        output_dir: Directory to save the plot
        top_n: Number of dimensions to plot
        threshold: Threshold for considering a difference minimal
    """
    layers = pair_data["layers"]
    
    all_dims = []
    
    for layer_name, layer_data in layers.items():
        dimensions = layer_data["dimensions"]
        
        for dim_name, dim_data in dimensions.items():
            diff = dim_data["activation_diff"]
            
            if diff < threshold:
                all_dims.append({
                    "layer": layer_name,
                    "dimension": dim_name,
                    "difference": diff,
                    "sentence1_activation": dim_data["sentence1_activation"],
                    "sentence2_activation": dim_data["sentence2_activation"]
                })
    
    all_dims.sort(key=lambda x: x["difference"])
    
    top_dims = all_dims[:top_n]
    
    if not top_dims:
        return  # No dimensions below threshold
    
    labels = [f"{d['layer']}/{d['dimension']}" for d in top_dims]
    diffs = [d["difference"] for d in top_dims]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, diffs)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            rotation=0
        )
    
    plt.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold ({threshold})")
    plt.title(f"Top {len(top_dims)} Dimensions with Minimal Activation Differences")
    plt.xlabel("Layer/Dimension")
    plt.ylabel("Activation Difference")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{pair_id}_top_minimal_dims.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, len(top_dims) * 0.5 + 2))
    plt.axis("off")
    
    table_data = [
        ["Layer", "Dimension", "Diff", "Activation 1", "Activation 2"]
    ]
    
    for dim in top_dims:
        table_data.append([
            dim["layer"],
            dim["dimension"],
            f"{dim['difference']:.6f}",
            f"{dim['sentence1_activation']:.6f}",
            f"{dim['sentence2_activation']:.6f}"
        ])
    
    table = plt.table(
        cellText=table_data,
        colLabels=None,
        cellLoc="center",
        loc="center",
        edges="horizontal"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title(f"Details of Top {len(top_dims)} Dimensions with Minimal Differences")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{pair_id}_top_minimal_dims_table.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_layer_statistics(
    results: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Plot statistics for each layer across all pairs.
    
    Args:
        results: Dictionary of activation differences
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    layer_stats = {}
    
    for pair_type, subtypes in results["diff_token_type"].items():
        for subtype, pairs in subtypes.items():
            for pair_id, pair_data in pairs.items():
                layers = pair_data["layers"]
                
                for layer_name, layer_data in layers.items():
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = {
                            "mean_diffs": [],
                            "min_diffs": [],
                            "max_diffs": []
                        }
                    
                    if "overall_mean_diff" in layer_data:
                        layer_stats[layer_name]["mean_diffs"].append(layer_data["overall_mean_diff"])
                    
                    if "overall_min_diff" in layer_data:
                        layer_stats[layer_name]["min_diffs"].append(layer_data["overall_min_diff"])
                    
                    if "overall_max_diff" in layer_data:
                        layer_stats[layer_name]["max_diffs"].append(layer_data["overall_max_diff"])
    
    layer_names = sorted(layer_stats.keys(), key=lambda x: int(x.split("_")[1]))
    mean_diffs = []
    min_diffs = []
    max_diffs = []
    
    for layer_name in layer_names:
        stats = layer_stats[layer_name]
        
        if stats["mean_diffs"]:
            mean_diffs.append(np.mean(stats["mean_diffs"]))
        else:
            mean_diffs.append(0)
        
        if stats["min_diffs"]:
            min_diffs.append(np.mean(stats["min_diffs"]))
        else:
            min_diffs.append(0)
        
        if stats["max_diffs"]:
            max_diffs.append(np.mean(stats["max_diffs"]))
        else:
            max_diffs.append(0)
    
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(layer_names))
    width = 0.25
    
    plt.bar(x - width, mean_diffs, width, label="Mean Difference")
    plt.bar(x, min_diffs, width, label="Min Difference")
    plt.bar(x + width, max_diffs, width, label="Max Difference")
    
    plt.xlabel("Layer")
    plt.ylabel("Activation Difference")
    plt.title("Average Activation Differences by Layer")
    plt.xticks(x, [f"Layer {i}" for i in range(len(layer_names))])
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "layer_statistics.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def visualize_results(
    results_path: str,
    output_dir: str,
    top_n: int = 10,
    threshold: float = 0.01
) -> None:
    """
    Visualize results from an activation analysis.
    
    Args:
        results_path: Path to the results JSON file
        output_dir: Directory to save the visualizations
        top_n: Number of dimensions with smallest differences to highlight
        threshold: Threshold for considering a difference minimal
    """
    with open(results_path, "r") as f:
        results = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    plot_activation_differences(
        results=results,
        output_dir=os.path.join(output_dir, "activation_differences"),
        top_n=top_n,
        threshold=threshold
    )
    
    plot_layer_statistics(
        results=results,
        output_dir=os.path.join(output_dir, "layer_statistics")
    )
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize activation differences")
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to the results JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory to save the visualizations"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of dimensions with smallest differences to highlight"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Threshold for considering a difference minimal"
    )
    
    args = parser.parse_args()
    visualize_results(
        results_path=args.results_path,
        output_dir=args.output_dir,
        top_n=args.top_n,
        threshold=args.threshold
    )
