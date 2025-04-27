"""
Analyze activations from the Llama-3-7B model for sentence pairs.
"""
import os
import json
import torch
import logging
import argparse
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from dotenv import load_dotenv
from src.data_generation.utils import load_dataset
from src.model_analysis.model_utils import (
    load_model_and_tokenizer,
    get_model_activations,
    calculate_activation_differences,
    find_minimal_difference_dimensions
)
from src.visualization.visualize_activations import visualize_results

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_sentence_pair(
    model: Any,
    tokenizer: Any,
    sentence1: str,
    sentence2: str,
    layer_indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Analyze activations for a pair of sentences.
    
    Args:
        model: The model to analyze
        tokenizer: The tokenizer to use
        sentence1: First sentence
        sentence2: Second sentence
        layer_indices: Indices of layers to analyze (None for all layers)
        
    Returns:
        Dictionary of activation differences
    """
    activations1 = get_model_activations(model, tokenizer, sentence1, layer_indices)
    activations2 = get_model_activations(model, tokenizer, sentence2, layer_indices)
    
    differences = calculate_activation_differences(activations1, activations2)
    
    return differences

def analyze_dataset(
    model: Any,
    tokenizer: Any,
    dataset: Dict[str, Any],
    layer_indices: Optional[List[int]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze activations for all sentence pairs in a dataset.
    
    Args:
        model: The model to analyze
        tokenizer: The tokenizer to use
        dataset: The dataset to analyze
        layer_indices: Indices of layers to analyze (None for all layers)
        output_path: Path to save the results
        
    Returns:
        Dictionary of activation differences for all sentence pairs
    """
    results = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "model_name": model.config._name_or_path,
            "device": model.device,
            "layer_indices": layer_indices
        },
        "diff_token_type": {}
    }
    
    for pair_type, subtypes in tqdm(dataset["diff_token_type"].items(), desc="Analyzing pair types"):
        results["diff_token_type"][pair_type] = {}
        
        for subtype, subtype_data in tqdm(subtypes.items(), desc=f"Processing {pair_type}"):
            results["diff_token_type"][pair_type][subtype] = {}
            
            for i, pair in enumerate(tqdm(subtype_data["sentence_pairs"], desc=f"Analyzing {subtype} pairs")):
                sentence1 = pair["sentence1"]
                sentence2 = pair["sentence2"]
                
                differences = analyze_sentence_pair(
                    model=model,
                    tokenizer=tokenizer,
                    sentence1=sentence1,
                    sentence2=sentence2,
                    layer_indices=layer_indices
                )
                
                results["diff_token_type"][pair_type][subtype][f"pair_{i}"] = {
                    "sentence_pair": {
                        "sentence1": sentence1,
                        "sentence2": sentence2
                    },
                    "layers": differences
                }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return results

def main(args):
    """
    Main function to analyze activations.
    """
    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name, 
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    layer_indices = None
    if args.layer_indices:
        layer_indices = [int(idx) for idx in args.layer_indices.split(',')]
    
    output_path = os.path.join(results_dir, f"activation_results_{timestamp}.json")
    minimal_diff_path = os.path.join(results_dir, f"minimal_diff_dimensions_{timestamp}.json")
    
    results = analyze_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_indices=layer_indices,
        output_path=output_path
    )
    
    # Find minimal difference dimensions
    minimal_diff_dimensions = {}
    
    for pair_type, subtypes in results["diff_token_type"].items():
        minimal_diff_dimensions[pair_type] = {}
        
        for subtype, pairs in subtypes.items():
            minimal_diff_dimensions[pair_type][subtype] = {}
            
            for pair_id, pair_data in pairs.items():
                differences = pair_data["layers"]
                minimal_dims = find_minimal_difference_dimensions(
                    differences=differences,
                    threshold=args.threshold
                )
                
                minimal_diff_dimensions[pair_type][subtype][pair_id] = minimal_dims
    
    # Save minimal difference dimensions
    with open(minimal_diff_path, 'w') as f:
        json.dump(minimal_diff_dimensions, f, indent=2)
    
    logger.info(f"Minimal difference dimensions saved to {minimal_diff_path}")
    
    if args.visualize:
        logger.info("Generating visualizations...")
        visualize_results(
            results_path=output_path,
            output_dir=figures_dir,
            top_n=args.top_n,
            threshold=args.threshold
        )
        logger.info(f"Visualizations saved to {figures_dir}")
    
    logger.info("Analysis complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze activations for sentence pairs")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3-7b-hf",
        help="Name of the model to analyze"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset of sentence pairs"
    )
    parser.add_argument(
        "--layer_indices",
        type=str,
        default=None,
        help="Comma-separated list of layer indices to analyze (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Threshold for considering a difference minimal"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save the results"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision to reduce memory usage"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit precision to reduce memory usage"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of the results"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of dimensions with smallest differences to highlight in visualizations"
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["none", "l2", "cosine"],
        default="none",
        help="Normalization method for activation differences"
    )
    
    args = parser.parse_args()
    main(args)
