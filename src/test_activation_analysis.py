"""
Test script for activation analysis.
"""
import os
import json
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from src.data_generation.generate_dataset import main as generate_dataset_main
from src.model_analysis.analyze_activations import analyze_dataset
from src.model_analysis.model_utils import load_model_and_tokenizer
from src.visualization.visualize_activations import visualize_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(args):
    """
    Main function to test activation analysis.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset_dir = os.path.join(args.output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    if args.dataset_path:
        logger.info(f"Using existing dataset: {args.dataset_path}")
        dataset_path = args.dataset_path
    else:
        logger.info("Generating new dataset...")
        
        # Create args for generate_dataset_main
        generate_args = argparse.Namespace(
            model=args.openai_model,
            tokenizer=args.model_name,
            num_pairs=args.num_pairs,
            output_dir=dataset_dir
        )
        
        generate_dataset_main(generate_args)
        
        dataset_files = list(Path(dataset_dir).glob("sentence_pairs_*.json"))
        if dataset_files:
            dataset_path = str(sorted(dataset_files, key=lambda x: x.stat().st_mtime, reverse=True)[0])
            logger.info(f"Dataset generated: {dataset_path}")
        else:
            raise FileNotFoundError("Dataset generation failed: No dataset file found")
    
    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    logger.info("Analyzing activations...")
    output_path = os.path.join(results_dir, f"activation_results_{timestamp}.json")
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    layer_indices = None
    if args.layer_indices:
        layer_indices = [int(idx) for idx in args.layer_indices.split(',')]
    
    results = analyze_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_indices=layer_indices,
        output_path=output_path
    )
    
    logger.info(f"Analysis results saved to: {output_path}")
    
    if args.visualize:
        logger.info("Generating visualizations...")
        visualize_results(
            results_path=output_path,
            output_dir=figures_dir,
            top_n=args.top_n,
            threshold=args.threshold
        )
        logger.info(f"Visualizations saved to: {figures_dir}")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test activation analysis")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3-7b-hf",
        help="Name of the model to analyze"
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model to use for dataset generation"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to existing dataset (if not provided, a new dataset will be generated)"
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=10,
        help="Number of sentence pairs to generate (if generating a new dataset)"
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
    
    args = parser.parse_args()
    main(args)
