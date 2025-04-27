"""
Generate a dataset of sentence pairs that differ by only one token using the ChatGPT API.
"""
import os
import json
import time
import logging
import argparse
from typing import Dict, List, Tuple, Any
from pathlib import Path
from datetime import datetime

import openai
from dotenv import load_dotenv
from transformers import AutoTokenizer
from tqdm import tqdm

from src.data_generation.utils import (
    SENTENCE_PAIR_TYPES,
    save_dataset,
    verify_token_difference
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

def generate_sentence_pairs(
    pair_type: str,
    subtype: str,
    description: str,
    num_pairs: int = 5,
    model: str = "gpt-3.5-turbo"
) -> List[Dict[str, str]]:
    """
    Generate sentence pairs using the ChatGPT API.
    
    Args:
        pair_type: Type of sentence pair (e.g., meaning_differences)
        subtype: Subtype of sentence pair (e.g., subject_differences)
        description: Description of the sentence pair type
        num_pairs: Number of pairs to generate
        model: Model to use for generation
        
    Returns:
        List of sentence pairs
    """
    logger.info(f"Generating {num_pairs} {pair_type}/{subtype} sentence pairs")
    
    prompt = f"""
    I need {num_pairs} pairs of sentences where each pair differs by exactly one token (word).
    
    Type: {pair_type}, Subtype: {subtype}
    Description: {description}
    
    Important requirements:
    1. Each pair should differ by EXACTLY ONE TOKEN when tokenized.
    2. The sentences should be natural and grammatically correct (unless the purpose is to show a grammatical error).
    3. The sentences should be diverse and cover different topics.
    4. Return the results in JSON format as a list of objects with 'sentence1' and 'sentence2' keys.
    
    Example format:
    [
        {{"sentence1": "She loves coffee.", "sentence2": "He loves coffee."}},
        {{"sentence1": "The cat sat on the mat.", "sentence2": "The cat sat on the hat."}}
    ]
    """
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates pairs of sentences that differ by exactly one token."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content
        
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        
        if start_idx == -1 or end_idx == 0:
            logger.error(f"Failed to parse JSON from response: {content}")
            return []
        
        json_str = content[start_idx:end_idx]
        sentence_pairs = json.loads(json_str)
        
        logger.info(f"Generated {len(sentence_pairs)} sentence pairs")
        return sentence_pairs
        
    except Exception as e:
        logger.error(f"Error generating sentence pairs: {e}")
        return []

def main(args):
    """
    Main function to generate the dataset.
    """
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    dataset = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "model": args.model,
            "tokenizer": args.tokenizer,
            "num_pairs_per_type": args.num_pairs
        },
        "diff_token_type": {}
    }
    
    total_pairs = 0
    valid_pairs = 0
    
    for pair_type, subtypes in tqdm(SENTENCE_PAIR_TYPES.items(), desc="Generating pairs"):
        dataset["diff_token_type"][pair_type] = {}
        
        for subtype, description in tqdm(subtypes.items(), desc=f"Processing {pair_type}"):
            sentence_pairs = generate_sentence_pairs(
                pair_type=pair_type,
                subtype=subtype,
                description=description,
                num_pairs=args.num_pairs,
                model=args.model
            )
            
            valid_sentence_pairs = []
            for pair in sentence_pairs:
                is_valid, diff_count = verify_token_difference(
                    pair["sentence1"],
                    pair["sentence2"],
                    tokenizer
                )
                
                total_pairs += 1
                
                if is_valid:
                    valid_pairs += 1
                    valid_sentence_pairs.append(pair)
                else:
                    logger.warning(
                        f"Pair differs by {diff_count} tokens, not 1: "
                        f"'{pair['sentence1']}' vs '{pair['sentence2']}'"
                    )
            
            dataset["diff_token_type"][pair_type][subtype] = {
                "sentence_pairs": valid_sentence_pairs
            }
            
            time.sleep(1)
    
    logger.info(f"Generated {total_pairs} total pairs")
    logger.info(f"Valid pairs (differing by exactly 1 token): {valid_pairs}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"sentence_pairs_{timestamp}.json")
    save_dataset(dataset, output_path)
    
    logger.info(f"Dataset generation complete. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentence pairs dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model to use for generation"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3-7b-hf",
        help="Tokenizer to use for verification"
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=17,  # ~17 pairs per type to get ~100 total
        help="Number of pairs to generate per type"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save the dataset"
    )
    
    args = parser.parse_args()
    main(args)
