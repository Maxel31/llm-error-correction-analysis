"""
Utility functions for dataset generation.
"""
import os
import json
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SENTENCE_PAIR_TYPES = {
    "meaning_differences": {
        "subject_differences": "Generate pairs of sentences where only the subject differs (e.g., 'She loves coffee.' vs 'He loves coffee.')",
        "location_differences": "Generate pairs of sentences where only a location word differs (e.g., 'The cat sat on the mat.' vs 'The cat sat on the hat.')",
        "adjective_differences": "Generate pairs of sentences where only an adjective differs (e.g., 'This is absolutely amazing.' vs 'This is absolutely terrible.')"
    },
    "grammatical_errors": {
        "tense_errors": "Generate pairs of sentences where one has the correct tense and the other has an incorrect tense (e.g., 'She was very tired yesterday.' vs 'She is very tired yesterday.')",
        "article_errors": "Generate pairs of sentences where one has the correct article and the other has an incorrect article (e.g., 'He adopted a dog from the shelter.' vs 'He adopted the dog from the shelter.')",
        "preposition_errors": "Generate pairs of sentences where one has the correct preposition and the other has an incorrect preposition (e.g., 'She is good at math.' vs 'She is good in math.')"
    }
}

def save_dataset(dataset: Dict[str, Any], output_path: str) -> None:
    """
    Save the generated dataset to a JSON file.
    
    Args:
        dataset: The dataset to save
        output_path: Path to save the dataset
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"Dataset saved to {output_path}")

def load_dataset(input_path: str) -> Dict[str, Any]:
    """
    Load a dataset from a JSON file.
    
    Args:
        input_path: Path to the dataset
        
    Returns:
        The loaded dataset
    """
    with open(input_path, 'r') as f:
        dataset = json.load(f)
    
    logger.info(f"Dataset loaded from {input_path}")
    return dataset

def verify_token_difference(sentence1: str, sentence2: str, tokenizer) -> Tuple[bool, int]:
    """
    Verify that two sentences differ by only one token.
    
    Args:
        sentence1: First sentence
        sentence2: Second sentence
        tokenizer: Tokenizer to use for tokenization
        
    Returns:
        Tuple of (is_one_token_diff, num_different_tokens)
    """
    tokens1 = tokenizer.encode(sentence1)
    tokens2 = tokenizer.encode(sentence2)
    
    if len(tokens1) != len(tokens2):
        return False, abs(len(tokens1) - len(tokens2))
    
    diff_count = sum(1 for t1, t2 in zip(tokens1, tokens2) if t1 != t2)
    
    return diff_count == 1, diff_count
