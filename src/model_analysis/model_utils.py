"""
Utility functions for model analysis.
"""
import os
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_name: str, 
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> Tuple[Any, Any]:
    """
    Load a model and tokenizer.
    
    Args:
        model_name: Name of the model to load
        device: Device to load the model on (cpu, cuda, mps)
        load_in_8bit: Whether to load the model in 8-bit precision
        load_in_4bit: Whether to load the model in 4-bit precision
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"  # Use MPS on Apple Silicon if available
    
    logger.info(f"Loading model {model_name} on {device}")
    
    if device == "cuda":
        try:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            free_memory_gb = info.free / 1024**3
            
            logger.info(f"GPU free memory: {free_memory_gb:.2f} GB")
            
            if free_memory_gb < 15 and not (load_in_8bit or load_in_4bit):
                logger.warning(
                    f"Low GPU memory ({free_memory_gb:.2f} GB free). "
                    "Consider using --load_in_8bit or --load_in_4bit."
                )
        except:
            logger.warning("Could not check GPU memory. Continuing without memory check.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    
    if load_in_8bit and load_in_4bit:
        logger.warning("Both 8-bit and 4-bit quantization specified. Using 4-bit.")
        load_in_8bit = False
    
    if load_in_8bit:
        logger.info("Loading model in 8-bit precision")
        load_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        logger.info("Loading model in 4-bit precision")
        load_kwargs["load_in_4bit"] = True
        load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
    
    if device == "cpu":
        load_kwargs["low_cpu_mem_usage"] = True
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        logger.info(f"Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Trying to load with safetensors=False...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                **load_kwargs,
                safetensors=False
            )
            logger.info(f"Model loaded successfully with safetensors=False")
        except Exception as e2:
            logger.error(f"Error loading model with safetensors=False: {e2}")
            raise e2
    
    return model, tokenizer

def get_model_activations(
    model: Any,
    tokenizer: Any,
    text: str,
    layer_indices: Optional[List[int]] = None,
    max_length: int = 512,
    focus_token_idx: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Get activations from a model for a given text.
    
    Args:
        model: The model to get activations from
        tokenizer: The tokenizer to use
        text: The text to get activations for
        layer_indices: Indices of layers to get activations for (None for all layers)
        max_length: Maximum sequence length to process
        focus_token_idx: If provided, only return activations for this token index
                         (None means return activations for the last token)
        
    Returns:
        Dictionary of activations for each layer
    """
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length,
        padding="max_length"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    if focus_token_idx is None:
        seq_len = attention_mask.sum(dim=1)[0].item()
        focus_token_idx = seq_len - 1
    
    if layer_indices is None:
        layer_indices = list(range(len(model.model.layers)))
    
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output[0].detach()
        return hook
    
    # Register hooks for each layer
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        hook = layer.register_forward_hook(get_activation(f"layer_{layer_idx}"))
        hooks.append(hook)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    token_ids = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    logger.debug(f"Input tokens: {tokens}")
    logger.debug(f"Focus token index: {focus_token_idx}, Token: {tokens[focus_token_idx]}")
    
    return activations

def calculate_activation_differences(
    activations1: Dict[str, torch.Tensor],
    activations2: Dict[str, torch.Tensor],
    focus_token_idx: Optional[int] = None,
    normalization: str = "none"
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate activation differences between two sets of activations.
    
    Args:
        activations1: First set of activations
        activations2: Second set of activations
        focus_token_idx: If provided, use this token index for comparison
                         (None means use the last token)
        normalization: Normalization method for differences:
                       - "none": raw differences
                       - "l2": L2 normalization
                       - "cosine": cosine similarity
        
    Returns:
        Dictionary of activation differences for each layer and dimension
    """
    differences = {}
    
    common_layers = set(activations1.keys()).intersection(set(activations2.keys()))
    
    for layer_name in common_layers:
        act1 = activations1[layer_name]
        act2 = activations2[layer_name]
        
        if act1.shape != act2.shape:
            logger.warning(f"Activation shapes differ for {layer_name}: {act1.shape} vs {act2.shape}")
            continue
        
        if focus_token_idx is None:
            focus_token_idx = -1  # Last token
        
        act1_token = act1[0, focus_token_idx, :]
        act2_token = act2[0, focus_token_idx, :]
        
        # Calculate differences based on normalization method
        if normalization == "l2":
            act1_norm = torch.nn.functional.normalize(act1_token, p=2, dim=0)
            act2_norm = torch.nn.functional.normalize(act2_token, p=2, dim=0)
            diff = torch.abs(act1_norm - act2_norm)
        elif normalization == "cosine":
            cos_sim = torch.nn.functional.cosine_similarity(
                act1_token.unsqueeze(0), 
                act2_token.unsqueeze(0)
            ).item()
            diff = torch.abs(act1_token - act2_token)  # Still compute abs diff for per-dimension analysis
            differences[layer_name] = {
                "dimensions": {},
                "overall_cosine_similarity": cos_sim,
                "overall_cosine_difference": 1.0 - cos_sim
            }
        else:
            # Raw absolute differences
            diff = torch.abs(act1_token - act2_token)
            
            # Calculate overall statistics
            differences[layer_name] = {
                "dimensions": {},
                "overall_mean_diff": diff.mean().item(),
                "overall_max_diff": diff.max().item(),
                "overall_min_diff": diff.min().item()
            }
        
        # Store per-dimension differences
        for dim_idx in range(act1_token.shape[-1]):
            act1_dim = act1_token[dim_idx].item()
            act2_dim = act2_token[dim_idx].item()
            diff_dim = diff[dim_idx].item()
            
            differences[layer_name]["dimensions"][f"dim_{dim_idx}"] = {
                "sentence1_activation": act1_dim,
                "sentence2_activation": act2_dim,
                "activation_diff": diff_dim
            }
    
    return differences

def find_minimal_difference_dimensions(
    differences: Dict[str, Dict[str, Any]],
    threshold: float = 0.01
) -> Dict[str, List[int]]:
    """
    Find dimensions with minimal activation differences.
    
    Args:
        differences: Dictionary of activation differences
        threshold: Threshold for considering a difference minimal
        
    Returns:
        Dictionary mapping layer names to lists of dimensions with minimal differences
    """
    minimal_diff_dimensions = {}
    
    for layer_name, layer_data in differences.items():
        minimal_dims = []
        
        for dim_name, dim_data in layer_data["dimensions"].items():
            if dim_data["activation_diff"] < threshold:
                dim_idx = int(dim_name.split("_")[1])
                minimal_dims.append(dim_idx)
        
        minimal_diff_dimensions[layer_name] = minimal_dims
    
    return minimal_diff_dimensions
