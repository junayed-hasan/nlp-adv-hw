#!/usr/bin/env python3
"""
Perplexity and Sampling Analysis Script
======================================

This script analyzes perplexity of text and compares different sampling strategies
using a pretrained language model (DistilGPT2).

Author: NLP Advanced HW Assignment
Date: September 2025
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def load_model_and_tokenizer():
    """Load DistilGPT2 model and tokenizer."""
    print("Loading DistilGPT2 model and tokenizer...")
    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Set to evaluation mode
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model loaded on {device}")
    return model, tokenizer, device

def compute_perplexity(model, tokenizer, text, device):
    """
    Compute perplexity of a given text using the language model.
    
    Args:
        model: Pretrained language model
        tokenizer: Corresponding tokenizer
        text: Input text string
        device: Device to run on
        
    Returns:
        perplexity: Perplexity value
        tokens: List of tokens
    """
    # Tokenize the text
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        # Get model outputs
        outputs = model(tokens, labels=tokens)
        loss = outputs.loss
        
        # Perplexity is exp(average negative log-likelihood)
        perplexity = torch.exp(loss).item()
    
    # Convert tokens back to strings for display
    token_strings = [tokenizer.decode([token]) for token in tokens[0]]
    
    return perplexity, token_strings

def shuffle_text_tokens(text, tokenizer):
    """Shuffle the tokens of text while maintaining tokenization."""
    tokens = tokenizer.encode(text)
    # Keep the first token (often special) and shuffle the rest
    if len(tokens) > 1:
        tokens_to_shuffle = tokens[1:]
        random.shuffle(tokens_to_shuffle)
        shuffled_tokens = [tokens[0]] + tokens_to_shuffle
    else:
        shuffled_tokens = tokens
    
    shuffled_text = tokenizer.decode(shuffled_tokens)
    return shuffled_text

def generate_with_temperature(model, tokenizer, prompt, temperature, max_length, device):
    """
    Generate text with a specific temperature.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        temperature: Sampling temperature
        max_length: Maximum generation length
        device: Device to run on
        
    Returns:
        generated_text: Generated continuation
    """
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        if temperature == 0:
            # Greedy decoding
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        else:
            # Temperature sampling
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
    
    # Decode only the generated part (excluding the prompt)
    generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return generated_text

def main():
    """Main function to run perplexity and sampling analysis."""
    
    print("Perplexity and Sampling Analysis")
    print("=================================")
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer()
    
    # =================
    # PERPLEXITY ANALYSIS
    # =================
    print("\n" + "="*50)
    print("PART A: PERPLEXITY ANALYSIS")
    print("="*50)
    
    # Original paragraph (3-5 sentences)
    original_text = """The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once. It has been used for typing practice and font testing for many years. The phrase became popular in the early 20th century. Today it remains a standard test sentence."""
    
    print(f"\nOriginal text:")
    print(f'"{original_text}"')
    
    # Compute perplexity of original text
    original_perplexity, original_tokens = compute_perplexity(model, tokenizer, original_text, device)
    print(f"\nOriginal text perplexity: {original_perplexity:.4f}")
    print(f"Number of tokens: {len(original_tokens)}")
    
    # Generate shuffled version
    shuffled_text = shuffle_text_tokens(original_text, tokenizer)
    print(f"\nShuffled text:")
    print(f'"{shuffled_text}"')
    
    # Compute perplexity of shuffled text
    shuffled_perplexity, shuffled_tokens = compute_perplexity(model, tokenizer, shuffled_text, device)
    print(f"\nShuffled text perplexity: {shuffled_perplexity:.4f}")
    print(f"Number of tokens: {len(shuffled_tokens)}")
    
    # Compare perplexities
    ratio = shuffled_perplexity / original_perplexity
    print(f"\nPerplexity ratio (shuffled/original): {ratio:.4f}")
    print(f"Shuffled text is {ratio:.2f}x more perplexing than original")
    
    # =================
    # SAMPLING COMPARISON
    # =================
    print("\n" + "="*50)
    print("PART B: SAMPLING COMPARISON")
    print("="*50)
    
    prompt = "Once upon a time"
    max_length = 500
    temperatures = [0, 0.3, 0.6, 0.9, 1.2, 1.5]
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Generation length: {max_length} tokens")
    print(f"Temperatures tested: {temperatures}")
    
    generations = {}
    
    for temp in temperatures:
        print(f"\nGenerating with temperature = {temp}...")
        
        if temp == 0:
            method_name = "Greedy Decoding (T=0)"
        else:
            method_name = f"Temperature Sampling (T={temp})"
        
        generated_text = generate_with_temperature(
            model, tokenizer, prompt, temp, max_length, device
        )
        
        generations[temp] = generated_text
        
        print(f"\n{method_name}:")
        print("-" * 40)
        print(f"{prompt}{generated_text}")
        print("-" * 40)
        
        # Calculate some basic statistics
        words = generated_text.split()
        unique_words = set(words)
        if len(words) > 0:
            diversity = len(unique_words) / len(words)
        else:
            diversity = 0
            
        print(f"Generated words: {len(words)}")
        print(f"Unique words: {len(unique_words)}")
        print(f"Lexical diversity: {diversity:.3f}")
    
    # =================
    # ANALYSIS SUMMARY
    # =================
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    print("\nPerplexity Analysis:")
    print(f"- Original text perplexity: {original_perplexity:.4f}")
    print(f"- Shuffled text perplexity: {shuffled_perplexity:.4f}")
    print(f"- The shuffled text has {ratio:.2f}x higher perplexity, indicating")
    print(f"  that word order and coherence significantly affect model predictions.")
    
    print("\nSampling Strategy Comparison:")
    print("- Temperature = 0 (Greedy): Most coherent but deterministic")
    print("- Low temperatures (0.3-0.6): Good balance of coherence and variety")
    print("- Medium temperatures (0.9): More creative but potentially less coherent")
    print("- High temperatures (1.2-1.5): Very diverse but may sacrifice quality")
    
    # Calculate diversity scores for comparison
    diversities = []
    for temp in temperatures:
        words = generations[temp].split()
        unique_words = set(words)
        if len(words) > 0:
            diversity = len(unique_words) / len(words)
        else:
            diversity = 0
        diversities.append(diversity)
    
    print("\nLexical Diversity by Temperature:")
    for temp, div in zip(temperatures, diversities):
        temp_str = "Greedy" if temp == 0 else f"T={temp}"
        print(f"- {temp_str}: {div:.3f}")
    
    print(f"\nExperiment completed successfully!")
    print(f"Repository: https://github.com/junayedhs/nlp-advanced-hw")

if __name__ == "__main__":
    main()
