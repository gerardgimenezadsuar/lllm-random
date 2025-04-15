from openai import OpenAI
import anthropic
import numpy as np
from scipy import stats
import json
import os
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import re
# Add Google Gemini imports
import base64
from google import genai
from google.genai import types

# Configure API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Add Gemini API key

def get_random_binary_from_llm(
    batch_size: int = 100, 
    conversation_history=None, 
    is_first_batch=True, 
    model_name="gemma-3-4b-it",
    provider="gemini"
) -> List[int]:
    """
    Query the LLM to generate random binary values (0 or 1)
    Uses conversation history to maintain context of previously generated numbers
    Supports OpenAI, Anthropic, and Google Gemini models
    """
    # Initialize conversation history if None
    if conversation_history is None:
        if provider == "openai":
            conversation_history = [
                {"role": "system", "content": "You are a helpful assistant that generates random numbers."}
            ]
        elif provider == "anthropic":  # anthropic
            conversation_history = [
                {"role": "assistant", "content": "I'm a helpful assistant that generates random numbers."}
            ]
        else:  # gemini
            conversation_history = [
                {"role": "user", "content": "You are a helpful assistant that generates random numbers."},
                {"role": "model", "content": "I'm a helpful assistant that generates random numbers."}
            ]
    
    # Add the new request to conversation history
    if is_first_batch:
        prompt = f"""Generate {batch_size} random binary values (either 0 or 1), like flipping a coin.
        Format them as a comma-separated list without any additional text or explanation.
        Example format: 0,1,0,1,1,0"""
    else:
        prompt = "Generate a new batch."
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": prompt})
    
    # Get response based on provider
    if provider == "openai":
        return _get_openai_response(conversation_history, model_name, batch_size)
    elif provider == "anthropic":  # anthropic
        return _get_anthropic_response(conversation_history, model_name, batch_size)
    else:  # gemini
        return _get_gemini_response(conversation_history, model_name, batch_size)

def _get_openai_response(conversation_history, model_name, batch_size):
    """Handle OpenAI API requests"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    response = client.chat.completions.create(
        model=model_name,
        messages=conversation_history,
        response_format={"type": "text"},
        temperature=0.5,
    )

    # Parse the response
    numbers_str = response.choices[0].message.content.strip()
    
    # Add the model's response to conversation history
    conversation_history.append({"role": "assistant", "content": numbers_str})
    
    # Process the response
    numbers = _process_response(numbers_str, batch_size)
    
    return numbers, conversation_history

def _get_anthropic_response(conversation_history, model_name, batch_size):
    """Handle Anthropic API requests"""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Convert OpenAI-style messages to Anthropic format if needed
    anthropic_messages = []
    for msg in conversation_history:
        # Skip system messages for Anthropic (or convert to assistant)
        if msg["role"] == "system":
            anthropic_messages.append({"role": "assistant", "content": msg["content"]})
        else:
            anthropic_messages.append(msg)
    
    response = client.messages.create(
        model=model_name,
        max_tokens=4096,
        temperature=0.5,
        messages=anthropic_messages
    )

    # Parse the response
    numbers_str = response.content[0].text
    
    # Add the model's response to conversation history
    conversation_history.append({"role": "assistant", "content": numbers_str})
    
    # Process the response
    numbers = _process_response(numbers_str, batch_size)
    
    return numbers, conversation_history

def _get_gemini_response(conversation_history, model_name, batch_size):
    """Handle Google Gemini API requests"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Convert conversation history to Gemini format
    gemini_messages = []
    for msg in conversation_history:
        role = msg["role"]
        # Map roles to Gemini format
        if role == "system" or role == "assistant":
            role = "model"
        gemini_messages.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=msg["content"])]
        ))
    
    # Get the most recent user message
    user_message = conversation_history[-1]["content"]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        temperature=0.5,
    )
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=gemini_messages,
            config=generate_content_config,
        )
        
        # Parse the response
        numbers_str = response.text
        
        # Add the model's response to conversation history
        conversation_history.append({"role": "model", "content": numbers_str})
        
        # Process the response
        numbers = _process_response(numbers_str, batch_size)
        
        return numbers, conversation_history
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        # Generate fallback random values
        numbers = np.random.randint(0, 2, size=batch_size).tolist()
        conversation_history.append({"role": "model", "content": ",".join(map(str, numbers))})
        return numbers, conversation_history

def _process_response(numbers_str, batch_size):
    """Process and validate the response from any model"""
    # Add validation to ensure we only get valid binary values
    try:
        # Clean the response - sometimes models add extra text
        # Try to extract just the comma-separated list using regex
        match = re.search(r'(\d+(?:,\s*\d+)+)', numbers_str)
        if match:
            numbers_str = match.group(1)
        
        raw_numbers = numbers_str.split(',')
        numbers = []
        for x in raw_numbers:
            value = int(x.strip())
            if value not in [0, 1]:
                print(f"Warning: Found non-binary value {value}, skipping")
                continue
            numbers.append(value)
            
        if len(numbers) == 0:
            raise ValueError("No valid binary values found in response")
            
        print(f"Successfully parsed {len(numbers)} binary values")
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {numbers_str}")
        # Return a small set of random values as fallback
        numbers = np.random.randint(0, 2, size=batch_size).tolist()
        print(f"Using fallback random values instead")
    
    return numbers

def run_binomial_test(numbers: List[int]) -> Tuple[float, bool]:
    """
    Run binomial test to check if the distribution of 0s and 1s is random
    Returns: (p_value, is_random)
    """
    # Count the number of 1s
    count_ones = sum(numbers)
    n = len(numbers)
    
    # Expected probability for a fair coin is 0.5
    expected_p = 0.5
    
    # Binomial test - using the correct function name for newer scipy versions
    result = stats.binomtest(count_ones, n, expected_p)
    p_value = result.pvalue
    is_random = p_value > 0.05  # Using 5% significance level
    
    return p_value, is_random

def plot_binary_distribution(numbers: List[int], batch_number: int, p_value: float, model_name="gpt-4o"):
    """
    Create a bar chart of the binary distribution and save it
    """
    plt.figure(figsize=(10, 6))
    
    # Count occurrences
    count_zeros = numbers.count(0)
    count_ones = numbers.count(1)
    total = len(numbers)
    
    # Create bar chart
    plt.bar(['0', '1'], [count_zeros/total, count_ones/total], color=['blue', 'green'], alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Expected (p=0.5)')
    
    plt.title(f'Batch {batch_number} Distribution - {model_name} (p-value: {p_value:.4f})')
    plt.xlabel('Value')
    plt.ylabel('Proportion')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{model_name}_batch_{batch_number}_distribution.png')
    plt.close()

def main():
    # Set the model name and provider
    provider = "gemini"  # Change to "openai" for OpenAI models or "gemini" for Gemini models
    
    # Choose model based on provider
    if provider == "openai":
        model_name = "gpt-4o"
    elif provider == "anthropic":  # anthropic
        model_name = "claude-3-5-haiku-20241022"  # or any other Claude model
    else:  # gemini
        model_name = "gemma-3-27b-it"  # or other available Gemini models
    
    results = []
    batch_number = 1
    max_batches = 30  # Maximum number of batches to test
    
    # Initialize conversation history based on provider
    if provider == "openai":
        conversation_history = [
            {"role": "system", "content": "You are a helpful assistant that generates random numbers."}
        ]
    elif provider == "anthropic":  # anthropic
        conversation_history = [
            {"role": "assistant", "content": "I'm a helpful assistant that generates random numbers."}
        ]
    else:  # gemini
        conversation_history = [
            {"role": "user", "content": "You are a helpful assistant that generates random numbers."},
            {"role": "model", "content": "I'm a helpful assistant that generates random numbers."}
        ]
    
    # Keep track of all numbers across batches for global test
    all_numbers = []
    
    # Add metadata to results
    metadata = {
        "provider": provider,
        "model": model_name,
        "test_date": str(np.datetime64('now')),
        "batches": []
    }
    
    # Track when we detect non-randomness
    stopped_early = False
    stopping_reason = ""
    
    while batch_number <= max_batches:
        print(f"\nTesting batch {batch_number} with {provider} model {model_name}...")
        
        # Get random binary values from LLM with conversation history
        is_first_batch = (batch_number == 1)
        numbers, conversation_history = get_random_binary_from_llm(
            conversation_history=conversation_history,
            is_first_batch=is_first_batch,
            model_name=model_name,
            provider=provider
        )
        
        # Add to global collection
        all_numbers.extend(numbers)
        
        # Run binomial test on current batch
        p_value, is_random = run_binomial_test(numbers)
        
        # Run global binomial test on all numbers so far
        global_p_value, global_is_random = run_binomial_test(all_numbers)
        
        # Plot the distribution
        plot_binary_distribution(numbers, batch_number, p_value, model_name)
        
        # Store results - convert all values to JSON-serializable types
        batch_result = {
            "batch_number": int(batch_number),
            "p_value": float(p_value),
            "is_random": bool(is_random),
            "count_ones": int(sum(numbers)),
            "count_zeros": int(len(numbers) - sum(numbers)),
            "global_p_value": float(global_p_value),
            "global_is_random": bool(global_is_random),
            "global_count_ones": int(sum(all_numbers)),
            "global_count_zeros": int(len(all_numbers) - sum(all_numbers)),
            "numbers": [int(n) for n in numbers]
        }
        metadata["batches"].append(batch_result)
        results.append(batch_result)
        
        print(f"Batch P-value: {p_value:.4f}")
        print(f"Batch follows random binomial distribution: {is_random}")
        print(f"Global P-value (all batches): {global_p_value:.4f}")
        print(f"Global follows random binomial distribution: {global_is_random}")
        print(f"Global counts: {sum(all_numbers)} ones, {len(all_numbers) - sum(all_numbers)} zeros")
        
        # Check stopping conditions
        if not is_random:
            print("\nFound non-random distribution in current batch! Stopping test.")
            stopped_early = True
            stopping_reason = f"Batch {batch_number} showed non-random distribution (p-value: {p_value:.4f})"
            break
        
        # Check if global p-value is below threshold
        if not global_is_random:
            print("\nGlobal p-value below threshold (0.05)! Stopping test.")
            stopped_early = True
            stopping_reason = f"Global p-value fell below threshold after batch {batch_number} (p-value: {global_p_value:.4f})"
            break
            
        batch_number += 1
    
    # Create a global distribution plot
    plt.figure(figsize=(10, 6))
    count_zeros = all_numbers.count(0)
    count_ones = all_numbers.count(1)
    total = len(all_numbers)
    
    plt.bar(['0', '1'], [count_zeros/total, count_ones/total], color=['blue', 'green'], alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Expected (p=0.5)')
    
    plt.title(f'Global Distribution - {model_name} (p-value: {global_p_value:.4f})')
    plt.xlabel('Value')
    plt.ylabel('Proportion')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{model_name}_global_distribution.png')
    plt.close()
    
    # Add final global stats to metadata
    metadata["global_stats"] = {
        "total_samples": len(all_numbers),
        "count_ones": sum(all_numbers),
        "count_zeros": len(all_numbers) - sum(all_numbers),
        "proportion_ones": sum(all_numbers)/len(all_numbers),
        "p_value": float(global_p_value),
        "is_random": bool(global_is_random),
        "stopped_early": stopped_early,
        "stopping_reason": stopping_reason,
        "total_batches": batch_number
    }
    
    # Save all results to a JSON file
    with open(f'{provider}_{model_name}_binomial_test_results.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Total batches tested: {batch_number}")
    if stopped_early:
        print(f"Test stopped early: {stopping_reason}")
    print(f"Last batch p-value: {results[-1]['p_value']:.4f}")
    print(f"Global p-value: {global_p_value:.4f}")
    print(f"Global counts: {sum(all_numbers)} ones, {len(all_numbers) - sum(all_numbers)} zeros")
    print(f"Global proportion of ones: {sum(all_numbers)/len(all_numbers):.4f}")
    print(f"Results saved to '{provider}_{model_name}_binomial_test_results.json'")
    print(f"Distribution plots saved as '{model_name}_batch_X_distribution.png' and '{model_name}_global_distribution.png'")
    
    # Save the conversation history
    with open(f'{provider}_{model_name}_conversation_history.json', 'w') as f:
        json.dump(conversation_history, f, indent=2)
    print(f"Conversation history saved to '{provider}_{model_name}_conversation_history.json'")

if __name__ == "__main__":
    main()