#!/usr/bin/env python3

import os
# Set environment variables before importing numpy
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# Set C compiler environment variable
os.environ['CC'] = 'gcc'  # or 'clang' depending on your system

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config file: {e}")

def load_prompts(prompts_path: str) -> List[Dict]:
    """Load prompts from JSON file."""
    try:
        with open(prompts_path, 'r') as f:
            data = json.load(f)
            if 'prompts' not in data:
                raise ValueError("JSON file must contain a 'prompts' key")
            return data['prompts']
    except Exception as e:
        raise RuntimeError(f"Failed to load prompts file: {e}")

def calculate_entropy(logits: np.ndarray) -> float:
    """Calculate entropy from logits."""
    try:
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return -np.sum(probs * np.log(probs))
    except Exception:
        return 0.0  # Return 0 if calculation fails

def escape_regex(text: str) -> str:
    """Escape special regex characters in a string."""
    return re.escape(text)

def extract_answer(text: str, answer_tag: str) -> Optional[str]:
    """Extract answer from text using XML tags."""
    try:
        pattern = f"{escape_regex(answer_tag)}(.*?)</{escape_regex(answer_tag[1:])}"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    except Exception:
        return None

def score_exact_match(
    answer: str,
    canonical: str,
    case_sensitive: bool,
    allow_partial: bool
) -> bool:
    """Score exact match answers."""
    try:
        if not case_sensitive:
            answer = answer.lower()
            canonical = canonical.lower()
        
        if allow_partial:
            return canonical in answer
        return answer == canonical
    except Exception:
        return False

def score_numeric(
    answer: str,
    canonical: Union[int, float],
    tolerance: float,
    allow_scientific: bool
) -> bool:
    """Score numeric answers."""
    try:
        # Clean the answer string
        answer = answer.strip()
        
        # Handle scientific notation
        if not allow_scientific:
            # Remove scientific notation indicators
            answer = re.sub(r'[eE][+-]?\d+', '', answer)
        
        # Try to parse the answer as a number
        parsed = float(answer)
        
        # Handle integer comparison if canonical is an integer
        if isinstance(canonical, int):
            parsed = int(parsed)
        
        return abs(parsed - canonical) <= tolerance
    except (ValueError, TypeError):
        return False

def score_list(
    answer: str,
    canonical: List[Any],
    order_matters: bool,
    allow_partial: bool,
    separators: List[str]
) -> bool:
    """Score list answers."""
    try:
        # Try to split the answer using different separators
        for sep in separators:
            try:
                # Split and clean each item
                items = [item.strip() for item in answer.split(sep)]
                items = [item for item in items if item]  # Remove empty items
                
                # Try to convert items to numbers if canonical is numeric
                if all(isinstance(x, (int, float)) for x in canonical):
                    items = [float(item) for item in items]
                
                if order_matters:
                    if allow_partial:
                        return all(item in canonical for item in items)
                    return items == canonical
                else:
                    if allow_partial:
                        return all(item in canonical for item in items)
                    return set(items) == set(canonical)
            except (ValueError, TypeError):
                continue
        
        return False
    except Exception:
        return False

def check_format_correctness(text: str, answer_tag: str) -> bool:
    """Check if the response contains properly formatted answer tags."""
    try:
        return f"{answer_tag}" in text and f"</{answer_tag[1:]}" in text
    except Exception:
        return False

def check_content_correctness(
    text: str,
    scoring_config: Dict,
    answer_tag: str
) -> bool:
    """Check if the response matches the scoring criteria."""
    try:
        answer = extract_answer(text, answer_tag)
        if answer is None:
            return False
        
        scoring_type = scoring_config['type']
        canonical = scoring_config['canonical_answer']
        
        if scoring_type == 'exact_match':
            return score_exact_match(
                answer,
                canonical,
                scoring_config.get('case_sensitive', False),
                scoring_config.get('allow_partial', False)
            )
        elif scoring_type == 'numeric':
            return score_numeric(
                answer,
                canonical,
                scoring_config.get('tolerance', 0),
                scoring_config.get('allow_scientific_notation', True)
            )
        elif scoring_type == 'list':
            return score_list(
                answer,
                canonical,
                scoring_config.get('order_matters', False),
                scoring_config.get('allow_partial', False),
                scoring_config.get('separators', [',', ';', 'and', '&'])
            )
        else:
            raise ValueError(f"Unknown scoring type: {scoring_type}")
    except Exception:
        return False

def process_completion(
    completion: Dict,
    answer_tag: str,
    scoring_config: Dict
) -> Dict:
    """Process a single completion and extract relevant information."""
    try:
        # Extract the actual completion text from the RequestOutput object
        if hasattr(completion, 'outputs') and completion.outputs:
            text = completion.outputs[0].text
        else:
            text = str(completion)
        
        # For now, we'll just store the text without detailed token information
        # since VLLM doesn't provide it by default
        tokens = [{
            'text': text,
            'start': 0,
            'end': len(text)
        }]
        
        # Check correctness
        format_correct = check_format_correctness(text, answer_tag)
        content_correct = check_content_correctness(text, scoring_config, answer_tag)
        
        return {
            'text': text,
            'tokens': tokens,
            'format_correct': format_correct,
            'content_correct': content_correct
        }
    except Exception as e:
        print(f"Error processing completion: {e}")
        return {
            'text': str(completion),
            'tokens': [],
            'format_correct': False,
            'content_correct': False
        }

def generate_completions(
    model: LLM,
    prompt: str,
    sampling_params: SamplingParams,
    num_completions: int
) -> List[Dict]:
    """Generate multiple completions for a single prompt."""
    try:
        outputs = []
        for _ in range(num_completions):
            output = model.generate(prompt, sampling_params)
            outputs.extend(output)
        return outputs
    except Exception as e:
        print(f"Error generating completions: {e}")
        return []

def save_results(
    results: Dict,
    output_dir: Path,
    prompt_id: str,
    save_individual: bool
) -> None:
    """Save results to file(s)."""
    try:
        if save_individual:
            prompt_file = output_dir / f"prompt_{prompt_id}.json"
            with open(prompt_file, 'w') as f:
                json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {e}")

def create_sampling_params(strategy: Dict) -> SamplingParams:
    """Create sampling parameters from strategy configuration."""
    params = {
        'temperature': strategy['temperature'],
        'top_p': strategy['top_p'],
        'max_tokens': strategy['max_tokens']
    }
    
    # Add optional parameters if they exist
    if 'min_p' in strategy:
        params['min_p'] = strategy['min_p']
    
    return SamplingParams(**params)

def validate_model_path(model_path: str) -> bool:
    """Validate that the model path exists and contains necessary files."""
    path = Path(model_path)
    if not path.exists():
        return False
    
    # Check for common model files
    required_files = ['config.json', 'pytorch_model.bin']
    return all((path / file).exists() for file in required_files)

def main():
    parser = argparse.ArgumentParser(description='Generate completions using VLLM')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--prompts', required=True, help='Path to prompts JSON file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--num-completions', type=int, default=5, help='Number of completions per prompt')
    args = parser.parse_args()

    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config(args.config)
        
        # Initialize model
        model_path = config['model']['name']
        print(f"Initializing model from: {model_path}")
        
        # Check if it's a local path
        if os.path.exists(model_path):
            print("Using local model files")
            if not validate_model_path(model_path):
                print("Warning: Local model path may be incomplete. Required files:")
                print("  - config.json")
                print("  - pytorch_model.bin")
        else:
            print("Model not found locally, will download from HuggingFace")
        
        print("This may take a few minutes for large models...")
        start_time = time.time()
        try:
            model = LLM(
                model=model_path,
                dtype=config['model']['dtype'],
                trust_remote_code=config['model']['trust_remote_code']
            )
            load_time = time.time() - start_time
            print(f"Model loaded successfully in {load_time:.2f} seconds")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")
        
        # Load prompts
        print("Loading prompts...")
        prompts = load_prompts(args.prompts)
        print(f"Loaded {len(prompts)} prompts")
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'prompts').mkdir(exist_ok=True)
        
        # Initialize results storage
        all_results = []
        
        # Process each prompt
        print("\nGenerating completions...")
        for prompt in tqdm(prompts, desc="Processing prompts"):
            prompt_results = {
                'prompt': {
                    'id': prompt['id'],
                    'text': prompt['text'],
                    'scoring': prompt['scoring'],
                    'tokens': []  # TODO: Add tokenization of prompt
                },
                'completions': []
            }
            
            # Generate completions for each sampling strategy
            for strategy in config['sampling_strategies']:
                sampling_params = create_sampling_params(strategy)
                
                completions = generate_completions(
                    model,
                    prompt['text'],
                    sampling_params,
                    args.num_completions
                )
                
                for completion in completions:
                    processed = process_completion(
                        completion,
                        config['scoring']['format']['answer_tag'],
                        prompt['scoring']
                    )
                    
                    completion_result = {
                        'model': config['model']['name'],
                        'sampling_args': strategy,
                        **processed
                    }
                    
                    prompt_results['completions'].append(completion_result)
            
            # Save individual results
            if config['output']['save_individual']:
                save_results(
                    prompt_results,
                    output_dir / 'prompts',
                    prompt['id'],
                    True
                )
            
            all_results.append(prompt_results)
        
        # Save consolidated results
        if config['output']['save_consolidated']:
            print("\nSaving consolidated results...")
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'model': config['model']['name'],
                'num_prompts': len(prompts),
                'num_completions_per_prompt': args.num_completions,
                'sampling_strategies': config['sampling_strategies']
            }
            
            consolidated = {
                'metadata': metadata,
                'results': all_results
            }
            
            with open(output_dir / 'all_results.json', 'w') as f:
                json.dump(consolidated, f, indent=2)
            print("Done!")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == '__main__':
    main() 