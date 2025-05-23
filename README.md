# VLLM Data Generation Tool

A tool for generating and evaluating completions from language models using VLLM. This tool supports:
- Multiple completions per prompt
- Configurable sampling strategies
- Automated scoring with format and content validation
- Detailed token-level metrics including entropy
- Flexible output formats

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your sampling strategies in `config.yaml`

3. Create your prompts file in JSON format (see `sample_prompts.json` for example)

4. Run the generation script:
```bash
python generate.py --config config.yaml --prompts prompts.json --output output/
```

### Troubleshooting

If you encounter an error about MKL threading layer incompatibility, the script will automatically handle this by setting the appropriate environment variables. If you need to set these manually, you can do so before running the script:

```bash
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
python generate.py --config config.yaml --prompts prompts.json --output output/
```

## Configuration

The `config.yaml` file allows you to specify:
- Model configuration
- Sampling strategies
- Output settings
- Scoring strategy defaults

## Prompt Format

Prompts should be provided in a JSON file with the following structure:
```json
{
  "prompts": [
    {
      "id": "unique_id",
      "text": "Your prompt text here",
      "scoring": {
        "type": "exact_match|numeric|list",
        "canonical_answer": "expected answer",
        // Additional scoring parameters based on type
      }
    }
  ]
}
```

### Scoring Types

1. `exact_match`:
   - `canonical_answer`: Expected string answer
   - `case_sensitive`: Whether to match case (default: false)
   - `allow_partial`: Whether to allow partial matches (default: false)

2. `numeric`:
   - `canonical_answer`: Expected number
   - `tolerance`: Allowed difference from canonical answer
   - `allow_scientific_notation`: Whether to allow scientific notation

3. `list`:
   - `canonical_answer`: Expected list of items
   - `order_matters`: Whether order is important
   - `allow_partial`: Whether to allow partial matches
   - `separators`: List of possible separators

## Output Format

Results are stored in both individual JSON files per prompt and a consolidated file. Each completion includes:
- Generated text
- Token-level information
- Scoring results (format and content correctness)
- Sampling parameters used

The output structure is:
```
output/
  ├── metadata.json
  ├── all_results.json
  └── prompts/
      ├── prompt_id1.json
      ├── prompt_id2.json
      └── ...
``` 