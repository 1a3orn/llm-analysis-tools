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

3. Run the generation script:
```bash
python generate.py --config config.yaml --prompts prompts.txt --output output/
```

## Configuration

The `config.yaml` file allows you to specify:
- Model configuration
- Sampling strategies
- Output settings
- Scoring criteria

## Output Format

Results are stored in both individual JSON files per prompt and a consolidated file. Each completion includes:
- Generated text
- Token-level information including entropy
- Scoring results (format and content correctness)
- Sampling parameters used 