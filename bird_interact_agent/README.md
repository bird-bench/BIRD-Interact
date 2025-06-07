# Bird-Interact Agentic Evaluation 

A repository for running and evaluating BIRD-Interact Agent.

## Repository Structure

```
.
├── batch_run_bird_interact/    # Code for batch-running experiments (primary results)
├── src/                        # Core source code for single-run experiments
├── experiments/                # Main code for single-run experiments
├── data/                       # Dataset storage
├── outputs/                    # Experiment results: single-run and batch-run results
├── postgre_table_dumps/        # PostgreSQL database dumps
├── Dockerfile.postgresql       # PostgreSQL container configuration
├── Dockerfile.so_eval         # Evaluation environment configuration
├── docker-compose.yml         # Container orchestration
├── requirements.txt           # Python dependencies
├── run_batch_experiments.sh   # Batch experiment runner
└── run_experiment.sh          # Single experiment runner
```

## Quick Start

### 1. Data Preparation

```bash
cd data
git clone https://huggingface.co/datasets/birdsql/bird-interact-lite
# Combine with GT fields (contact us for access) into bird_interact_data.jsonl
```

### 2. Environment Setup

1. Download the database dumps:
   - Get from: [Google Drive](https://drive.google.com/file/d/1KABce6czIqL9kMyIX7i-_A0CIQoDnmyW/view)
   - Move to working directory and rename to `postgre_table_dumps`

2. Build and run Docker containers:
   ```bash
   docker compose up --build
   ```
   This launches two containers:
   - PostgreSQL database
   - Evaluation environment (so_eval_env)

### 3. API Configuration

#### VertexAI Setup

> Current user simulator is based on gemini-2.0-flash-001, which is called by VertexAI. If you are new customer to google cloud, you will get [$300 in free credits](https://cloud.google.com/vertex-ai?hl=en), and could use it to call vertex API.

If you want to use VertexAI, you should configure in `src/llm_utils/call_api_batch.py` and `src/llm_utils/vertex_ai_simple.py`:
```python
GCP_PROJECT = "Your GCP Project ID"
GCP_REGION = "us-central1"
GCP_CREDENTIALS_PATH = "Your GCP Credentials Path"
```
> If you find it hard to configure this, you could also try other API providers to use gemini-2.0-flash-001, or use other models.

#### OpenAI/Third-party API Setup
Configure in `src/llm_utils/config.py`:
- Set `base_url`
- Set `api_key`

## Running Experiments

### Single Sample Mode
Single sample mode (`src/` and `experiments/`) is useful for debugging and workflow understanding
```bash
bash run_experiment.sh
```
Output directory: `outputs/single_runs/`

### Batch Mode
Batch mode (`batch_run_bird_interact/`) is recommended for production runs
```bash
bash run_batch_experiments.sh
```
Output directory: `outputs/batch_runs/`
Default user patience is set to 6.

