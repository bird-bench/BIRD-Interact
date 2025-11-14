# Mini-Interact Conversational Mode

A lightweight conversational interaction system for natural language database queries using SQLite, focusing on ambiguity resolution through multi-turn dialogue.

## Overview

**What is it?**  
A simplified conversational database interaction system that evaluates how well a system (S) can communicate with a user simulator (U) to resolve ambiguities and generate correct SQL queries for SQLite databases.

---

#### Step-by-step flow

1. **Ambiguous task**  
   *U* starts with a task containing several ambiguities.

2. **Clarification round**  
   *S* can ask clarifying questions until it thinks it understands, then sends a SQL query.  
   • If the query is wrong, *S* can have one more chance (a "debugging") to fix it. (Optional)
   • If attempts fail, the test stops.

3. **Turn limit**  
   The back-and-forth in the clarification round is limited by:  
   • The number of ambiguity points marked in the task, plus  
   • A small extra "user patience" allowance (default: 3 extra turns).

---

#### How scoring works

For a batch of tasks we count how many times *S* succeeds:

| Situation | What counts as a win? | Reward added per win |
|-----------|----------------------|----------------------|
| First try | Solved without using the debugging | **1.0** |
| After debugging | Solved on the second try | **0.5** |

Add up the points and divide by the number of tasks.  
The closer the final number is to **1**, the better the system:

* High points = solves quickly with few mistakes.  
* Low points = needs debugging or can't solve at all.


## Directory Structure

```
mini_interact_conv/
├── code/
│   ├── sql_parser_sqlite.py         # SQL query parsing utilities for SQLite
│   ├── wrap_up_sql_sqlite.py        # SQL query formatting and validation
│   ├── infer_api_system_sqlite.py   # System-side API inference
│   ├── infer_api_user_1_sqlite.py   # User simulation (step 1: parsing)
│   ├── infer_api_user_2_sqlite.py   # User simulation (step 2: response)
│   ├── collect_response.py          # Response collection utilities
│   ├── call_api.py                  # API calling interface
│   └── config.py                    # Configuration file for API keys
├── pipeline/
│   └── run_Mini_Interact.sh         # Pipeline execution script
├── prompts/
│   └── prompts.py                   # Conversation prompts and templates
├── evaluation/
│   ├── db_utils.py                  # SQLite database utilities
│   ├── single_instance_eval_sqlite.py  # Single instance evaluation
│   ├── wrapper_evaluation_sqlite.py    # Evaluation wrapper
│   └── run_eval.sh                  # Evaluation script (optional)
├── data/
│   ├── DBs ...      # SQLite databases
│   └── mini_interact.jsonl     # Mini-Interact dataset
└── results/                         # Output directory for results
```

## Setup and Usage

### 1. Prerequisites

```bash
conda create -n bird_interact python=3.10 -y
conda activate bird_interact
cd BIRD-Interact/env
pip install -r requirements.txt
```

### 2. Data Preparation

Please download the SQLite databases in the `data/` directory and the mini_interact data in `data/mini_interact.jsonl`. Please download [HERE](https://huggingface.co/datasets/birdsql/mini-interact).

### 3. API Configuration

You need to set up the model name and API key in the `mini_interact/knowledge_based/mini_interact_conv/code/config.py` file:

```python
# Example configuration
API_KEY = "your-api-key-here"
MODEL_NAME = "gpt-4o-2024-08-06"  # or other supported models
```


## Running Experiments

1. **Configure the pipeline script:**
   ```bash
   cd mini_interact/knowledge_based/mini_interact_conv/pipeline
   # The virtual environment is the same with `mini_interact_agent`.
   # Edit run_Mini_Interact.sh:
   # - Line 12: Set project_root to your project directory (e.g., "/Users/yourname/mini_interact_conv")
   # - Line 9: Adjust patience parameter (default: 3)
   # - Line 10-11: Set user simulator and system model names
   ```

2. **Run the conversation pipeline:**
   ```bash
   bash run_Mini_Interact.sh
   ```

3. **Output:**
   Results will be saved in `mini_interact/knowledge_based/mini_interact_conv/results/patience_{patience}/{model_name}/`

### Key Parameters

- **patience**: Extra turns allowed beyond ambiguity points (default: 3)
- **US_model_name**: Model for user simulator (e.g., "claude-4-5-haiku")
- **system_model_name**: Model for the evaluated system (e.g., "qwen3-coder")
- **project_root**: Absolute path to the project directory

## Components

### Code Module
- **sql_parser_sqlite.py**: Parse SQL queries into snippets for our proposed user simulater (Used in LLM as parser below)
- **wrap_up_sql_sqlite.py**: Formats and finalizes SQL queries
- **infer_api_system_sqlite.py**: Manages system-side conversation logic
- **infer_api_user_1_sqlite.py**: User simulator step 1 (LLM as parser)
- **infer_api_user_2_sqlite.py**: User simulator step 2 (response generation)
- **collect_response.py**: Utilities for gathering and processing responses
- **call_api.py**: Interface for API interactions
- **config.py**: Configuration file for API keys and model settings

### Pipeline
The `run_Mini_Interact.sh` script orchestrates the conversation flow through multiple turns:
- Ambiguity resolution phase with iterative clarification
- System response generation
- SQL query execution on SQLite databases
- Response collection and evaluation

### Prompts
The `prompts.py` file contains templates and prompts for:
- User query generation
- System response formatting
- SQL query construction
- Error handling and clarification
- You are welcome to use your own prompts when meeting the task settings.

### Results
Results from conversation runs are stored in the `results/` directory, organized by:
- Patience parameter
- Model name
- Interaction type (system/user)



