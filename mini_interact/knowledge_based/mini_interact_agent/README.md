# Mini-Interact Agentic Mode Evaluation

A lightweight agentic interaction system for natural language database queries using SQLite, focusing on ambiguity resolution through dynamic interaction with user simulator and database environment.

## Unique Features compared to [Bird-Interact](../../../bird_interact_agent/README.md)

- **DB Backend**: Use SQLite as the database backend instead of PostgreSQL. No need to setup the docker. Speed up the evaluation. The user simulator is also adapted to SQLite (sql parser and prompt).
- **Ambiguous User Query Task**: Focusing on the 1st subtask, i.e. ambiguous user query, without the follow-up query.
- **Query Type**: Focusing on the "Query" category without "Management" category. Thus, the DB cannot be modified, which facilitates multiple parallel evaluation experiments.


## Mini-Interact Agentic Evaluation Setting
### Task Setting
A task consists of only one phase: an **ambiguous user query**. All these queries require the understanding of the user real intent and external knowledge. To complete these queries, agents could interact with two environments: a **Database Environment** and a **User Simulator Environment**. Each interaction is constrained by a **budget** measured in virtual *bird-coin*s <img src="materials/bird-coin.png" style="height: 1em; vertical-align: middle;">, with different actions incurring varying *bird-coin*s <img src="materials/bird-coin.png" style="height: 1em; vertical-align: middle;">. 

### Metrics:

**Success rate (SR)**: The percentage of tasks that are solved successfully.
**Reward**: In mini-interact, there is no follow-up query, so we set the reward for the 1st subtask to 1.0. That means reward = SR.

### More details about Action Space, Budget Calculation, etc

Could refer to the [Bird-Interact Agentic Mode](../../../bird_interact_agent/README.md) for more details about Action Space, Budget Calculation, etc.


## Setup and Usage

### 1. Prerequisites

```bash
conda create -n bird_interact python=3.10 -y
conda activate bird_interact
cd BIRD-Interact/env
pip install -r requirements.txt
```

### 2. Data Preparation

Download the mini-interact dataset containing both the DBs, metafiles and the task file. 

### 3. API Configuration

You need to set up the model name and API key in the `code/config.py` file:

```python
# Example configuration
API_KEY = "your-api-key-here"
MODEL_NAME = "gpt-4o-2024-08-06"  # or other supported models
```

### 4. Running Experiments

1. **Run the experiment:**

   ```bash
   bash run_experiment.sh
   ```

2. **Run the batch experiments:**

   ```bash
   bash run_batch_mini_interact_experiments.sh --data_path=/path/to/mini-interact-dataset/bird_interact_data.jsonl
   ```

   You could also set your agent/user simulator model (`--agent_model` and `--user_model`), user patience budget (`--budgets`), etc. Details can be found in the [run_batch_mini_interact_experiments.sh](run_batch_mini_interact_experiments.sh) file.

3. **Output:**
   Results will be saved in `outputs/batch_runs/` directory.
