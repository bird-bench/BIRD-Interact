# BIRD-INTERACT <img src="materials/hku-logo.jpg" alt="HKU Logo" width="50" style="vertical-align:middle;margin-left:10px;"> <img src="materials/google-cloud-logo.png" alt="Google Cloud Logo" width="50" style="vertical-align:middle;margin-left:10px;">

<p align="center">
  <img src="materials/bird_interact.png" 
       style="width: 30%; min-width: 100px; display: block; margin: auto; border-radius: 15px !important;">
</p>


<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <a href="https://creativecommons.org/licenses/by-sa/4.0/deed.en">
    <img src="https://img.shields.io/badge/License-CC%20By%20SA%204.0-orange.svg" alt="License">
  </a>
  <a href="https://bird-interact.github.io/">
    <img src="https://img.shields.io/badge/Leaderboard-2025-28a745.svg" alt="Leaderboard">
  </a>
  <a href="https://huggingface.co/datasets/birdsql/bird-interact-lite/tree/main">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-FFD21E.svg" alt="HuggingFace">
  </a>
  <a href="https://www.python.org/downloads/release/python-310/">
    <img src="https://img.shields.io/badge/Python-3.10+-teal.svg" alt="Python">
  </a>
  <a href="https://pypi.org/project/openai/">
    <img src="https://img.shields.io/badge/OpenAI-1.40+-beige.svg" alt="OpenAI">
  </a>
</div>

## 🧸 Overview

BIRD-INTERACT, an interactive text-to-SQL benchmark, **re-imagines Text-to-SQL evaluation via lens of dynamic interactions**.
The environment blends a hierarchical knowledge base, database documentation and a function-driven user simulator to recreate authentic enterprise environments across full **CRUD** operations.
It offers two rigorous test modes: (1) passive **Conversational Interaction** and (2) active **Agentic Interaction**, spanning 600 annotated tasks including Business Intelligence (BI), CRUD operations and etc., each guarded by executable test cases.
Typical evaluations trigger 1,968-5,496 interaction turns between model and user simulator, while state-of-the-art reasoning models currently solve only **≈24%** and **≈18%** of tasks, underscoring the benchmark's challenge.

<p align="center">
  <img src="materials/workflow.png" 
       style="width: 100%; min-width: 100px; display: block; margin: auto; ">
</p>

### ✅ Two Evaluation Modes

BIRD-INTERACT supports two evaluation modes as mentioned above:

   - **c-Interact**: Conversational Interaction which is a passive mode and the workflow is fixed. The code and detailed information can be found in `bird_interact_conv`.
   - **a-Interact**: Agentic Interaction which is an embodied active mode where the workflow is dynamic and led by models. The code and detailed information can be found in `bird_interact_agent`.


### 🐣 Lite Version

We are releasing a lite version of BIRD-INTERACT, `bird-interact-lite-exp`, which includes 270 high-quality real-world tasks specifically for PostgreSQL. This is a good starting point for quick experimentation. 

### 🦜 Full Version

The full version of BIRD-INTERACT, `bird-interact-full`, is a comprehensive benchmark that includes 600 tasks for PostgreSQL. It covers a wide range of SQL operations and user queries. The full version is coming soon.

## 📦 Dataset Details

### Dataset Description

- **Database:** The complete database can be download from [the Google Drive](https://drive.google.com/drive/folders/1nJReLrvZVVrnfgBYwwNEgYvLroPGbcPD?usp=sharing). Check the [Quick Eval](#quick-eval) section for more details.
- **data:** Each data instance contain the following main parts:
   - `selected_database`: The name of the database.  
   - `query`: The unambiguous user query.  
   - `amb_user_query`: The user query with injected ambiguities.
   - `user_query_ambiguity`: The ambiguities injected into the user query.
   - `non_critical_ambiguity`: The non-critical ambiguities like order, limit, etc.
   - `knowledge_ambiguity`: The ambiguities created by masked external knowledges. 
   - `sol_sql`: The ground truth SQL solution.  
   - `preprocess_sql`: SQL queries to run before executing the solution or prediction.  
   - `clean_up_sql`: SQL queries to run after the test cases to revert any changes made to the database.  
   - `test_cases`: A set of test cases to validate the predicted corrected SQL.
   - `follow_up`: The labeled follow up questions.
   - `external_knowledge`: The external knowledge related to the specific task.

- **evaluation:** The evaluation code is available in the [`./evaluation`](./evaluation) directory.
- **Curated by:** BIRD Team & Google Cloud
- **License:** [cc-by-sa-4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **HuggingFace Dataset Card:** [bird-interact-lite](https://huggingface.co/datasets/birdsql/bird-interact-lite)

### Dataset Uses

To avoid data leakage by auto-crawling, we do not include GT solution sqls and test cases along with data.
please email [bird.bench23@gmail.com](mailto:bird.bench23@gmail.com) or [bird.bench25@gmail.com](mailto:bird.bench25@gmail.com) for full set, which will be sent automatically.


### Use the Dataset from HuggingFace

You can download the dataset from HuggingFace using the following command:
```bash
from datasets import load_dataset
# Load the flash version of the dataset
dataset = load_dataset("birdsql/bird-interact-lite")
print(dataset["lite"][0])

# Load the full version of the dataset (coming soon)
dataset = load_dataset("birdsql/bird-interact-full")
print(dataset["full"][0])
```

Or you can use the provided script to download the full version of the dataset and split it into different dialects.
```bash
cd baseline/data
python pull_data.py \
  --schema_path path/to/full_schema.jsonl \
  --input_path path/to/input.jsonl \ # Path to the input JSONL file (may be empty if you want to download the dataset from HuggingFace)
  --output_folder path/to/output_dir # output folder of the split files
```

## 💨 Quick Eval

### Folder Structure
```ultree
.
├── LICENSE
├── README.md
├── bird_interact_conv
│   ├── ...
│   └── README.md
├── bird_interact_agent
│   ├── ...
│   └── README.md
├── evaluation
│   ├── docker-compose.yml
│   ├── env
│   ├── mssql_table_dumps
│   ├── mysql_table_dumps
│   ├── oracle_table_dumps
│   ├── postgre_table_dumps
│   ├── run
│   └── src
├── materials
│   ├── ...
└── requirements.txt
```

### Environment Setup
To run the baseline code you need to install the following dependencies:
```bash
conda create -n bird_interact python=3.10 -y
conda activate bird_interact
pip install -r requirements.txt
```

### Evaluation
We use **docker** to provide a consistent environment for running the benchmark. To set up the environment, follow these steps:

1. First download the PostgreSQL database from [the Google Drive](https://drive.google.com/file/d/1KABce6czIqL9kMyIX7i-_A0CIQoDnmyW/view).
2. Unzip the folder and save it in the [`./evaluation/postgre_table_dumps`](./evaluation/postgre_table_dumps).
3. Build the docker compose
```bash
cd evaluation
docker compose up --build
```
4. Interact with the database
You can use the `perform_query_on_postgresql_databases()` function in the `evaluation/src/postgresql_utils.py` file to interact with the each database. The function will return the result of the query.
5. Run the evaluation script inside the so_eval_env container
```bash
docker compose exec so_eval_env bash
cd run
bash run_eval.sh 
```

The output report file will be saved in the same folder as your input file. 
If you want the log file for each instance, you can set the `--logging` to `true` in the `run_eval.sh` script.

## 📋 Todo Lists

- [x] Release lite version, bird-interact-lite (270).
- [x] Release conversational version, bird-interact-conv.
- [x] Release agent version, bird-interact-agent.
- [ ] Release Full bird-interact-full (600).
- [ ] SFT / RL an User Simulator

## Created By:
BIRD Team & Google Cloud
