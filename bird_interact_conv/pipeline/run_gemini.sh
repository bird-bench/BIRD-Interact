#!/usr/bin/env bash
set -e

#####################################
# Basic Configurations
#####################################

# Parameters:
patience=3
US_model_name="gemini-2.0-flash-001"
system_model_name="gemini-2.0-flash-001"
project_root="YOUR-ROOT"
# ===========================================: Phase 1 (Ambiguity Resolution) :===========================================
# Phase 1: Ambiguity Resolution
## Turn 1
### parameters setting
turn_num=1
result_dir="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/"
mkdir -p "$result_dir"
result_path_prompt="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_prompt.jsonl"
DB_schema_path="${project_root}/bird_interact_conv/data/bird-interact-lite/[[DB_name]]/[[DB_name]]_schema.txt"
external_kg_path="${project_root}/bird_interact_conv/data/bird-interact-lite/[[DB_name]]/[[DB_name]]_kb.jsonl"
FILE_PATH="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
if [ -f "$FILE_PATH" ]; then
    data_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
else
    data_path="${project_root}/bird_interact_conv/data/bird-interact-lite/bird_interact_data.jsonl"
fi
user_resp_path="${project_root}/bird_interact_conv/data/bird-interact-lite/bird_interact_data.jsonl"
### System: Prompt Generation + Infer API + Response Collection
python ${project_root}/bird_interact_conv/code/infer_api_system.py --prompt_path ${data_path} --result_path ${result_path_prompt} --user_resp_path ${user_resp_path} --DB_schema_path ${DB_schema_path} --external_kg_path ${external_kg_path} --patience ${patience} --turn_num ${turn_num}
wait 

if [ ! -s "$result_path_prompt" ]; then
    echo "Empty File!"
else
    result_path_response="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_response.jsonl"
    python ${project_root}/bird_interact_conv/code/call_api.py --model_name ${system_model_name} --prompt_path ${result_path_prompt} --output_path ${result_path_response}
    wait

    result_path_selected_llm="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
    python ${project_root}/bird_interact_conv/code/collect_response.py --source_path ${user_resp_path} --response_path ${result_path_response} --result_path ${result_path_selected_llm}
    wait
fi

### User Simulator Step 1 (LLM as Parser): Prompt Generation + Infer API + Response Collection
FILE_PATH="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_1_interaction.jsonl"
if [ -f "$FILE_PATH" ]; then
    user_1_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_1_interaction.jsonl"
else
    user_1_path="${project_root}/bird_interact_conv/data/bird-interact-lite/bird_interact_data.jsonl"
fi
result_path_prompt="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_1_interaction_prompt.jsonl"
sys_resp_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
python ${project_root}/bird_interact_conv/code/infer_api_user_1.py --prompt_path ${user_1_path} --result_path ${result_path_prompt} --sys_resp_path ${sys_resp_path} --DB_schema_path ${DB_schema_path} --turn_num ${turn_num}
wait
if [ ! -s "$result_path_prompt" ]; then
    echo "Empty File!"
else
    result_path_response="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_1_interaction_response.jsonl"
    python ${project_root}/bird_interact_conv/code/call_api.py --model_name ${US_model_name} --prompt_path ${result_path_prompt} --output_path ${result_path_response}
    wait

    result_path_selected_llm="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_1_interaction.jsonl"
    python ${project_root}/bird_interact_conv/code/collect_response.py --source_path ${user_resp_path} --response_path ${result_path_response} --result_path ${result_path_selected_llm}
    wait
fi
### User Simulator Step 2 (LLM as Generator): Prompt Generation + Infer API + Response Collection
FILE_PATH="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_2_interaction.jsonl"
if [ -f "$FILE_PATH" ]; then
    user_2_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_2_interaction.jsonl"
else
    user_2_path="${project_root}/bird_interact_conv/data/bird-interact-lite/bird_interact_data.jsonl"
fi
result_path_prompt="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_2_interaction_prompt.jsonl"
user_1_resp_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_1_interaction.jsonl"
python ${project_root}/bird_interact_conv/code/infer_api_user_2.py --prompt_path ${user_2_path} --result_path ${result_path_prompt} --sys_resp_path ${sys_resp_path} --user_1_resp_path ${user_1_resp_path} --DB_schema_path ${DB_schema_path} --turn_num ${turn_num}
wait
if [ ! -s "$result_path_prompt" ]; then
    echo "Empty File!"
else
    result_path_response="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_2_interaction_response.jsonl"
    python ${project_root}/bird_interact_conv/code/call_api.py --model_name ${US_model_name} --prompt_path ${result_path_prompt} --output_path ${result_path_response}
    wait

    result_path_selected_llm="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_2_interaction.jsonl"
    python ${project_root}/bird_interact_conv/code/collect_response.py --source_path ${user_resp_path} --response_path ${result_path_response} --result_path ${result_path_selected_llm}
    wait
fi

## Remaining Turns 
jsonl_file="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
max_turn=$(awk '
    index($0, "\"max_turn\"") {
        match($0, /"max_turn"[ \t]*:[ \t]*[0-9]+/)
        val = substr($0, RSTART, RLENGTH)
        gsub(/[^0-9]/, "", val)
        if (val+0 > max) max = val+0
    }
    END { print max }
' "$jsonl_file")

for ((i=2; i<max_turn; i++)); do
    turn_num=${i}
    result_path_prompt="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_prompt.jsonl"
    DB_schema_path="${project_root}/bird_interact_conv/data/bird-interact-lite/[[DB_name]]/[[DB_name]]_schema.txt"
    external_kg_path="${project_root}/bird_interact_conv/data/bird-interact-lite/[[DB_name]]/[[DB_name]]_kb.jsonl"
    data_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
    user_resp_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_2_interaction.jsonl"
    user_1_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_1_interaction.jsonl"

    ### System: Prompt Generation + Infer API + Response Collection
    python ${project_root}/bird_interact_conv/code/infer_api_system.py --prompt_path ${data_path} --result_path ${result_path_prompt} --user_resp_path ${user_resp_path} --DB_schema_path ${DB_schema_path} --external_kg_path ${external_kg_path} --patience ${patience} --turn_num ${turn_num}
    wait 
    if [ ! -s "$result_path_prompt" ]; then
        echo "Empty File!"
    else
        result_path_response="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_response.jsonl"
        python ${project_root}/bird_interact_conv/code/call_api.py --model_name ${system_model_name} --prompt_path ${result_path_prompt} --output_path ${result_path_response}
        wait

        result_path_selected_llm="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
        python ${project_root}/bird_interact_conv/code/collect_response.py --source_path ${result_path_selected_llm} --response_path ${result_path_response} --result_path ${result_path_selected_llm}
        wait
    fi
    ### User Simulator Step 1 (LLM as Parser): Prompt Generation + Infer API + Response Collection
    result_path_prompt="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_1_interaction_prompt.jsonl"
    sys_resp_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
    python ${project_root}/bird_interact_conv/code/infer_api_user_1.py --prompt_path ${user_1_path} --result_path ${result_path_prompt} --sys_resp_path ${sys_resp_path} --DB_schema_path ${DB_schema_path} --turn_num ${turn_num}
    wait
    if [ ! -s "$result_path_prompt" ]; then
        echo "Empty File!"
    else
        result_path_response="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_1_interaction_response.jsonl"
        python ${project_root}/bird_interact_conv/code/call_api.py --model_name ${US_model_name} --prompt_path ${result_path_prompt} --output_path ${result_path_response}
        wait

        result_path_selected_llm="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_1_interaction.jsonl"
        python ${project_root}/bird_interact_conv/code/collect_response.py --source_path ${result_path_selected_llm} --response_path ${result_path_response} --result_path ${result_path_selected_llm}
        wait
    fi
    ### User Simulator Step 2 (LLM as Generator): Prompt Generation + Infer API + Response Collection
    result_path_prompt="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_2_interaction_prompt.jsonl"
    user_1_resp_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_1_interaction.jsonl"
    user_2_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_2_interaction.jsonl"
    python ${project_root}/bird_interact_conv/code/infer_api_user_2.py --prompt_path ${user_2_path} --result_path ${result_path_prompt} --sys_resp_path ${sys_resp_path} --user_1_resp_path ${user_1_resp_path} --DB_schema_path ${DB_schema_path} --turn_num ${turn_num}
    wait
    if [ ! -s "$result_path_prompt" ]; then
        echo "Empty File!"
    else
        result_path_response="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_2_interaction_response.jsonl"
        python ${project_root}/bird_interact_conv/code/call_api.py --model_name ${US_model_name} --prompt_path ${result_path_prompt} --output_path ${result_path_response}
        wait

        result_path_selected_llm="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_2_interaction.jsonl"
        python ${project_root}/bird_interact_conv/code/collect_response.py --source_path ${result_path_selected_llm} --response_path ${result_path_response} --result_path ${result_path_selected_llm}
        wait
    fi
done

## Final Turn: Gen SQL
turn_num=${max_turn}
result_path_prompt="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_prompt.jsonl"
DB_schema_path="${project_root}/bird_interact_conv/data/bird-interact-lite/[[DB_name]]/[[DB_name]]_schema.txt"
external_kg_path="${project_root}/bird_interact_conv/data/bird-interact-lite/[[DB_name]]/[[DB_name]]_kb.jsonl"
data_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
user_resp_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/user_2_interaction.jsonl"

### System (final turn): Prompt Generation + Infer API + Response Collection
python ${project_root}/bird_interact_conv/code/infer_api_system.py --prompt_path ${data_path} --result_path ${result_path_prompt} --user_resp_path ${user_resp_path} --DB_schema_path ${DB_schema_path} --external_kg_path ${external_kg_path} --patience ${patience} --turn_num ${turn_num}
wait 
if [ ! -s "$result_path_prompt" ]; then
    echo "Empty File!"
else
    result_path_response="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_response.jsonl"
    python ${project_root}/bird_interact_conv/code/call_api.py --model_name ${system_model_name} --prompt_path ${result_path_prompt} --output_path ${result_path_response}
    wait

    result_path_selected_llm="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
    python ${project_root}/bird_interact_conv/code/collect_response.py --source_path ${result_path_selected_llm} --response_path ${result_path_response} --result_path ${result_path_selected_llm}
    wait
fi

## SQL extract
data_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
result_path_sql="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/sql_results.jsonl"
python ${project_root}/bird_interact_conv/code/wrap_up_sql.py --data_path ${data_path} --result_path ${result_path_sql}
wait

## Eval
jsonl_file="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/sql_results.jsonl"
python ${project_root}/evaluation/src/eval_bird_interact_batch.py --jsonl "$jsonl_file"

# ===========================================: Phase 1 Debugging: One More Chance for Debugging ===========================================
### Parameters Setting
exec_report="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/sql_results_output_with_status.jsonl"
data_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
result_path_prompt="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_prompt.jsonl"
phase='debug'

### System: Prompt Generation + Infer API + Response Collection
python ${project_root}/bird_interact_conv/code/infer_api_system.py --prompt_path ${data_path} --result_path ${result_path_prompt} --user_resp_path ${exec_report} --DB_schema_path ${DB_schema_path} --external_kg_path ${external_kg_path} --patience ${patience} --turn_num ${turn_num} --phase ${phase}
wait 
if [ ! -s "$result_path_prompt" ]; then
    echo "Empty File!"
else
    result_path_response="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_response.jsonl"
    python ${project_root}/bird_interact_conv/code/call_api.py --model_name ${system_model_name} --prompt_path ${result_path_prompt} --output_path ${result_path_response}
    wait

    result_path_selected_llm="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
    python ${project_root}/bird_interact_conv/code/collect_response.py --source_path ${result_path_selected_llm} --response_path ${result_path_response} --result_path ${result_path_selected_llm}
    wait
fi

## SQL extract
data_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
result_path_sql="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/sql_results_debug.jsonl"
python ${project_root}/bird_interact_conv/code/wrap_up_sql.py --data_path ${data_path} --result_path ${result_path_sql}
wait

## Eval
jsonl_file="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/sql_results_debug.jsonl"
python ${project_root}/evaluation/src/eval_bird_interact_batch.py --jsonl "$jsonl_file"


# ===========================================: Phase 2 (Follow Up Question) :===========================================
# Phase 2: Follow Up question

### Parameters Setting
exec_report="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/sql_results_debug_output_with_status.jsonl"
data_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
result_path_prompt="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_prompt.jsonl"
phase='follow'

### System: Prompt Generation + Infer API + Response Collection
python ${project_root}/bird_interact_conv/code/infer_api_system.py --prompt_path ${data_path} --result_path ${result_path_prompt} --user_resp_path ${exec_report} --DB_schema_path ${DB_schema_path} --external_kg_path ${external_kg_path} --patience ${patience} --turn_num ${turn_num} --phase ${phase}
wait 
if [ ! -s "$result_path_prompt" ]; then
    echo "Empty File!"
else
    result_path_response="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_response.jsonl"
    python ${project_root}/bird_interact_conv/code/call_api.py --model_name ${system_model_name} --prompt_path ${result_path_prompt} --output_path ${result_path_response}
    wait

    result_path_selected_llm="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
    python ${project_root}/bird_interact_conv/code/collect_response.py --source_path ${result_path_selected_llm} --response_path ${result_path_response} --result_path ${result_path_selected_llm}
    wait
fi

## SQL extract
data_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
result_path_sql="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/sql_results_fu.jsonl"
follow_up_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_prompt.jsonl"
python ${project_root}/bird_interact_conv/code/wrap_up_sql.py --data_path ${data_path} --result_path ${result_path_sql} --follow_up_path ${follow_up_path}
wait

## Eval
jsonl_file="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/sql_results_fu.jsonl"
python ${project_root}/evaluation/src/eval_bird_interact_batch.py --jsonl "$jsonl_file"


# ===========================================: Phase 2 Debugging: One more chance for Follow Up question Debugging ===========================================
### Parameters Setting
exec_report="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/sql_results_fu_output_with_status.jsonl"
data_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
result_path_prompt="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_prompt.jsonl"
phase='debug'

### System: Prompt Generation + Infer API + Response Collection
python ${project_root}/bird_interact_conv/code/infer_api_system.py --prompt_path ${data_path} --result_path ${result_path_prompt} --user_resp_path ${exec_report} --DB_schema_path ${DB_schema_path} --external_kg_path ${external_kg_path} --patience ${patience} --turn_num ${turn_num} --phase ${phase}
wait 
if [ ! -s "$result_path_prompt" ]; then
    echo "Empty File!"
else
    result_path_response="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_response.jsonl"
    python ${project_root}/bird_interact_conv/code/call_api.py --model_name ${system_model_name} --prompt_path ${result_path_prompt} --output_path ${result_path_response}
    wait

    result_path_selected_llm="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
    python ${project_root}/bird_interact_conv/code/collect_response.py --source_path ${result_path_selected_llm} --response_path ${result_path_response} --result_path ${result_path_selected_llm}
    wait
fi

## SQL extract
data_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction.jsonl"
result_path_sql="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/sql_results_fu_debug.jsonl"
follow_up_path="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/system_interaction_prompt.jsonl"
python ${project_root}/bird_interact_conv/code/wrap_up_sql.py --data_path ${data_path} --result_path ${result_path_sql} --follow_up_path ${follow_up_path}
wait

## Eval
jsonl_file="${project_root}/bird_interact_conv/results/patience_${patience}/${system_model_name}/sql_results_fu_debug.jsonl"
python ${project_root}/evaluation/src/eval_bird_interact_batch.py --jsonl "$jsonl_file"
