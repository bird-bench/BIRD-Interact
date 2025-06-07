#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os  
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse  
import json  
from sql_parser import segment_sql


def process_batch_data(data, batch_size):
    # Assuming process_batch_data is a custom function to split data into batches
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def extract_response(original_response):
    cut_idx = original_response.find("</s>")
    if cut_idx != -1:
        extracted_response = original_response[:cut_idx].strip()
    else:
        extracted_response = original_response
        
    if "<s>" in extracted_response:
        cut_idx_1 = extracted_response.find("<s>") 
        extracted_response = extracted_response[cut_idx_1:].replace("<s>", "").strip()
        
    return extracted_response

def wrap_up_prompt(data, DB_schema_path, prompt_template, turn_i, data_sys, data_user_1):
    
    if "prompt" in data:
        del data["prompt"]
    
    # re-run error cases: set flg  
    if 'prediction_turn_'+str(turn_i) in data and "Error:" in data['prediction_turn_'+str(turn_i)]:
        error_flg = True
        data["error_flg"] = error_flg
    elif data_sys["error_flg"] == True or data_user_1["error_flg"] == True:
        error_flg = True
        data["error_flg"] = error_flg
    else:
        error_flg = False
        data["error_flg"] = error_flg
    return_flg = 'prediction_turn_'+str(turn_i) in data and error_flg == False
    
    if "Terminate_flg" in data_sys and data_sys["Terminate_flg"] == True:
        terminate_flag = True
        data["Terminate_flg"] = True
        data["final_turn"] = data_sys["final_turn"]
        return data
    elif return_flg:
        return data
    else:
        terminate_flag = False
        data["final_turn"] = data_sys["final_turn"]
    
    # Start
    db_name = data.get('selected_database', '')    
    with open(DB_schema_path.replace("[[DB_name]]", db_name), 'r', encoding='utf-8') as file:
        DB_schema = file.read()

    ### prompt filling
    prompt = prompt_template.replace('[[clarification_Q]]', extract_response(data_sys['prediction_turn_'+str(turn_i)]))
    prompt = prompt.replace('[[Action]]', extract_response(data_user_1['prediction_turn_'+str(turn_i)]))
    prompt = prompt.replace('[[amb_json]]', "user_query_ambiguity: \n" + json.dumps(data["user_query_ambiguity"], indent=4) + '\n\nknowledge_ambiguity: \n' + json.dumps(data["knowledge_ambiguity"], indent=4))
    
    sql_segs = ""
    sol_sql_all = ""
    cnt = 0
    for sol_sql_i in data["sol_sql"]: 
        sol_sql_all = sol_sql_all + sol_sql_i + "\n\n"
        cnt += 1 
        if cnt > 1:
            sql_segs = sql_segs + "\n===\n"
        for clause, text in segment_sql(sol_sql_i):
            sql_segs = sql_segs + clause+ ":\n" + text + "\n\n"    

    prompt = prompt.replace('[[SQL_Glot]]', sql_segs.strip())
    prompt = prompt.replace('[[GT_SQL]]', sol_sql_all.strip())
    prompt = prompt.replace('[[DB_schema]]', DB_schema)
    prompt = prompt.replace('[[clear_query]]', data.get("query", ""))

    if terminate_flag != True:  
        data['prompt_turn_'+str(turn_i)] = prompt
        data["prompt"] = prompt

    return data

def load_from_jsonl_dataset(prompt_path, sys_resp_path, user_1_resp_path, result_path, DB_schema_path, prompt_template, turn_i):

    with open(sys_resp_path, 'r') as f:
        dataset_sys = [json.loads(line) for line in f]    
    with open(user_1_resp_path, 'r') as f:
        dataset_user_1 = [json.loads(line) for line in f]    
    with open(prompt_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
        dataset = [wrap_up_prompt(dataset[j], DB_schema_path, prompt_template, turn_i, dataset_sys[j], dataset_user_1[j]) for j in range(len(dataset))]
    
    with open(result_path, "w", encoding="utf-8") as f:
        for item in dataset:
            if "prompt" in item:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + "\n")
  
def inference():  
    parser = argparse.ArgumentParser(description='Call OpenAI API with specified parameters and configurations.')  
    parser.add_argument('--turn_num', type=int, help='Turn number.')   
    parser.add_argument('--prompt_path', type=str, required=True, help='Path to the input .jsonl file containing prompts.')  
    parser.add_argument('--sys_resp_path', type=str, required=True, help='Path to the sys_resp.jsonl file containing user responses.')
    parser.add_argument('--user_1_resp_path', type=str, required=True, help='Path to the user_1_resp.jsonl file containing user responses.')
    parser.add_argument('--result_path', type=str, required=True, help='Path where the output .jsonl file with results will be saved.')  
    parser.add_argument('--DB_schema_path', type=str, required=True, help='Path where the DB_schema.json file will be saved.')

    args = parser.parse_args()  
        
    from bird_interact.prompts.prompts import user_simulator_decoder
    prompt_template = user_simulator_decoder

    load_from_jsonl_dataset(prompt_path=args.prompt_path, sys_resp_path=args.sys_resp_path, user_1_resp_path=args.user_1_resp_path, result_path=args.result_path, DB_schema_path=args.DB_schema_path, prompt_template=prompt_template, turn_i=args.turn_num)
 
if __name__ == "__main__":  
    inference()  
