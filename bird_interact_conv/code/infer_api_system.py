#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os  
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse  
import json   


def process_batch_data(data, batch_size):
    # Assuming process_batch_data is a custom function to split data into batches
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def extract_user_response(original_response):
    cut_idx = original_response.find("</s>")
    if cut_idx != -1:
        extracted_response = original_response[:cut_idx].strip()
    else:
        extracted_response = original_response
        
    if "<s>" in extracted_response:
        cut_idx_1 = extracted_response.find("<s>") 
        extracted_response = extracted_response[cut_idx_1:].replace("<s>", "").strip()
        
    return extracted_response

def extract_system_response(original_response):
    cut_prep = original_response.find("### Turn ")
    if cut_prep != -1:
        original_response = original_response[:cut_prep]
    if "</s>" in original_response:
        sep_char = "s"
        terminate_flag = False
    elif "</t>" in original_response:
        sep_char = "t"
        terminate_flag = True
    else:
        terminate_flag = False
        return original_response, terminate_flag
    
    cut_idx = original_response.find("</"+sep_char+">")
    extracted_response = original_response[:cut_idx].strip()
    if "<"+sep_char+">" in extracted_response:
        cut_idx_1 = extracted_response.find("<"+sep_char+">") 
        extracted_response = extracted_response[cut_idx_1:].replace("<"+sep_char+">", "").strip()
        
    return extracted_response, terminate_flag

def wrap_up_prompt(data, DB_schema_path, external_kg_path, prompt_template, patience, turn_i, data_user_dict, phase="amb"):    
    if "prompt" in data:
        del data["prompt"]
    
    # re-run error cases: set flg
    if 'prediction_turn_'+str(turn_i) in data and "Error:" in data['prediction_turn_'+str(turn_i)]:
        error_flg = True
        data["error_flg"] = error_flg
    else:
        error_flg = False
        data["error_flg"] = error_flg
    return_flg = 'prediction_turn_'+str(turn_i) in data and "Error:" not in data['prediction_turn_'+str(turn_i)]
    
    # Start
    try:
        data_user = data_user_dict[data["instance_id"]]
    except KeyError:
        data_user = {}
    
    if phase == "amb":
        if "Terminate_flg" in data or return_flg:
            terminate_flag = True
            return data
        else:
            terminate_flag = False
        
        max_turn = len(data["user_query_ambiguity"]["critical_ambiguity"]) + len(data["knowledge_ambiguity"]) + patience
        
        ### If first turn:
        if turn_i == 1:
            db_name = data.get('selected_database', '')
            question = data.get('amb_user_query', '')
            with open(DB_schema_path.replace("[[DB_name]]", db_name), 'r', encoding='utf-8') as file:
                DB_schema = file.read()
            
            ### Exclude masked knowledge
            external_kg_list = []
            exclude_ids = []
            for knowledge_amb_i in data["knowledge_ambiguity"]:
                exclude_ids.append(knowledge_amb_i["deleted_knowledge"])
                
            with open(external_kg_path.replace("[[DB_name]]", db_name), "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if obj.get("id") not in exclude_ids:
                        external_kg_list.append(json.dumps(obj))

            external_kg = "\n".join(external_kg_list)

            ### prompt filling
            prompt = prompt_template.replace('[[user_query]]', question)
            prompt = prompt.replace('[[DB_name]]', db_name)
            prompt = prompt.replace('[[max_turn]]', str(max_turn))
            prompt = prompt.replace('[[DB_schema]]', DB_schema)
            prompt = prompt.replace('[[external_kg]]', external_kg)
            data['max_turn'] = max_turn
            
        ### If not first turn:
        else:
            prompt = data.get('prompt_turn_'+str(turn_i-1), '')
            response_prev = data.get('prediction_turn_'+str(turn_i-1), '')
            response_user_prev = data_user.get('prediction_turn_'+str(turn_i-1), '')
            if max_turn > turn_i:
                sys_response, terminate_flag = extract_system_response(response_prev)
                if terminate_flag == True:
                    data["Terminate_flg"] = True
                user_response = extract_user_response(response_user_prev)
                prompt = prompt + sys_response + "\n- User: " + user_response + '\n\n### Turn [[turn_i]] ([[turn_left]] turns left): \n# Format: "<s>[YOUR-ONLY-ONE-QUESTION]]</s>" if you choose to ask for clarification; or "<t>```postgresql [YOUR-SQL] ```</t>" if you choose to generate final SQL.\n- You: '.replace('[[turn_i]]', str(turn_i)).replace('[[turn_left]]', str(max_turn-turn_i+1))
            else:
                sys_response, terminate_flag = extract_system_response(response_prev)
                if terminate_flag == True:
                    data["Terminate_flg"] = True
                user_response = extract_user_response(response_user_prev)
                prompt = prompt + sys_response + "\n- User: " + user_response + '\n\n### Turn [[turn_i]] (1 turn left): \n# It is the final turn. You MUST provide the final PostgreSQL and follow the format: "<t>```postgresql [YOUR-SQL] ```</t>"\n- You: <t>'.replace('[[turn_i]]', str(turn_i))
        
        if terminate_flag != True:  
            data['prompt_turn_'+str(turn_i)] = prompt
            data["prompt"] = prompt
            data['final_turn'] = turn_i
            
    elif phase=="debug" and data_user!={} and data_user["status"]=="failed":
        data_sql_report = data_user
        prompt = data.get('prompt_turn_'+str(data['final_turn']), '')
        data['final_turn'] = data['final_turn'] + 1
        
        if "[exec_err_flg]" in data_sql_report["error_msg"]:
            error_msg = "Your SQL is not executable and raises the following error: " + data_sql_report["error_msg"]
        else:
            error_msg = "Are you sure about your SQL? You have one more chance to update your SQL now."
            
        prompt = prompt.replace("- You: <t>", "- You: \n```postgresql \n") + data_sql_report.get('pred_sqls', '')[0] + '\n``` \n\n### Turn [[turn_i]]: \n# Your sql in previous turn may have problem. You MUST provide the updated PostgreSQL and follow the format: "<t>```postgresql [YOUR-SQL] ```</t>"\n-User: '.replace('[[turn_i]]', str(data['final_turn'])) + error_msg.strip() + '\n- You: <t>'
        
        data['prompt_turn_'+str(data['final_turn'])] = prompt
        data["prompt"] = prompt
    
    elif phase=="follow" and data_user!={} and data_user["status"]=="success":
        data_sql_report = data_user
        prompt = data.get('prompt_turn_'+str(data['final_turn']), '')
        data['final_turn'] = data['final_turn'] + 1
        follow_up_Q = data['follow_up']['query']
        
        prompt = prompt.replace("- You: <t>", "- You: \n```postgresql \n") + data_sql_report.get('pred_sqls', '')[0] + '\n``` \n\n### Turn [[turn_i]]: \n# Here is a follow up question. You MUST provide the PostgreSQ to solve this question and follow the format: "<t>```postgresql [YOUR-SQL] ```</t>"\n-User: ```text \n'.replace('[[turn_i]]', str(data['final_turn'])) + follow_up_Q + '\n```\n\n- You: <t>'
        data['prompt_turn_'+str(data['final_turn'])] = prompt
        data["prompt"] = prompt
        
    else:
        pass

    return data

def load_from_jsonl_dataset(prompt_path, user_resp_path, result_path, DB_schema_path, external_kg_path, prompt_template, patience, turn_i, phase="amb"):

    with open(user_resp_path, 'r') as f:
        dataset_user = {}
        for line in f:
            dict = json.loads(line)
            dataset_user[dict["instance_id"]] = dict
    with open(prompt_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
        dataset = [wrap_up_prompt(dataset[j], DB_schema_path, external_kg_path, prompt_template, patience, turn_i, dataset_user, phase=phase) for j in range(len(dataset))]
        
    with open(result_path, "w", encoding="utf-8") as f:
        for item in dataset:
            if "prompt" in item:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + "\n")


def inference():  
    parser = argparse.ArgumentParser(description='Call OpenAI API with specified parameters and configurations.')  
    parser.add_argument('--patience', type=int, default=6, help='Maximum turn.')   
    parser.add_argument('--turn_num', type=int, help='Turn number.')   
    parser.add_argument('--prompt_path', type=str, required=True, help='Path to the input .jsonl file containing prompts.')  
    parser.add_argument('--user_resp_path', type=str, required=True, help='Path to the user_resp.jsonl file containing user responses.')
    parser.add_argument('--result_path', type=str, required=True, help='Path where the output .jsonl file with results will be saved.')  
    parser.add_argument('--DB_schema_path', type=str, required=True, help='Path where the DB_schema.json file will be saved.')
    parser.add_argument('--external_kg_path', type=str, required=True, help='Path where the external_kg.json file will be saved.')  
    parser.add_argument('--phase', type=str, required=False, default='amb', help='The phase you want to proceed: ["amb", "debug", "follow"]')

    args = parser.parse_args()  
        
    from bird_interact_conv.prompts.prompts import system_react
    prompt_template = system_react

    load_from_jsonl_dataset(prompt_path=args.prompt_path, user_resp_path=args.user_resp_path, result_path=args.result_path, DB_schema_path=args.DB_schema_path, external_kg_path=args.external_kg_path, prompt_template=prompt_template, patience=args.patience, turn_i=args.turn_num, phase=args.phase)
 
if __name__ == "__main__":  
    inference()  
