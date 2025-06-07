#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os  
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse  
import json  

def merge_jsonl_by_instance_id(source_path, response_path, output_path):
    id_to_response = {}
    with open(response_path, "r", encoding="utf-8") as resp_file:
        for line in resp_file:
            dict_i = json.loads(line)
            dict_i['prediction_turn_'+str(dict_i["final_turn"])] = dict_i["response"]
            
            del dict_i["response"]
            if "reasoning_content" in dict_i:
                del dict_i["reasoning_content"]
            if "token_usage" in dict_i:
                del dict_i["token_usage"]
            
            instance_id = dict_i.get("instance_id")
            if instance_id is not None:
                id_to_response[instance_id] = dict_i

    with open(source_path, "r", encoding="utf-8") as f:
        src_file = [json.loads(line) for line in f]    
        
    with open(output_path, "w", encoding="utf-8") as out_file:
        for item in src_file:
            instance_id = item.get("instance_id")
            if instance_id in id_to_response:
                item = id_to_response[instance_id]
                item["error_flg"] = False
            else:
                if item["error_flg"] == False:
                    item["Terminate_flg"] = True
            if "prompt" in item:
                del item["prompt"]
            out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
  
def inference():  
    parser = argparse.ArgumentParser(description='Call OpenAI API with specified parameters and configurations.')  
    parser.add_argument('--source_path', type=str, required=True, help='Path to the prompt.jsonl file containing user responses.')
    parser.add_argument('--response_path', type=str, required=True, help='Path to the response.jsonl file containing user responses.')
    parser.add_argument('--result_path', type=str, required=True, help='Path where the output.jsonl file with results will be saved.')  
    
    args = parser.parse_args()  
    merge_jsonl_by_instance_id(args.source_path, args.response_path, args.result_path)    

if __name__ == "__main__":  
    inference()  
