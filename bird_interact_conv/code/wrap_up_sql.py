import os  
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse  
import json  

def extract_system_response(original_response):
    cut_prep = original_response.find("### Turn ")
    if cut_prep != -1:
        original_response = original_response[:cut_prep]
        
    if "```sql" in original_response:
        cut_idx = original_response.find("```sql")
        extracted_response = original_response[cut_idx:].replace("```sql", "").strip()
        if "```" in extracted_response:
            cut_idx_1 = extracted_response.find("```") 
            extracted_response = extracted_response[:cut_idx_1].strip()
    else: 
        return "ERROR!"
        
    return extracted_response

def get_max_turn_prediction(d):
    max_turn = -1
    max_prediction = None
    
    for key, value in d.items():
        if key.startswith("prediction_turn_"):
            try:
                turn_num = int(key[len("prediction_turn_"):])
                if turn_num > max_turn:
                    max_turn = turn_num
                    max_prediction = value
            except ValueError:
                continue 

    return max_prediction
        
def wrap_up_prompt(data):         
    response_sql = get_max_turn_prediction(data)
    sql_extracted = extract_system_response(response_sql)
    test_cases = data.get('test_cases', '')
    if not test_cases:
        if not data["conditions"]["distinct"]:
            test_cases = ["\ndef test_case(pred_sqls, sol_sqls, db_name, conn):\n    pred_sqls = remove_distinct(pred_sqls)\n    sol_sqls = remove_distinct(sol_sqls)\n    result = ex_base(pred_sqls, sol_sqls, db_name, conn)\n    assert result == 1, f\"ex_base returned {result} but expected 1.\"\n    return result"]
        else:
            test_cases = ["\ndef test_case(pred_sqls, sol_sqls, db_name, conn):\n    result = ex_base(pred_sqls, sol_sqls, db_name, conn)\n    assert result == 1, f\"ex_base returned {result} but expected 1.\"\n    return result"]
        data['test_cases'] = test_cases
        
    data['pred_sqls'] = [sql_extracted]
    
    return data

def process_jsonl():
    parser = argparse.ArgumentParser(description='Wrap up SQL.')  
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input .jsonl file containing prompts.')  
    parser.add_argument('--result_path', type=str, required=True, help='Path where the output .jsonl file with results will be saved.')  
    parser.add_argument('--follow_up_path', type=str, required=False, default=None, help='Path where the output .jsonl file with results will be saved.')  
    args = parser.parse_args()  
    
    input_file = args.data_path
    output_file = args.result_path
    instance_list = []
    if args.follow_up_path:
        with open(args.follow_up_path, 'r', encoding='utf-8') as f_p:
            for line in f_p:
                if not line.strip():
                    continue  
                data = json.loads(line)
                instance_list.append(data["instance_id"])
                
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue  
            data = json.loads(line)
            modified_data = wrap_up_prompt(data)
            if instance_list:
                if data["instance_id"] in instance_list:
                    fout.write(json.dumps(modified_data, ensure_ascii=False) + '\n')
            else:
                fout.write(json.dumps(modified_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":  
    process_jsonl()  