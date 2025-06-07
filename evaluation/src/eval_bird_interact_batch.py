import argparse
import json
from io import StringIO
from contextlib import redirect_stdout
from eval_bird_interact import test_case_default
from postgresql_utils import execute_queries, reset_and_restore_database, get_connection_for_phase


def compute_score_rate(results):
    total = len(results)
    if total == 0:
        return 0.0
    success_count = sum(1 for status in results.values() if status == 'success')
    return success_count / total


def ensure_list(val):
    """
    Coerce val into a list of strings.
    """
    if isinstance(val, list):
        return [s for s in val if isinstance(s, str)]
    if isinstance(val, str):
        text = val.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [s for s in parsed if isinstance(s, str)]
        except json.JSONDecodeError:
            return [text]
    return []


def batch_eval(jsonl_path, pg_password="123123"):
    output_path = jsonl_path.replace('.jsonl', '_output_with_status.jsonl')
    results = {}
    total = 0

    with open(jsonl_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            instance_id = rec.get('instance_id', '<no_id>')
            db_name = rec.get('selected_database')
            pred_sqls = rec.get('pred_sqls', [])
            conditions = rec.get("conditions", None)
            if not isinstance(pred_sqls, list):
                pred_sqls = ensure_list(pred_sqls)
            sol_sqls = rec.get('sol_sql') if '_fu' not in jsonl_path else rec.get('follow_up', {}).get('sol_sql')
            if not isinstance(sol_sqls, list):
                sol_sqls = ensure_list(sol_sqls)
            
            test_cases = rec.get('test_cases') if '_fu' not in jsonl_path else rec.get('follow_up', {}).get('test_cases')
            if not isinstance(test_cases, list):
                test_cases = [test_cases]
                
            if '_fu' not in jsonl_path:
                preprocess_sql = rec.get("preprocess_sql")
                clean_up_sqls = rec.get("clean_up_sqls")
            else:
                preprocess_sql = rec.get("preprocess_sql")
                preprocess_sql.extend(rec.get('sol_sql'))
                preprocess_sql.extend(rec.get("clean_up_sqls"))
                clean_up_sqls = []

            status = 'failed'
            error_msg = ''
            pred_query_result = None

            # reset DB
            try:
                reset_and_restore_database(db_name, pg_password)
            except Exception as e:
                error_msg = f'DB reset error: {e}'
                rec.update({'status': status, 'error_msg': error_msg})
                results[instance_id] = status
                outfile.write(json.dumps(rec, ensure_ascii=False) + '\n')
                continue

            # fresh connection
            try:
                conn = get_connection_for_phase(db_name)
            except Exception as e:
                error_msg = f'DB connection error: {e}'
                rec.update({'status': status, 'error_msg': error_msg})
                results[instance_id] = status
                outfile.write(json.dumps(rec, ensure_ascii=False) + '\n')
                continue

            # execute preprocess_sql
            try:
                prep_query_result, prep_err, prep_to = execute_queries(preprocess_sql, db_name, conn)
            except:
                pass
            
            # execute pred_sqls
            try:
                pred_query_result, pred_err, pred_to = execute_queries(pred_sqls, db_name, conn)
                if pred_err:
                    raise RuntimeError(f"Execution_Error: {pred_err}")
                if pred_to:
                    raise RuntimeError("Execution_Error: timeout_error")
            except Exception as e:
                error_msg = f"[exec_err_flg] - {e}"
                rec.update({'status': status, 'error_msg': error_msg})
                results[instance_id] = status
                outfile.write(json.dumps(rec, ensure_ascii=False) + '\n')
                try:
                    conn.close()
                except:
                    pass
                continue

            # default test functions list
            test_funcs = []

            # choose and load external test cases
            if '_M_' in instance_id:
                # make pred_query_result available globally
                globals()['pred_query_result'] = pred_query_result
                for code_snippet in test_cases:
                    local_ns = {}
                    try:
                        exec(code_snippet, globals(), local_ns)
                        # collect any function defined
                        for v in local_ns.values():
                            if callable(v):
                                test_funcs.append(v)
                    except Exception as e:
                        error_msg = f'Compiling test_cases error: {e}'
                        
            # always include default if no external funcs
            cond_flg = False
            if not test_funcs:
                test_funcs = [test_case_default]
                cond_flg = True

            # run all test functions
            overall_pass = True
            for func in test_funcs:
                buf = StringIO()
                try:
                    with redirect_stdout(buf):
                        if cond_flg:
                            res = func(pred_sqls, sol_sqls, db_name, conn, conditions)
                        else:
                            res = func(pred_sqls, sol_sqls, db_name, conn)
                    out = buf.getvalue().strip()
                except Exception as e:
                    overall_pass = False
                    out = buf.getvalue().strip()
                    error_msg = out or str(e)
                    break

            status = 'success' if overall_pass else 'failed'
            if overall_pass:
                error_msg = ''
                
            # execute clean_up_sql
            try:
                clean_query_result, clean_err, clean_to = execute_queries(clean_up_sqls, db_name, conn)
            except:
                pass

            # close connection
            try:
                conn.close()
            except:
                pass

            rec.update({'status': status, 'error_msg': error_msg})
            results[instance_id] = status
            outfile.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # summary
    success_count = sum(1 for s in results.values() if s == 'success')
    rate = compute_score_rate(results) * 100
    print(f'Batch evaluation complete. Results written to {output_path}')
    print(f'Total: {total}, Successes: {success_count}, Success rate: {rate:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch evaluate JSONL of SQL cases.')
    parser.add_argument('--jsonl', required=True, help='Path to JSONL input file')
    parser.add_argument('--pg_password', default='123123', help='Password for resetting DB')
    args = parser.parse_args()

    batch_eval(args.jsonl, args.pg_password)
