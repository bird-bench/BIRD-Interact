import re
import json
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from postgresql_utils import execute_queries
import logging 
logging.basicConfig(level=logging.WARNING)


# Use centralized database configuration
# DB_CONFIG = get_db_config()

# === Helper Functions ===
def process_decimals_recursive(item, decimal_places):
    """
    Recursively process decimals in any data structure (list, dict, tuple).
    Returns a new structure with all decimals rounded to specified places.
    """
    quantizer = Decimal(1).scaleb(-decimal_places)
    
    if isinstance(item, Decimal):
        return item.quantize(quantizer, rounding=ROUND_HALF_UP)
    elif isinstance(item, float):
        return round(item, decimal_places)
    elif isinstance(item, (list, tuple)):
        return type(item)(process_decimals_recursive(x, decimal_places) for x in item)
    elif isinstance(item, dict):
        return {k: process_decimals_recursive(v, decimal_places) for k, v in item.items()}
    else:
        return item

def preprocess_results(results, decimal_places=2):
    """
    Process the result set:
    - Replace dates with normalized string: YYYY-MM-DD
    - Convert tuples to lists for JSON serializability
    - Convert any unhashable types (dicts, lists) to their string representation for comparison
    - Process decimals recursively in all nested structures
    """
    processed = []
    for result in results:
        processed_result = []
        for item in result:
            if isinstance(item, (date, datetime)):
                processed_result.append(item.strftime('%Y-%m-%d'))
            else:
                # Process decimals recursively first
                processed_item = process_decimals_recursive(item, decimal_places)
                if isinstance(processed_item, (dict, list)):
                    # Convert unhashable types to their string representation with sorted keys
                    processed_result.append(json.dumps(processed_item, sort_keys=True))
                else:
                    processed_result.append(processed_item)
        processed.append(tuple(processed_result))
    return processed


def remove_comments(sql_list):
    """
    Remove all SQL comments from each query string in the list.
    - Block comments: /* ... */
    - Line comments: -- ... (to end of line)
    Also collapses multiple blank lines into one, and strips leading/trailing whitespace.
    """
    cleaned = []
    for sql in sql_list:
        # remove block comments
        no_block = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        # remove line comments, keep newline
        no_line  = re.sub(r'--.*?(\r\n|\r|\n)', r'\1', no_block)
        # collapse extra blank lines
        no_blank = re.sub(r'\n\s*\n+', '\n', no_line)
        cleaned.append(no_blank.strip())
    return cleaned


def remove_distinct(sql_list):
    """
    Strip out all DISTINCT tokens (case-insensitive).
    """
    cleaned = []
    for q in sql_list:
        tokens = q.split(" ")
        tokens = [t for t in tokens if t.lower() != 'distinct']
        cleaned.append(' '.join(tokens))
    return cleaned


def remove_round_functions(sql_string):
    """
    Remove all ROUND() function calls from a SQL string, including nested ones.
    This regex properly handles nested functions with commas.
    """
    
    def find_matching_paren(text, start_pos):
        """Find the position of the matching closing parenthesis."""
        paren_count = 0
        for i in range(start_pos, len(text)):
            if text[i] == '(':
                paren_count += 1
            elif text[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    return i
        return -1
    
    def find_first_arg_end(text, start_pos):
        """Find the end of the first argument, accounting for nested parentheses."""
        paren_count = 0
        for i in range(start_pos, len(text)):
            if text[i] == '(':
                paren_count += 1
            elif text[i] == ')':
                if paren_count == 0:
                    return i  # End of ROUND function
                paren_count -= 1
            elif text[i] == ',' and paren_count == 0:
                return i  # End of first argument
        return len(text)
    
    result = sql_string
    
    while True:
        # Find ROUND function (case insensitive)
        pattern = re.compile(r'ROUND\s*\(', re.IGNORECASE)
        match = pattern.search(result)
        
        if not match:
            break
            
        start_pos = match.start()
        open_paren_pos = match.end() - 1
        
        # Find the end of the first argument
        first_arg_end = find_first_arg_end(result, open_paren_pos + 1)
        
        # Find the matching closing parenthesis
        close_paren_pos = find_matching_paren(result, open_paren_pos)
        
        if close_paren_pos == -1:
            break  # Malformed SQL, can't find closing paren
        
        # Extract the first argument
        first_arg = result[open_paren_pos + 1:first_arg_end].strip()
        
        # Replace ROUND(...) with just the first argument
        result = result[:start_pos] + first_arg + result[close_paren_pos + 1:]
    
    return result


def remove_round_functions_regex(sql_string):
    pattern = r'ROUND\s*\(([^,()]*(?:\([^()]*\)[^,()]*)*?)(?:,[^)]*)?\)'
    while True:
        new_result = re.sub(pattern, r'\1', sql_string, flags=re.IGNORECASE)
        if new_result == sql_string:  # No more changes made
            break
        sql_string = new_result
    return sql_string


def remove_round(sql_list):
    """
    Remove ROUND function calls while preserving the inner expression.
    For example: 
    - ROUND(column, 2) -> column
    - ROUND(ROUND(price, 2), 1) -> ROUND(price, 2) -> price (handles nested ROUNDs)
    
    Uses non-greedy matching (? after +) to handle nested expressions:
    - Greedy: ROUND(.*, n) would match ROUND(ROUND(price, 2), 1) as one match
    - Non-greedy: ROUND(.*?, n) matches ROUND(price, 2) first, then ROUND(..., 1)
    """
    cleaned = []
    for sql in sql_list:
        result = sql
        result = remove_round_functions(result)
        cleaned.append(result)
        if "ROUND" in result:
            logging.warning(f"ROUND found in {result}")
    return cleaned


def remove_order_by(sql_list):
    """
    Remove all ORDER BY ... clauses (up to next LIMIT/FETCH/OFFSET/)/; or end-of-string).
    """
    pattern = re.compile(
        r'\bORDER\s+BY\b'       # ORDER BY
        r'.*?'                     # non-greedy
        r'(?=(?:\bLIMIT\b|\bFETCH\b|\bOFFSET\b|\)|;|$))',
        flags=re.IGNORECASE | re.DOTALL
    )
    cleaned = []
    for sql in sql_list:
        no_ob = pattern.sub('', sql)
        # normalize whitespace
        no_ob = re.sub(r'\s+', ' ', no_ob).strip()
        cleaned.append(no_ob)
    return cleaned


def process_decimals(results, decimal_places):
    """
    Round any Decimal or float in the result set to `decimal_places`.
    """
    quantizer = Decimal(1).scaleb(-decimal_places)
    rounded = []
    for row in results:
        new_row = []
        for item in row:
            if isinstance(item, Decimal):
                new_row.append(item.quantize(quantizer, rounding=ROUND_HALF_UP))
            elif isinstance(item, float):
                new_row.append(round(item, decimal_places))
            else:
                new_row.append(item)
        rounded.append(tuple(new_row))
    return rounded


def ex_base(pred_sqls, sol_sqls, db_name, conn, conditions=None):
    """
    Compare result-sets of two lists of SQL queries:
    - Strip comments, DISTINCT, and ORDER BY
    - Execute
    - Normalize dates and optionally round decimals
    - Check equality (either ordered or unordered based on conditions)
    Return 1 on match, else 0.
    """
    if not pred_sqls or not sol_sqls:
        return 0

    # execute
    predicted_res, pred_err, pred_to = execute_queries(pred_sqls, db_name, conn, None, "")
    ground_res, gt_err, gt_to      = execute_queries(sol_sqls,  db_name, conn, None, "")
    if any([pred_err, pred_to, gt_err, gt_to]):
        return 0

    predicted_res = preprocess_results(predicted_res)
    ground_res    = preprocess_results(ground_res)
    if not predicted_res or not ground_res:
        return 0
    
    # Check if we should compare with order
    if conditions is not None and conditions.get("order", False):
        # Compare as lists to preserve order
        return 1 if predicted_res == ground_res else 0
    else:
        # Default: compare as sets (order doesn't matter)
        return 1 if set(predicted_res) == set(ground_res) else 0


def test_case_default(pred_sqls, sol_sqls, db_name, conn, conditions=None):
    """
    Default test_case: pytest-style assertion.
    """
    # clean queries
    pred_sqls = remove_comments(pred_sqls)
    sol_sqls  = remove_comments(sol_sqls)
    pred_sqls = remove_distinct(pred_sqls)
    pred_sqls = remove_round(pred_sqls)
    sol_sqls  = remove_distinct(sol_sqls)
    sol_sqls  = remove_round(sol_sqls)

    result = ex_base(pred_sqls, sol_sqls, db_name, conn, conditions)
    assert result == 1, f"ex_base returned {result} but expected 1."
    return result


if __name__ == "__main__":
    tested_sql = """
-- This query evaluates optimal observatory conditions using Atmospheric Observability Index (AOI)
-- and generates a JSON report of environmental factors using JSON aggregation functions.
-- Using KB knowledge: Atmospheric Observability Index (AOI) [id: 1], Optimal Observing Window (OOW) [id: 13]

-- Step 1: Calculate AOI for each observatory and determine if conditions meet Optimal Observing Window criteria
WITH observatory_conditions AS (
    SELECT
        o.ObservStation,
        o.AtmosTransparency * (1 - o.HumidityRate/100) * (1 - 0.02 * o.WindSpeedMs) AS aoi,  -- AOI formula from KB
        o.LunarStage,
        o.LunarDistDeg,
        o.SolarStatus,
        -- Check if conditions meet OOW criteria
        CASE WHEN
            o.AtmosTransparency * (1 - o.HumidityRate/100) * (1 - 0.02 * o.WindSpeedMs) > 0.85 AND
            (o.LunarStage = 'New' OR o.LunarStage = 'First Quarter') AND
            o.LunarDistDeg > 45 AND
            (o.SolarStatus = 'Low' OR o.SolarStatus = 'Moderate')
        THEN TRUE ELSE FALSE END AS is_optimal_window
    FROM
        Observatories o
)
-- Step 2: Generate JSON report grouping observatories by optimal condition status
SELECT
    is_optimal_window,
    COUNT(*) AS station_count,
    ROUND(AVG(aoi)::numeric, 3) AS avg_aoi,
    -- Use JSON aggregation to create detailed environmental factors report
    jsonb_agg(jsonb_build_object(
        'station', ObservStation,
        'aoi', ROUND(aoi::numeric, 3),
        'lunar_factors', jsonb_build_object(
            'stage', LunarStage,
            'distance', LunarDistDeg
        ),
        'solar_status', SolarStatus
    )) AS observatory_details
FROM
    observatory_conditions
GROUP BY
    is_optimal_window;
    """
    sqls = [tested_sql]
    # test remove roudn 
    sqls = remove_round(sqls)
    print("\n".join(sqls))
