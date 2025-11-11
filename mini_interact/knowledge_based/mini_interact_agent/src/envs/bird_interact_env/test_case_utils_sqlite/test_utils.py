import re
from datetime import date, datetime
from .db_utils import perform_query_on_sqlite_databases, execute_queries
import sqlite3
import json
from decimal import Decimal, ROUND_HALF_UP
import logging


def process_decimals(results, decimal_places):
    """
    将结果集中的任何Decimal或float舍入到decimal_places位
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


def remove_round_functions(sql_string):
    """
    从SQL字符串中删除所有ROUND()函数调用，包括嵌套的
    此正则表达式正确处理带逗号的嵌套函数
    """

    def find_matching_paren(text, start_pos):
        """找到匹配的右括号位置"""
        paren_count = 0
        for i in range(start_pos, len(text)):
            if text[i] == "(":
                paren_count += 1
            elif text[i] == ")":
                paren_count -= 1
                if paren_count == 0:
                    return i
        return -1

    def find_first_arg_end(text, start_pos):
        """找到第一个参数的结尾，考虑嵌套括号"""
        paren_count = 0
        for i in range(start_pos, len(text)):
            if text[i] == "(":
                paren_count += 1
            elif text[i] == ")":
                if paren_count == 0:
                    return i  # ROUND函数结尾
                paren_count -= 1
            elif text[i] == "," and paren_count == 0:
                return i  # 第一个参数结尾
        return len(text)

    result = sql_string

    while True:
        # 查找ROUND函数(不区分大小写)
        pattern = re.compile(r"ROUND\s*\(", re.IGNORECASE)
        match = pattern.search(result)

        if not match:
            break

        start_pos = match.start()
        open_paren_pos = match.end() - 1

        # 找到第一个参数的
        # 找到第一个参数的结尾
        first_arg_end = find_first_arg_end(result, open_paren_pos + 1)

        # 找到匹配的右括号
        close_paren_pos = find_matching_paren(result, open_paren_pos)

        if close_paren_pos == -1:
            break  # 格式错误的SQL，找不到右括号

        # 提取第一个参数
        first_arg = result[open_paren_pos + 1 : first_arg_end].strip()

        # 用第一个参数替换ROUND(...)
        result = result[:start_pos] + first_arg + result[close_paren_pos + 1 :]

    return result


def remove_round_functions_regex(sql_string):
    pattern = r"ROUND\s*\(([^,()]*(?:\([^()]*\)[^,()]*)*?)(?:,[^)]*)?\)"
    while True:
        new_result = re.sub(pattern, r"\1", sql_string, flags=re.IGNORECASE)
        if new_result == sql_string:  # 没有更多更改
            break
        sql_string = new_result
    return sql_string


def remove_round(sql_list):
    """
    删除ROUND函数调用，同时保留内部表达式
    例如:
    - ROUND(column, 2) -> column
    - ROUND(ROUND(price, 2), 1) -> ROUND(price, 2) -> price (处理嵌套的ROUND)
    """
    cleaned = []
    for sql in sql_list:
        result = sql
        result = remove_round_functions(result)
        cleaned.append(result)
        if "ROUND" in result:
            logging.warning(f"ROUND found in {result}")
    return cleaned


def process_decimals_recursive(item, decimal_places):
    """
    递归处理任何数据结构(list, dict, tuple)中的小数
    返回一个新结构，其中所有小数都舍入到指定位数
    """
    quantizer = Decimal(1).scaleb(-decimal_places)

    if isinstance(item, Decimal):
        return item.quantize(quantizer, rounding=ROUND_HALF_UP)
    elif isinstance(item, float):
        return round(item, decimal_places)
    elif isinstance(item, (list, tuple)):
        return type(item)(process_decimals_recursive(x, decimal_places) for x in item)
    elif isinstance(item, dict):
        return {
            k: process_decimals_recursive(v, decimal_places) for k, v in item.items()
        }
    else:
        return item


def preprocess_results(results, decimal_places=2):
    """
    处理结果集:
    - 用规范化字符串替换日期: YYYY-MM-DD
    - 将元组转换为列表以实现JSON序列化
    - 将任何不可哈希类型(dicts, lists)转换为其字符串表示形式以进行比较
    - 递归处理所有嵌套结构中的小数
    """
    processed = []
    for result in results:
        processed_result = []
        for item in result:
            if isinstance(item, (date, datetime)):
                processed_result.append(item.strftime("%Y-%m-%d"))
            else:
                # 首先递归处理小数
                processed_item = process_decimals_recursive(item, decimal_places)
                if isinstance(processed_item, (dict, list)):
                    # 将不可哈希类型转换为其字符串表示形式，使用排序的键
                    processed_result.append(json.dumps(processed_item, sort_keys=True))
                else:
                    processed_result.append(processed_item)
        processed.append(tuple(processed_result))
    return processed


def remove_distinct(sql_list):
    """
    从SQL查询字符串列表中删除所有DISTINCT关键字(任何大小写形式)
    这是一种不使用正则表达式的暴力方法

    参数:
    -----------
    sql_list : list of str
        SQL查询列表(字符串)

    返回:
    --------
    list of str
        删除了所有'DISTINCT'关键字的新SQL查询列表
    """

    cleaned_queries = []
    for query in sql_list:
        tokens = query.split(" ")
        filtered_tokens = []
        for token in tokens:
            # 检查此标记是否为'distinct'(不区分大小写)
            if token.lower() != "distinct":
                filtered_tokens.append(token)
        cleaned_query = " ".join(filtered_tokens)
        cleaned_queries.append(cleaned_query)

    return cleaned_queries


def check_sql_function_usage(sqls, required_keywords):
    """
    检查预测的SQL查询列表是否使用了所有指定的关键字或函数
    如果所有必需的关键字都出现，返回1；否则返回0

    参数:
        sqls (list[str]): 预测的SQL查询列表
        required_keywords (list[str]): 必需的关键字或函数列表

    返回:
        int: 如果所有必需的关键字都出现返回1，如果至少缺少一个返回0
    """
    # 如果sqls为空或None，立即返回0
    if not sqls:
        return 0

    # 将所有SQL查询合并为一个字符串并转换为小写
    combined_sql = " ".join(sql.lower() for sql in sqls)

    # 检查所有必需的关键字是否出现在combined_sql中
    for kw in required_keywords:
        if kw.lower() not in combined_sql:
            return 0

    return 1


def ex_base(pred_sqls, sol_sqls, db_path, conn, conditions=None):
    """
    比较两个SQL查询列表的结果集:
    - 去除注释、DISTINCT和ORDER BY
    - 执行
    - 规范化日期并可选地舍入小数
    - 检查相等性(根据条件进行有序或无序比较)
    匹配时返回1，否则返回0
    """
    if not pred_sqls or not sol_sqls:
        return 0

    # 执行
    predicted_res, pred_err, pred_to = execute_queries(
        pred_sqls, db_path, conn, None, ""
    )
    ground_res, gt_err, gt_to = execute_queries(sol_sqls, db_path, conn, None, "")
    if any([pred_err, pred_to, gt_err, gt_to]):
        return 0

    predicted_res = preprocess_results(predicted_res)
    ground_res = preprocess_results(ground_res)
    if not predicted_res or not ground_res:
        return 0

    # 检查是否应该比较顺序
    if conditions is not None and conditions.get("order", False):
        # 作为列表比较以保留顺序
        return 1 if predicted_res == ground_res else 0
    else:
        # 默认：作为集合比较(顺序无关紧要)
        return 1 if set(predicted_res) == set(ground_res) else 0


def performance_compare_by_qep(old_sqls, sol_sqls, db_path, conn):
    """
    在一个连接中比较old_sqls与sol_sqls的总计划成本
    通过使用事务+ROLLBACK确保每组看到相同的初始状态

    如果sol_sqls总计划成本较低返回1，否则返回0

    注意:
      - 如果old_sqls/sol_sqls包含模式更改或数据修改，
        我们依靠事务回滚在测量另一侧之前丢弃这些更改
      - EXPLAIN不执行查询；它只返回计划和成本估算
      - 这种方法确保两组看到相同的起始状态进行成本比较
    """

    if not old_sqls or not sol_sqls:
        print("Either old_sqls or sol_sqls is empty. Returning 0.")
        return 0
    print(f"Old SQLs are {old_sqls}")
    print(f"New SQLs are {sol_sqls}")

    def measure_sqls_cost(sql_list):
        """
        通过EXPLAIN (查询计划)测量sql_list中每个DML语句的'Total Cost'总和
        非DML语句只执行，但不包括在总成本中
        """
        total_cost = 0.0
        for sql in sql_list:
            upper_sql = sql.strip().upper()
            # 我们只测量SELECT/INSERT/UPDATE/DELETE的DML成本
            if not (
                upper_sql.startswith("SELECT")
                or upper_sql.startswith("INSERT")
                or upper_sql.startswith("UPDATE")
                or upper_sql.startswith("DELETE")
            ):
                print(f"[measure_sqls_cost] Skip EXPLAIN for non-DML: {sql}")
                try:
                    perform_query_on_sqlite_databases(sql, db_path, conn=conn)
                except Exception as exc:
                    print(f"[measure_sqls_cost] Error executing non-DML '{sql}': {exc}")
                continue

            explain_sql = f"EXPLAIN QUERY PLAN {sql}"
            try:
                result_rows, _ = perform_query_on_sqlite_databases(
                    explain_sql, db_path, conn=conn
                )
                if not result_rows:
                    print(f"[measure_sqls_cost] No result returned for EXPLAIN: {sql}")
                    continue

                # SQLite的EXPLAIN QUERY PLAN返回文本描述，而不是JSON
                # 我们需要解析成本信息(如果可用)
                # 对于SQLite，我们可能需要使用不同的方法来估算成本
                # 这里我们使用一个简化的方法
                total_cost_part = 1.0  # 默认成本为1

                total_cost += float(total_cost_part)

            except sqlite3.Error as e:
                print(f"[measure_sqls_cost] SQLite Error on SQL '{sql}': {e}")
            except Exception as e:
                print(f"[measure_sqls_cost] Unexpected error on SQL '{sql}': {e}")

        return total_cost

    # 测量old_sqls的成本
    try:
        perform_query_on_sqlite_databases("BEGIN", db_path, conn=conn)
        old_total_cost = measure_sqls_cost(old_sqls)
        print(f"Old SQLs total plan cost: {old_total_cost}")
    finally:
        perform_query_on_sqlite_databases("ROLLBACK", db_path, conn=conn)

    # 测量sol_sqls的成本
    try:
        perform_query_on_sqlite_databases("BEGIN", db_path, conn=conn)
        sol_total_cost = measure_sqls_cost(sol_sqls)
        print(f"Solution SQLs total plan cost: {sol_total_cost}")
    finally:
        perform_query_on_sqlite_databases("ROLLBACK", db_path, conn=conn)

    # 比较最终成本
    print(
        f"[performance_compare_by_qep] Compare old({old_total_cost}) vs. sol({sol_total_cost})"
    )
    return 1 if sol_total_cost < old_total_cost else 0


def remove_comments(sql_list):
    """
    从列表中的每个查询字符串中删除所有SQL注释
    - 块注释: /* ... */
    - 行注释: -- ... (到行尾)
    还将多个空行折叠为一个，并去除前导/尾随空格
    """
    cleaned = []
    for sql in sql_list:
        # 删除块注释
        no_block = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        # 删除行注释，保留换行符
        no_line = re.sub(r"--.*?(\r\n|\r|\n)", r"\1", no_block)
        # 折叠额外的空行
        no_blank = re.sub(r"\n\s*\n+", "\n", no_line)
        cleaned.append(no_blank.strip())
    return cleaned


def test_case_default(pred_sqls, sol_sqls, db_path, conn, conditions):
    """
    默认test_case: pytest风格的断言
    """
    # 清理查询
    pred_sqls = remove_comments(pred_sqls)
    sol_sqls = remove_comments(sol_sqls)
    pred_sqls = remove_distinct(pred_sqls)
    pred_sqls = remove_round(pred_sqls)
    sol_sqls = remove_distinct(sol_sqls)
    sol_sqls = remove_round(sol_sqls)

    result = ex_base(pred_sqls, sol_sqls, db_path, conn, conditions)
    assert result == 1, f"ex_base returned {result} but expected 1."
    return result


# 注意: 函数名应该是`test_case`，而不是`test_case_default`
TEST_CASE_DEFAULT = """
def test_case(pred_sqls, sol_sqls, db_path, conn, conditions):
   pred_sqls = remove_comments(pred_sqls)
   sol_sqls  = remove_comments(sol_sqls)
   pred_sqls = remove_distinct(pred_sqls)
   pred_sqls = remove_round(pred_sqls)
   sol_sqls  = remove_distinct(sol_sqls)
   sol_sqls  = remove_round(sol_sqls)
   result = ex_base(pred_sqls, sol_sqls, db_path, conn, conditions)
   assert result == 1, f"ex_base returned {result} but expected 1."
   return result
"""
