"""
SQL Parser - Utility for segmenting SQL into clauses.

This module provides functionality to parse SQL statements and segment them
into their constituent clauses (SELECT, FROM, WHERE, etc.).
"""

import re
from typing import List, Tuple


# SQL clause keywords in order of typical appearance
SQL_CLAUSES = [
    "WITH",
    "SELECT",
    "FROM",
    "JOIN",
    "LEFT JOIN",
    "RIGHT JOIN",
    "INNER JOIN",
    "OUTER JOIN",
    "CROSS JOIN",
    "WHERE",
    "GROUP BY",
    "HAVING",
    "ORDER BY",
    "LIMIT",
    "OFFSET",
    "UNION",
    "INTERSECT",
    "EXCEPT",
]


def segment_sql(sql: str) -> List[Tuple[str, str]]:
    """
    Segment a SQL statement into its constituent clauses.

    Args:
        sql: A SQL statement string to segment.

    Returns:
        A list of tuples, where each tuple contains:
        - clause_name: The SQL clause keyword (e.g., "SELECT", "FROM")
        - clause_content: The content of that clause

    Example:
        >>> sql = "SELECT id, name FROM users WHERE active = true"
        >>> segments = segment_sql(sql)
        >>> for clause, content in segments:
        ...     print(f"{clause}: {content}")
        SELECT: id, name
        FROM: users
        WHERE: active = true
    """
    if not sql or not sql.strip():
        return []

    # Normalize whitespace
    sql = " ".join(sql.split())

    segments = []

    # Build pattern to find clause boundaries
    clause_pattern = "|".join([re.escape(c) for c in SQL_CLAUSES])
    pattern = rf'\b({clause_pattern})\b'

    # Find all clause positions
    matches = list(re.finditer(pattern, sql, re.IGNORECASE))

    if not matches:
        # No recognizable clauses, return entire SQL as unknown
        return [("SQL", sql.strip())]

    for i, match in enumerate(matches):
        clause_name = match.group(1).upper()
        start_pos = match.end()

        # Find end position (start of next clause or end of string)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(sql)

        # Extract and clean clause content
        content = sql[start_pos:end_pos].strip()

        # Remove trailing comma if present
        if content.endswith(","):
            content = content[:-1].strip()

        if content:
            segments.append((clause_name, content))

    return segments


def format_sql_segments(sql: str, separator: str = "\n\n") -> str:
    """
    Format SQL into readable segments with clause labels.

    Args:
        sql: A SQL statement string to format.
        separator: String to use between segments.

    Returns:
        Formatted string with labeled clauses.

    Example:
        >>> sql = "SELECT id FROM users WHERE active = true"
        >>> print(format_sql_segments(sql))
        SELECT:
        id

        FROM:
        users

        WHERE:
        active = true
    """
    segments = segment_sql(sql)
    formatted_parts = []

    for clause, content in segments:
        formatted_parts.append(f"{clause}:\n{content}")

    return separator.join(formatted_parts)


def format_multiple_sqls(sqls: List[str], sql_separator: str = "\n===\n") -> str:
    """
    Format multiple SQL statements into readable segments.

    Args:
        sqls: List of SQL statement strings.
        sql_separator: String to use between different SQL statements.

    Returns:
        Formatted string with all SQLs segmented and labeled.
    """
    formatted_sqls = []

    for sql in sqls:
        formatted_sqls.append(format_sql_segments(sql))

    return sql_separator.join(formatted_sqls)


if __name__ == "__main__":
    # Example usage
    test_sql = """
    SELECT f.userregistry, f.nicklabel, ROUND(f.flv, 2) AS flv
    FROM fan_lifetime_value f, percentile_values p
    WHERE f.flv > p.p90
    ORDER BY f.flv DESC
    LIMIT 10
    """

    print("=== SQL Segmentation Example ===")
    print(format_sql_segments(test_sql))
