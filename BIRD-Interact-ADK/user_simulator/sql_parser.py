"""SQL segmentation using sqlglot tokenizer."""

from typing import List, Tuple
from sqlglot import tokenize
from sqlglot.tokens import TokenType

_CLAUSE_KEYWORDS = {
    TokenType.SELECT: "SELECT",
    TokenType.FROM: "FROM",
    TokenType.WHERE: "WHERE",
    TokenType.HAVING: "HAVING",
    TokenType.GROUP_BY: "GROUP BY",
    TokenType.ORDER_BY: "ORDER BY",
    TokenType.LIMIT: "LIMIT",
    TokenType.OFFSET: "OFFSET",
    TokenType.JOIN: "JOIN",
}


def segment_sql(sql: str, dialect: str = "postgres") -> List[Tuple[str, str]]:
    try:
        tokens = tokenize(sql, read=dialect)
        starts = []
        for tok in tokens:
            name = _CLAUSE_KEYWORDS.get(tok.token_type)
            if name:
                starts.append((tok.start, name))
        if not starts:
            return [("STATEMENT", sql.strip())]
        starts.sort(key=lambda x: x[0])
        segments = []
        for idx, (pos, name) in enumerate(starts):
            end = starts[idx + 1][0] if idx + 1 < len(starts) else len(sql)
            segments.append((name, sql[pos:end].strip()))
        return segments
    except Exception:
        parts = [p.strip() for p in sql.split(";")]
        return [("STATEMENT", p + ";" if not p.endswith(";") else p) for p in parts if p]
