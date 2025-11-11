from sqlglot import tokenize
from sqlglot.tokens import TokenType

# Core clauses we’ll slice out when present
_CLAUSE_KEYWORDS = {
    TokenType.SELECT:     "SELECT",
    TokenType.FROM:       "FROM",
    TokenType.WHERE:      "WHERE",
    TokenType.HAVING:     "HAVING",
    TokenType.GROUP_BY:   "GROUP BY",    # SQLGlot treats this as one token 
    TokenType.ORDER_BY:   "ORDER BY",    # likewise 
    TokenType.LIMIT:      "LIMIT",
    TokenType.OFFSET:     "OFFSET",
    TokenType.JOIN:       "JOIN",
    TokenType.STRAIGHT_JOIN: "STRAIGHT JOIN",
    # (You can extend this mapping if you want to catch more multiword joins, e.g. LEFT_OUTER_JOIN, etc.)
}

def segment_sql(sql: str, dialect: str = "sqlite") -> list[tuple[str, str]]:
    """
    Always returns a list of (clause_name, clause_text) for the input SQL.
    
    - If known clauses appear, slices out each one exactly as in the original.
    - On *any* error (unknown token type, lexing glitch, etc.) falls back to splitting
      on semicolons and returning each full statement as ("STATEMENT", stmt).
    """
    try:
        tokens = tokenize(sql, read=dialect)
        starts: list[tuple[int, str]] = []
        
        for tok in tokens:
            name = _CLAUSE_KEYWORDS.get(tok.token_type)
            if name:
                starts.append((tok.start, name))
        
        if not starts:
            # no recognized clauses → treat the whole SQL as one statement
            return [("STATEMENT", sql.strip())]
        
        # build segments by slicing from one clause start to the next
        starts.sort(key=lambda x: x[0])
        segments: list[tuple[str, str]] = []
        
        for idx, (pos, name) in enumerate(starts):
            end = starts[idx + 1][0] if idx + 1 < len(starts) else len(sql)
            seg = sql[pos:end].strip()
            segments.append((name, seg))
        
        return segments
    
    except Exception:
        # Fallback: split on semicolons (preserves each statement roughly)
        parts = [p.strip() for p in sql.split(";")]
        return [("STATEMENT", p + ";" if not p.endswith(";") else p) 
                for p in parts if p]

if __name__ == "__main__":  
    pass
