"""
Data Loader - Utilities for loading and merging data from different sources.

This module handles the complexity of different data formats and provides
unified data loading functionality.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dictionaries, one per line.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """
    Save data to a JSONL file.

    Args:
        data: List of dictionaries to save.
        path: Path to the output JSONL file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_source_data(
    source_path: Union[str, Path],
    index_by: str = "instance_id",
) -> Dict[str, Dict[str, Any]]:
    """
    Load source data and index by a key field.

    Args:
        source_path: Path to source JSONL file.
        index_by: Field to use as index key.

    Returns:
        Dictionary mapping index values to data records.
    """
    data = load_jsonl(source_path)
    return {item[index_by]: item for item in data if index_by in item}


def normalize_instance_id(instance_id: str) -> str:
    """
    Normalize instance_id to match source data format.

    Handles variations like 'mental_M_5' -> 'mental_5'.

    Args:
        instance_id: The instance ID to normalize.

    Returns:
        Normalized instance ID.
    """
    if "_M_" in instance_id:
        return instance_id.replace("_M_", "_")
    return instance_id


def merge_with_source(
    current_data: List[Dict[str, Any]],
    source_data: Dict[str, Dict[str, Any]],
    required_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Merge current data with source data by instance_id.

    Args:
        current_data: List of current data records.
        source_data: Dictionary of source data indexed by instance_id.
        required_fields: List of fields to copy from source (None = all).

    Returns:
        List of merged data records.
    """
    if required_fields is None:
        required_fields = [
            "amb_user_query",
            "user_query_ambiguity",
            "knowledge_ambiguity",
            "sol_sql",
            "external_knowledge",
        ]

    merged = []
    for item in current_data:
        instance_id = item.get("instance_id", "")
        normalized_id = normalize_instance_id(instance_id)

        merged_item = item.copy()

        source_item = source_data.get(instance_id) or source_data.get(normalized_id)
        if source_item:
            for field in required_fields:
                if field in source_item and field not in merged_item:
                    merged_item[field] = source_item[field]

        merged.append(merged_item)

    return merged


class DataLoader:
    """
    HuggingFace-style data loader for user simulator data.

    Handles loading data from various formats and merging with source data.

    Example:
        >>> loader = DataLoader(
        ...     source_path="data/bird_interact_data.jsonl",
        ...     db_base_path="data/databases"
        ... )
        >>> data = loader.load("data/data_labeled.jsonl")
        >>> for item in data:
        ...     print(item["instance_id"], item["clarification_question"])
    """

    def __init__(
        self,
        source_path: Optional[Union[str, Path]] = None,
        db_base_path: Optional[Union[str, Path]] = None,
        auto_merge: bool = True,
    ):
        """
        Initialize the data loader.

        Args:
            source_path: Path to source JSONL file with full data.
            db_base_path: Base path to DBs directory.
            auto_merge: Whether to automatically merge with source data.
        """
        self.source_path = Path(source_path) if source_path else None
        self.db_base_path = Path(db_base_path) if db_base_path else None
        self.auto_merge = auto_merge
        self._source_data: Optional[Dict[str, Dict[str, Any]]] = None
        self._schema_cache: Dict[str, str] = {}

    @property
    def source_data(self) -> Dict[str, Dict[str, Any]]:
        """Lazy load and cache source data."""
        if self._source_data is None:
            if self.source_path and self.source_path.exists():
                self._source_data = load_source_data(self.source_path)
            else:
                self._source_data = {}
        return self._source_data

    def load_schema(self, db_name: str) -> str:
        """Load database schema with caching."""
        if db_name not in self._schema_cache:
            if self.db_base_path:
                schema_path = self.db_base_path / db_name / f"{db_name}_schema.txt"
                if schema_path.exists():
                    self._schema_cache[db_name] = schema_path.read_text(encoding="utf-8")
                else:
                    self._schema_cache[db_name] = ""
            else:
                self._schema_cache[db_name] = ""
        return self._schema_cache[db_name]

    def load(
        self,
        path: Union[str, Path],
        merge_source: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load data from a JSONL file.

        Args:
            path: Path to the JSONL file.
            merge_source: Whether to merge with source data (None = use auto_merge).

        Returns:
            List of data records.
        """
        data = load_jsonl(path)

        should_merge = merge_source if merge_source is not None else self.auto_merge

        if should_merge and self.source_data:
            data = merge_with_source(data, self.source_data)

        return data

    def load_with_schemas(
        self,
        path: Union[str, Path],
        merge_source: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load data and add database schemas.

        Args:
            path: Path to the JSONL file.
            merge_source: Whether to merge with source data.

        Returns:
            List of data records with db_schema field added.
        """
        data = self.load(path, merge_source)

        for item in data:
            db_name = item.get("selected_database", "")
            if db_name and "db_schema" not in item:
                item["db_schema"] = self.load_schema(db_name)

        return data

    def iterate(
        self,
        path: Union[str, Path],
        merge_source: Optional[bool] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over data records from a JSONL file.

        Args:
            path: Path to the JSONL file.
            merge_source: Whether to merge with source data.

        Yields:
            Data records one at a time.
        """
        path = Path(path)
        should_merge = merge_source if merge_source is not None else self.auto_merge

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                if should_merge and self.source_data:
                    instance_id = item.get("instance_id", "")
                    if instance_id in self.source_data:
                        source_item = self.source_data[instance_id]
                        for field in ["amb_user_query", "user_query_ambiguity",
                                     "knowledge_ambiguity", "sol_sql", "external_knowledge"]:
                            if field in source_item and field not in item:
                                item[field] = source_item[field]

                yield item


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_data_loader(
    source_path: Optional[Union[str, Path]] = None,
    db_base_path: Optional[Union[str, Path]] = None,
) -> DataLoader:
    """
    Factory function to create a DataLoader.

    Args:
        source_path: Path to source data file.
        db_base_path: Path to DBs directory.

    Returns:
        Configured DataLoader instance.
    """
    return DataLoader(
        source_path=source_path,
        db_base_path=db_base_path,
    )
