# DiaGuardianAI Repository Manager
# Manages the storage, retrieval, and updating of "successful" patterns.

import sys
import os
from typing import List, Dict, Any, Optional
import numpy as np # For potential future use with more complex retrieval
import json
import sqlite3 # Added import

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.base_classes import BasePatternRepository

class RepositoryManager(BasePatternRepository):
    """Manages a repository of learned or predefined treatment patterns using SQLite."""
    
    def __init__(self, db_path: str = "patterns.sqlite", **kwargs):
        """Initializes the RepositoryManager with SQLite persistence.
        
        Args:
            db_path (str): Path to the SQLite database file.
            **kwargs: For BasePatternRepository compatibility.
        """
        super().__init__(**kwargs)
        self.db_path: str = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self._connect_db()
        self._create_table()
        self.next_pattern_id: int = self._get_next_pattern_id_from_db()
        
        print(f"RepositoryManager (SQLite) initialized. DB: '{self.db_path}'. Next pattern ID: {self.next_pattern_id}")

    def _connect_db(self):
        """Establishes a connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"RepositoryManager: Connected to SQLite DB at {self.db_path}")
        except sqlite3.Error as e:
            print(f"RepositoryManager: Error connecting to SQLite DB at {self.db_path}: {e}")
            self.conn = None # Ensure conn is None if connection failed
            self.cursor = None

    def _create_table(self):
        """Creates the patterns table if it doesn't exist."""
        if not self.cursor:
            print("RepositoryManager: No DB cursor available. Cannot create table.")
            return
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    data TEXT,
                    state_representation_text TEXT,
                    metadata TEXT,
                    effectiveness_score REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            if self.conn: # Check if connection exists before committing
                self.conn.commit()
            else:
                print("RepositoryManager: No DB connection available to commit table creation.")
        except sqlite3.Error as e:
            print(f"RepositoryManager: Error creating 'patterns' table: {e}")

    def _get_next_pattern_id_from_db(self) -> int:
        """Determines the next pattern ID based on existing data."""
        if not self.cursor:
            return 1
        try:
            self.cursor.execute("SELECT MAX(id) FROM patterns")
            max_id_result = self.cursor.fetchone()
            if max_id_result and max_id_result[0] is not None:
                return int(max_id_result[0]) + 1
            return 1
        except sqlite3.Error as e:
            print(f"RepositoryManager: Error fetching max pattern ID: {e}. Defaulting to 1.")
            return 1
        except ValueError: # Handle case where max_id_result[0] might not be convertible to int
            print(f"RepositoryManager: Error converting max pattern ID. Defaulting to 1.")
            return 1


    def add_pattern(self, pattern_data: Dict[str, Any],
                    metadata: Optional[Dict[str, Any]] = None):
        """Adds a new pattern to the SQLite repository.

        Args:
            pattern_data (Dict[str, Any]): The core data of the pattern.
            metadata (Optional[Dict[str, Any]]): Additional metadata.
        """
        if not self.conn or not self.cursor:
            print("RepositoryManager: No DB connection. Cannot add pattern.")
            return

        pattern_id_to_insert = self.next_pattern_id
        
        try:
            data_json = json.dumps(pattern_data.get("data", {}))
            metadata_json = json.dumps(metadata if metadata else {})
            
            self.cursor.execute("""
                INSERT INTO patterns (id, pattern_type, data, state_representation_text, metadata, effectiveness_score, usage_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_id_to_insert,
                pattern_data.get("pattern_type", "unknown"),
                data_json,
                pattern_data.get("state_representation_text", ""),
                metadata_json,
                0.0, # Initial effectiveness_score
                0    # Initial usage_count
            ))
            self.conn.commit()
            print(f"RepositoryManager: Pattern ID {pattern_id_to_insert} ('{pattern_data.get('pattern_type', 'unknown')}') added to DB.")
            self.next_pattern_id += 1
        except sqlite3.Error as e:
            print(f"RepositoryManager: Error adding pattern ID {pattern_id_to_insert} to DB: {e}")
        except TypeError as e: # For json.dumps errors
            print(f"RepositoryManager: Error serializing pattern data/metadata to JSON for ID {pattern_id_to_insert}: {e}")
        

    def retrieve_relevant_patterns(self, current_state_features: Any,
                                   n_top_patterns: int = 1
                                  ) -> List[Dict[str, Any]]:
        """Retrieves patterns relevant to the current state features from SQLite DB.
        
        - Filters by 'pattern_type' using SQL if present in `current_state_features`.
        - Fetches candidates and then filters in Python by matching other key-value
          pairs from `current_state_features` against the deserialized `pattern['data']`.
        - Returns patterns sorted by effectiveness_score and id (recency).
        """
        if not self.cursor:
            print("RepositoryManager: No DB cursor available. Cannot retrieve patterns.")
            return []

        print(f"RepositoryManager: Retrieving patterns from DB. Query features: {current_state_features}")
        
        query_params = []
        sql_query = "SELECT id, pattern_type, data, state_representation_text, metadata, effectiveness_score, usage_count FROM patterns"
        
        where_clauses = []
        if isinstance(current_state_features, dict):
            query_type = current_state_features.get("pattern_type")
            if query_type:
                where_clauses.append("pattern_type = ?")
                query_params.append(query_type)
        
        if where_clauses:
            sql_query += " WHERE " + " AND ".join(where_clauses)
            
        # Initial sort by score and ID to get potentially relevant candidates
        # We might fetch more than n_top_patterns initially if further Python filtering is needed
        # For simplicity, let's fetch a moderate amount more, e.g., n_top_patterns * 5 or a fixed limit.
        sql_query += " ORDER BY effectiveness_score DESC, id DESC LIMIT ?"
        query_params.append(n_top_patterns * 5 + 10) # Fetch more to allow for python-side filtering

        try:
            self.cursor.execute(sql_query, tuple(query_params))
            rows = self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"RepositoryManager: Error retrieving patterns from DB: {e}")
            return []

        candidate_patterns = []
        for row in rows:
            try:
                pattern = {
                    "id": row[0],
                    "pattern_type": row[1],
                    "data": json.loads(row[2]) if row[2] else {},
                    "state_representation_text": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "effectiveness_score": row[5],
                    "usage_count": row[6]
                }
                candidate_patterns.append(pattern)
            except json.JSONDecodeError as e:
                print(f"RepositoryManager: Error decoding JSON for pattern ID {row[0]} during retrieval: {e}")
                continue # Skip this pattern

        # Further Python-side filtering for non-pattern_type features
        final_filtered_patterns = []
        if isinstance(current_state_features, dict):
            features_to_match = {
                k: v for k, v in current_state_features.items() if k != "pattern_type"
            }
            if features_to_match:
                print(f"  Attempting to match additional features in Python: {features_to_match}")
                for pattern in candidate_patterns:
                    pattern_data_dict = pattern.get("data", {})
                    is_a_match = True
                    if isinstance(pattern_data_dict, dict):
                        for query_key, query_value in features_to_match.items():
                            if query_key not in pattern_data_dict or pattern_data_dict[query_key] != query_value:
                                is_a_match = False
                                break
                    else: # pattern['data'] is not a dict
                        is_a_match = False
                    
                    if is_a_match:
                        final_filtered_patterns.append(pattern)
                print(f"  After Python feature matching, {len(final_filtered_patterns)} candidates.")
            else: # No additional features to match, all SQL-retrieved candidates are final (after sorting)
                final_filtered_patterns = candidate_patterns
        else: # If current_state_features is not a dict, no specific filtering
            final_filtered_patterns = candidate_patterns

        # The SQL query already sorted, but if Python filtering changed the set, re-sort.
        # However, the primary sort was already done by SQL.
        # If we fetched more than n_top_patterns and then filtered, we just need to take the top N.
        return final_filtered_patterns[:n_top_patterns]


    def update_pattern_effectiveness(self, pattern_id: Any,
                                     new_outcome_data: Dict[str, Any]):
        """Updates the stored effectiveness or outcome of a pattern in the SQLite DB.
        
        Args:
            pattern_id (Any): The ID of the pattern to update.
            new_outcome_data (Dict[str, Any]): Dictionary containing data like
                'reward' or 'effectiveness_score', and optionally 'notes'.
        """
        if not self.conn or not self.cursor:
            print("RepositoryManager: No DB connection. Cannot update pattern.")
            return

        try:
            # First, fetch the current score, usage_count, and metadata
            self.cursor.execute("SELECT effectiveness_score, usage_count, metadata FROM patterns WHERE id = ?", (pattern_id,))
            row = self.cursor.fetchone()

            if not row:
                print(f"RepositoryManager: Pattern ID {pattern_id} not found for update.")
                return

            current_score = float(row[0])
            current_usage = int(row[1])
            metadata_json = row[2]
            metadata_dict = json.loads(metadata_json) if metadata_json else {}

            new_score_from_outcome = new_outcome_data.get("reward",
                                                         new_outcome_data.get("effectiveness_score"))
            
            updated_score = current_score
            if new_score_from_outcome is not None:
                # Simple weighted average update: (old_total_score + new_score) / (old_usage + 1)
                updated_score = (current_score * current_usage + float(new_score_from_outcome)) / (current_usage + 1)
            
            updated_usage_count = current_usage + 1
            
            if "notes" in new_outcome_data and isinstance(metadata_dict, dict):
                metadata_dict["last_outcome_notes"] = new_outcome_data["notes"]
            
            updated_metadata_json = json.dumps(metadata_dict)

            self.cursor.execute("""
                UPDATE patterns
                SET effectiveness_score = ?, usage_count = ?, metadata = ?
                WHERE id = ?
            """, (updated_score, updated_usage_count, updated_metadata_json, pattern_id))
            
            self.conn.commit()
            print(f"RepositoryManager: Updated pattern ID {pattern_id} in DB. New score: {updated_score:.2f}, Usage: {updated_usage_count}")

        except sqlite3.Error as e:
            print(f"RepositoryManager: Error updating pattern ID {pattern_id} in DB: {e}")
        except json.JSONDecodeError as e:
            print(f"RepositoryManager: Error decoding/encoding JSON for pattern ID {pattern_id} metadata during update: {e}")
        except TypeError as e: # For json.dumps errors if metadata_dict is not serializable
            print(f"RepositoryManager: Error serializing metadata to JSON for pattern ID {pattern_id} during update: {e}")


    def get_pattern_by_id(self, pattern_id: Any) -> Optional[Dict[str, Any]]:
        """Retrieves a specific pattern by its unique ID from the SQLite database.

        Args:
            pattern_id (Any): The unique identifier of the pattern to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The pattern dictionary if found, else None.
        """
        if not self.cursor:
            print("RepositoryManager: No DB cursor available. Cannot get pattern by ID.")
            return None
        try:
            self.cursor.execute("SELECT id, pattern_type, data, state_representation_text, metadata, effectiveness_score, usage_count FROM patterns WHERE id = ?", (pattern_id,))
            row = self.cursor.fetchone()
            if row:
                pattern = {
                    "id": row[0],
                    "pattern_type": row[1],
                    "data": json.loads(row[2]) if row[2] else {},
                    "state_representation_text": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "effectiveness_score": row[5],
                    "usage_count": row[6]
                }
                return pattern
            return None
        except sqlite3.Error as e:
            print(f"RepositoryManager: Error fetching pattern ID {pattern_id} from DB: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"RepositoryManager: Error decoding JSON for pattern ID {pattern_id} from DB: {e}")
            return None # Or handle more gracefully, e.g., return partial data with warning

    def get_patterns_by_type(self, pattern_type: str) -> List[Dict[str, Any]]:
        """Retrieves all patterns of a specific type from the SQLite database.

        Args:
            pattern_type (str): The type of patterns to retrieve.

        Returns:
            List[Dict[str, Any]]: List of pattern dictionaries.
        """
        if not self.cursor:
            print("RepositoryManager: No DB cursor available. Cannot get patterns by type.")
            return []

        try:
            self.cursor.execute("SELECT id, pattern_type, data, state_representation_text, metadata, effectiveness_score, usage_count FROM patterns WHERE pattern_type = ?", (pattern_type,))
            rows = self.cursor.fetchall()

            patterns = []
            for row in rows:
                pattern = {
                    "id": row[0],
                    "pattern_type": row[1],
                    "data": row[2],  # Keep as string for now
                    "state_representation_text": row[3],
                    "metadata": row[4],  # Keep as string for now
                    "effectiveness_score": row[5],
                    "usage_count": row[6],
                    "timestamp": "",  # Add default timestamp
                    "quality_score": row[5] if row[5] is not None else 0.5  # Use effectiveness_score as quality_score
                }
                patterns.append(pattern)

            return patterns

        except sqlite3.Error as e:
            print(f"RepositoryManager: Error fetching patterns by type '{pattern_type}' from DB: {e}")
            return []

    def get_all_patterns(self) -> List[Dict[str, Any]]:
        """Retrieves all patterns from the SQLite database.

        Returns:
            List[Dict[str, Any]]: List of all pattern dictionaries.
        """
        if not self.cursor:
            print("RepositoryManager: No DB cursor available. Cannot get all patterns.")
            return []

        try:
            self.cursor.execute("SELECT id, pattern_type, data, state_representation_text, metadata, effectiveness_score, usage_count FROM patterns")
            rows = self.cursor.fetchall()

            patterns = []
            for row in rows:
                pattern = {
                    "id": row[0],
                    "pattern_type": row[1],
                    "data": row[2],  # Keep as string for now
                    "state_representation_text": row[3],
                    "metadata": row[4],  # Keep as string for now
                    "effectiveness_score": row[5],
                    "usage_count": row[6],
                    "timestamp": "",  # Add default timestamp
                    "quality_score": row[5] if row[5] is not None else 0.5  # Use effectiveness_score as quality_score
                }
                patterns.append(pattern)

            return patterns

        except sqlite3.Error as e:
            print(f"RepositoryManager: Error fetching all patterns from DB: {e}")
            return []

    def update_pattern(self, pattern_id: str, updated_data: Dict[str, Any]):
        """Updates a pattern with new data.

        Args:
            pattern_id (str): The ID of the pattern to update.
            updated_data (Dict[str, Any]): The updated pattern data.
        """
        if not self.cursor or not self.conn:
            print("RepositoryManager: No DB connection available. Cannot update pattern.")
            return

        try:
            # Update the pattern data
            data_json = updated_data.get("data", "{}")
            quality_score = updated_data.get("quality_score", 0.5)

            self.cursor.execute("""
                UPDATE patterns
                SET data = ?, effectiveness_score = ?
                WHERE id = ?
            """, (data_json, quality_score, pattern_id))

            self.conn.commit()

        except sqlite3.Error as e:
            print(f"RepositoryManager: Error updating pattern {pattern_id}: {e}")
        except Exception as e:
            print(f"RepositoryManager: Unexpected error updating pattern {pattern_id}: {e}")

if __name__ == '__main__':
    print("--- RepositoryManager (SQLite) Standalone Example ---")
    
    example_db_file = "example_patterns.sqlite"
    
    if os.path.exists(example_db_file):
        os.remove(example_db_file)
        print(f"Removed existing '{example_db_file}' for a fresh start.")

    repo = RepositoryManager(db_path=example_db_file)
    
    # Add some patterns
    repo.add_pattern(
        {"pattern_type": "meal_bolus", "data": {"carbs": 50, "bolus_calc": "standard_cr"}},
        {"source": "initial_rules", "meal_type": "breakfast"}
    )
    repo.add_pattern(
        {"pattern_type": "correction_bolus", "data": {"bg_target": 100, "isf": 50}},
        {"source": "expert_system_v1"}
    )
    repo.add_pattern(
        {"pattern_type": "meal_bolus", "data": {"carbs": 75, "bolus_calc": "aggressive_cr"}},
        {"source": "RL_agent_run_123", "meal_type": "lunch"}
    )
    
    # Example: Retrieve all patterns (not implemented yet, would need a new method)
    # print("\nAll patterns after adding (from DB - requires specific retrieval method):")
    # all_db_patterns = repo.get_all_patterns() # Assuming such a method exists
    # for p in all_db_patterns:
    #     print(f"  ID: {p['id']}, Type: {p['pattern_type']}, Data: {p['data']}, Score: {p.get('effectiveness_score',0.0):.2f}, Usage: {p.get('usage_count',0)}")

    # The old save/load methods are no longer applicable for SQLite version.
    # Persistence is handled by DB writes.
    
    # To demonstrate loading, we can close the connection and re-initialize
    if repo.conn:
        repo.conn.close()
        print("\nDB connection closed.")

    print("\n--- Simulating new session: Initializing new RepositoryManager with same DB ---")
    repo_new_session = RepositoryManager(db_path=example_db_file)
    
    # Add another pattern to see if next_pattern_id is correct
    repo_new_session.add_pattern(
        {"pattern_type": "basal_adjustment", "data": {"reason": "high_overnight", "adjustment_factor": 1.1}},
        {"source": "manual_review"}
    )
    
    # Retrieve patterns (example, actual retrieval needs full implementation)
    print("\nRetrieving 'meal_bolus' patterns (top 1) from new session:")
    # This retrieve_relevant_patterns method needs to be updated for SQLite
    # For now, this will likely not work as expected without DB query logic.
    # meal_patterns_new = repo_new_session.retrieve_relevant_patterns({"pattern_type": "meal_bolus"}, n_top_patterns=1)
    # for p in meal_patterns_new:
    #     print(f"  Retrieved: {p}")

    if repo_new_session.conn:
        repo_new_session.conn.close()

    # Clean up the example file
    if os.path.exists(example_db_file):
        os.remove(example_db_file)
        print(f"\nCleaned up '{example_db_file}'.")

    print("\nRepositoryManager (SQLite) example run modified (retrieval/full display needs DB query implementation).")