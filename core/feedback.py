from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict

FEEDBACK_CSV = os.path.join("data", "feedback.csv")


def append_feedback_row(row: Dict[str, Any]) -> None:
    """
    Appends one feedback/training row into data/feedback.csv.
    Creates the file with headers on first write.

    Row should include:
      - resume_id (str)
      - job_id (str)
      - label (int: 1 relevant, 0 not relevant)
      - plus any feature columns you want (semantic_score, lexical_score, etc.)
    """
    os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)

    # Always add timestamp
    row = dict(row)
    row["timestamp_utc"] = datetime.utcnow().isoformat()

    file_exists = os.path.exists(FEEDBACK_CSV)

    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)