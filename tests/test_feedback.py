import os

from core.feedback import append_feedback_row


def test_append_feedback_row_creates_file(tmp_path, monkeypatch):
    # Redirect feedback.csv into a temp folder so we don't touch real data/
    fake_csv = tmp_path / "feedback.csv"
    monkeypatch.setattr("core.feedback.FEEDBACK_CSV", str(fake_csv))

    row = {
        "resume_id": "resume123",
        "job_id": "job456",
        "label": 1,
        "semantic_score": 80,
        "lexical_score": 60,
    }

    append_feedback_row(row)

    assert os.path.exists(fake_csv)

    # file should not be empty
    assert fake_csv.read_text(encoding="utf-8").strip() != ""