from core.job_sources import get_job_source_label
from core.job_sources import get_job_link

def test_detect_linkedin_from_extensions():
    job = {
        "title": "AI Engineer",
        "extensions": ["Full-time", "via LinkedIn"]
    }

    label = get_job_source_label(job)

    assert label == "LinkedIn"

def test_get_job_link_prefers_matching_apply_option():
    job = {
        "link": "https://jooble.org/somejob",  # aggregator main link
        "extensions": ["via LinkedIn"],
        "apply_options": [
            {"title": "Apply on Jooble", "link": "https://jooble.org/apply/123"},
            {"title": "Apply on LinkedIn", "link": "https://www.linkedin.com/jobs/view/999"},
        ],
    }

    link = get_job_link(job)

    assert "linkedin.com/jobs/view" in link