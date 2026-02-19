from core.job_dedupe import dedupe_jobs


def test_dedupe_jobs_removes_exact_duplicates():
    jobs = [
        {
            "title": "Psychologist",
            "company_name": "ABC",
            "location": "Kuala Lumpur",
            "link": "https://jooble.org/x",
            "extensions": ["via LinkedIn"],
            "apply_options": [{"title": "Apply on LinkedIn", "link": "https://www.linkedin.com/jobs/view/1"}],
        },
        {
            "title": "Psychologist",
            "company_name": "ABC",
            "location": "Kuala Lumpur",
            "link": "https://jooble.org/x",
            "extensions": ["via LinkedIn"],
            "apply_options": [{"title": "Apply on LinkedIn", "link": "https://www.linkedin.com/jobs/view/1"}],
        },
    ]

    out = dedupe_jobs(jobs)
    assert len(out) == 1
