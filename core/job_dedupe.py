from core.job_sources import get_job_source_label, get_job_link


def dedupe_jobs(all_jobs: list[dict]) -> list[dict]:
    """
    Deduplicate jobs using title+company+location+source+link.
    Keeps the first instance of each unique job.
    """
    unique_jobs = {}
    for j in all_jobs:
        title = (j.get("title") or "").strip().lower()
        company = (j.get("company_name") or "").strip().lower()
        location = (j.get("location") or "").strip().lower()
        source = (get_job_source_label(j) or "").strip().lower()
        link = (get_job_link(j) or j.get("link") or "").strip().lower()

        key = f"{title}|{company}|{location}|{source}|{link}"

        if key not in unique_jobs:
            unique_jobs[key] = j

    return list(unique_jobs.values())
