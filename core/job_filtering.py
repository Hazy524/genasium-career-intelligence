TRUSTED_JOB_DOMAINS = [
    "linkedin.com",
    "indeed.com",
    "jobstreet.com",
    "glassdoor.com",
    "jobsdb.com",
]

# --- Map label -> domains we consider "matching" that label ---
TRUSTED_JOB_DOMAINS_MAP = {
    "LinkedIn": ["linkedin.com"],
    "Indeed": ["indeed.com"],
    "JobStreet": ["jobstreet.com"],
    "Glassdoor": ["glassdoor.com"],
    "JobsDB": ["jobsdb.com"],
}

def _link_has_any_domain(link: str, domains: list[str]) -> bool:
    if not link:
        return False
    lk = link.lower()
    return any(d in lk for d in domains)

def pick_apply_link_for_source(job: dict, source_label: str) -> str:
    """Pick an apply_options link that matches the displayed source label (if possible)."""
    opts = job.get("apply_options") or []
    domains = TRUSTED_JOB_DOMAINS_MAP.get(source_label, [])
    for opt in opts:
        lnk = (opt.get("link") or "").strip()
        if lnk and _link_has_any_domain(lnk, domains):
            return lnk
    return ""

def has_trusted_destination(job: dict) -> bool:
    """Strict: job must have at least one destination URL in trusted domains."""
    main = (job.get("link") or "").strip()
    if _link_has_any_domain(main, TRUSTED_JOB_DOMAINS):
        return True
    for opt in (job.get("apply_options") or []):
        lnk = (opt.get("link") or "").strip()
        if _link_has_any_domain(lnk, TRUSTED_JOB_DOMAINS):
            return True
    return False
