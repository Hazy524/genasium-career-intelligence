# core/job_sources.py

TRUSTED_JOB_DOMAINS = [
    "linkedin.com",
    "indeed.com",
    "jobstreet.com",
    "glassdoor.com",
    "jobsdb.com",
]

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
    """Check if job has at least one trusted destination domain."""
    main = (job.get("link") or "").strip()
    if _link_has_any_domain(main, TRUSTED_JOB_DOMAINS):
        return True

    for opt in (job.get("apply_options") or []):
        lnk = (opt.get("link") or "").strip()
        if _link_has_any_domain(lnk, TRUSTED_JOB_DOMAINS):
            return True

    return False

def get_job_source_label(job: dict) -> str:
    """
    Detect job source label from serpapi/google_jobs result.
    Returns: LinkedIn / Indeed / JobStreet / Glassdoor / JobsDB / Company Site / Other
    """
    parts = []

    # link fields
    for k in ["link", "url", "job_url"]:
        v = job.get(k) or ""
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    # apply options
    for opt in (job.get("apply_options") or []):
        if isinstance(opt, dict):
            lnk = (opt.get("link") or "").strip()
            ttl = (opt.get("title") or "").strip()
            if lnk:
                parts.append(lnk)
            if ttl:
                parts.append(ttl)

    # via/source fields
    for k in ["via", "source"]:
        v = job.get(k) or ""
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    # extensions (google_jobs often puts "via Indeed" here)
    exts = job.get("extensions") or []
    if isinstance(exts, list):
        for e in exts:
            if isinstance(e, str) and e.strip():
                parts.append(e.strip())

    blob = " | ".join(parts).lower()

    if "linkedin" in blob:
        return "LinkedIn"
    if "indeed" in blob:
        return "Indeed"
    if "jobstreet" in blob:
        return "JobStreet"
    if "glassdoor" in blob:
        return "Glassdoor"
    if "jobsdb" in blob:
        return "JobsDB"

    # company ATS patterns
    if any(x in blob for x in ["workday", "greenhouse", "lever", "careers", "myworkdayjobs"]):
        return "Company Site"

    return "Other"

def get_job_link(job: dict) -> str:
    """
    Return the best job URL.
    Priority:
      1) apply_options link that matches detected source label
      2) main job["link"]
      3) first apply_options link
    """
    if not isinstance(job, dict):
        return ""

    source = get_job_source_label(job)

    # 1) prefer matching apply link
    matched_apply = pick_apply_link_for_source(job, source)
    if matched_apply:
        return matched_apply

    # 2) fallback to main link
    main = (job.get("link") or "").strip()
    if main:
        return main

    # 3) fallback to any apply option
    opts = job.get("apply_options") or []
    if isinstance(opts, list) and opts:
        return (opts[0].get("link") or "").strip()

    return ""

def is_trusted_job(job: dict) -> bool:
    """
    Option 2: "Trusted" means attributed to major platforms
    even if the outbound URL opens on an aggregator.
    """
    label = get_job_source_label(job)
    return label in {"LinkedIn", "Indeed", "JobStreet", "Glassdoor", "JobsDB"}
