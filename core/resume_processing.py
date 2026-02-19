def build_full_resume_representation(raw_resume_text: str, memory_bank: dict, role: str = "", core_field: str = "") -> str:
    """
    Creates a single unified resume string for consistent use across:
    - LLM scoring
    - TF-IDF lexical score
    - Vector embeddings (later step)
    """
    raw_resume_text = (raw_resume_text or "").strip()
    memory_bank = memory_bank or {}

    # --- skills (flatten all skill groups)
    skills_flat = []
    skills_dict = memory_bank.get("skills", {}) or {}
    for group in skills_dict.values():
        if isinstance(group, list):
            for s in group:
                if isinstance(s, str) and s.strip():
                    skills_flat.append(s.strip())
    # de-dupe while preserving order
    seen = set()
    skills_flat = [x for x in skills_flat if not (x.lower() in seen or seen.add(x.lower()))]

    # --- experience (role/company/duration + achievements + technologies)
    exp_lines = []
    for e in (memory_bank.get("experience") or []):
        if not isinstance(e, dict):
            continue
        head = " | ".join([str(e.get("role") or "").strip(),
                           str(e.get("company") or "").strip(),
                           str(e.get("duration") or "").strip()]).strip(" |")
        if head:
            exp_lines.append(head)

        ach = e.get("achievements") or []
        if isinstance(ach, list):
            for a in ach[:4]:
                if isinstance(a, str) and a.strip():
                    exp_lines.append(f"- {a.strip()}")

        tech = e.get("technologies_used") or []
        if isinstance(tech, list) and tech:
            tech_clean = [str(t).strip() for t in tech if str(t).strip()]
            if tech_clean:
                exp_lines.append("Tech/Tools: " + ", ".join(tech_clean[:15]))

    # --- education
    edu_lines = []
    for ed in (memory_bank.get("education") or []):
        if not isinstance(ed, dict):
            continue
        line = " | ".join([str(ed.get("degree") or "").strip(),
                           str(ed.get("institution") or "").strip(),
                           str(ed.get("years") or "").strip()]).strip(" |")
        if line:
            edu_lines.append(line)

    # --- projects
    proj_lines = []
    for p in (memory_bank.get("projects") or []):
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()
        desc = str(p.get("description") or "").strip()
        tech = p.get("tech_stack") or []
        tech_clean = [str(t).strip() for t in tech if str(t).strip()] if isinstance(tech, list) else []
        block = " | ".join([x for x in [name, desc] if x])
        if block:
            proj_lines.append(block)
        if tech_clean:
            proj_lines.append("Stack: " + ", ".join(tech_clean[:15]))

    strengths = memory_bank.get("strengths") or []
    weaknesses = memory_bank.get("weaknesses") or []

    strengths = [s.strip() for s in strengths if isinstance(s, str) and s.strip()][:8]
    weaknesses = [s.strip() for s in weaknesses if isinstance(s, str) and s.strip()] [:8]

    # --- final unified string
    out = f"""
RESUME SUMMARY:
Role (detected): {role or "Unknown"}
Core field: {core_field or "Unknown"}

Cleaned Resume Text:
{raw_resume_text}

Skills:
{", ".join(skills_flat) if skills_flat else "None listed"}

Education:
{"; ".join(edu_lines) if edu_lines else "None listed"}

Work Experience:
{"; ".join(exp_lines) if exp_lines else "None listed"}

Projects:
{"; ".join(proj_lines) if proj_lines else "None listed"}

Strengths:
{", ".join(strengths) if strengths else "N/A"}

Weaknesses:
{", ".join(weaknesses) if weaknesses else "N/A"}
""".strip()

    return out
