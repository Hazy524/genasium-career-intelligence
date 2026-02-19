import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd
from groq import Groq
import fitz  # PyMuPDF for faster text extraction
from serpapi import GoogleSearch
import json
import re
from init_chroma import get_chroma
import time
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.job_sources import get_job_source_label, get_job_link, is_trusted_job
from core.logger import get_logger
from core.scoring import stabilize_scores
from core.resume_processing import build_full_resume_representation
from core.job_dedupe import dedupe_jobs
from core.job_filtering import (
    pick_apply_link_for_source,
    has_trusted_destination,
)
logger = get_logger()

st.caption("✅ job_sources module imported")


def clamp_int(x, lo=0, hi=100, default=50) -> int:
    try:
        v = int(float(x))
        return max(lo, min(hi, v))
    except Exception:
        return default

def distances_to_score_and_conf(distances: list[float]) -> tuple[int, int]:
    """
    distances: list of cosine distances (lower = better), usually ~0..1
    Returns: (semantic_score 0-100, confidence 0-100)
    """
    if not distances:
        return 0, 0

    # clamp distances safely
    ds = [max(0.0, min(1.0, float(d))) for d in distances if d is not None]
    if not ds:
        return 0, 0

    ds_sorted = sorted(ds)

    # --- Trimmed mean: drop the worst match (reduces noise)
    if len(ds_sorted) >= 4:
        ds_used = ds_sorted[:-1]   # drop largest distance
    else:
        ds_used = ds_sorted

    avg_d = sum(ds_used) / len(ds_used)

    # --- Score: calibrated (your realism curve)
    raw = (1 - avg_d)
    calibrated = raw ** 1.8
    score = int(round(calibrated * 100))
    score = max(0, min(100, score))

    # --- Confidence: tighter distances = higher confidence
    # Use range as a simple spread proxy (robust + cheap)
    spread = ds_used[-1] - ds_used[0] if len(ds_used) > 1 else 0.0
    spread = max(0.0, min(0.35, spread))           # cap spread impact
    conf = int(round((1 - (spread / 0.35)) * 100)) # 0..100
    conf = max(0, min(100, conf))

    # Penalize low sample size
    if len(ds_used) < 3:
        conf = int(conf * 0.7)

    return score, conf

# --- Semantic Similarity Calculator ---
def compute_semantic_match(job_text, vector_collection):
    """
    Takes a job description, compares it with resume embeddings,
    and returns a similarity score between 0–100.
    """

    if not job_text or len(job_text.strip()) < 40:
        return 0  # no job text = no semantic signal

    try:
        results = vector_collection.query(
            query_texts=[job_text[:2000]],  # keep it stable + cheap
            n_results=5,
            where={"type": "resume_intake"}
        )

        distances = (results.get("distances") or [[]])[0]
        score, _conf = distances_to_score_and_conf(distances)
        return score

    except Exception:
        return 0  # safe fallback
    
def compute_semantic_match_from_job_id(job_id: str, job_text: str, vector_collection, n_results: int = 5) -> int:
    """
    Uses the stored job vector embedding (job_id) to query resume vectors.
    Falls back to text-based semantic match if embedding can't be retrieved.
    """
    if not job_id:
        return compute_semantic_match(job_text, vector_collection)

    try:
        job_row = vector_collection.get(ids=[job_id], include=["embeddings"])
        job_embs = job_row.get("embeddings") or []

        if not job_embs or job_embs[0] is None:
            return compute_semantic_match(job_text, vector_collection)

        job_embedding = job_embs[0]

        results = vector_collection.query(
            query_embeddings=[job_embedding],
            n_results=n_results,
            where={"type": "resume_intake"}
        )

        distances = (results.get("distances") or [[]])[0]
        score, _conf = distances_to_score_and_conf(distances)
        return score

    except Exception:
        return compute_semantic_match(job_text, vector_collection)
    
def compute_semantic_score_and_conf(job_id: str, job_text: str, vector_collection, n_results: int = 5) -> tuple[int, int]:

    """
    Returns (semantic_score 0–100, semantic_confidence 0–100)
    Uses stored job embedding if available, otherwise falls back to text-based query.
    """
    if vector_collection is None:
        return 0, 0
    
    if not job_id:
        score = compute_semantic_match(job_text, vector_collection)
        return score, 50

    try:
        job_row = vector_collection.get(ids=[job_id], include=["embeddings"])
        job_embs = job_row.get("embeddings") or []
        if not job_embs or job_embs[0] is None:
            score = compute_semantic_match(job_text, vector_collection)
            return score, 50

        results = vector_collection.query(
            query_embeddings=[job_embs[0]],
            n_results=n_results,
            where={"type": "resume_intake"}
        )

        distances = (results.get("distances") or [[]])[0]
        return distances_to_score_and_conf(distances)

    except Exception:
        score = compute_semantic_match(job_text, vector_collection)
        return score, 50

def clean_job_description(text: str) -> str:
    """
    Light job-description cleaner:
    - removes excessive whitespace
    - strips weird symbols
    - keeps domain words intact (works for psychology/medical/business/etc)
    - keeps it safe for TF-IDF + LLM + embeddings
    """
    t = (text or "").strip()
    if not t:
        return ""

    # normalize whitespace
    t = re.sub(r"\s+", " ", t)

    # remove repeated junk characters
    t = re.sub(r"[•◦●▪■►]+", " ", t)

    # remove super long dashed separators
    t = re.sub(r"[-=_]{6,}", " ", t)

    # keep it reasonable
    return t.strip()


def job_source_priority(job: dict) -> int:
    source = get_job_source_label(job)

    priority = {
        "LinkedIn": 0,
        "Indeed": 1,
        "JobStreet": 2,
        "Glassdoor": 3,
        "JobsDB": 4,
        "Other": 99
    }

    return priority.get(source, 99)

def make_job_id(j: dict) -> str:
    title = (j.get("title") or "").strip().lower()
    company = (j.get("company_name") or "").strip().lower()
    location = (j.get("location") or "").strip().lower()
    link = (j.get("link") or (j.get("apply_options", [{}])[0].get("link")) or "").strip().lower()
    base = f"{title}|{company}|{location}|{link}"

    stable = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"job_{stable}"

def store_job_vector(j: dict, vector_collection) -> str:

    if vector_collection is None:
        return make_job_id(j)

    job_id = make_job_id(j)
    job_desc_raw = (j.get("description") or "")
    job_desc_clean = clean_job_description(job_desc_raw)

    if len(job_desc_clean) < 80:
        job_desc_clean = clean_job_description(build_job_text(j))

    job_desc = job_desc_clean.strip()

    if len(job_desc) < 40:
        return job_id

    # Skip if already stored
    try:
        existing = vector_collection.get(ids=[job_id])
        if existing and existing.get("ids"):
            return job_id
    except Exception:
        pass

    vector_collection.add(
        ids=[job_id],
        documents=[job_desc[:4000]],
        metadatas=[{
            "type": "job_posting",
            "title": j.get("title"),
            "company": j.get("company_name"),
            "location": j.get("location"),
        }]
    )
    return job_id

def build_job_text(j: dict) -> str:
    desc = (j.get("description") or "").strip()
    if len(desc) >= 120:
        return desc

    title = (j.get("title") or "").strip()
    company = (j.get("company_name") or "").strip()
    location = (j.get("location") or "").strip()

    # SerpAPI sometimes provides extra fields depending on result
    snippet = (j.get("snippet") or "").strip()

    highlights = []
    for h in (j.get("job_highlights") or []):
        items = h.get("items") or []
        highlights.extend([str(x).strip() for x in items if str(x).strip()])

    parts = [
        f"Title: {title}",
        f"Company: {company}",
        f"Location: {location}",
    ]

    if snippet:
        parts.append(f"Snippet: {snippet}")
    if highlights:
        parts.append("Highlights: " + " | ".join(highlights[:12]))

    fallback = "\n".join([p for p in parts if p.strip()])
    return fallback

def repair_job_description(j: dict) -> tuple[str, bool]:
    """
    Normalizes and expands job description using safe fallback data.
    Returns: (job_text, used_fallback)
    """
    raw_desc = (j.get("description") or "").strip()
    title = (j.get("title") or "").strip()
    company = (j.get("company_name") or "").strip()
    location = (j.get("location") or "").strip()

    # If SERP description is short/empty, build a neutral, domain-agnostic placeholder
    used_fallback = len(raw_desc) < 250

    if used_fallback:
        expanded = (
            f"Job Title: {title}\n"
            f"Company: {company}\n"
            f"Location: {location}\n\n"
            "Role overview:\n"
            "Responsibilities may include supporting day-to-day operations, collaborating with stakeholders, "
            "managing tasks and documentation, delivering outcomes aligned to team goals, and maintaining professional standards.\n\n"
            "What employers often look for:\n"
            "Relevant domain knowledge, communication, organization, problem-solving, attention to detail, and reliability."
        )
        return expanded, True

    return raw_desc, False

def compute_lexical_tfidf_score(resume_text: str, job_text: str) -> int:
    """
    TF-IDF cosine similarity between resume_text and job_text.
    Returns a 0–100 score.
    """
    resume_text = (resume_text or "").strip()
    job_text = (job_text or "").strip()

    if len(resume_text) < 50 or len(job_text) < 50:
        return 0

    # keep it stable + fast
    r = resume_text[:6000]
    j = job_text[:6000]

    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)
        X = vec.fit_transform([r, j])  # 0 = resume, 1 = job
        sim = cosine_similarity(X[0], X[1])[0][0]  # 0..1
        score = int(round(sim * 100))
        return max(0, min(100, score))
    except Exception:
        return 0
    
def compute_keyword_overlap_score(memory_bank: dict, job_text: str) -> int:
    """
    Computes keyword overlap between structured memory bank skills
    and job description text.
    Returns score 0–100.
    """
    if not memory_bank or not job_text:
        return 0

    job_text_lower = job_text.lower()

    # collect all skills into one flat list
    skills = []
    skill_groups = memory_bank.get("skills", {})

    for group in skill_groups.values():
        if isinstance(group, list):
            skills.extend(group)

    if not skills:
        return 0

    matched = 0
    for skill in skills:
        if isinstance(skill, str) and skill.lower() in job_text_lower:
            matched += 1

    score = int((matched / len(skills)) * 100)
    return max(0, min(100, score))

def normalize_skill_label(s: str) -> str:
    """
    Normalize skill labels so counts don't split across variants.
    Keeps it lightweight + domain-agnostic.
    """
    if not isinstance(s, str):
        return ""
    x = s.strip().lower()

    # basic cleanup
    x = re.sub(r"[\(\)\[\]\{\}\|/\\]", " ", x)
    x = re.sub(r"[^a-z0-9\+\#\.\- ]+", "", x)  # keep c++, c#, node.js etc.
    x = re.sub(r"\s+", " ", x).strip()

    # small common normalizations (safe + general)
    aliases = {
        "ms excel": "excel",
        "microsoft excel": "excel",
        "powerbi": "power bi",
        "node js": "node.js",
        "nodejs": "node.js",
        "react js": "react",
    }
    return aliases.get(x, x)
    
def analyze_resume_strengths_and_gaps(resume_text: str, client) -> dict:
    """
    One-shot resume analysis. Returns a dict in a fixed JSON schema.
    Keep it stable, realistic, and ATS-style.
    """
    base = (resume_text or "").strip()
    if len(base) < 200:
        return {
            "ats_score": 0,
            "seniority": "Unknown",
            "experience_depth": 0,
            "missing_hard_skills": [],
            "missing_soft_skills": [],
            "missing_keywords": [],
            "improvement_tips": ["Resume text too short to analyze reliably."],
            "growth_plan": []
        }

    # Keep prompt cheap + stable
    sample = base[:4500]
    st.session_state["resume_analysis_last_error"] = ""

    core_field = (st.session_state.get("core_field") or "").strip()

    prompt = f"""
You are a Resume Intelligence Engine.

You MUST adapt your analysis to the candidate's domain.
Candidate domain (if provided): {core_field if core_field else "Infer from resume"}

Return ONLY a JSON object with this exact structure and types:

{{
  "ats_score": 0-100,
  "seniority": "Entry Level|Intermediate|Advanced|Senior|Lead / Specialist",
  "experience_depth": 0-100,
  "missing_hard_skills": ["..."],
  "missing_soft_skills": ["..."],
  "missing_keywords": ["..."],
  "improvement_tips": ["..."],
  "growth_plan": ["..."]
}}

Rules (critical):
- Output must be valid JSON. No extra keys.
- ats_score: readability, structure, keyword coverage, section completeness.
- experience_depth: quantify impact, action verbs, outcomes, and depth (domain-appropriate).
- DO NOT recommend software/AI/programming/cloud skills unless the resume clearly shows a tech role.
- missing_hard_skills MUST be domain-relevant (psychology/medical/business/fashion/etc).
- missing_keywords MUST be domain keywords expected in that field.
- growth_plan MUST be realistic for that field (certs/tools/courses typical for that career).
- Keep lists concise (max 8 items each). Deduplicate items.

RESUME TEXT:
{sample}
"""


    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)

        # light safety clamps
        data["ats_score"] = clamp_int(data.get("ats_score"), default=50)
        data["experience_depth"] = clamp_int(data.get("experience_depth"), default=50)

        if not isinstance(data.get("missing_hard_skills"), list): data["missing_hard_skills"] = []
        if not isinstance(data.get("missing_soft_skills"), list): data["missing_soft_skills"] = []
        if not isinstance(data.get("missing_keywords"), list): data["missing_keywords"] = []
        if not isinstance(data.get("improvement_tips"), list): data["improvement_tips"] = []
        if not isinstance(data.get("growth_plan"), list): data["growth_plan"] = []

        if not isinstance(data.get("seniority"), str) or not data["seniority"].strip():
            data["seniority"] = "Unknown"

        return data

    except Exception as e:
        st.session_state["resume_analysis_last_error"] = str(e)
        return {
            "ats_score": 50,
            "seniority": "Unknown",
            "experience_depth": 50,
            "missing_hard_skills": [],
            "missing_soft_skills": [],
            "missing_keywords": [],
            "improvement_tips": [f"Resume analysis temporarily unavailable: {type(e).__name__}"],
            "growth_plan": []
        }


def extract_resume_memory(raw_text: str, client) -> dict:
    """
    Extracts structured resume memory (skills, experience, education, projects, strengths/weaknesses).
    Returns JSON-safe dict only.
    Uses Groq once.
    """

    text = (raw_text or "").strip()
    if len(text) < 200:
        return {
            "skills": {
                "programming_languages": [],
                "frameworks": [],
                "ai_ml_tools": [],
                "cloud_platforms": [],
                "soft_skills": []
            },
            "experience": [],
            "education": [],
            "projects": [],
            "strengths": [],
            "weaknesses": []
        }

    # keep prompt stable + cheap
    sample = text[:5000]

    prompt = f"""
You are a Resume Knowledge Graph Extractor.
Return ONLY a JSON object in this exact structure:

{{
  "skills": {{
    "programming_languages": [],
    "frameworks": [],
    "ai_ml_tools": [],
    "cloud_platforms": [],
    "soft_skills": []
  }},
  "experience": [
    {{
      "role": "",
      "company": "",
      "duration": "",
      "achievements": ["", ""],
      "technologies_used": []
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "years": ""
    }}
  ],
  "projects": [
    {{
      "name": "",
      "description": "",
      "tech_stack": []
    }}
  ],
  "strengths": [],
  "weaknesses": []
}}

Rules:
- Output MUST be valid JSON (no markdown).
- Do NOT add extra keys.
- Keep lists clean and deduplicated.
- Keep each list max 10 items.
- experience achievements: max 3 bullets per role.
- If unknown, use "" or [].

RESUME TEXT:
{sample}
"""

    try:
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)

        # --- Minimal schema hardening (avoid crashes) ---
        if not isinstance(data, dict):
            raise ValueError("Memory bank output is not a dict")

        if "skills" not in data or not isinstance(data["skills"], dict):
            data["skills"] = {}

        for k in ["programming_languages", "frameworks", "ai_ml_tools", "cloud_platforms", "soft_skills"]:
            if not isinstance(data["skills"].get(k), list):
                data["skills"][k] = []

        if not isinstance(data.get("experience"), list): data["experience"] = []
        if not isinstance(data.get("education"), list): data["education"] = []
        if not isinstance(data.get("projects"), list): data["projects"] = []
        if not isinstance(data.get("strengths"), list): data["strengths"] = []
        if not isinstance(data.get("weaknesses"), list): data["weaknesses"] = []

        return data

    except Exception:
        return {
            "skills": {
                "programming_languages": [],
                "frameworks": [],
                "ai_ml_tools": [],
                "cloud_platforms": [],
                "soft_skills": []
            },
            "experience": [],
            "education": [],
            "projects": [],
            "strengths": [],
            "weaknesses": []
        }

# --- 1. THE OS CORE ---
st.set_page_config(page_title="GENASIUM OS", layout="wide", page_icon="📟")

# --- 1. Deep State Initialization
if "logic" not in st.session_state:
    st.session_state.update({
        "logic": 0, "creative": 0, "leadership": 0,
        "job": "System Standby", "resume_text": "",
        "messages": [], "file_id": None
    })

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "semantic_cache" not in st.session_state:
    st.session_state.semantic_cache = {}

if "resume_analysis_cache" not in st.session_state:
    st.session_state.resume_analysis_cache = {}

if "lexical_cache" not in st.session_state:
    st.session_state.lexical_cache = {}

if "memory_bank" not in st.session_state:
    st.session_state.memory_bank = None

if "resume_fingerprint" not in st.session_state:
    st.session_state.resume_fingerprint = None

if "full_resume_representation" not in st.session_state:
    st.session_state.full_resume_representation = ""

if "full_resume_rep_cache" not in st.session_state:
    st.session_state.full_resume_rep_cache = {}

# Optional: cache per resume fingerprint so you don’t call Groq again for the same resume
if "memory_bank_cache" not in st.session_state:
    st.session_state.memory_bank_cache = {}

if "global_skill_stats" not in st.session_state:
    st.session_state.global_skill_stats = {
        "matched": {},   # skill -> count
        "missing": {},   # skill -> count
        "jobs_count": 0  # how many jobs contributed
    }


# --- 2. ChromaDB Initialization ---
try:
    chroma_client, vector_collection = get_chroma()
except Exception:
    vector_collection = None
    st.sidebar.warning("Vault Offline: ChromaDB not initialized.")

# --- AI & SEARCH CONFIG ---
# Using get for safer secrets handling
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

st.write("DEBUG GROQ key loaded:", "YES" if GROQ_API_KEY else "NO")
st.write("DEBUG GROQ key prefix:", (GROQ_API_KEY[:6] + "..." if GROQ_API_KEY else "None"))

if not GROQ_API_KEY or not SERPAPI_API_KEY:
    st.error("Credential Error: Missing API keys in secrets.toml.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# --- 2. THE OS SIDEBAR ---
with st.sidebar:
    st.title("📟 GENASIUM OS")
    st.caption("Version 2.1.1 - Secure AI Intake")
    st.markdown("---")
    
    if st.button("🔄 Hard System Reboot"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Data Intake")
    resume_file = st.file_uploader("Upload Profile (PDF)", type="pdf")
    file_bytes = None
    if resume_file:
        file_bytes = resume_file.getvalue()
        current_hash = hashlib.sha1(file_bytes).hexdigest()[:16]

        # detect real resume change (not just filename)
        if st.session_state.get("file_hash") != current_hash:
            st.session_state.update({
                "logic": 0, "creative": 0, "leadership": 0,
                "resume_text": "",
                "job": "Analyzing...",
                "messages": [],
                "file_hash": current_hash,
                "file_id": resume_file.name
            })

    
    # --- FEATURE: 500KB CAPACITY LIMIT ---
    if resume_file:
        file_size_kb = resume_file.size / 1024
        if file_size_kb > 500:
            st.error(f"❌ File too large ({file_size_kb:.1f}KB). Max limit is 500KB.")
            # This prevents further execution if the file is too big
            resume_file = None 
            st.stop()

    if resume_file and st.session_state.get("file_id") != resume_file.name:
        st.session_state.update({
            "logic": 0, "creative": 0, "leadership": 0, 
            "resume_text": "", "job": "Analyzing...", 
            "file_id": resume_file.name
        })

    market_mode = st.selectbox("Market Mode", ["Internship", "Full-Time", "Freelance"])

    # --- CLEAN SYSTEM MONITOR (Replaces Vault Monitor) ---
    with st.sidebar.expander("🛠️ System Diagnostics"):
        if vector_collection is None:
            st.error("Vault Connection: OFFLINE")
        else:
            vault_stats = vector_collection.count()
            st.caption(f"Vector Database: **ACTIVE**")
            st.caption(f"Knowledge Fragments: **{vault_stats}**")
            st.caption(f"Engine: **Llama-3.3-70B**")

            if vault_stats > 0:
                st.success("Precision Retrieval Enabled", icon="🎯")

    st.markdown("---")
    st.caption("© 2026 GENASIUM OS | Secure Career Intelligence")

# --- 3. THE "INTELLIGENT" GATEKEEPER (MERGED & UPGRADED) ---
if resume_file and st.session_state.resume_text == "":
    with st.spinner("📟 System analyzing document authenticity & calibrating metrics..."):
        try:
            # A. Extract Text
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            raw_text = " ".join([page.get_text() for page in doc]).strip()
            st.session_state.resume_fingerprint = hashlib.sha1(raw_text[:4500].encode("utf-8")).hexdigest()[:16]

                        # --- SAFETY: if Chroma is offline, we can still validate resume,
            # but we cannot store chunks. Stop early to avoid Gatekeeper Error.
            if vector_collection is None:
                st.error("Vault Offline: cannot store resume vectors. Please restart the app or fix ChromaDB path.")
                st.stop()

            # B. SUSTAINED FEATURE: HARD LOGIC ANCHOR CHECK
            # (Saves API costs by rejecting junk before even calling the AI)
            anchors = ["education", "experience", "skills", "projects", "work", "contact", "summary", "profile"]
            found_anchors = [a for a in anchors if a in raw_text.lower()]
            
            if len(found_anchors) < 2 and len(raw_text) < 2000: 
                st.error("🚫 Document Rejected: Non-Resume Detected.")
                st.info("The system couldn't find standard sections like 'Experience' or 'Education'.")
                st.stop()

            # C. UPGRADED: AI REASONING VIA JSON
            gatekeeper_prompt = f"""
            You are a Career Intelligence Model trained to evaluate resumes.
        Return ONLY a JSON object in this exact structure:

        {{
        "is_resume": "YES",
        "quality_score": 0,
        "title": "",
        "core_field": "",
        "justification": ["", "", ""],
        "scores": {{
            "logic": 0,
            "creative": 0,
            "leadership": 0
        }}
        }}

        Rules:
        - Do NOT include explanations outside the JSON.
        - Fill values realistically.
        - All numbers must be integers.
        - justification must contain exactly 3 bullet points.

        Resume Text:
        {raw_text[:4000]}
        """
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": gatekeeper_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            res_data = json.loads(response.choices[0].message.content)

            # D. SUSTAINED FEATURE: REJECTION LOGIC
            if res_data.get("is_resume") == "NO":
                st.error("⚠️ Document Rejected: AI Audit failed.")
                # We use the first bullet point from justification as the reason
                reason = res_data.get('justification', ["No reason provided"])[0]
                st.info(f"**AI Reason:** {reason}")
                st.stop()

            # E. SUSTAINED FEATURE: LOW MATCH WARNING (The tips)
            q_score = res_data.get("quality_score", 0)
            if q_score < 6:
                st.warning(f"⚠️ **Low Match Warning:** Profile seems weak ({q_score}/10).")
                with st.expander("🛠️ Recommended Fixes"):
                    st.write("- Use 'Action Verbs' (Developed, Engineered).")
                    st.write("- Quantify results (e.g., 'Improved speed by 20%').")

            # Clean unreliable AI titles
# F. STATE UPDATE & VECTOR STORAGE (With Chunking & Auto-Purge)
            raw_role = res_data.get("title", "Professional")
            clean_role = re.sub(r"(?i)curriculum vitae.*", "", raw_role).strip() or "Professional"

            # 1. THE CHUNKER: Slicing the text for better AI "vision"
            chunk_size = 600
            overlap = 150
            # ✅ Always use a non-empty resume source for embeddings
            resume_for_vectors = (
                st.session_state.get("full_resume_representation") or raw_text
            )

            chunks = [
                resume_for_vectors[i:i + chunk_size]
                for i in range(0, len(resume_for_vectors), chunk_size - overlap)
            ]



            # 2. AUTO-PURGE: Clear previous resume chunks to avoid name-swapping
            existing_docs = vector_collection.get(where={"type": "resume_intake"})
            if existing_docs and len(existing_docs['ids']) > 0:
                vector_collection.delete(ids=existing_docs['ids'])

            # 3. VECTOR STORAGE LOOP: Save each slice separately
            import time
            current_ts = int(time.time())
            
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{resume_file.name}_chunk_{idx}_{current_ts}"
                vector_collection.add(
                    documents=[chunk], 
                    ids=[chunk_id],
                    metadatas=[{
                        "role": clean_role, 
                        "type": "resume_intake",
                        "upload_time": current_ts,
                        "chunk_index": idx
                    }]
                )

            # 4. UPDATE SESSION STATE (Keeping your scores and UI happy)
            st.session_state.update({
                "logic": res_data["scores"]["logic"],
                "creative": res_data["scores"]["creative"],
                "leadership": res_data["scores"]["leadership"],
                "job": clean_role,
                "core_field": (res_data.get("core_field") or "").strip(),
                "resume_text": raw_text
            })

            # --- Step 5 + Step 6: Resume fingerprint, Memory Bank, Full Resume Representation (ONE clean pipeline) ---
            resume_text_full = st.session_state.get("resume_text", "") or ""
            resume_fingerprint = hashlib.sha1(resume_text_full[:4500].encode("utf-8")).hexdigest()[:16]
            st.session_state.resume_fingerprint = resume_fingerprint  # keep it globally available

            core_field = (st.session_state.get("core_field") or "").strip()
            role_detected = (st.session_state.get("job") or "").strip()

            # 1) Memory bank (cached)
            if resume_fingerprint in st.session_state.memory_bank_cache:
                mb = st.session_state.memory_bank_cache[resume_fingerprint]
            else:
                mb = extract_resume_memory(resume_text_full, client)
                st.session_state.memory_bank_cache[resume_fingerprint] = mb

            st.session_state.memory_bank = mb

            # 2) Full resume representation (cached)
            if resume_fingerprint in st.session_state.full_resume_rep_cache:
                rep = st.session_state.full_resume_rep_cache[resume_fingerprint]
            else:
                rep = build_full_resume_representation(
                    raw_resume_text=resume_text_full,   # ✅ correct arg name
                    memory_bank=mb,
                    role=role_detected,
                    core_field=core_field
                )
                st.session_state.full_resume_rep_cache[resume_fingerprint] = rep

            st.session_state.full_resume_representation = rep or resume_text_full
            # harden: never allow blank rep
            st.session_state.full_resume_representation = (st.session_state.full_resume_representation or "").strip() or resume_text_full

            # Show a little success badge for the chunks
            st.sidebar.caption(f"🧬 Profile fragmented into {len(chunks)} searchable vectors.")
                
            justification_str = ", ".join(res_data.get("justification", []))
            st.sidebar.success(f"🎯 **Evidence:** {justification_str}")
                
            st.rerun()

        except Exception as ex:
            st.error(f"Gatekeeper Error: {ex}")

# --- 4. THE DASHBOARD ---
st.title("🚀 Career Intelligence Dashboard")

with st.expander("🧾 Debug: Full Resume Representation"):
    rep = (st.session_state.get("full_resume_representation") or "").strip()
    fallback = (st.session_state.get("resume_text") or "").strip()
    st.text((rep or fallback)[:1500])

if st.session_state.resume_text:
        # --- Step 4: Resume Strength Analysis (cached per resume) ---
    resume_text_full = st.session_state.get("resume_text", "") or ""
    resume_fingerprint = hashlib.sha1(resume_text_full[:4500].encode("utf-8")).hexdigest()[:16]

    fp = st.session_state.get("resume_fingerprint") or resume_fingerprint

    if fp and fp in st.session_state.resume_analysis_cache:
        resume_analysis = st.session_state.resume_analysis_cache[fp]
    else:
        resume_analysis = analyze_resume_strengths_and_gaps(resume_text_full, client)
        if fp:
            st.session_state.resume_analysis_cache[fp] = resume_analysis

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Pillar Analysis")
        chart_data = pd.DataFrame({
            "Pillar": ["Logic", "Creative", "Leadership"],
            "Score": [st.session_state.logic, st.session_state.creative, st.session_state.leadership]
        }).set_index("Pillar")
        st.bar_chart(chart_data)

    with col2:
        st.subheader("System Forecast")
        st.metric("Analyzed Role", st.session_state.job)
        st.info(f"OS recommends pursuing {st.session_state.job} roles in the {market_mode} sector.")

    st.divider()

    st.subheader("🔍 Resume Strength Analysis & ATS Score")

    ats = resume_analysis.get("ats_score", 50)
    depth = resume_analysis.get("experience_depth", 50)
    seniority = resume_analysis.get("seniority", "Unknown")

    a1, a2, a3 = st.columns(3)
    with a1:
        st.caption("ATS Compatibility")
        st.progress(ats / 100)
        st.write(f"**{ats}/100**")

    with a2:
        st.caption("Experience Depth")
        st.progress(depth / 100)
        st.write(f"**{depth}/100**")

    with a3:
        st.caption("Seniority (AI Estimate)")
        st.info(seniority)

    left, right = st.columns(2)

    with left:
        st.markdown("### ❌ Missing Hard Skills")
        for x in (resume_analysis.get("missing_hard_skills") or []):
            st.write(f"- {x}")

        st.markdown("### ❌ Missing Soft Skills")
        for x in (resume_analysis.get("missing_soft_skills") or []):
            st.write(f"- {x}")

    with right:
        st.markdown("### 🔑 Missing Keywords")
        for x in (resume_analysis.get("missing_keywords") or []):
            st.write(f"- {x}")

        st.markdown("### 🛠️ Improvement Tips")
        for x in (resume_analysis.get("improvement_tips") or []):
            st.write(f"- {x}")

    with st.expander("📈 Growth Plan (AI Generated)"):
        for x in (resume_analysis.get("growth_plan") or []):
            st.write(f"- {x}")

    with st.expander("🛠 Debug: Resume Analysis Error", expanded=False):
        err = st.session_state.get("resume_analysis_last_error", "")
        if err:
            st.code(err)
        else:
            st.caption("No error captured.")

    with st.expander("🧪 Debug: Last LLM Error (Job Match)", expanded=False):
        err = st.session_state.get("last_llm_error", "")
        if err:
            st.code(err)
        else:
            st.caption("No error captured.")

    with st.expander("🧠 Resume Memory Bank"):
        mb = st.session_state.get("memory_bank", None)
        st.write("DEBUG memory_bank:", st.session_state.get("memory_bank"))
        if mb is not None:
            st.json(mb)
        else:
            st.info("Memory bank not generated yet.")

    st.divider()


# --- 5. MALAYSIAN CAREER PATH & JOB ENGINE (V8 - FLEXIBLE CITY FILTER + SMART FALLBACK) ---
st.subheader("🇲🇾 Targeted Malaysian Job Search")

CITY_OPTIONS = [
    "Kuala Lumpur", "Petaling Jaya", "Shah Alam", "Cyberjaya",
    "George Town", "Johor Bahru", "Ipoh", "Melaka",
    "Kuching", "Kota Kinabalu"
]

location_input = st.selectbox("📍 Select City in Malaysia:", CITY_OPTIONS)

trusted_only = st.sidebar.toggle(
    "✅ Show jobs attributed to LinkedIn / Indeed / JobStreet",
    value=True
)


# --- MALAYSIAN CITY + ALIASES MAP ---
CITY_MAP = {
    "kuala lumpur": ["kl", "ampang", "cheras", "setapak", "wangsa maju", "kerinchi", "bangsar"],
    "petaling jaya": ["pj", "subang", "damansara", "sunway", "kota damansara"],
    "shah alam": ["uitm", "seksyen 7", "seksyen 13"],
    "george town": ["penang", "pulau pinang", "butterworth", "seberang jaya"],
    "johor bahru": ["jb", "skudai", "nusajaya"],
    "ipoh": ["perak"],
    "melaka": ["malacca"],
    "kuching": ["sarawak"],
    "kota kinabalu": ["sabah"],
    "cyberjaya": ["mmu", "mdec", "setia eco glades"]
}

def normalize_city(city):
    c = city.lower().strip()
    for main, alts in CITY_MAP.items():
        if c == main or c in alts:
            return main
    return c

# --- BUTTON ---
if st.button("Find Placements"):
    # Clear old job results so each search is fresh
    st.session_state.pop("last_jobs", None)
    st.session_state.pop("jobs_cache", None)
    st.session_state.pop("all_jobs", None)
    st.session_state.pop("filtered_jobs", None)
    st.session_state.pop("candidate_jobs", None)

    st.session_state.searching = True
    # --- Heatmap reset for this search run ---
    st.session_state.global_skill_stats = {"matched": {}, "missing": {}, "jobs_count": 0}
    raw_job = st.session_state.get("job", "Professional")
    clean_job = " ".join(raw_job.split()[:3])

    if not (st.session_state.get("resume_text") or "").strip():
        st.warning("Please upload a resume first so I can infer your target role accurately.")
        st.stop()

    user_city = normalize_city(location_input)
    st.write(f"DEBUG: Normalized city → {user_city}")

    # --- SERPAPI SEARCH FUNCTION ---
    def serp_search(city_query):
        # Try progressively broader queries (fixes "no jobs" for weird titles)
        role_words = clean_job.split()
        short_role = " ".join(role_words[:2]) if role_words else clean_job

        query_variants = [
            f"{clean_job} {market_mode} jobs {city_query}",
            f"{short_role} {market_mode} jobs {city_query}",
            f"{clean_job} jobs {city_query}",
            f"{short_role} jobs {city_query}",
            f"{market_mode} jobs {city_query}",
            f"entry level {short_role} jobs {city_query}" if market_mode != "Freelance" else f"{short_role} contract jobs {city_query}",
        ]

        last_result = {}
        for q in query_variants:
            search = GoogleSearch({
                "q": q,
                "engine": "google_jobs",
                "google_domain": "google.com.my",
                "gl": "my",
                "hl": "en",
                "location": city_query,
                "api_key": SERPAPI_API_KEY
            })
            result = search.get_dict()
            last_result = result

            jobs = result.get("jobs_results", [])
            if jobs:
                return result  # ✅ return first successful query

        return last_result  # return last for debugging


    # --- SEARCH CHAIN (user city → state/metro → Malaysia fallback) ---
    search_chain = [user_city]

    # add nearby state/metro fallback for major metro areas
    if user_city in ["kuala lumpur", "petaling jaya", "cyberjaya"]:
        search_chain.append("Selangor")
    if user_city == "george town":
        search_chain.append("Penang")
    
    search_chain.append("Malaysia")  # last fallback

    all_jobs = []
    for area in search_chain:
        st.write(f"DEBUG: Searching → {area}")
        try:
            result = serp_search(area)
            jobs = result.get("jobs_results", [])
            for j in jobs:
                if j not in all_jobs:
                    all_jobs.append(j)
            if len(all_jobs) >= 10:
                break
        except Exception:
            continue

    all_jobs = dedupe_jobs(all_jobs)

    # --- SORT BY SOURCE PRIORITY (LinkedIn/Indeed/etc first) ---
    all_jobs = sorted(all_jobs, key=job_source_priority)

    # --- OPTIONAL: TRUSTED SOURCES ONLY (STRICT) ---
    if trusted_only:
        st.write("DEBUG jobs before trusted filter:", len(all_jobs))
        all_jobs = [j for j in all_jobs if is_trusted_job(j)]
        st.write("DEBUG jobs after trusted filter:", len(all_jobs))
        if not all_jobs:
            st.warning("No trusted-source jobs found for this query/city. Try a different city/title or broaden the query.")
            st.stop()

    if not all_jobs:
        st.warning("⚠️ No trusted-source jobs found (LinkedIn/Indeed/JobStreet/Glassdoor/JobsDB) for this search. Try a different city/title or broaden the query.")

    # No st.stop() here!

    # --- FLEXIBLE LOCATION FILTER ---
    def job_matches_city(job, city):
        loc = (job.get("location") or "").lower()
        city = city.lower()

        # exact city or alias
        for main, alts in CITY_MAP.items():
            if city == main:
                if main in loc or any(a in loc for a in alts):
                    return True

        # state-level fallback for major metro
        if city in ["kuala lumpur", "petaling jaya", "cyberjaya"]:
            if "selangor" in loc:
                return True
        if city == "george town" and "penang" in loc:
            return True

        return False

    filtered_jobs = [j for j in all_jobs if job_matches_city(j, user_city)]

    # --- IF NOT ENOUGH CITY JOBS, INCLUDE STATE FALLBACKS ---
    if len(filtered_jobs) < 10:
        remaining = 10 - len(filtered_jobs)
        # include other jobs from all_jobs not already in filtered_jobs
        for j in all_jobs:
            if j not in filtered_jobs:
                filtered_jobs.append(j)
            if len(filtered_jobs) >= 10:
                break

    # --- avoid repeating jobs already shown in this session ---
    shown = st.session_state.get("shown_job_keys", set())
    fresh = []
    for j in all_jobs:
        title = (j.get("title") or "").strip().lower()
        company = (j.get("company_name") or "").strip().lower()
        location = (j.get("location") or "").strip().lower()
        key = f"{title}|{company}|{location}"

        if key in shown:
            continue
        fresh.append(j)

    # If everything is a repeat, fall back to original list
    if fresh:
        all_jobs = fresh

    # Score MORE than 10 first, then rank by match score
    candidate_jobs = filtered_jobs[:30]

    st.success(f"✅ Found {len(candidate_jobs)} jobs in **{user_city.title()}** (including nearby areas if needed).")
    # Reset global heatmap stats for THIS search run
    st.session_state.global_skill_stats = {"matched": {}, "missing": {}, "jobs_count": 0}

    scored_rows = []

    # Ensure memory bank exists once resume text exists
    if "build_memory_bank" in globals():
        if (st.session_state.get("resume_text") or st.session_state.get("full_resume_representation")) and not st.session_state.get("memory_bank"):
            st.warning("Generating resume memory bank...")
            st.session_state["memory_bank"] = build_memory_bank(
                st.session_state.get("full_resume_representation") or st.session_state.get("resume_text", "")
            )

    # --- SCORE JOBS ---
    for j in candidate_jobs:

        job_id = store_job_vector(j, vector_collection)
        job_desc, used_fallback = repair_job_description(j)

        job_desc_clean = clean_job_description(job_desc)
        job_desc_clean = (job_desc_clean or "").strip()

        job_text = clean_job_description(build_job_text(j))

        resume_text_full = st.session_state.get("full_resume_representation") or st.session_state.get("resume_text", "")

        # Fallback: if description is empty, still give the engines something stable
        if len(job_desc_clean) < 80:
            job_desc_clean = clean_job_description(build_job_text(j))

        if job_id in st.session_state.lexical_cache:
            lexical_score = st.session_state.lexical_cache[job_id]
        else:
            resume_for_matching = st.session_state.get(
                "full_resume_representation",
                st.session_state.get("resume_text", "")
            ) or ""

            tfidf_score = compute_lexical_tfidf_score(resume_text_full, job_text)
            keyword_score = compute_keyword_overlap_score(st.session_state.get("memory_bank"), job_text)

            lexical_score = int(round((0.7 * tfidf_score) + (0.3 * keyword_score)))
            lexical_score = max(0, min(100, lexical_score))
            st.session_state.lexical_cache[job_id] = lexical_score

        # ✅ use stable job_id for caching
        if job_id in st.session_state.semantic_cache:
            semantic_score, semantic_conf = st.session_state.semantic_cache[job_id]
        else:
            semantic_score, semantic_conf = compute_semantic_score_and_conf(job_id, job_text, vector_collection)
            st.session_state.semantic_cache[job_id] = (semantic_score, semantic_conf)

        # --- INSIDE THE JOB LOOP ---
        match_prompt = f"""
        Compare the Resume and Job Description. Return a JSON object:
        {{
            "score": 0-100,
            "hard_skills_score": 0-100,
            "soft_skills_score": 0-100,
            "market_demand_score": 0-100,
            "matched_skills": ["skill1", "skill2"],
            "missing_skills": ["skill1", "skill2"]
        }}
        Resume: {st.session_state.get("full_resume_representation", st.session_state.resume_text)[:1400]}
        Job: {job_text[:1200]}
        """

        try:
            m_res = client.chat.completions.create(
                messages=[{"role": "user", "content": match_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            m_data = json.loads(m_res.choices[0].message.content)

            score = clamp_int(m_data.get("score"), default=50)
            h_score = clamp_int(m_data.get("hard_skills_score"), default=50)
            s_score = clamp_int(m_data.get("soft_skills_score"), default=50)
            m_score = clamp_int(m_data.get("market_demand_score"), default=50)

            llm_match_score = int(round((h_score * 0.5) + (s_score * 0.3) + (m_score * 0.2)))
            llm_match_score = max(0, min(100, llm_match_score))

            matched = ", ".join(m_data.get("matched_skills", []))
            missing = ", ".join(m_data.get("missing_skills", []))

            matched_list = m_data.get("matched_skills", []) or []
            missing_list = m_data.get("missing_skills", []) or []

            semantic_score, lexical_score, llm_match_score = stabilize_scores(
                semantic_score,
                semantic_conf,
                lexical_score,
                llm_match_score,
                used_fallback
            )


            # --- Heatmap accumulation (dynamic, career-agnostic) ---
            seen_m = set()
            for sk in matched_list:
                k = normalize_skill_label(sk)
                if k and k not in seen_m:
                    st.session_state.global_skill_stats["matched"][k] = st.session_state.global_skill_stats["matched"].get(k, 0) + 1
                    seen_m.add(k)

            seen_x = set()
            for sk in missing_list:
                k = normalize_skill_label(sk)
                if k and k not in seen_x:
                    st.session_state.global_skill_stats["missing"][k] = st.session_state.global_skill_stats["missing"].get(k, 0) + 1
                    seen_x.add(k)

            st.session_state.global_skill_stats["jobs_count"] += 1

        except Exception as e:
            logger.exception("LLM match scoring failed (fallback used): %s", e)
            # --- Token-safe fallback: do NOT break heatmap when Groq 429 happens ---
            st.session_state["last_llm_error"] = str(e)

            score = h_score = s_score = m_score = 50
            llm_match_score = 50

            # Heuristic matched skills from memory_bank (no extra API calls)
            matched_list = []
            missing_list = []

            mb = st.session_state.get("memory_bank") or {}
            skill_groups = (mb.get("skills") or {}) if isinstance(mb, dict) else {}
            flat_skills = []
            for group in skill_groups.values():
                if isinstance(group, list):
                    flat_skills.extend([x for x in group if isinstance(x, str)])

            jt = (job_desc_clean or "").lower()
            seen = set()
            for sk in flat_skills:
                k = normalize_skill_label(sk)
                if k and k not in seen and k in jt:
                    matched_list.append(sk)
                    seen.add(k)
                if len(matched_list) >= 6:
                    break

            matched = ", ".join(matched_list) if matched_list else "LLM unavailable (rate limit)."
            missing = "—"

            # ✅ still accumulate + increment so the heatmap renders
            seen_m = set()
            for sk in matched_list:
                k = normalize_skill_label(sk)
                if k and k not in seen_m:
                    st.session_state.global_skill_stats["matched"][k] = st.session_state.global_skill_stats["matched"].get(k, 0) + 1
                    seen_m.add(k)

            # (Optional) don’t invent missing skills without LLM — leave missing empty
            st.session_state.global_skill_stats["jobs_count"] += 1

        final_smart_score = int(round(
            (0.45 * llm_match_score) +
            (0.35 * semantic_score) +
            (0.20 * lexical_score)
        ))

        logger.info(
            "Scoring | Title=%s | LLM=%s | Semantic=%s | Lexical=%s | Final=%s",
            j.get("title"),
            llm_match_score,
            semantic_score,
            lexical_score,
            final_smart_score,
        )

        final_smart_score = max(0, min(100, final_smart_score))

        scored_rows.append({
            "job": j,
            "final_smart_score": final_smart_score,
            "llm_match_score": llm_match_score,
            "semantic_score": semantic_score,
            "semantic_conf": semantic_conf,
            "lexical_score": lexical_score,
            "used_fallback": used_fallback,
            "h_score": h_score,
            "s_score": s_score,
            "m_score": m_score,
            "matched": matched,
            "missing": missing,
        })

    # --- SORT + DISPLAY TOP 10 ---
    scored_rows.sort(key=lambda r: r["final_smart_score"], reverse=True)
    top_rows = scored_rows[:10]

    # remember shown jobs so next click rotates results
    shown = st.session_state.get("shown_job_keys", set())
    for row in top_rows:
        j = row["job"]
        title = (j.get("title") or "").strip().lower()
        company = (j.get("company_name") or "").strip().lower()
        location = (j.get("location") or "").strip().lower()
        shown.add(f"{title}|{company}|{location}")
    st.session_state["shown_job_keys"] = shown

    for row in top_rows:
        j = row["job"]
        final_smart_score = row["final_smart_score"]
        llm_match_score = row["llm_match_score"]
        semantic_score = row["semantic_score"]
        semantic_conf = row["semantic_conf"]
        lexical_score = row["lexical_score"]
        used_fallback = row["used_fallback"]
        h_score = row["h_score"]
        s_score = row["s_score"]
        m_score = row["m_score"]
        matched = row["matched"]
        missing = row["missing"]

        # --- YOUR EXISTING UI BLOCK GOES HERE ---
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])

            with c1:
                source = get_job_source_label(j)

                st.markdown(f"**{j.get('title')}**")
                st.write(f"🏢 {j.get('company_name')} | 📍 {j.get('location')}")
                if source == "Other":
                    st.caption("⚠️ Source: **Other / Unknown site**")
                else:
                    st.caption(f"🔗 Source: **{source}**")

                actual_link = get_job_link(j)
                if actual_link:
                    domain = actual_link.split("/")[2] if "://" in actual_link else actual_link
                    if source != "Other" and source.lower() not in domain.lower():
                        st.caption(f"↪ Opens via: {domain}")

                if used_fallback:
                    st.caption("⚠️ Job description incomplete — using fallback text for scoring.")

                st.progress(final_smart_score / 100)

            with c2:
                st.metric("Smart Match", f"{final_smart_score}%")
                link = get_job_link(j)
                if link:
                    st.link_button("View 🚀", link)

            with st.expander("📊 Detailed Intelligence & Skill Gap"):
                st.caption("Hybrid Match Breakdown")

                b1, b2, b3, b4 = st.columns(4)

                with b1:
                    st.metric("LLM Score (normalized)", f"{llm_match_score}%")

                with b2:
                    st.metric("Semantic Score (stabilized)", f"{semantic_score}%")

                with b3:
                    st.metric("Lexical Score (stabilized)", f"{lexical_score}%")

                with b4:
                    st.metric("Smart Score (final)", f"{final_smart_score}%")

                st.caption(f"Semantic confidence: **{semantic_conf}%**")

                st.divider()

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.caption("Hard Skills")
                    st.progress(h_score / 100)
                    st.write(f"**{h_score}%**")
                with col_b:
                    st.caption("Soft Skills")
                    st.progress(s_score / 100)
                    st.write(f"**{s_score}%**")
                with col_c:
                    st.caption("Market Demand")
                    st.progress(m_score / 100)
                    st.write(f"**{m_score}%**")

                st.divider()

                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("✅ **Matched Competencies**")
                    st.caption(matched)
                with col_right:
                    st.markdown("❌ **Identified Gaps**")
                    st.caption(missing)


    with st.expander("🌍 Global Skill Heatmap (Across This Job Search)", expanded=True):
        stats = st.session_state.get("global_skill_stats", {})
        matched = stats.get("matched", {}) or {}
        missing = stats.get("missing", {}) or {}
        jobs_count = int(stats.get("jobs_count") or 0)

        st.caption(f"Signals aggregated across **{jobs_count} jobs**. (Counts = how many job postings mention the skill.)")
        # --- GLOBAL SUMMARY ---
        top_missing = sorted(missing.items(), key=lambda x: x[1], reverse=True)[:5]
        top_matched = sorted(matched.items(), key=lambda x: x[1], reverse=True)[:5]

        miss_list = ", ".join([f"{k} ({v})" for k, v in top_missing]) if top_missing else "None"
        match_list = ", ".join([f"{k} ({v})" for k, v in top_matched]) if top_matched else "None"

        st.markdown("### 📌 Global Summary")

        cA, cB, cC = st.columns(3)

        with cA:
            st.metric("Jobs Analyzed", jobs_count)

        with cB:
            st.metric("Matched Skills Found", len(matched))

        with cC:
            st.metric("Missing Skills Found", len(missing))

        st.info(
            f"""
        ✅ **Top Strengths:** {match_list}  
        ❌ **Top Market Gaps:** {miss_list}
        """
        )

        if jobs_count == 0 or (not matched and not missing):
            st.info("No heatmap data yet. Run a job search first.")
        else:
            col1, col2 = st.columns(2)

            def top_items(d: dict, n: int = 15):
                return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]

            with col1:
                st.markdown("### ❌ Top Missing Skills (Market Gaps)")
                miss_top = top_items(missing, 15)
                miss_df = pd.DataFrame(miss_top, columns=["Skill", "Count"])
                miss_df["% of Jobs"] = miss_df["Count"].apply(lambda c: int(round((c / jobs_count) * 100)) if jobs_count else 0)
                st.dataframe(miss_df, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("### ✅ Top Matched Skills (Your Strengths)")
                match_top = top_items(matched, 15)
                match_df = pd.DataFrame(match_top, columns=["Skill", "Count"])
                match_df["% of Jobs"] = match_df["Count"].apply(lambda c: int(round((c / jobs_count) * 100)) if jobs_count else 0)
                st.dataframe(match_df, use_container_width=True, hide_index=True)

            st.divider()

            st.markdown("### 🔥 Heat Intensity View (Missing Skills)")
            miss_top = top_items(missing, 20)
            if miss_top:
                heat_df = pd.DataFrame(miss_top, columns=["Skill", "Count"])
                heat_df["Heat"] = heat_df["Count"].apply(lambda c: (c / jobs_count) if jobs_count else 0.0)
                # simple color heatmap using dataframe styling
                st.dataframe(
                    heat_df[["Skill", "Count", "Heat"]].style.background_gradient(subset=["Heat"]),
                    use_container_width=True
                )
            else:
                st.caption("No missing-skill heat data.")

            st.divider()
            st.markdown("## 🧠 Strategic Career Insights")

            HIGH_DEMAND_THRESHOLD = 0.4  # 40% of jobs

            high_demand_missing = []
            high_demand_strengths = []

            for skill, count in missing.items():
                if jobs_count and (count / jobs_count) >= HIGH_DEMAND_THRESHOLD:
                    high_demand_missing.append(skill)

            for skill, count in matched.items():
                if jobs_count and (count / jobs_count) >= HIGH_DEMAND_THRESHOLD:
                    high_demand_strengths.append(skill)

            if not high_demand_missing and not high_demand_strengths:
                st.info("Not enough dominant signals yet. Try analyzing more jobs.")
            else:
                if high_demand_missing:
                    st.warning(
                        f"""
🚨 **Priority Market Gaps Identified**

The following skills appear in over {int(HIGH_DEMAND_THRESHOLD*100)}% of analyzed jobs 
but are missing or weak in your profile:

{", ".join(high_demand_missing[:6])}

👉 Recommendation: Prioritize strengthening at least 1-2 of these areas.
"""
                )

            if high_demand_strengths:
                st.success(
                    f"""
💪 **Dominant Career Strengths**

These skills appear frequently in job postings AND are already strong in your profile:

{", ".join(high_demand_strengths[:6])}

👉 Strategy: Emphasize these in your resume headline, summary, and interviews.
"""
                )


# --- 6. CHAT SUPPORT (Moved outside of loops for stability) ---
if st.session_state.resume_text:
    st.divider()
    st.subheader("💬 Genesis Career AI")

    if st.button("🗑️ Clear Chat History", type="secondary"):
       st.session_state.messages = []
       st.rerun()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): 
            st.markdown(msg["content"])

    # --- CHAT ENGINE (Upgraded for Vector Chunks) ---
    prompt = st.chat_input("Ask about your career or job matches...")
    if isinstance(prompt, str) and prompt.strip():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)
    
        # 1. RETRIEVE RELEVANT CHUNKS (Searching the 12+ slices)
        try:
            # We ask for the 5 most relevant chunks to answer the question
            search_results = vector_collection.query(
                query_texts=[prompt], 
                n_results=5,
                where={"type": "resume_intake"}
            )
            vault_docs = search_results.get('documents', [[]])[0]
            vault_context = "\n\n---\n\n".join(vault_docs) if vault_docs else "No specific data found."
        except Exception:
            vault_context = "Vault search unavailable."

        # 2. THE GREETING LOGIC (Keep as is)
        if len(prompt.split()) < 3 and prompt.lower().strip() in ["hello", "hi", "hey", "greetings"]:
            ai_res = f"Hello! I'm the Genesis Career AI. I've analyzed your profile as a **{st.session_state.get('job', 'Professional')}**. How can I help you today?"
            st.chat_message("assistant").markdown(ai_res)
        
        else:
            # 3. FULL AI COMPLETION
            with st.chat_message("assistant"):
                # We still include the main text + the specific chunks for maximum accuracy
                current_resume = st.session_state.get('resume_text', 'No resume currently uploaded.')
                
                full_prompt = f"""
                You are the Genesis Career AI. 

                CRITICAL CONTEXT:
                Below are the most relevant sections found in the user's resume for this specific question.
                
                RELEVANT RESUME CHUNKS:
                {vault_context}

                FULL RESUME REFERENCE:
                {current_resume[:2000]}

                USER CURRENT ROLE: {st.session_state.get('job', 'Professional')}
                
                INSTRUCTIONS: 
                - Use the chunks to provide specific evidence-based answers.
                - If the information isn't in the chunks, check the 'Full Resume Reference'.
                - Speak directly to the user with a sharp, professional wit.

                USER QUESTION: {prompt}
                """
                
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": full_prompt}],
                    model="llama-3.3-70b-versatile"
                )
                ai_res = response.choices[0].message.content
                st.markdown(ai_res)
                
        # 4. SAVE AND REFRESH
        st.session_state.messages.append({"role": "assistant", "content": ai_res})
        st.rerun()

# --- 7. FINAL GLOBAL ELSE ---
if not st.session_state.resume_text:
    st.info("📟 OS STANDBY: Please upload a Resume PDF in the sidebar to initialize intelligence.")