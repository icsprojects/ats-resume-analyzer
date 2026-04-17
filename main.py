"""
ATS Resume Analyzer — Backend v7 (Multi-Provider, HR-Grade)
============================================================
AI Providers supported (set via .env):
  - ollama   : Free, local, fully private (default)
  - groq     : Free API, fast (llama-3 / mixtral)
  - openai   : GPT-4o-mini (very cheap)
  - anthropic: Claude Haiku

Configuration: copy backend/.env.example → backend/.env and fill in values.
Never hardcode API keys in this file.
"""

import os, io, json, re
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Environment config (never hardcode) ──────────────────────────────────────
AI_PROVIDER      = os.getenv("AI_PROVIDER", "ollama").lower()   # ollama|groq|openai|anthropic
OLLAMA_BASE      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "phi:latest")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama3-8b-8192")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL  = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
CORS_ORIGINS     = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost:4173").split(",")
MAX_RESUME_CHARS = int(os.getenv("MAX_RESUME_CHARS", "5000"))
MAX_JD_CHARS     = int(os.getenv("MAX_JD_CHARS", "3000"))

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

import httpx

app = FastAPI(title="ATS Resume Analyzer", version="7.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# ═════════════════════════════════════════════════════════════════════════════
# SENIORITY TABLE
# ═════════════════════════════════════════════════════════════════════════════

SENIORITY_LEVELS = {
    "intern": 0, "internship": 0, "trainee": 1, "fresher": 1, "entry": 1,
    "junior": 2, "associate": 2, "mid": 3, "mid-level": 3, "intermediate": 3,
    "senior": 4, "lead": 5, "principal": 5, "staff": 5, "manager": 6,
    "head": 7, "director": 8, "vp": 8, "vice president": 8,
    "cto": 9, "ceo": 9, "executive": 9,
}

def detect_seniority_level(text: str) -> int:
    text_lower = text.lower()
    best = -1
    for keyword, level in SENIORITY_LEVELS.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            if level > best:
                best = level
    return best if best >= 0 else 3

def years_to_seniority(years: Optional[int]) -> int:
    if years is None: return 3
    if years == 0:    return 1
    if years <= 1:    return 2
    if years <= 3:    return 3
    if years <= 6:    return 4
    if years <= 10:   return 5
    return 6


# ═════════════════════════════════════════════════════════════════════════════
# AI PROVIDER ABSTRACTION — add new providers here without touching anything else
# ═════════════════════════════════════════════════════════════════════════════

async def call_ai(prompt: str, max_tokens: int = 1000) -> str:
    """Route to the configured AI provider."""
    if AI_PROVIDER == "ollama":
        return await _call_ollama(prompt, max_tokens)
    elif AI_PROVIDER == "groq":
        return await _call_groq(prompt, max_tokens)
    elif AI_PROVIDER == "openai":
        return await _call_openai(prompt, max_tokens)
    elif AI_PROVIDER == "anthropic":
        return await _call_anthropic(prompt, max_tokens)
    else:
        raise HTTPException(status_code=500, detail=f"Unknown AI_PROVIDER: '{AI_PROVIDER}'. Use: ollama|groq|openai|anthropic")


async def _call_ollama(prompt: str, max_tokens: int) -> str:
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": max_tokens}}
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error: {resp.text[:300]}")
        return resp.json().get("response", "")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Run 'ollama serve' first.")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama timed out. Try a smaller model.")


async def _call_groq(prompt: str, max_tokens: int) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": GROQ_MODEL, "max_tokens": max_tokens, "temperature": 0.1,
                      "messages": [{"role": "user", "content": prompt}]}
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Groq error: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq call failed: {e}")


async def _call_openai(prompt: str, max_tokens: int) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in .env")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={"model": OPENAI_MODEL, "max_tokens": max_tokens, "temperature": 0.1,
                      "messages": [{"role": "user", "content": prompt}]}
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")


async def _call_anthropic(prompt: str, max_tokens: int) -> str:
    if not ANTHROPIC_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set in .env")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json"},
                json={"model": ANTHROPIC_MODEL, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]}
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Anthropic error: {resp.text[:300]}")
        return resp.json()["content"][0]["text"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anthropic call failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# FILE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not PDF_SUPPORT:
        raise HTTPException(status_code=500, detail="Run: pip install pdfplumber")
    parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t: parts.append(t)
    return "\n".join(parts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    if not DOCX_SUPPORT:
        raise HTTPException(status_code=500, detail="Run: pip install python-docx")
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def get_resume_text(filename: str, file_bytes: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":    return extract_text_from_pdf(file_bytes)
    elif ext == "docx": return extract_text_from_docx(file_bytes)
    elif ext == "txt":  return file_bytes.decode("utf-8", errors="ignore")
    else: raise HTTPException(status_code=400, detail=f"Unsupported file '.{ext}'. Use PDF, DOCX, or TXT.")

def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars: return text
    trimmed = text[:max_chars]
    cut = trimmed.rfind('.')
    return trimmed[:cut + 1] + "\n[trimmed]" if cut > max_chars * 0.8 else trimmed + "\n[trimmed]"


# ═════════════════════════════════════════════════════════════════════════════
# LLM PROMPTS
# ═════════════════════════════════════════════════════════════════════════════

def prompt_extract_resume(resume_text: str) -> str:
    return f"""Extract structured data from this resume. Output ONLY valid JSON, no markdown, no explanation.

RESUME:
{resume_text}

Required JSON format:
{{
  "full_name": "candidate full name",
  "years_of_experience": 0,
  "current_or_last_role": "most recent job title",
  "seniority_in_resume": "intern/junior/mid/senior/lead/manager/director/unknown",
  "domain": "primary domain e.g. Graphic Design, Software Engineering, Marketing, Finance",
  "skills": ["every skill, tool, software mentioned"],
  "education": "highest degree and field",
  "certifications": ["certifications listed"],
  "achievements": ["quantified achievements"],
  "industries": ["industries worked in"],
  "career_summary": "2 sentence factual summary"
}}

Output only JSON:"""


def prompt_extract_jd(jd_text: str) -> str:
    return f"""Extract structured requirements from this job description. Output ONLY valid JSON.

JOB DESCRIPTION:
{jd_text}

Required JSON format:
{{
  "job_title": "exact job title",
  "seniority_required": "intern/junior/mid/senior/lead/manager/director",
  "min_years_experience": 0,
  "max_years_experience": 99,
  "domain": "primary domain e.g. Graphic Design, Software Engineering, Marketing",
  "required_skills": ["must-have skills and tools"],
  "preferred_skills": ["nice-to-have skills"],
  "education_required": "minimum education or none",
  "responsibilities": ["key responsibilities"],
  "industry": "industry or sector"
}}

min_years_experience and max_years_experience must be integers. Output only JSON:"""


def prompt_evaluate(resume_data: dict, jd_data: dict, rule_scores: dict) -> str:
    alignment      = rule_scores.get("domain_alignment", "unknown")
    equiv_notes    = rule_scores.get("equivalent_notes", [])
    inferred_soft  = rule_scores.get("inferred_soft_skills", [])
    verdict_reason = rule_scores.get("verdict_reason", "")
    flags          = rule_scores.get("flags", [])

    context_block = ""
    if equiv_notes:
        context_block += f"\nSKILL EQUIVALENCES FOUND: {', '.join(equiv_notes)}"
    if inferred_soft:
        context_block += f"\nINFERRED SOFT SKILLS: {', '.join(inferred_soft)}"
    if verdict_reason:
        context_block += f"\nSCORING SYSTEM REASON: {verdict_reason}"

    overqualified_note = (
        "\n⚠️  Candidate is overqualified. Frame this as a RETENTION RISK, not a skill problem. "
        "Mention it in recommendation but do not list it as a skills gap."
        if "RETENTION_RISK" in flags else ""
    )
    domain_note = (
        f"\n⚠️  Domain alignment is '{alignment}'. "
        + ("Use language like 'adjacent field' or 'overlapping skill set', not 'mismatch'."
           if alignment == "partial" else
           "Be honest that this is a significant domain shift." if alignment == "weak" else "")
    )

    return f"""You are an experienced, empathetic HR professional writing a candidate evaluation brief.
Your tone is like a trusted recruiter talking to a hiring manager — direct, fair, human.
A scoring system has already produced numeric scores. Your job is to add the human layer.
Output ONLY valid JSON.
{overqualified_note}{domain_note}{context_block}

RULES FOR YOUR RESPONSE:
1. Write like a human recruiter, not a system. Use natural language.
2. Strengths: be specific. "Has Photoshop and Illustrator" beats "design skills present."
3. Gaps: only list real skill gaps. Overqualification is NOT a skill gap — it goes in recommendation.
4. If equivalent tools were found (e.g. CapCut for Premiere), call them out as POSITIVES in strengths.
5. Inferred soft skills (creativity, social media savvy, etc.) can appear in strengths if relevant to JD.
6. Summary: start with what the candidate IS good at, then mention risks.
7. Recommendation: should sound like what you'd actually say to a hiring manager.
   Bad: "Not recommended due to overqualification."
   Good: "Strong designer — but given their lead-level background, I'd want to understand why
          they're applying for a junior role before moving forward. High retention risk."

CANDIDATE PROFILE:
{json.dumps(resume_data, indent=2)}

JOB REQUIREMENTS:
{json.dumps(jd_data, indent=2)}

SCORING CONTEXT:
{json.dumps({k: v for k, v in rule_scores.items() if k not in ("extracted_resume","extracted_jd")}, indent=2)}

Output this JSON:
{{
  "summary": "2-3 sentences — start with strengths, end with the main risk or concern",
  "strengths": ["specific, evidence-backed strength relevant to this JD"],
  "gaps": ["genuine skill or experience gap only — no seniority complaints here"],
  "key_skills_found": ["tools/skills the candidate has that directly match JD needs"],
  "missing_skills": ["skills JD needs that candidate genuinely lacks — can be empty"],
  "recommendation": "what you'd actually say to a hiring manager — honest, human, actionable",
  "interview_questions": ["one targeted question per concern or strength to probe further"]
}}

Output only JSON:"""


# ═════════════════════════════════════════════════════════════════════════════
# JSON PARSING
# ═════════════════════════════════════════════════════════════════════════════

def repair_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    text = re.sub(r"(?<![\\])'", '"', text)
    for field in ["verdict", "confidence", "seniority_required", "seniority_in_resume"]:
        text = re.sub(rf'("{field}"\s*:\s*)([A-Z_][A-Z_]*)\b', r'\1"\2"', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text += ']' * max(0, text.count('[') - text.count(']'))
    text += '}' * max(0, text.count('{') - text.count('}'))
    return text

def parse_json_response(raw: str, label: str = "") -> dict:
    for attempt in [raw.strip(), repair_json(raw)]:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass
        m = re.search(r'\{.*\}', attempt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                try:
                    return json.loads(repair_json(m.group()))
                except json.JSONDecodeError:
                    pass
    print(f"[WARN] {label} JSON parse failed — using fallback extraction")
    return {}


# ═════════════════════════════════════════════════════════════════════════════
# SKILL EQUIVALENCE MAP
# If a candidate has ANY skill in a group, all group members count as covered.
# Add new tools/aliases here — no other code needs changing.
# ═════════════════════════════════════════════════════════════════════════════

SKILL_GROUPS: dict[str, list[str]] = {
    "video_editing":        ["adobe premiere pro", "premiere pro", "premiere", "capcut", "cap cut",
                             "filmora", "davinci resolve", "davinci", "final cut pro", "final cut",
                             "after effects", "vegas pro", "imovie", "kdenlive"],
    "adobe_creative_suite": ["adobe creative suite", "creative suite", "photoshop", "adobe photoshop",
                             "illustrator", "adobe illustrator", "indesign", "adobe indesign",
                             "lightroom", "adobe xd", "xd"],
    "motion_graphics":      ["after effects", "adobe after effects", "blender", "cinema 4d", "c4d",
                             "motion", "animate", "adobe animate"],
    "content_tools":        ["canva", "figma", "adobe express", "spark", "picsart", "midjourney",
                             "dall-e", "stable diffusion"],
    "frontend":             ["react", "reactjs", "react.js", "vue", "vuejs", "angular", "nextjs",
                             "next.js", "svelte", "html", "css", "javascript", "typescript"],
    "backend":              ["node", "nodejs", "node.js", "express", "django", "flask", "fastapi",
                             "spring", "laravel", "rails", "ruby on rails"],
    "python_ecosystem":     ["python", "django", "flask", "fastapi", "pandas", "numpy", "pytorch", "tensorflow"],
    "js_ecosystem":         ["javascript", "typescript", "node", "react", "vue", "angular"],
    "databases":            ["sql", "mysql", "postgresql", "postgres", "mongodb", "redis",
                             "sqlite", "oracle", "firebase", "supabase"],
    "cloud":                ["aws", "azure", "gcp", "google cloud", "heroku", "vercel", "netlify",
                             "docker", "kubernetes", "k8s", "terraform"],
    "office_suite":         ["microsoft office", "ms office", "word", "excel", "powerpoint",
                             "google docs", "google sheets", "google slides", "google workspace"],
    "digital_marketing":    ["seo", "sem", "google ads", "meta ads", "facebook ads", "instagram ads",
                             "email marketing", "mailchimp", "hubspot", "google analytics", "ga4"],
    "ux_design":            ["figma", "sketch", "adobe xd", "xd", "invision", "zeplin",
                             "wireframing", "prototyping", "user research"],
}

# Soft skills that can be INFERRED from job titles, industries, and achievements
# Format: { inferred_skill: [trigger_keywords] }
SOFT_SKILL_INFERENCE: dict[str, list[str]] = {
    "creativity":             ["graphic", "design", "creative", "brand", "content", "art", "visual",
                               "photography", "videograph", "motion", "illustrat"],
    "social media savvy":     ["social media", "instagram", "tiktok", "facebook", "youtube",
                               "content creator", "influencer", "community manager"],
    "brand awareness":        ["brand", "branding", "brand identity", "brand strategy", "marketing",
                               "advertising", "campaign"],
    "visual communication":   ["graphic", "design", "visual", "layout", "typography", "infographic",
                               "presentation", "deck"],
    "attention to detail":    ["audit", "quality", "qa", "testing", "proofreading", "editing",
                               "data entry", "compliance", "accounting"],
    "project management":     ["managed", "led", "coordinated", "delivered", "launched", "oversaw",
                               "supervised", "spearheaded"],
    "client communication":   ["client", "stakeholder", "customer", "account", "relationship",
                               "vendor", "partner"],
    "analytical thinking":    ["analytics", "data", "reporting", "kpi", "metrics", "dashboard",
                               "research", "insight", "analysis"],
    "adaptability":           ["freelance", "startup", "agency", "multiple clients", "fast-paced",
                               "cross-functional", "diverse"],
}

def _infer_soft_skills(resume: dict) -> list[str]:
    """Infer soft skills from job titles, achievements, summary, and industries."""
    # Build a corpus of text to scan
    corpus_parts = [
        resume.get("current_or_last_role", ""),
        resume.get("career_summary", ""),
        resume.get("domain", ""),
        " ".join(resume.get("achievements", [])),
        " ".join(resume.get("industries", [])),
        " ".join(resume.get("skills", [])),
    ]
    corpus = " ".join(corpus_parts).lower()

    inferred = []
    for soft_skill, triggers in SOFT_SKILL_INFERENCE.items():
        if any(t in corpus for t in triggers):
            inferred.append(soft_skill)
    return inferred


# ─────────────────────────────────────────────────────────────────────────────
# SKILL MATCHING (equivalence-aware, substring-tolerant)
# ─────────────────────────────────────────────────────────────────────────────

def _expand_skill_set(skills: set[str]) -> set[str]:
    """Expand a raw skill set with all group equivalents."""
    expanded = set(skills)
    for members in SKILL_GROUPS.values():
        members_lower = [m.lower() for m in members]
        if any(any(m in s or s in m for m in members_lower) for s in skills):
            expanded.update(members_lower)
    return expanded

def _skill_matches(candidate_skills_raw: set[str], jd_skill: str) -> bool:
    jd_lower = jd_skill.lower().strip()
    expanded = _expand_skill_set(candidate_skills_raw)
    return any(jd_lower in s or s in jd_lower for s in expanded)

def _match_skills(candidate_skills_raw: list, jd_skills: list) -> tuple[list, list]:
    c_set = set(s.lower().strip() for s in candidate_skills_raw)
    matched, missing = [], []
    for jd_skill in jd_skills:
        (matched if _skill_matches(c_set, jd_skill) else missing).append(jd_skill)
    return matched, missing


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN ALIGNMENT (3-level: strong / partial / weak)
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_CLUSTERS = [
    # Creative / design — content creation and graphic design overlap here
    {"graphic design", "visual design", "ui design", "ux design", "ui/ux", "multimedia",
     "content creation", "creative", "branding", "video production", "video editing",
     "digital media", "social media", "photography", "motion graphics", "advertising",
     "art direction", "illustration"},
    # Marketing
    {"digital marketing", "marketing", "brand management", "growth", "seo", "sem",
     "content marketing", "social media marketing", "ecommerce", "e-commerce",
     "performance marketing", "influencer marketing", "copywriting"},
    # Tech
    {"software engineering", "software development", "web development", "backend", "frontend",
     "fullstack", "mobile development", "devops", "data science", "machine learning",
     "artificial intelligence", "cybersecurity", "cloud", "it"},
    # Business
    {"finance", "accounting", "audit", "banking", "investment", "consulting",
     "business development", "sales", "operations", "supply chain", "logistics"},
    # People
    {"human resources", "hr", "talent acquisition", "recruitment", "people operations",
     "learning and development", "l&d"},
    # Healthcare
    {"healthcare", "medical", "nursing", "pharmacy", "clinical"},
    # Education
    {"education", "teaching", "training", "e-learning", "curriculum"},
]

def _classify_domain_alignment(candidate_domain: str, jd_domain: str) -> tuple[str, int, str]:
    """Returns (level, score, label). level: strong | partial | weak."""
    if not candidate_domain or not jd_domain:
        return "partial", 75, "Domain context unclear — scored conservatively"

    c = candidate_domain.lower().strip()
    j = jd_domain.lower().strip()

    # Direct word overlap → Strong
    if set(c.split()) & set(j.split()):
        return "strong", 95, f"Strong domain alignment ({candidate_domain} ↔ {jd_domain})"

    # Same cluster → Partial
    c_cluster = next((i for i, g in enumerate(DOMAIN_CLUSTERS) if any(t in c for t in g)), None)
    j_cluster = next((i for i, g in enumerate(DOMAIN_CLUSTERS) if any(t in j for t in g)), None)

    if c_cluster is not None and j_cluster is not None:
        if c_cluster == j_cluster:
            return "partial", 72, f"Adjacent domain — overlapping skills ({candidate_domain} ↔ {jd_domain})"
        else:
            return "weak", 38, f"Different domain areas ({candidate_domain} vs {jd_domain})"

    return "partial", 65, f"Domain overlap unclear ({candidate_domain} ↔ {jd_domain})"


# ═════════════════════════════════════════════════════════════════════════════
# HUMAN-LIKE SCORING ENGINE
# Philosophy: weighted reasoning, not hard rules.
# No score is a hard veto. Everything is a trade-off.
# ═════════════════════════════════════════════════════════════════════════════

def compute_rule_scores(resume: dict, jd: dict) -> dict:
    scores  = {}
    flags   = []
    notes   = []   # human-readable recruiter notes (not just issues)

    # ── 1. Skill Match (equivalence-aware) ────────────────────────────────────
    candidate_skills_raw = resume.get("skills", [])
    required_skills_raw  = jd.get("required_skills", [])
    preferred_skills_raw = jd.get("preferred_skills", [])

    matched_req,  missing_req  = _match_skills(candidate_skills_raw, required_skills_raw)
    matched_pref, _            = _match_skills(candidate_skills_raw, preferred_skills_raw)

    if required_skills_raw:
        req_ratio  = len(matched_req)  / len(required_skills_raw)
        pref_ratio = len(matched_pref) / max(len(preferred_skills_raw), 1)
        skill_score = min(100, int(req_ratio * 65 + pref_ratio * 25 + 10))  # +10 base goodwill
    else:
        skill_score = 60   # can't penalise without data

    # Soft skill inference bonus (up to +8 pts)
    inferred_soft = _infer_soft_skills(resume)
    soft_bonus = min(8, len(inferred_soft) * 2)
    skill_score = min(100, skill_score + soft_bonus)
    scores["skill_match"] = skill_score

    # Equivalence notes for transparency
    c_set = set(s.lower().strip() for s in candidate_skills_raw)
    equivalent_notes = []
    for jd_skill in matched_req:
        jd_lower = jd_skill.lower().strip()
        if not any(jd_lower in s or s in jd_lower for s in c_set):
            satisfiers = [s for s in c_set if _skill_matches({s}, jd_skill)]
            if satisfiers:
                equivalent_notes.append(f"{satisfiers[0].title()} accepted as equivalent to {jd_skill}")

    # ── 2. Seniority Fit (soft penalties, never a veto) ───────────────────────
    candidate_years = resume.get("years_of_experience") or 0
    jd_min_years    = jd.get("min_years_experience") or 0
    jd_max_years    = jd.get("max_years_experience") or 99
    candidate_level = detect_seniority_level(resume.get("seniority_in_resume", "")) or years_to_seniority(candidate_years)
    jd_level        = detect_seniority_level(jd.get("seniority_required", ""))
    seniority_diff  = candidate_level - jd_level
    overqualified   = False
    underqualified  = False

    if seniority_diff >= 2:
        overqualified   = True
        # Graduated penalty — severe mismatch scores lower, but never zero
        seniority_score = max(20, 65 - (seniority_diff - 1) * 12)
        flags.append("RETENTION_RISK")
        notes.append(
            f"Candidate is {seniority_diff} seniority level(s) above the role. "
            f"Strong skills present — but there is a real retention risk. "
            f"Suitable only if role has growth potential or this is a deliberate hire."
        )
    elif seniority_diff <= -2:
        underqualified  = True
        seniority_score = max(15, 65 + seniority_diff * 12)
        flags.append("EXPERIENCE_GAP")
        notes.append(
            f"Candidate has {abs(seniority_diff)} fewer seniority level(s) than required. "
            f"May need additional coaching to perform independently."
        )
    else:
        # Within 1 level — very acceptable
        seniority_score = 100 - abs(seniority_diff) * 15

    if candidate_years > 0 and jd_max_years < 99 and candidate_years > jd_max_years + 3:
        if not overqualified:
            overqualified = True
            flags.append("RETENTION_RISK")
            notes.append(
                f"With {candidate_years} years of experience, candidate may find this role below their trajectory. "
                f"Worth exploring their motivation for applying."
            )
    if candidate_years > 0 and jd_min_years > 0 and candidate_years < jd_min_years:
        underqualified = True
        if "EXPERIENCE_GAP" not in flags:
            flags.append("EXPERIENCE_GAP")
            notes.append(f"JD targets {jd_min_years}+ years; candidate has {candidate_years}. Gap is manageable if skills are strong.")

    scores["seniority_match"] = max(0, min(100, seniority_score))

    # ── 3. Domain Alignment (3-level, no binary reject) ───────────────────────
    candidate_domain  = (resume.get("domain") or "").strip()
    jd_domain         = (jd.get("domain") or "").strip()
    alignment_level, domain_score, domain_label = _classify_domain_alignment(candidate_domain, jd_domain)
    domain_mismatch   = (alignment_level == "weak")

    if alignment_level == "partial":
        flags.append("ADJACENT_DOMAIN")
        notes.append(domain_label)
    elif alignment_level == "weak":
        flags.append("DOMAIN_SHIFT")
        notes.append(domain_label + " — candidate would be making a significant domain shift.")

    scores["domain_match"] = domain_score

    # ── 4. Education (soft / hard requirement aware) ──────────────────────────
    edu_required  = (jd.get("education_required") or "none").lower()
    edu_candidate = (resume.get("education") or "").lower()
    is_soft_edu   = any(w in edu_required for w in ("preferred", "advantage", "plus", "desirable", "beneficial", "or equivalent"))
    is_hard_edu   = not is_soft_edu and edu_required not in ("none", "", "not specified", "n/a")

    degree_order = ["phd", "doctorate", "master", "mba", "bachelor", "degree",
                    "diploma", "bsc", "msc", "be", "btech", "mtech"]
    dr    = {k: i for i, k in enumerate(degree_order)}
    req_d = next((d for d in degree_order if d in edu_required),  None)
    cand_d= next((d for d in degree_order if d in edu_candidate), None)

    if is_soft_edu:
        edu_score = 100 if cand_d else 75   # having any degree is a plus, not having one is a minor gap
    elif is_hard_edu and req_d and cand_d:
        if   dr.get(cand_d, 5) > dr.get(req_d, 5):  edu_score = 100
        elif dr.get(cand_d, 5) == dr.get(req_d, 5): edu_score = 95
        else:
            edu_score = 60
            notes.append(f"Education gap: JD requires {req_d}, candidate has {cand_d}.")
    elif is_hard_edu and req_d and not cand_d:
        edu_score = 45
        notes.append(f"No matching degree found; JD requires {req_d}.")
    else:
        edu_score = 85   # no requirement stated — neutral

    scores["education_match"] = edu_score

    # ── 5. Weighted composite — no hard caps, soft penalties instead ──────────
    # Weights: skill 40%, domain 20%, seniority 20%, education 20%
    weights   = {"skill_match": 0.40, "domain_match": 0.20,
                 "seniority_match": 0.20, "education_match": 0.20}
    composite = int(sum(scores[k] * w for k, w in weights.items()))

    # SOFT adjustments (not hard caps):
    # Overqualification: -8 pts (signal, not punishment)
    if overqualified:
        composite = max(0, composite - 8)
    # Weak domain: -10 pts
    if domain_mismatch:
        composite = max(0, composite - 10)
    # Underqualification: -6 pts
    if underqualified:
        composite = max(0, composite - 6)

    # ── 6. Human-like verdict reasoning ──────────────────────────────────────
    # Determines verdict from the overall picture, not rigid thresholds.
    verdict_data = _reason_verdict(composite, scores, flags, overqualified,
                                   underqualified, domain_mismatch, alignment_level, skill_score)

    return {
        "composite_score":   composite,
        "skill_score":       scores["skill_match"],
        "seniority_score":   scores["seniority_match"],
        "domain_score":      scores["domain_match"],
        "education_score":   scores["education_match"],
        "rule_verdict":      verdict_data["verdict"],
        "confidence":        verdict_data["confidence"],
        "verdict_reason":    verdict_data["reason"],
        "flags":             flags,
        "notes":             notes,
        "inferred_soft_skills": inferred_soft,
        "equivalent_notes":  equivalent_notes,
        "overqualified":     overqualified,
        "underqualified":    underqualified,
        "domain_mismatch":   domain_mismatch,
        "domain_alignment":  alignment_level,
        "matched_skills":    [s.title() for s in matched_req[:10]],
        "missing_required":  [s.title() for s in missing_req[:8]],
        "candidate_level":   candidate_level,
        "jd_level":          jd_level,
        "seniority_diff":    seniority_diff,
    }


def _reason_verdict(composite: int, scores: dict, flags: list,
                    overqualified: bool, underqualified: bool,
                    domain_mismatch: bool, alignment_level: str,
                    skill_score: int) -> dict:
    """
    Human-like verdict reasoning matrix.
    Reads the overall picture instead of applying sequential rules.
    Returns verdict, confidence, and a plain-English reason.
    """
    has_retention_risk = "RETENTION_RISK" in flags
    has_domain_shift   = "DOMAIN_SHIFT" in flags
    has_exp_gap        = "EXPERIENCE_GAP" in flags

    # Strong hire — good skills, good fit, no major concerns
    if composite >= 72 and not has_retention_risk and not has_domain_shift:
        return {"verdict": "HIRE", "confidence": "HIGH",
                "reason": "Strong overall match across skills, domain, and seniority. Recommend moving forward."}

    # Moderate hire — good candidate, minor concerns
    if composite >= 62 and not has_domain_shift and not overqualified:
        return {"verdict": "HIRE", "confidence": "MEDIUM",
                "reason": "Good candidate with solid skill alignment. Minor gaps present but manageable with onboarding."}

    # Overqualified but skilled — classic retention risk scenario
    if overqualified and skill_score >= 60 and not has_domain_shift:
        if composite >= 55:
            return {"verdict": "MAYBE", "confidence": "MEDIUM",
                    "reason": (
                        "Candidate is overqualified for this seniority level but brings strong, relevant skills. "
                        "Proceed with caution — explore their motivation and long-term goals before deciding. "
                        "If the role has growth potential, this could be a strong hire."
                    )}
        else:
            return {"verdict": "NO_HIRE", "confidence": "MEDIUM",
                    "reason": (
                        "Candidate's experience significantly exceeds the role requirements. "
                        "Retention risk is high. Consider for a senior-level opening instead."
                    )}

    # Adjacent domain — good skills, just different field
    if alignment_level == "partial" and skill_score >= 55 and not overqualified:
        return {"verdict": "MAYBE", "confidence": "MEDIUM",
                "reason": (
                    "Candidate comes from an adjacent domain with transferable skills. "
                    "Worth an exploratory conversation to assess adaptability and role understanding."
                )}

    # Underqualified but promising
    if underqualified and skill_score >= 65:
        return {"verdict": "MAYBE", "confidence": "LOW",
                "reason": (
                    "Candidate is below the target seniority level but demonstrates strong relevant skills. "
                    "Could be considered for a junior variant of this role or with a development plan."
                )}

    # Domain shift — significant re-alignment required
    if has_domain_shift and skill_score < 50:
        return {"verdict": "NO_HIRE", "confidence": "HIGH",
                "reason": (
                    "The candidate's background is in a significantly different domain. "
                    "Transferable skills are limited. Not recommended for this role."
                )}

    if has_domain_shift and skill_score >= 50:
        return {"verdict": "NO_HIRE", "confidence": "MEDIUM",
                "reason": (
                    "Candidate would need to make a significant domain shift for this role. "
                    "Some transferable skills exist, but the gap is likely too large for a direct hire."
                )}

    # General scoring fallback
    if composite >= 55:
        return {"verdict": "MAYBE", "confidence": "MEDIUM",
                "reason": "Moderate match. Recommend a screening call to assess cultural fit and specific skill depth."}

    if composite >= 40:
        return {"verdict": "NO_HIRE", "confidence": "MEDIUM",
                "reason": "Below-average match across key criteria. Other candidates likely to be stronger fits."}

    return {"verdict": "NO_HIRE", "confidence": "HIGH",
            "reason": "Significant gaps across skills, domain, and/or seniority. Not recommended for this role."}


def combine_verdicts(rule_scores: dict, llm_eval: dict) -> dict:
    """
    Merge rule-based verdict with LLM qualitative read.
    Rules set the floor; LLM adds nuance. Neither fully overrides the other.
    """
    rule_verdict    = rule_scores["rule_verdict"]
    flags           = rule_scores["flags"]
    composite       = rule_scores["composite_score"]
    overqualified   = rule_scores["overqualified"]
    domain_mismatch = rule_scores["domain_mismatch"]

    # Only hard veto: completely misaligned domain + very low composite
    # Everything else goes through the reasoning verdict
    if domain_mismatch and composite < 38:
        final_verdict, final_confidence = "NO_HIRE", "HIGH"
    elif "EXPERIENCE_GAP" in flags and rule_scores["skill_score"] < 30:
        final_verdict, final_confidence = "NO_HIRE", "HIGH"
    else:
        final_verdict    = rule_verdict
        final_confidence = rule_scores["confidence"]

    s       = rule_scores["seniority_score"]
    exp_rel = "HIGH" if s >= 75 else ("MEDIUM" if s >= 45 else "LOW")

    return {
        "final_verdict":        final_verdict,
        "final_confidence":     final_confidence,
        "experience_relevance": exp_rel,
        "verdict_reason":       rule_scores.get("verdict_reason", ""),
    }


# ═════════════════════════════════════════════════════════════════════════════
# CORE ANALYSIS FUNCTION (reusable for single + batch)
# ═════════════════════════════════════════════════════════════════════════════

async def analyze_one(filename: str, file_bytes: bytes, jd_text: str) -> dict:
    raw_text = get_resume_text(filename, file_bytes)
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text found in resume. Scanned image PDFs not supported.")

    resume_text = trim_text(raw_text, MAX_RESUME_CHARS)
    jd_trimmed  = trim_text(jd_text, MAX_JD_CHARS)

    resume_raw  = await call_ai(prompt_extract_resume(resume_text), max_tokens=800)
    resume_data = parse_json_response(resume_raw, "RESUME")
    jd_raw      = await call_ai(prompt_extract_jd(jd_trimmed), max_tokens=600)
    jd_data     = parse_json_response(jd_raw, "JD")
    rule_scores = compute_rule_scores(resume_data, jd_data)
    eval_raw    = await call_ai(prompt_evaluate(resume_data, jd_data, rule_scores), max_tokens=900)
    llm_eval    = parse_json_response(eval_raw, "EVAL")
    combined    = combine_verdicts(rule_scores, llm_eval)

    candidate_name = resume_data.get("full_name") or "See resume"
    key_skills     = llm_eval.get("key_skills_found") or rule_scores.get("matched_skills") or []
    missing_sk     = llm_eval.get("missing_skills") or rule_scores.get("missing_required") or []
    strengths      = llm_eval.get("strengths") or []
    gaps           = llm_eval.get("gaps") or []

    # Add recruiter notes from rule engine (retention risk, domain shift, etc.)
    for note in rule_scores.get("notes", []):
        if not any(note[:30].lower() in g.lower() for g in gaps):
            gaps.append(note)

    # Surface inferred soft skills as strengths if LLM didn't already mention them
    inferred_soft = rule_scores.get("inferred_soft_skills", [])
    for soft in inferred_soft[:3]:
        if not any(soft.lower() in s.lower() for s in strengths):
            strengths.append(f"Demonstrates {soft} (inferred from background)")

    # Summary comes from LLM — trust it since we gave it full context
    summary = llm_eval.get("summary") or resume_data.get("career_summary") or ""

    # Recommendation: prefer LLM (it has human context), fallback to rule reasoning
    recommendation = (
        llm_eval.get("recommendation")
        or combined.get("verdict_reason")
        or rule_scores.get("verdict_reason", "")
    )

    interview_qs = llm_eval.get("interview_questions") or [
        f"Walk me through your experience as {resume_data.get('current_or_last_role','your last role')} and its relevance here.",
        "What specifically draws you to this position given your current seniority?",
        "What would you need to ramp up quickly in the first 60 days?",
    ]

    return {
        "verdict":              combined["final_verdict"],
        "confidence":           combined["final_confidence"],
        "match_score":          rule_scores["composite_score"],
        "experience_relevance": combined["experience_relevance"],
        "verdict_reason":       combined.get("verdict_reason", ""),
        "candidate_name":       candidate_name,
        "summary":              summary,
        "strengths":            strengths,
        "gaps":                 gaps,
        "key_skills_found":     key_skills,
        "missing_skills":       missing_sk,
        "recommendation":       recommendation,
        "interview_questions":  interview_qs,
        "score_breakdown": {
            "skill_match":     rule_scores["skill_score"],
            "seniority_match": rule_scores["seniority_score"],
            "domain_match":    rule_scores["domain_score"],
            "education_match": rule_scores["education_score"],
        },
        "flags":                rule_scores["flags"],
        "inferred_soft_skills": inferred_soft,
        "equivalent_notes":     rule_scores.get("equivalent_notes", []),
        "domain_alignment":     rule_scores.get("domain_alignment", "unknown"),
        "extracted_resume":     resume_data,
        "extracted_jd":         jd_data,
        "filename":             filename,
        "ai_provider":          AI_PROVIDER,
    }


# ═════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/analyze")
async def analyze_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
    """Analyze a single resume against a job description."""
    file_bytes = await resume.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return await analyze_one(resume.filename, file_bytes, job_description)


@app.post("/analyze-batch")
async def analyze_batch(resumes: List[UploadFile] = File(...), job_description: str = Form(...)):
    """Analyze multiple resumes and return ranked results."""
    if len(resumes) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 resumes per batch.")

    results = []
    errors  = []

    for upload in resumes:
        file_bytes = await upload.read()
        if not file_bytes:
            errors.append({"filename": upload.filename, "error": "Empty file"})
            continue
        try:
            result = await analyze_one(upload.filename, file_bytes, job_description)
            results.append(result)
        except HTTPException as e:
            errors.append({"filename": upload.filename, "error": e.detail})
        except Exception as e:
            errors.append({"filename": upload.filename, "error": str(e)})

    results.sort(key=lambda r: r.get("match_score", 0), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return {"results": results, "errors": errors, "total": len(results)}


@app.get("/health")
async def health_check():
    info = {"status": "ok", "ai_provider": AI_PROVIDER, "version": "7.1.0"}
    if AI_PROVIDER == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{OLLAMA_BASE}/api/tags")
                models = [m["name"] for m in r.json().get("models", [])]
            info["ollama"] = "connected"
            info["available_models"] = models
        except Exception as e:
            info["ollama"] = "unreachable"
            info["error"] = str(e)
    return info


@app.get("/config")
async def get_config():
    return {
        "ai_provider": AI_PROVIDER,
        "model": {
            "ollama": OLLAMA_MODEL, "groq": GROQ_MODEL,
            "openai": OPENAI_MODEL, "anthropic": ANTHROPIC_MODEL
        }.get(AI_PROVIDER, "unknown"),
        "max_resumes_per_batch": 10,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
