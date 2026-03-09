"""
LinkedIn Job Search and Application Agent
Built with LangGraph for agentic AI workflows
"""

from typing import Annotated, TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import os
import json
import re
import ast

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(MessagesState):
    """
    State schema for the LinkedIn job agent.
    Extends MessagesState to enable Chat mode in LangGraph Studio.
    """
    # MessagesState already includes: messages: Annotated[list, add_messages]
    job_search_params: dict
    found_jobs: list
    applied_jobs: list
    next_action: str


# ============================================================================
# CUSTOM TOOLS (NOW WITH REAL LINKEDIN SCRAPING)
# ============================================================================

# Import the real scraper and profile fetcher
from linkedin_agent.real_linkedin_scraper import create_linkedin_scraper
from linkedin_agent.profile_fetcher import get_user_profile
from linkedin_agent.resume_cover_generator import (
    generate_resume_for_job,
    generate_cover_letter_for_job,
    generate_full_application
)
from linkedin_agent.llm_factory import create_chat_model

# Initialize scraper globally
_scraper = None
_user_profile = None


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from model text output."""
    if not text:
        return None

    # Try raw JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Try fenced code block JSON
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Try first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def _coerce_text_tool_call(response: AIMessage) -> AIMessage:
    """
    Convert plain-text JSON tool intent into a real tool_call for local models.
    """
    if hasattr(response, "tool_calls") and response.tool_calls:
        return response

    content = response.content if isinstance(response.content, str) else ""
    payload = _extract_json_object(content)
    if not payload:
        return response

    # Supported shapes:
    # {"name": "tool_name", "arguments": {...}}
    # {"tool": "tool_name", "args": {...}}
    tool_name = payload.get("name") or payload.get("tool")
    args = payload.get("arguments") or payload.get("args") or {}

    if not tool_name or not isinstance(args, dict):
        return response

    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "local_tool_call_1",
                "name": tool_name,
                "args": args,
                "type": "tool_call",
            }
        ],
    )


def _latest_user_text(messages: list) -> str:
    """Get latest user message text."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            return str(content)
    return ""


def _latest_human_index(messages: list) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return -1


def _has_tool_after_latest_human(messages: list) -> bool:
    h_idx = _latest_human_index(messages)
    if h_idx == -1:
        return False
    for i in range(h_idx + 1, len(messages)):
        if isinstance(messages[i], ToolMessage):
            return True
    return False


def _extract_numeric_job_id(text: str) -> Optional[str]:
    """Extract numeric LinkedIn job id from plain id or full URL."""
    url_match = re.search(r"/jobs/view/(?:[^/?]+-)?(\d+)", text)
    if url_match:
        return url_match.group(1)
    id_match = re.search(r"\b(\d{6,})\b", text)
    if id_match:
        return id_match.group(1)
    return None


def _guess_experience_level(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["entry", "fresher", "junior", "0-1", "0 to 1", "1 year"]):
        return "entry"
    if any(k in t for k in ["senior", "lead", "staff", "principal", "5-7", "6 year", "7 year"]):
        return "senior"
    if "director" in t:
        return "director"
    if "executive" in t:
        return "executive"
    return "mid"


def _guess_job_type(text: str) -> str:
    t = text.lower()
    if "part-time" in t or "part time" in t:
        return "part-time"
    if "contract" in t:
        return "contract"
    if "temporary" in t:
        return "temporary"
    if "intern" in t:
        return "internship"
    return "full-time"


def _extract_search_keywords(text: str) -> str:
    """Extract role/keyword phrase from a user request."""
    t = text.strip()
    role_match = re.search(r"([a-zA-Z0-9/+\-\s]{2,60})\s+jobs?", t, re.IGNORECASE)
    if role_match:
        role = role_match.group(1)
        role = re.sub(r"^(find|search|show|list|get)\s+(me\s+)?", "", role, flags=re.IGNORECASE).strip()
        return role or "software engineer"
    return "software engineer"


def _extract_location(text: str) -> str:
    """Extract a probable location phrase from user request."""
    loc_match = re.search(
        r"\b(?:in|at|near)\s+([a-zA-Z][a-zA-Z\s\-/,]{1,60}?)(?:\s+(?:for|with|jobs?|role|position|remote)\b|$)",
        text,
        re.IGNORECASE,
    )
    if not loc_match:
        return ""
    return loc_match.group(1).strip(" ,")


def _latest_job_from_messages(messages: list) -> dict:
    """
    Best-effort extraction of most recent job object from tool outputs.
    """
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content
        payload = None
        if isinstance(content, str):
            payload = _extract_json_object(content)
        elif isinstance(content, dict):
            payload = content

        if not isinstance(payload, dict):
            continue

        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            jobs = payload.get("linkedin_jobs")
        if isinstance(jobs, list) and jobs:
            job = jobs[0]
            if isinstance(job, dict):
                return job
    return {}


def _latest_jobs_from_messages(messages: list) -> list[dict]:
    """
    Best-effort extraction of most recent jobs list from tool outputs.
    """
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content
        payload = None
        if isinstance(content, str):
            payload = _extract_json_object(content)
        elif isinstance(content, dict):
            payload = content

        if not isinstance(payload, dict):
            continue

        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            jobs = payload.get("linkedin_jobs")
        if isinstance(jobs, list) and jobs:
            return [j for j in jobs if isinstance(j, dict)]
    return []


def _latest_search_params_from_messages(messages: list) -> dict:
    """
    Extract latest search_params dict from tool outputs.
    """
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content
        payload = None
        if isinstance(content, str):
            payload = _extract_json_object(content)
        elif isinstance(content, dict):
            payload = content

        if not isinstance(payload, dict):
            continue
        params = payload.get("search_params")
        if isinstance(params, dict):
            return params
    return {}


def _extract_requested_sources(text: str) -> str:
    """
    Parse source filters from user text and return CSV suitable for tool args.
    """
    t = text.lower()
    sources = []
    if "linkedin" in t or "linked-in" in t or "linked in" in t:
        sources.append("linkedin")
    if "indeed" in t:
        sources.append("indeed")
    if "naukri" in t:
        sources.append("naukri")
    return ",".join(sources)


def _extract_company_name(text: str) -> str:
    patterns = [
        r"\bat\s+([A-Za-z0-9&.,\- ]{2,60})",
        r"\bfor\s+([A-Za-z0-9&.,\- ]{2,60})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            value = m.group(1).strip(" .,:")
            if value:
                return value
    return ""


def _extract_job_title(text: str) -> str:
    m = re.search(r"for\s+(.+?)\s+role", text, re.IGNORECASE)
    if m:
        title = m.group(1).strip(" .,:")
        if title:
            return title
    return _extract_search_keywords(text)


def _resolve_job_id_from_text_or_recent_jobs(messages: list, text: str) -> Optional[str]:
    """
    Resolve job id from:
    1) explicit numeric id / URL in user text
    2) index reference (job 1, 2nd job, index 3)
    3) title/company match from latest job list
    """
    # 1) direct numeric extraction
    direct = _extract_numeric_job_id(text)
    if direct:
        return direct

    jobs = _latest_jobs_from_messages(messages)
    if not jobs:
        return None

    # 2) index-based references (1-based)
    idx_match = re.search(r"\b(?:job|index|#)\s*(\d{1,2})\b", text, re.IGNORECASE)
    if not idx_match:
        idx_match = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s+job\b", text, re.IGNORECASE)
    if idx_match:
        idx = int(idx_match.group(1))
        if 1 <= idx <= len(jobs):
            raw_id = str(jobs[idx - 1].get("job_id", "")).strip()
            return _extract_numeric_job_id(raw_id) or raw_id or None

    # 3) title/company contains matching
    t = text.lower()
    best = None
    best_score = 0
    for job in jobs:
        title = str(job.get("title", "")).lower()
        company = str(job.get("company", "")).lower()
        score = 0
        if title and title in t:
            score += 2
        if company and company in t:
            score += 2
        # token overlap fallback
        for token in re.findall(r"[a-z0-9]+", t):
            if len(token) < 4:
                continue
            if token in title:
                score += 1
            if token in company:
                score += 1
        if score > best_score:
            best = job
            best_score = score

    if best and best_score > 0:
        raw_id = str(best.get("job_id", "")).strip()
        return _extract_numeric_job_id(raw_id) or raw_id or None

    return None


def _build_tool_summary_from_latest_tool(messages: list) -> Optional[str]:
    """
    Build a deterministic natural-language summary from the latest tool output.
    This avoids a second local LLM call that can stall on large tool payloads.
    """
    if not messages or not isinstance(messages[-1], ToolMessage):
        return None

    content = messages[-1].content
    payload = None
    if isinstance(content, str):
        payload = _extract_json_object(content)
    elif isinstance(content, dict):
        payload = content

    if not isinstance(payload, dict):
        return None

    # search_linkedin_jobs summary with dynamic aggregator sections (*_jobs)
    aggregator_sections: dict[str, list] = {}
    for key, value in payload.items():
        if key.endswith("_jobs") and isinstance(value, list):
            aggregator_name = key[:-5].strip().lower()
            aggregator_sections[aggregator_name] = value

    jobs = payload.get("jobs")
    if aggregator_sections or isinstance(jobs, list):
        display_name_map = {
            "linkedin": "Linked-In",
            "indeed": "Indeed",
        }

        lines = []
        if aggregator_sections:
            total_count = payload.get("count", sum(len(v) for v in aggregator_sections.values()))
            lines.append(f"Found {total_count} jobs total.")
            lines.append("")

            ordered_sources = []
            if "linkedin" in aggregator_sections:
                ordered_sources.append("linkedin")
            for src in sorted(aggregator_sections.keys()):
                if src != "linkedin":
                    ordered_sources.append(src)

            for source in ordered_sources:
                src_jobs = aggregator_sections.get(source, [])
                if not isinstance(src_jobs, list):
                    continue

                display = display_name_map.get(source, source.replace("_", " ").title())
                lines.append(f"{display} Jobs:")
                if not src_jobs:
                    lines.append("No jobs found.")
                    lines.append("")
                    continue
                for i, job in enumerate(src_jobs[:5], start=1):
                    title = job.get("title", "Unknown role")
                    company = job.get("company", "Unknown company")
                    location = job.get("location", "Unknown location")
                    raw_id = str(job.get("job_id", ""))
                    if source == "indeed":
                        description = str(job.get("description", "")).strip()
                        if len(description) > 240:
                            description = f"{description[:237]}..."
                        url = str(job.get("url", "")).strip() or "Not available"
                        id_value = raw_id
                        lines.append(f"{i}. {title} [{source}_id: {id_value}]")
                        if description:
                            lines.append(f"   Description: {description}")
                        lines.append(f"   URL: {url}")
                    elif source == "linkedin":
                        id_value = _extract_numeric_job_id(raw_id) or raw_id
                        id_label = "job_id"
                        lines.append(f"{i}. {title} at {company} ({location}) [{id_label}: {id_value}]")
                    else:
                        id_value = raw_id
                        id_label = f"{source}_id"
                        lines.append(f"{i}. {title} at {company} ({location}) [{id_label}: {id_value}]")
                lines.append("")
        else:
            count = payload.get("count", len(jobs))
            lines.append(f"Found {count} jobs.")
            lines.append("")
            for i, job in enumerate(jobs[:5], start=1):
                title = job.get("title", "Unknown role")
                company = job.get("company", "Unknown company")
                location = job.get("location", "Unknown location")
                raw_id = str(job.get("job_id", ""))
                numeric_id = _extract_numeric_job_id(raw_id) or raw_id
                lines.append(f"{i}. {title} at {company} ({location}) [job_id: {numeric_id}]")
            lines.append("")

        lines.append("You can ask:")
        lines.append("- Get details for Linked-In job 1")
        lines.append("- Show 20 more jobs")
        # Disabled for now:
        # lines.append("- Generate resume and cover letter for a chosen role")
        return "\n".join(lines)

    # get_job_details summary
    if "full_description" in payload or "criteria" in payload:
        job_id = payload.get("job_id", "unknown")
        desc = str(payload.get("full_description", "")).strip()
        excerpt = (desc[:700] + "...") if len(desc) > 700 else desc
        criteria = payload.get("criteria", {})
        criteria_text = ""
        if isinstance(criteria, dict) and criteria:
            preview = []
            for k, v in list(criteria.items())[:5]:
                preview.append(f"- {k}: {v}")
            criteria_text = "\n".join(preview)
        parts = [f"Fetched details for job_id {job_id}."]
        if excerpt:
            parts.extend(["", "Job description excerpt:", excerpt])
        if criteria_text:
            parts.extend(["", "Key criteria:", criteria_text])
        parts.extend(["", "You can ask me to show more jobs or fetch details for another role."])
        return "\n".join(parts)

    return None


def _build_local_intent_tool_call(messages: list) -> Optional[AIMessage]:
    """
    Deterministic local router for frequent intents when local LLMs miss tool routing.
    """
    user_text = _latest_user_text(messages)
    if not user_text:
        return None

    lower = user_text.lower()

    # 1) Job details intent
    if any(k in lower for k in ["job detail", "details", "description", "full description"]):
        job_id = _resolve_job_id_from_text_or_recent_jobs(messages, user_text)
        if job_id:
            return AIMessage(
                content="",
                tool_calls=[{
                    "id": "local_intent_get_job_details",
                    "name": "get_job_details",
                    "args": {"job_id": job_id},
                    "type": "tool_call",
                }],
            )

    # Disabled intent routes (kept as comments for quick restore):
    # 2) Profile intent
    # if any(k in lower for k in ["my profile", "my linkedin profile", "show profile", "get profile"]):
    #     return AIMessage(
    #         content="",
    #         tool_calls=[{
    #             "id": "local_intent_get_profile",
    #             "name": "get_my_profile",
    #             "args": {},
    #             "type": "tool_call",
    #         }],
    #     )
    #
    # 3) Full application package intent
    # if any(k in lower for k in ["application package", "full application", "resume and cover", "resume + cover"]):
    #     latest_job = _latest_job_from_messages(messages)
    #     args = {
    #         "job_title": _extract_job_title(user_text) or latest_job.get("title", "Software Engineer"),
    #         "company_name": _extract_company_name(user_text) or latest_job.get("company", "Target Company"),
    #         "job_description": "Use job details from previous context. If missing, fetch details first.",
    #         "save_files": True,
    #     }
    #     return AIMessage(
    #         content="",
    #         tool_calls=[{
    #             "id": "local_intent_generate_package",
    #             "name": "generate_application_package",
    #             "args": args,
    #             "type": "tool_call",
    #         }],
    #     )
    #
    # 4) Cover letter intent
    # if "cover letter" in lower:
    #     latest_job = _latest_job_from_messages(messages)
    #     args = {
    #         "job_title": _extract_job_title(user_text) or latest_job.get("title", "Software Engineer"),
    #         "company_name": _extract_company_name(user_text) or latest_job.get("company", "Target Company"),
    #         "job_description": "Use job details from previous context. If missing, fetch details first.",
    #     }
    #     return AIMessage(
    #         content="",
    #         tool_calls=[{
    #             "id": "local_intent_generate_cover_letter",
    #             "name": "generate_cover_letter",
    #             "args": args,
    #             "type": "tool_call",
    #         }],
    #     )
    #
    # 5) Resume intent
    if "resume" in lower and "cover letter" not in lower:
        format_choice = "professional"
        if "ats" in lower:
            format_choice = "ats"
        elif "technical" in lower:
            format_choice = "technical"

        return AIMessage(
            content="",
            tool_calls=[{
                "id": "local_intent_generate_resume",
                "name": "generate_resume",
                "args": {
                    "job_description": "Use job details from previous context. If missing, fetch details first.",
                    "format": format_choice,
                },
                "type": "tool_call",
            }],
        )

    return None


def _build_search_intent_tool_call(messages: list) -> Optional[AIMessage]:
    """
    Cross-provider deterministic search routing.
    If user asks to search/list/show jobs, force search_linkedin_jobs tool call.
    """
    user_text = _latest_user_text(messages)
    if not user_text:
        return None
    lower = user_text.lower()

    if "job" not in lower:
        return None
    if not any(k in lower for k in ["find", "search", "show", "list", "more"]):
        return None

    prev = _latest_search_params_from_messages(messages)
    source_filter = _extract_requested_sources(user_text)
    is_more = "more" in lower

    if is_more and prev:
        keywords = prev.get("keywords", "") or _extract_search_keywords(user_text)
        location = prev.get("location", "") or _extract_location(user_text)
        experience_level = prev.get("experience_level", "mid")
        job_type = prev.get("job_type", "full-time")
        remote = bool(prev.get("remote", False))
    else:
        keywords = _extract_search_keywords(user_text)
        location = _extract_location(user_text) or prev.get("location", "")
        experience_level = _guess_experience_level(user_text) if user_text else prev.get("experience_level", "mid")
        job_type = _guess_job_type(user_text) if user_text else prev.get("job_type", "full-time")
        remote = "remote" in lower or bool(prev.get("remote", False))

    # Keep tool arguments concrete.
    if not keywords:
        keywords = prev.get("keywords", "software engineer")

    args = {
        "keywords": keywords,
        "location": location,
        "experience_level": experience_level,
        "job_type": job_type,
        "remote": remote,
        "limit": 25 if is_more else 10,
        "sources": source_filter,
    }
    return AIMessage(
        content="",
        tool_calls=[{
            "id": "deterministic_search_jobs",
            "name": "search_linkedin_jobs",
            "args": args,
            "type": "tool_call",
        }],
    )

def get_scraper():
    """Lazy initialization of scraper"""
    global _scraper
    if _scraper is None:
        method = os.getenv("LINKEDIN_SCRAPER_METHOD", "auto")
        _scraper = create_linkedin_scraper(method=method)
    return _scraper

def get_cached_user_profile():
    """Get user profile (cached)"""
    global _user_profile
    if _user_profile is None:
        _user_profile = get_user_profile()
        if _user_profile:
            print(f"✅ Loaded profile for: {_user_profile.get('name', 'User')}")
        else:
            print("⚠️ Could not load user profile. Set LINKEDIN_USER_HANDLE in .env")
    return _user_profile


@tool
def search_linkedin_jobs(
    keywords: str,
    location: str = "",
    experience_level: str = "mid",
    job_type: str = "full-time",
    remote: bool = False,
    limit: int = 10,
    sources: str = ""
) -> dict:
    """
    Search for REAL jobs on LinkedIn based on criteria.
    This tool scrapes actual, current job listings from LinkedIn.
    
    Args:
        keywords: Job title or keywords to search for
        location: Location for the job (city, state, or remote)
        experience_level: Experience level (entry, mid, senior, director, executive)
        job_type: Type of job (full-time, part-time, contract, temporary, internship)
        remote: Filter for remote jobs only
        limit: Maximum number of jobs to return (default 10)
        sources: Optional CSV source filter when aggregator is enabled.
                 Example: "linkedin", "linkedin,naukri", "indeed,naukri"
    
    Returns:
        Dictionary containing list of real jobs found
    """
    try:
        scraper = get_scraper()
        
        search_aggregator = _env_true("SEARCH_AGGREGATOR")
        requested_sources = {
            s.strip().lower()
            for s in (sources or "").split(",")
            if s.strip()
        }
        if requested_sources:
            requested_sources = {
                "linkedin" if s in {"linked-in", "linked in"} else s
                for s in requested_sources
            }

        include_linkedin = (not search_aggregator) or (not requested_sources) or ("linkedin" in requested_sources)
        include_indeed = search_aggregator and ((not requested_sources) or ("indeed" in requested_sources))
        include_naukri = search_aggregator and ((not requested_sources) or ("naukri" in requested_sources))
        source_filter_fallback = False

        linkedin_jobs = []
        if include_linkedin:
            linkedin_jobs = scraper.search_jobs(
                keywords=keywords,
                location=location,
                experience_level=experience_level,
                job_type=job_type,
                remote=remote,
                limit=limit
            )

        indeed_jobs = []
        naukri_jobs = []
        if include_indeed and hasattr(scraper, "search_indeed_jobs"):
            indeed_jobs = scraper.search_indeed_jobs(
                keywords=keywords,
                location=location,
                experience_level=experience_level,
                limit=limit,
            )
        if include_naukri and hasattr(scraper, "search_naukri_jobs"):
            naukri_jobs = scraper.search_naukri_jobs(
                keywords=keywords,
                location=location,
                experience_level=experience_level,
                limit=limit,
            )

        combined_jobs = linkedin_jobs + indeed_jobs + naukri_jobs

        # If user explicitly requested one/more sources but no jobs came back,
        # fallback to all aggregators.
        if search_aggregator and requested_sources and not combined_jobs:
            source_filter_fallback = True
            include_linkedin = True
            include_indeed = True
            include_naukri = True

            linkedin_jobs = scraper.search_jobs(
                keywords=keywords,
                location=location,
                experience_level=experience_level,
                job_type=job_type,
                remote=remote,
                limit=limit
            )
            indeed_jobs = scraper.search_indeed_jobs(
                keywords=keywords,
                location=location,
                experience_level=experience_level,
                limit=limit,
            ) if hasattr(scraper, "search_indeed_jobs") else []
            naukri_jobs = scraper.search_naukri_jobs(
                keywords=keywords,
                location=location,
                experience_level=experience_level,
                limit=limit,
            ) if hasattr(scraper, "search_naukri_jobs") else []
            combined_jobs = linkedin_jobs + indeed_jobs + naukri_jobs

        result = {
            "success": True,
            "jobs": combined_jobs if search_aggregator else linkedin_jobs,
            "count": len(combined_jobs) if search_aggregator else len(linkedin_jobs),
            "search_params": {
                "keywords": keywords,
                "location": location,
                "experience_level": experience_level,
                "job_type": job_type,
                "remote": remote,
                "sources": sources,
            },
            "source": "LinkedIn (live scraping)" if not search_aggregator else "LinkedIn + Indeed + Naukri (aggregated)"
        }

        if search_aggregator:
            counts = {}
            if include_linkedin:
                result["linkedin_jobs"] = linkedin_jobs
                counts["linkedin"] = len(linkedin_jobs)
            if include_indeed:
                result["indeed_jobs"] = indeed_jobs
                counts["indeed"] = len(indeed_jobs)
            if include_naukri:
                result["naukri_jobs"] = naukri_jobs
                counts["naukri"] = len(naukri_jobs)

            result["counts"] = counts
            result["aggregated"] = True
            result["aggregator_debug"] = getattr(scraper, "last_aggregator_debug", {})
            result["selected_sources"] = sorted(
                [src for src, ok in {
                    "linkedin": include_linkedin,
                    "indeed": include_indeed,
                    "naukri": include_naukri,
                }.items() if ok]
            )
            result["source_filter_fallback"] = source_filter_fallback

        return result
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "jobs": [],
            "count": 0,
            "message": f"Failed to fetch jobs from LinkedIn: {str(e)}"
        }


@tool
def get_job_details(job_id: str) -> dict:
    """
    Get detailed information about a specific job posting from LinkedIn.
    This fetches the full job description and details from the live listing.
    
    Args:
        job_id: Unique identifier for the job
    
    Returns:
        Detailed job information including full description
    """
    try:
        scraper = get_scraper()
        
        # Get real job details
        details = scraper.get_job_details(job_id)
        
        if details:
            return {
                "success": True,
                "job_id": job_id,
                "full_description": details.get("full_description", ""),
                "criteria": details.get("criteria", {}),
                "url": details.get("url", f"https://www.linkedin.com/jobs/view/{job_id}"),
                "source": "LinkedIn (live scraping)"
            }
        else:
            return {
                "success": False,
                "error": "Could not fetch job details",
                "job_id": job_id
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "job_id": job_id,
            "message": f"Failed to fetch job details: {str(e)}"
        }


@tool
def apply_to_job(job_id: str, cover_letter: str = "") -> dict:
    """
    Apply to a job on LinkedIn (Easy Apply).
    
    Args:
        job_id: Unique identifier for the job
        cover_letter: Optional cover letter text
    
    Returns:
        Application status
    """
    # TODO: Implement actual job application
    # This would require:
    # 1. LinkedIn authentication
    # 2. Browser automation (Playwright/Selenium)
    # 3. Form filling and submission
    
    return {
        "success": True,
        "job_id": job_id,
        "status": "applied",
        "message": f"Successfully applied to job {job_id}",
        "timestamp": "2025-11-16T10:30:00Z"
    }


@tool
def get_my_profile() -> dict:
    """
    Fetch and display the user's LinkedIn profile information.
    Shows current skills, experience, education that will be used for applications.
    
    Returns:
        Dictionary containing user's profile data
    """
    try:
        user_profile = get_cached_user_profile()
        
        if not user_profile:
            return {
                "success": False,
                "error": "Could not load profile. Set LINKEDIN_USER_HANDLE in .env"
            }
        
        return {
            "success": True,
            "profile": {
                "name": user_profile.get('name'),
                "headline": user_profile.get('headline'),
                "location": user_profile.get('location'),
                "about": user_profile.get('about', '')[:200] + "..." if user_profile.get('about') else "No summary",
                "skills": user_profile.get('skills', [])[:15],
                "experience_count": len(user_profile.get('experience', [])),
                "education_count": len(user_profile.get('education', [])),
                "certifications_count": len(user_profile.get('certifications', [])),
                "url": user_profile.get('url')
            },
            "message": "Profile loaded successfully. This data will be used to generate personalized application materials."
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error fetching profile: {str(e)}"
        }


@tool
def generate_cover_letter(job_title: str, company_name: str, job_description: str) -> str:
    """
    Generate a personalized cover letter based on user's profile and job description.
    Uses the user's real LinkedIn profile data to create a tailored cover letter.
    
    Args:
        job_title: The job title
        company_name: The company name
        job_description: Full job description text
    
    Returns:
        Generated cover letter text
    """
    try:
        # Get user profile
        user_profile = get_cached_user_profile()
        
        if not user_profile:
            return "Error: Could not load user profile. Please set LINKEDIN_USER_HANDLE in .env"
        
        # Generate cover letter
        cover_letter = generate_cover_letter_for_job(
            user_profile=user_profile,
            job_title=job_title,
            company_name=company_name,
            job_description=job_description,
            tone="professional"
        )
        
        return cover_letter
        
    except Exception as e:
        return f"Error generating cover letter: {str(e)}"


@tool
def generate_resume(job_description: str, format: str = "professional") -> str:
    """
    Generate a tailored resume based on user's profile and job requirements.
    Uses the user's real LinkedIn profile data to create an optimized resume.
    
    Args:
        job_description: Full job description text
        format: Resume format (professional, ats, technical)
    
    Returns:
        Generated resume text
    """
    try:
        # Get user profile
        user_profile = get_cached_user_profile()
        
        if not user_profile:
            return "Error: Could not load user profile. Please set LINKEDIN_USER_HANDLE in .env"
        
        # Generate resume
        resume = generate_resume_for_job(
            user_profile=user_profile,
            job_description=job_description,
            format=format
        )
        
        return resume
        
    except Exception as e:
        return f"Error generating resume: {str(e)}"


@tool
def generate_application_package(
    job_title: str,
    company_name: str,
    job_description: str,
    save_files: bool = True
) -> dict:
    """
    Generate complete application package (resume + cover letter) for a job.
    Uses the user's real LinkedIn profile to create tailored materials.
    Optionally saves to files.
    
    Args:
        job_title: The job title
        company_name: The company name
        job_description: Full job description text
        save_files: Whether to save materials to files
    
    Returns:
        Dictionary with resume and cover letter
    """
    try:
        # Get user profile
        user_profile = get_cached_user_profile()
        
        if not user_profile:
            return {
                "success": False,
                "error": "Could not load user profile. Set LINKEDIN_USER_HANDLE in .env"
            }
        
        # Generate package
        package = generate_full_application(
            user_profile=user_profile,
            job_title=job_title,
            company_name=company_name,
            job_description=job_description,
            save_to_files=save_files
        )
        
        result = {
            "success": True,
            "candidate": package['candidate'],
            "job_title": package['job_title'],
            "company": package['company'],
            "resume": package.get('resume', ''),
            "cover_letter": package.get('cover_letter', ''),
        }
        
        if save_files and 'saved_files' in package:
            result['saved_files'] = package['saved_files']
            result['message'] = f"Application materials saved to {len(package['saved_files'])} files"
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error generating application package: {str(e)}"
        }


@tool
def get_my_profile() -> dict:
    """
    Fetch and display the user's LinkedIn profile information.
    Shows current skills, experience, education that will be used for applications.
    
    Returns:
        Dictionary containing user's profile data
    """
    try:
        user_profile = get_cached_user_profile()
        
        if not user_profile:
            return {
                "success": False,
                "error": "Could not load profile. Set LINKEDIN_USER_HANDLE in .env"
            }
        
        return {
            "success": True,
            "profile": {
                "name": user_profile.get('name'),
                "headline": user_profile.get('headline'),
                "location": user_profile.get('location'),
                "about": user_profile.get('about', '')[:200] + "...",
                "skills": user_profile.get('skills', [])[:15],
                "experience_count": len(user_profile.get('experience', [])),
                "education_count": len(user_profile.get('education', [])),
                "url": user_profile.get('url')
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error fetching profile: {str(e)}"
        }


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def agent_node(state: AgentState) -> AgentState:
    """
    Main agent node that decides what action to take.
    Uses the LLM to determine next steps based on conversation.
    """
    messages = state["messages"]
    llm_provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    search_aggregator_enabled = _env_true("SEARCH_AGGREGATOR")

    # Deterministic post-tool summary for all providers to keep output stable
    # and ensure aggregator sections are properly separated.
    if llm_provider == "ollama" and messages and isinstance(messages[-1], ToolMessage):
        deterministic_summary = _build_tool_summary_from_latest_tool(messages)
        if deterministic_summary:
            return {"messages": [AIMessage(content=deterministic_summary)]}

    # Local models sometimes miss structured tool routing; use deterministic intent fallback.
    if llm_provider == "ollama":
        # If tools already ran for the latest user request, force a final summary step
        # without tool binding to prevent re-calling the same tools in a loop.
        if messages and isinstance(messages[-1], ToolMessage):
            summarizer = create_chat_model(temperature=0, max_tokens=2048)
            no_tool_system = SystemMessage(content="""
            You are a LinkedIn job assistant.
            The latest tool result is already available in conversation.
            Summarize the result clearly for the user and suggest concise next actions.
            Do NOT call tools in this response.
            """)
            summary_response = summarizer.invoke([no_tool_system] + messages)
            return {"messages": [summary_response]}

        # Deterministic intent routing should only happen on fresh user input,
        # not after tool outputs for the same turn.
        if not _has_tool_after_latest_human(messages):
            search_tool_call = _build_search_intent_tool_call(messages)
            if search_tool_call is not None:
                return {"messages": [search_tool_call]}
            routed_tool_call = _build_local_intent_tool_call(messages)
            if routed_tool_call is not None:
                return {"messages": [routed_tool_call]}
    
    # Initialize the LLM with tools - Using Claude Sonnet 4
    llm = create_chat_model(temperature=0, max_tokens=4096)
    tools = [
        search_linkedin_jobs,
        get_job_details,
        generate_resume,
        # Disabled for now (re-enable later if needed):
        # apply_to_job,
        # generate_cover_letter,
        # generate_application_package,
        # get_my_profile,
    ]
    llm_with_tools = llm.bind_tools(tools)
    
    local_tool_call_instructions = ""
    if llm_provider == "ollama":
        local_tool_call_instructions = """
    IMPORTANT TOOL-CALLING RULES (LOCAL MODEL):
    - If a tool is needed, CALL the tool via the model's tool-calling interface.
    - Do NOT return a JSON blob in normal text like {"name": "...", "arguments": {...}}.
    - Do NOT describe a tool call. Actually perform the tool call.
    - If user asks to search jobs, call search_linkedin_jobs immediately with concrete arguments.
    - For follow-up source filters, pass sources CSV:
      linkedin / indeed / naukri (e.g., "linkedin,naukri").
    - After tool results are returned, summarize results clearly for the user.
    """

    aggregator_instructions = ""
    if search_aggregator_enabled:
        aggregator_instructions = """
    SEARCH AGGREGATOR MODE IS ENABLED.
    - When user asks to search jobs, call search_linkedin_jobs.
    - The tool aggregates from multiple providers (Linked-In, Indeed, Naukri).
    - Respect source-specific follow-ups by passing sources CSV in tool args:
      "linkedin", "indeed", "naukri", or combinations like "linkedin,naukri".
    - For "show me more ..." follow-ups, reuse previous search criteria unless user changes them.
    - If tool output contains one or more *_jobs lists, present each source in separate sections.
    - Keep the source section headers explicit, e.g., "Linked-In Jobs", "Indeed Jobs", "Naukri Jobs".
    """

    # System message for the agent
    system_message = SystemMessage(content=f"""
    You are an intelligent LinkedIn job search assistant.
    
    Your capabilities:
    1. Search for real jobs on LinkedIn based on user criteria
    2. Get detailed information about specific jobs
    3. Generate a tailored resume when the user explicitly asks for it
    
    Always:
    - Provide clear summaries of job matches
    - Help users refine their search criteria
    - Be proactive in suggesting relevant actions
    - Keep suggestions limited to:
      - Get job details
      - Show more jobs
    - Do not proactively suggest cover letter, profile fetch, or application package actions
    - If tool output includes one or more aggregator lists like *_jobs, present each source in a separate section.
    {aggregator_instructions}
    
    When searching for jobs, consider:
    - Keywords/job titles
    - Location preferences
    - Experience level
    - Job type (full-time, contract, etc.)
    - Remote options
    {local_tool_call_instructions}
    """)
    
    # Invoke the LLM
    response = llm_with_tools.invoke([system_message] + messages)
    response = _coerce_text_tool_call(response)
    
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Conditional edge function to determine if we should continue to tools or end.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, continue to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the conversation
    return "end"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_linkedin_agent() -> StateGraph:
    """
    Create the LinkedIn job search agent graph.
    """
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    
    # Create tools node
    tools = [
        search_linkedin_jobs,
        get_job_details,
        generate_resume,
        # Disabled for now (re-enable later if needed):
        # apply_to_job,
        # generate_cover_letter,
        # generate_application_package,
        # get_my_profile,
    ]
    workflow.add_node("tools", ToolNode(tools))
    
    # Add edges
    workflow.add_edge(START, "agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After tools, go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    return workflow.compile()


# ============================================================================
# MAIN GRAPH INSTANCE
# ============================================================================

# Create the compiled graph for LangGraph server
graph = create_linkedin_agent()


# ============================================================================
# TESTING / USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage for testing the agent locally.
    When using LangGraph Studio, you can test directly in Chat mode!
    """
    print("LinkedIn Job Search Agent - Testing Mode")
    print("=" * 50)
    print("\n💡 TIP: Run 'langgraph dev' and use Chat mode in LangGraph Studio")
    print("   for the best experience!\n")
    
    # Test the agent with a sample query
    initial_state = {
        "messages": [
            HumanMessage(content="Find me AI engineer jobs in Noida with remote options")
        ],
        "job_search_params": {},
        "found_jobs": [],
        "applied_jobs": [],
        "next_action": ""
    }
    
    # Run the agent
    result = graph.invoke(initial_state)
    
    # Print results
    print("\nAgent Response:")
    print("-" * 50)
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(f"AI: {message.content}")
        elif isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
    
    print("\n" + "=" * 50)
    print("✅ Test complete!")
    print("\n🚀 To use Chat mode:")
    print("   1. Run: langgraph dev")
    print("   2. Click the 'Chat' tab in LangGraph Studio")
    print("   3. Start chatting with your agent!")
