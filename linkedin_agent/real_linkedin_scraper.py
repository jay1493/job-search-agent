"""
Real LinkedIn Job Scraper Implementation
Fetches actual job data from LinkedIn using multiple methods
"""

import os
import time
import json
import ast
import re
from typing import List, Dict, Optional
from urllib.parse import quote, parse_qs, urlparse
import requests
from bs4 import BeautifulSoup

# ============================================================================
# METHOD 1: Public LinkedIn Jobs Scraper (No Authentication Required)
# ============================================================================

class LinkedInJobScraper:
    """
    Scrapes public LinkedIn job listings without authentication.
    Uses the public jobs search page.
    """
    
    def __init__(self):
        self.base_url = "https://in.linkedin.com/jobs/search"
        self.see_more_url = "https://in.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
        self.typeahead_url = "https://in.linkedin.com/jobs-guest/api/typeaheadHits"
        self.search_strategy = os.getenv("LINKEDIN_SEARCH_STRATEGY", "auto").strip().lower()
        self.session = requests.Session()
        self.last_aggregator_debug: Dict[str, Dict] = {}
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    def _extract_exp_years_from_text(self, text: str) -> Optional[int]:
        """Extract explicit years from user keyword text, e.g. '6 years'."""
        if not text:
            return None
        m = re.search(r"\b(\d{1,2})\s*(?:\+?\s*)?(?:year|years|yr|yrs)\b", text.lower())
        if m:
            return int(m.group(1))
        return None

    def _normalize_naukri_location(self, location: str) -> str:
        """Normalize location as Naukri commonly expects (e.g., 'delhi / ncr')."""
        if not location:
            return ""
        loc = re.sub(r"\s+", " ", location.strip())
        loc_lower = loc.lower()
        if loc_lower in {"delhi ncr", "delhi-ncr", "delhi/ncr", "delhi / ncr", "ncr", "delhi ncr location"}:
            return "delhi / ncr"
        return loc_lower

    def _normalize_serpapi_location(self, location: str) -> str:
        """Normalize location for SerpAPI geo hint."""
        if not location:
            return "India"
        loc = re.sub(r"\s+", " ", location.strip())
        loc_lower = loc.lower()
        if loc_lower in {"delhi ncr", "delhi-ncr", "delhi/ncr", "delhi / ncr", "ncr", "delhi ncr location"}:
            return "Delhi, India"
        if "," in loc:
            return loc
        return f"{loc}, India"

    def _safe_json_response(self, response: requests.Response):
        """Parse JSON with fallbacks for prefixed/non-strict payloads."""
        text = response.text or ""

        # 1) Strict JSON parse.
        try:
            return response.json()
        except Exception:
            pass

        # 2) Strip common anti-JSON prefixes and parse object slice.
        cleaned = text.lstrip()
        cleaned = re.sub(r"^\)\]\}',?\s*", "", cleaned)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start:end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                pass

            # 3) Python/JSON-ish fallback (single quotes etc.).
            try:
                py_like = re.sub(r"\btrue\b", "True", snippet, flags=re.IGNORECASE)
                py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
                py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
                parsed = ast.literal_eval(py_like)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            # 4) As a last resort, extract jobDetails array only.
            m = re.search(r'(?:"jobDetails"|\'jobDetails\'|jobDetails)\s*:\s*\[', snippet)
            if m:
                array_start = m.end() - 1
                depth = 0
                in_string = False
                escaped = False
                quote_char = ""
                for idx in range(array_start, len(snippet)):
                    ch = snippet[idx]
                    if in_string:
                        if escaped:
                            escaped = False
                        elif ch == "\\":
                            escaped = True
                        elif ch == quote_char:
                            in_string = False
                        continue
                    if ch in {'"', "'"}:
                        in_string = True
                        quote_char = ch
                        continue
                    if ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0:
                            arr_text = snippet[array_start:idx + 1]
                            try:
                                return {"jobDetails": json.loads(arr_text)}
                            except Exception:
                                try:
                                    py_like = re.sub(r"\btrue\b", "True", arr_text, flags=re.IGNORECASE)
                                    py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
                                    py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
                                    return {"jobDetails": ast.literal_eval(py_like)}
                                except Exception:
                                    break

        raise ValueError("Unable to parse Naukri response JSON payload.")

    def _parse_indeed_jobs_html(self, html: str, limit: int) -> List[Dict]:
        """Parse Indeed SRP HTML into normalized job objects."""
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.find_all("li", class_=re.compile(r"css-1ac2h1w"))
        if not cards:
            cards = soup.find_all("div", class_=re.compile(r"\bjob_seen_beacon\b"))

        jobs: List[Dict] = []
        for card in cards:
            if len(jobs) >= limit:
                break

            anchor = card.find("a", attrs={"data-jk": True}) or card.find("a", href=True)
            if not anchor:
                continue

            job_jk = anchor.get("data-jk", "").strip()
            href = anchor.get("href", "").strip()
            if not job_jk and "/rc/clk" not in href and "/viewjob" not in href:
                continue

            title_span = anchor.find("span")
            title = ""
            if title_span and title_span.get_text(strip=True):
                title = title_span.get_text(strip=True)
            elif anchor.get("title"):
                title = anchor.get("title").strip()
            if not title:
                continue

            company_elem = card.find("span", attrs={"data-testid": "company-name"})
            location_elem = card.find("div", attrs={"data-testid": "text-location"})
            snippet_elem = card.find("div", attrs={"data-testid": "belowJobSnippet"})

            company = company_elem.get_text(strip=True) if company_elem else "Unknown"
            job_location = location_elem.get_text(strip=True) if location_elem else "Not specified"
            snippet = snippet_elem.get_text(" ", strip=True) if snippet_elem else ""

            if href.startswith("/"):
                job_url = f"https://in.indeed.com{href}"
            elif href.startswith("http"):
                job_url = href
            else:
                job_url = f"https://in.indeed.com/{href.lstrip('/')}"

            jobs.append({
                "job_id": job_jk or href,
                "title": title,
                "company": company,
                "location": job_location,
                "description": snippet,
                "url": job_url,
                "posted_date": "Unknown",
                "easy_apply": False,
                "source": "Indeed",
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        return jobs

    def _slugify_indeed_part(self, value: str) -> str:
        """Convert free text to Indeed path-style slug."""
        if not value:
            return ""
        slug = re.sub(r"[^a-zA-Z0-9\s\-]+", "", value.lower())
        slug = re.sub(r"\s+", "-", slug).strip("-")
        return slug

    def _is_indeed_url(self, url: str) -> bool:
        """Return True for Indeed domains only."""
        if not url:
            return False
        try:
            host = (urlparse(url).netloc or "").lower()
        except Exception:
            return False
        return bool(re.search(r"(^|\.)indeed\.[a-z.]+$", host))

    def _extract_indeed_job_id(self, url: str) -> str:
        """Extract Indeed job identifier from common query params."""
        if not url:
            return ""

        try:
            parsed = urlparse(url)
            query = parse_qs(parsed.query or "")
            for key in ("jk", "vjk"):
                values = query.get(key) or []
                if values and values[0]:
                    return values[0]
        except Exception:
            pass

        match = re.search(r"(?:jk|vjk)=([a-zA-Z0-9]+)", url)
        if match:
            return match.group(1)
        return url

    def _search_indeed_jobs_via_serpapi(
        self,
        keywords: str,
        location: str = "",
        experience_level: str = "",
        limit: int = 10,
    ) -> Dict:
        """
        Search Indeed job URLs using SerpAPI (Google engine).
        Returns both normalized jobs and debug metadata.
        """
        api_key = os.getenv("SERPAPI_API_KEY") or os.getenv("SERP_API_KEY")

        if not api_key:
            return {
                "jobs": [],
                "debug": {
                    "selected_strategy": "serpapi_google_search",
                    "api_configured": False,
                    "query": None,
                    "parsed_count": 0,
                },
            }

        query_parts = [keywords.strip()]
        if experience_level:
            query_parts.append(experience_level.strip())
        query_base = " ".join(part for part in query_parts if part).strip()
        query = f"{query_base} site:(indeed.com OR in.indeed.com)".strip()
        serp_location = self._normalize_serpapi_location(location)

        jobs: List[Dict] = []
        start_index = 0
        page_count = 0
        raw_item_count = 0

        while len(jobs) < limit:
            per_page = min(10, limit - len(jobs))
            params = {
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": per_page,
                "start": start_index,
                "location": serp_location,
            }

            try:
                response = requests.get(
                    "https://serpapi.com/search.json",
                    params=params,
                    timeout=15,
                )
                if response.status_code in {401, 403}:
                    reason = ""
                    try:
                        err_payload = response.json()
                        if isinstance(err_payload, dict):
                            reason = str(
                                err_payload.get("error")
                                or err_payload.get("message")
                                or err_payload
                            )
                    except Exception:
                        reason = (response.text or "").strip()[:400]

                    return {
                        "jobs": jobs[:limit],
                        "debug": {
                            "selected_strategy": "serpapi_google_search",
                            "api_configured": True,
                            "query": query,
                            "location": serp_location,
                            "page_count": page_count,
                            "raw_item_count": raw_item_count,
                            "parsed_count": len(jobs[:limit]),
                            "status_code": response.status_code,
                            "error": "SerpAPI authorization failed",
                            "serpapi_error": reason,
                            "hint": (
                                "Verify SERPAPI_API_KEY is valid, active, and has remaining credits."
                            ),
                        },
                    }

                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and data.get("error"):
                    return {
                        "jobs": jobs[:limit],
                        "debug": {
                            "selected_strategy": "serpapi_google_search",
                            "api_configured": True,
                            "query": query,
                            "location": serp_location,
                            "page_count": page_count,
                            "raw_item_count": raw_item_count,
                            "parsed_count": len(jobs[:limit]),
                            "error": f"SerpAPI error: {data.get('error')}",
                        },
                    }
            except Exception as e:
                return {
                    "jobs": jobs[:limit],
                    "debug": {
                        "selected_strategy": "serpapi_google_search",
                        "api_configured": True,
                        "query": query,
                        "location": serp_location,
                        "page_count": page_count,
                        "raw_item_count": raw_item_count,
                        "parsed_count": len(jobs[:limit]),
                        "error": str(e),
                    },
                }

            items = data.get("organic_results", []) if isinstance(data, dict) else []
            if not items:
                break

            page_count += 1
            raw_item_count += len(items)

            for item in items:
                url = item.get("link", "").strip()
                if not self._is_indeed_url(url):
                    continue

                title = (item.get("title") or "Indeed Job").strip()
                title = re.sub(
                    r"\s*[-|]\s*Indeed(?:\.com| India| Canada| UK)?\s*$",
                    "",
                    title,
                    flags=re.IGNORECASE,
                ).strip()
                snippet = (item.get("snippet") or "").strip()
                job_id = self._extract_indeed_job_id(url)

                jobs.append({
                    "job_id": job_id,
                    "title": title or "Indeed Job",
                    "company": "Unknown",
                    "location": location or "Not specified",
                    "description": snippet,
                    "url": url,
                    "posted_date": "Unknown",
                    "easy_apply": False,
                    "source": "Indeed (SerpAPI)",
                    "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

                if len(jobs) >= limit:
                    break

            if len(items) < per_page:
                break
            start_index += per_page

        return {
            "jobs": jobs[:limit],
            "debug": {
                "selected_strategy": "serpapi_google_search",
                "api_configured": True,
                "query": query,
                "location": serp_location,
                "page_count": page_count,
                "raw_item_count": raw_item_count,
                "parsed_count": len(jobs[:limit]),
            },
        }

    def search_indeed_jobs(
        self,
        keywords: str,
        location: str = "",
        experience_level: str = "",
        limit: int = 10,
        max_retries: int = 5,
    ) -> List[Dict]:
        """
        Search Indeed jobs via SerpAPI and return Indeed URLs only.
        Direct Indeed scraping is avoided due anti-bot/captcha blocks.
        """
        _ = max_retries  # Retained for interface compatibility.
        if not keywords.strip():
            self.last_aggregator_debug["indeed"] = {
                "selected_strategy": "serpapi_google_search",
                "api_configured": bool(os.getenv("SERPAPI_API_KEY") or os.getenv("SERP_API_KEY")),
                "query": None,
                "parsed_count": 0,
                "error": "Missing keywords",
            }
            return []

        result = self._search_indeed_jobs_via_serpapi(
            keywords=keywords,
            location=location,
            experience_level=experience_level,
            limit=limit,
        )

        debug = result.get("debug", {})
        debug["input"] = {
            "keywords": keywords,
            "location": location,
            "experience_level": experience_level,
            "limit": limit,
        }
        self.last_aggregator_debug["indeed"] = debug
        return result.get("jobs", [])

    def _infer_experience_years(self, keywords: str, experience_level: str = "") -> int:
        """Infer numeric experience years only from explicit numeric text."""
        text = f"{keywords} {experience_level}".lower()
        m = re.search(r"\b(\d{1,2})\s*(?:\+?\s*)?(?:year|years|yr|yrs)\b", text)
        if m:
            return int(m.group(1))
        m_range = re.search(r"\b(\d{1,2})\s*(?:-|to)\s*(\d{1,2})\b", text)
        if m_range:
            return int(m_range.group(1))
        m_plain = re.search(r"\b(\d{1,2})\b", (experience_level or "").strip())
        if m_plain:
            return int(m_plain.group(1))
        return 0

    def search_naukri_jobs(
        self,
        keywords: str,
        location: str = "",
        experience_level: str = "",
        limit: int = 10,
    ) -> List[Dict]:
        """
        Search jobs from Naukri SRP API.
        Supports page-based pagination via pageNo.
        """
        base_url = "https://www.naukri.com/jobapi/v3/search"
        jobs: List[Dict] = []
        page_no = 1

        debug = {
            "input": {
                "keywords": keywords,
                "location": location,
                "normalized_location": self._normalize_naukri_location(location),
                "experience_level": experience_level,
            },
            "pages": [],
        }

        max_pages = max(1, ((limit - 1) // 20) + 1)
        normalized_location = self._normalize_naukri_location(location)
        normalized_keywords = re.sub(r"\s+", " ", (keywords or "").strip()).lower()

        while len(jobs) < limit and page_no <= max_pages:
            params = {
                "noOfResults": 20,
                "urlType": "search_by_key_loc",
                "searchType": "adv",
                "location": normalized_location or location,
                "keyword": normalized_keywords or keywords,
                "pageNo": page_no,
                "k": normalized_keywords or keywords,
                "l": normalized_location or location,
                "src": "directSearch",
            }
            experience_years = self._infer_experience_years(keywords, experience_level)
            if experience_years > 0:
                # Keep duplicate experience param as observed in provided curl.
                params["experience"] = [str(experience_years), str(experience_years)]

            headers = {
                "appid": "109",
                "systemid": "Naukri",
            }

            try:
                # Keep request structure aligned with known-good curl shape.
                response = requests.get(base_url, params=params, headers=headers, timeout=15)
                response.raise_for_status()
                data = self._safe_json_response(response)
                debug["pages"].append({
                    "pageNo": page_no,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                    "params": {
                        "keyword": params["keyword"],
                        "location": params["location"],
                        "pageNo": params["pageNo"],
                        "experience": params.get("experience"),
                    },
                })
            except Exception as e:
                print(f"Naukri request failed on page {page_no}: {e}")
                debug["pages"].append({
                    "pageNo": page_no,
                    "error": str(e),
                    "params": {
                        "keyword": params["keyword"],
                        "location": params["location"],
                        "pageNo": params["pageNo"],
                        "experience": params.get("experience"),
                    },
                })
                break

            if not isinstance(data, dict):
                debug["pages"].append({
                    "pageNo": page_no,
                    "error": "Naukri response was not an object",
                })
                break

            details = data.get("jobDetails", [])
            if not isinstance(details, list) or not details:
                break

            for item in details:
                if len(jobs) >= limit:
                    break
                if not isinstance(item, dict):
                    continue

                job_id = str(item.get("jobId", "")).strip()
                title = item.get("title", "Unknown")
                company = item.get("companyName", "Unknown")

                placeholders = item.get("placeholders", [])
                location_value = "Not specified"
                if isinstance(placeholders, list):
                    for ph in placeholders:
                        if isinstance(ph, dict) and ph.get("type") == "location":
                            location_value = ph.get("label", location_value)
                            break

                jd_url = item.get("jdURL", "")
                if isinstance(jd_url, str) and jd_url:
                    url = f"https://www.naukri.com{jd_url}" if jd_url.startswith("/") else jd_url
                else:
                    url = f"https://www.naukri.com/{job_id}" if job_id else "https://www.naukri.com"

                raw_description = item.get("jobDescription", "")
                description = BeautifulSoup(str(raw_description), "html.parser").get_text(" ", strip=True)

                jobs.append({
                    "job_id": job_id or url,
                    "title": title,
                    "company": company,
                    "location": location_value,
                    "description": description,
                    "url": url,
                    "posted_date": item.get("footerPlaceholderLabel", "Unknown"),
                    "easy_apply": False,
                    "source": "Naukri",
                    "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

            page_no += 1
            time.sleep(0.5)

        debug["parsed_count"] = len(jobs[:limit])
        self.last_aggregator_debug["naukri"] = debug
        return jobs[:limit]

    def _normalize_india_location(self, location: str) -> str:
        """Append ', India' when location is provided without country suffix."""
        if not location:
            return location
        loc = location.strip()
        loc_lower = loc.lower()
        if loc_lower.endswith(", india") or loc_lower == "india":
            return loc
        return f"{loc}, India"

    def _resolve_india_geo_id(self, location: str) -> Optional[str]:
        """
        Resolve geoId via LinkedIn typeahead and keep only results ending with ', India'.
        """
        if not location:
            return None

        try:
            response = self.session.get(
                self.typeahead_url,
                params={
                    "query": location,
                    "typeaheadType": "GEO",
                    "geoTypes": "POPULATED_PLACE,ADMIN_DIVISION_2,MARKET_AREA,COUNTRY_REGION",
                },
                timeout=10,
            )
            response.raise_for_status()
            hits = response.json()
        except Exception as e:
            print(f"Typeahead geo lookup failed: {e}")
            return None

        if not isinstance(hits, list):
            return None

        india_hits = [
            item for item in hits
            if str(item.get("displayName", "")).strip().endswith(", India")
        ]
        if not india_hits:
            return None

        location_lower = location.lower().strip()
        for item in india_hits:
            display = str(item.get("displayName", "")).lower()
            if display.startswith(location_lower):
                return str(item.get("id"))

        return str(india_hits[0].get("id"))
    
    def search_jobs(
        self,
        keywords: str,
        location: str = "",
        experience_level: str = "",
        job_type: str = "",
        remote: bool = False,
        limit: int = 25
    ) -> List[Dict]:
        """
        Search for jobs on LinkedIn public jobs page.
        
        Args:
            keywords: Job title or keywords
            location: Location string
            experience_level: Entry level, Mid-Senior level, etc.
            job_type: Full-time, Part-time, Contract, etc.
            remote: Filter for remote jobs
            limit: Maximum number of jobs to return
            
        Returns:
            List of job dictionaries
        """
        if self.search_strategy == "google":
            return self._search_jobs_via_google(
                keywords=keywords,
                location=location,
                experience_level=experience_level,
                job_type=job_type,
                remote=remote,
                limit=limit,
            )

        location_with_country = self._normalize_india_location(location)
        geo_id = self._resolve_india_geo_id(location)

        # Build search URL
        params = {
            'keywords': keywords,
            'location': location_with_country,
            'trk': 'public_jobs_jobs-search-bar_search-submit',
            'position': 1,
            'pageNum': 0,
            'start': 0
        }
        if geo_id:
            params['geoId'] = geo_id
        
        # Add filters
        filters = []
        
        # Experience level mapping
        experience_map = {
            'entry': '2',
            'mid': '3',
            'senior': '4',
            'director': '5',
            'executive': '6'
        }
        
        # Job type mapping
        job_type_map = {
            'full-time': 'F',
            'part-time': 'P',
            'contract': 'C',
            'temporary': 'T',
            'internship': 'I'
        }
        
        if experience_level and experience_level.lower() in experience_map:
            filters.append(f"f_E={experience_map[experience_level.lower()]}")
        
        if job_type and job_type.lower() in job_type_map:
            filters.append(f"f_JT={job_type_map[job_type.lower()]}")
        
        if remote:
            filters.append("f_WT=2")  # Remote work
        
        if filters:
            params['f'] = '&'.join(filters)
        
        jobs = []
        page = 0
        
        while len(jobs) < limit:
            if page == 0:
                request_url = self.base_url
                params['position'] = 1
                params['pageNum'] = 0
                params['start'] = 0
            else:
                request_url = self.see_more_url
                params['position'] = page * 25
                params['pageNum'] = page
                params['start'] = page * 25
            
            try:
                # Make request
                response = self.session.get(
                    request_url,
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find job cards
                job_cards = soup.find_all('div', class_='base-card')
                
                if not job_cards:
                    # Try alternative class name
                    job_cards = soup.find_all('div', class_='job-search-card')
                
                if not job_cards:
                    print(f"No jobs found on page {page + 1}")
                    break
                
                for card in job_cards:
                    if len(jobs) >= limit:
                        break
                    
                    try:
                        job = self._parse_job_card(card)
                        if job:
                            jobs.append(job)
                    except Exception as e:
                        print(f"Error parsing job card: {e}")
                        continue
                
                page += 1
                
                # Be respectful - add delay
                time.sleep(2)
                
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                break
            except Exception as e:
                print(f"Error fetching jobs: {e}")
                break
        
        jobs = jobs[:limit]

        # Auto mode: if LinkedIn page scrape did not return enough jobs,
        # top up with Google Search API LinkedIn URLs (if configured).
        if self.search_strategy == "auto" and len(jobs) < limit:
            google_jobs = self._search_jobs_via_google(
                keywords=keywords,
                location=location_with_country,
                experience_level=experience_level,
                job_type=job_type,
                remote=remote,
                limit=limit,
            )
            if google_jobs:
                existing_keys = {j.get("job_id") or j.get("url") for j in jobs}
                for g_job in google_jobs:
                    key = g_job.get("job_id") or g_job.get("url")
                    if key in existing_keys:
                        continue
                    jobs.append(g_job)
                    existing_keys.add(key)
                    if len(jobs) >= limit:
                        break

        return jobs[:limit]

    def _search_jobs_via_google(
        self,
        keywords: str,
        location: str = "",
        experience_level: str = "",
        job_type: str = "",
        remote: bool = False,
        limit: int = 25,
    ) -> List[Dict]:
        """
        Search LinkedIn job URLs through Google Custom Search JSON API.

        Requires:
        - GOOGLE_SEARCH_API_KEY
        - GOOGLE_SEARCH_ENGINE_ID
        """
        api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        cse_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

        if not api_key or not cse_id:
            print("Google Search API not configured. Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID.")
            return []

        query_parts = [keywords, "LinkedIn jobs", "site:linkedin.com/jobs/view"]
        if location:
            query_parts.append(location)
        if experience_level:
            query_parts.append(f"{experience_level} experience")
        if job_type:
            query_parts.append(job_type)
        if remote:
            query_parts.append("remote")

        query = " ".join(query_parts)
        jobs = []
        start_index = 1

        while len(jobs) < limit:
            per_page = min(10, limit - len(jobs))
            params = {
                "key": api_key,
                "cx": cse_id,
                "q": query,
                "num": per_page,
                "start": start_index,
            }

            try:
                response = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=params,
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print(f"Google Search API error: {e}")
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                url = item.get("link", "")
                if "linkedin.com/jobs/view" not in url:
                    continue

                title = item.get("title", "LinkedIn Job")
                snippet = item.get("snippet", "")
                job_id = self._extract_job_id(url)

                jobs.append({
                    "job_id": job_id,
                    "title": title.replace(" | LinkedIn", "").replace(" - LinkedIn", ""),
                    "company": "Unknown",
                    "location": location or "Not specified",
                    "description": snippet,
                    "url": url,
                    "posted_date": "Unknown",
                    "easy_apply": False,
                    "source": "Google Search API",
                    "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

                if len(jobs) >= limit:
                    break

            next_page = data.get("queries", {}).get("nextPage", [])
            if not next_page:
                break
            start_index = next_page[0].get("startIndex", start_index + per_page)

        return jobs[:limit]
    
    def _parse_job_card(self, card) -> Optional[Dict]:
        """Parse individual job card from HTML"""
        try:
            # Extract job ID and link
            link_elem = card.find('a', class_='base-card__full-link') or \
                       card.find('a', href=re.compile(r'/jobs/view/'))
            
            if not link_elem:
                return None
            
            job_link = link_elem.get('href', '')
            job_id = self._extract_job_id(job_link)
            
            # Extract title
            title_elem = card.find('h3', class_='base-search-card__title') or \
                        card.find('span', class_='sr-only')
            title = title_elem.get_text(strip=True) if title_elem else "Unknown"
            
            # Extract company
            company_elem = card.find('h4', class_='base-search-card__subtitle') or \
                          card.find('a', class_='hidden-nested-link')
            company = company_elem.get_text(strip=True) if company_elem else "Unknown"
            
            # Extract location
            location_elem = card.find('span', class_='job-search-card__location')
            location = location_elem.get_text(strip=True) if location_elem else "Not specified"
            
            # Extract date posted
            date_elem = card.find('time')
            posted_date = date_elem.get('datetime', 'Unknown') if date_elem else "Unknown"
            
            # Extract description snippet (if available)
            desc_elem = card.find('p', class_='base-search-card__snippet')
            description = desc_elem.get_text(strip=True) if desc_elem else ""
            
            return {
                'job_id': job_id,
                'title': title,
                'company': company,
                'location': location,
                'description': description,
                'url': f"https://in.linkedin.com{job_link}" if job_link.startswith('/') else job_link,
                'posted_date': posted_date,
                'easy_apply': False,  # Can't determine from public page
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        except Exception as e:
            print(f"Error parsing job card: {e}")
            return None
    
    def _extract_job_id(self, url: str) -> str:
        """Extract job ID from URL"""
        match = re.search(r'/jobs/view/(\d+)', url)
        return match.group(1) if match else url.split('/')[-1]
    
    def get_job_details(self, job_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific job.
        
        Args:
            job_id: LinkedIn job ID
            
        Returns:
            Detailed job information
        """
        url = f"https://in.linkedin.com/jobs/view/{job_id}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract full description
            desc_elem = soup.find('div', class_='show-more-less-html__markup')
            full_description = desc_elem.get_text(strip=True) if desc_elem else ""
            
            # Extract criteria (seniority, employment type, etc.)
            criteria = {}
            criteria_items = soup.find_all('li', class_='description__job-criteria-item')
            for item in criteria_items:
                label = item.find('h3')
                value = item.find('span')
                if label and value:
                    criteria[label.get_text(strip=True)] = value.get_text(strip=True)
            
            return {
                'job_id': job_id,
                'full_description': full_description,
                'criteria': criteria,
                'url': url
            }
        
        except Exception as e:
            print(f"Error fetching job details: {e}")
            return None


# ============================================================================
# METHOD 2: LinkedIn Jobs API (Using linkedin-jobs-scraper library)
# ============================================================================

class LinkedInJobsLibraryScraper:
    """
    Uses the linkedin-jobs-scraper library for more robust scraping.
    Requires: pip install linkedin-jobs-scraper
    """
    
    def __init__(self):
        try:
            from linkedin_jobs_scraper import LinkedinScraper
            from linkedin_jobs_scraper.events import Events, EventData
            from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
            from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters, ExperienceLevelFilters
            
            self.LinkedinScraper = LinkedinScraper
            self.Events = Events
            self.Query = Query
            self.QueryOptions = QueryOptions
            self.QueryFilters = QueryFilters
            self.RelevanceFilters = RelevanceFilters
            self.TimeFilters = TimeFilters
            self.TypeFilters = TypeFilters
            self.ExperienceLevelFilters = ExperienceLevelFilters
            
            self.scraper = None
            self.jobs_data = []
            
        except ImportError:
            print("linkedin-jobs-scraper not installed. Install with: pip install linkedin-jobs-scraper")
            raise
    
    def search_jobs(
        self,
        keywords: str,
        location: str = "",
        limit: int = 25,
        experience_level: str = "",
        job_type: str = ""
    ) -> List[Dict]:
        """
        Search for jobs using linkedin-jobs-scraper library.
        """
        self.jobs_data = []
        
        # Event handlers
        def on_data(data):
            """Called when job data is scraped"""
            job = {
                'job_id': data.job_id,
                'title': data.title,
                'company': data.company,
                'location': data.place,
                'description': data.description,
                'description_html': data.description_html,
                'url': data.link,
                'apply_link': data.apply_link,
                'posted_date': data.date,
                'insights': data.insights,
                'company_link': data.company_link,
                'company_img_link': data.company_img_link,
            }
            self.jobs_data.append(job)
            
            if len(self.jobs_data) >= limit:
                self.scraper.close()
        
        def on_error(error):
            print(f"Scraper error: {error}")
        
        def on_end():
            print(f"Scraping completed. Found {len(self.jobs_data)} jobs.")
        
        # Initialize scraper
        self.scraper = self.LinkedinScraper(
            chrome_options=None,  # Use default Chrome options
            headless=True,  # Run in headless mode
            max_workers=1,  # Number of concurrent workers
            slow_mo=0.5  # Slow down scraping to avoid rate limits
        )
        
        # Add event listeners
        self.scraper.on(self.Events.DATA, on_data)
        self.scraper.on(self.Events.ERROR, on_error)
        self.scraper.on(self.Events.END, on_end)
        
        # Build query with filters
        query_filters = self.QueryFilters()
        
        if experience_level:
            exp_map = {
                'entry': self.ExperienceLevelFilters.ENTRY_LEVEL,
                'mid': self.ExperienceLevelFilters.MID_SENIOR,
                'senior': self.ExperienceLevelFilters.MID_SENIOR,
                'director': self.ExperienceLevelFilters.DIRECTOR,
                'executive': self.ExperienceLevelFilters.EXECUTIVE
            }
            if experience_level.lower() in exp_map:
                query_filters.experience_level = [exp_map[experience_level.lower()]]
        
        if job_type:
            type_map = {
                'full-time': self.TypeFilters.FULL_TIME,
                'part-time': self.TypeFilters.PART_TIME,
                'contract': self.TypeFilters.CONTRACT,
                'temporary': self.TypeFilters.TEMPORARY,
                'internship': self.TypeFilters.INTERNSHIP
            }
            if job_type.lower() in type_map:
                query_filters.type = [type_map[job_type.lower()]]
        
        # Create queries
        queries = [
            self.Query(
                query=keywords,
                options=self.QueryOptions(
                    locations=[location] if location else [],
                    limit=limit,
                    filters=query_filters
                )
            )
        ]
        
        # Run scraper
        try:
            self.scraper.run(queries)
        except Exception as e:
            print(f"Error running scraper: {e}")
        finally:
            if self.scraper:
                self.scraper.close()
        
        return self.jobs_data


# ============================================================================
# METHOD 3: Using Unofficial LinkedIn API (linkedin-api library)
# ============================================================================

class LinkedInAPIClient:
    """
    Uses linkedin-api library to access LinkedIn data.
    Requires LinkedIn credentials.
    Requires: pip install linkedin-api
    """
    
    def __init__(self, email: str = None, password: str = None):
        try:
            from linkedin_api import Linkedin
            
            email = email or os.getenv('LINKEDIN_EMAIL')
            password = password or os.getenv('LINKEDIN_PASSWORD')
            
            if not email or not password:
                raise ValueError("LinkedIn credentials required. Set LINKEDIN_EMAIL and LINKEDIN_PASSWORD env vars.")
            
            # Authenticate
            self.api = Linkedin(email, password)
            
        except ImportError:
            print("linkedin-api not installed. Install with: pip install linkedin-api")
            raise
    
    def search_jobs(
        self,
        keywords: str,
        location: str = "",
        limit: int = 25,
        **kwargs
    ) -> List[Dict]:
        """
        Search for jobs using linkedin-api library.
        Note: This method requires authentication and may be rate-limited.
        """
        try:
            # Search jobs
            jobs = self.api.search_jobs(
                keywords=keywords,
                location_name=location,
                limit=limit
            )
            
            # Format results
            formatted_jobs = []
            for job in jobs:
                formatted_job = {
                    'job_id': job.get('entityUrn', '').split(':')[-1],
                    'title': job.get('title', 'Unknown'),
                    'company': job.get('companyDetails', {}).get('company', {}).get('name', 'Unknown'),
                    'location': job.get('formattedLocation', 'Not specified'),
                    'description': job.get('description', {}).get('text', ''),
                    'url': f"https://www.linkedin.com/jobs/view/{job.get('entityUrn', '').split(':')[-1]}",
                    'posted_date': job.get('listedAt', 'Unknown'),
                    'easy_apply': job.get('easyApply', False),
                }
                formatted_jobs.append(formatted_job)
            
            return formatted_jobs
        
        except Exception as e:
            print(f"Error searching jobs with LinkedIn API: {e}")
            return []


# ============================================================================
# FACTORY FUNCTION - Choose Best Available Method
# ============================================================================

def create_linkedin_scraper(method: str = "auto") -> object:
    """
    Factory function to create the best available LinkedIn scraper.
    
    Args:
        method: "auto", "public", "library", or "api"
        
    Returns:
        Instance of a LinkedIn scraper
    """
    if method == "api":
        return LinkedInAPIClient()
    
    elif method == "library":
        return LinkedInJobsLibraryScraper()
    
    elif method == "public":
        return LinkedInJobScraper()
    
    else:  # auto or # google
        # Use public scraper first because it supports Google API fallback
        # and does not require extra credentials/dependencies.
        try:
            print("Using public LinkedIn scraper with optional Google fallback...")
            return LinkedInJobScraper()
        except:
            try:
                print("Falling back to linkedin-jobs-scraper library...")
                return LinkedInJobsLibraryScraper()
            except:
                print("Falling back to linkedin-api...")
                return LinkedInAPIClient()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the scraper
    print("Testing LinkedIn Job Scraper")
    print("=" * 60)
    
    # Use public scraper (no auth required)
    scraper = LinkedInJobScraper()
    
    print("\nSearching for AI Engineer jobs in San Francisco...")
    jobs = scraper.search_jobs(
        keywords="AI Engineer",
        location="San Francisco, CA",
        remote=False,
        limit=5
    )
    
    print(f"\nFound {len(jobs)} jobs:")
    print("-" * 60)
    
    for i, job in enumerate(jobs, 1):
        print(f"\n{i}. {job['title']}")
        print(f"   Company: {job['company']}")
        print(f"   Location: {job['location']}")
        print(f"   Posted: {job['posted_date']}")
        print(f"   URL: {job['url']}")
        if job.get('description'):
            print(f"   Description: {job['description'][:100]}...")
    
    # Get details for first job
    if jobs:
        print("\n" + "=" * 60)
        print("Fetching details for first job...")
        details = scraper.get_job_details(jobs[0]['job_id'])
        if details:
            print(f"\nFull Description Length: {len(details['full_description'])} characters")
            print(f"Criteria: {details['criteria']}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
