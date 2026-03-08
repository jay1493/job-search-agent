"""
Real LinkedIn Job Scraper Implementation
Fetches actual job data from LinkedIn using multiple methods
"""

import os
import time
import json
import re
from typing import List, Dict, Optional
from urllib.parse import quote
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
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

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
