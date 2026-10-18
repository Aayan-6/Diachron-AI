import os
import logging
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from dotenv import load_dotenv

# Ensure root .env is loaded
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

logger = logging.getLogger("tinyfish_client")

try:
    from tinyfish import TinyFish, AsyncTinyFish
    from tinyfish.fetch import FetchFormat
    TINYFISH_AVAILABLE = True
except ImportError:
    TINYFISH_AVAILABLE = False
    logger.warning("tinyfish package not found. Running in fallback mode.")


class TinyFishClientService:
    """
    Central wrapper for TinyFish AI Web Agent Infrastructure.
    Integrates:
      1. Search API (Intent-driven, geo-anchored queries)
      2. Fetch API (Dynamic JS/SPA rendering to clean markdown)
      3. Web Agent Automation API (Goal-driven headless browser actions)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TINYFISH_API_KEY")
        self._client: Optional[Any] = None
        self._init_client()

    def _init_client(self):
        if not TINYFISH_AVAILABLE:
            logger.error("TinyFish SDK is not installed.")
            return

        if not self.api_key:
            logger.warning("TINYFISH_API_KEY is not set. API calls will fail unless provided.")
            return

        try:
            self._client = TinyFish(api_key=self.api_key)
            logger.info("TinyFish client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TinyFish client: {e}")
            self._client = None

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def search_news(
        self,
        query: str,
        purpose: Optional[str] = None,
        location: Optional[str] = "IN",
        language: Optional[str] = "en",
        after_date: Optional[str] = None,
        before_date: Optional[str] = None,
        include_domains: Optional[str] = None,
        exclude_domains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes an intent-driven web search using TinyFish Search API.
        """
        if not self._client:
            logger.warning("TinyFish client not initialized. Cannot perform search.")
            return []

        try:
            response = self._client.search.query(
                query=query,
                purpose=purpose or "Autonomous Ground Truth Investigation for Satellite Change Detection",
                location=location,
                language=language,
                after_date=after_date,
                before_date=before_date,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
            
            results = []
            if hasattr(response, "results") and response.results:
                for item in response.results:
                    results.append({
                        "title": getattr(item, "title", "No Title"),
                        "url": getattr(item, "url", ""),
                        "snippet": getattr(item, "snippet", "") or getattr(item, "description", ""),
                        "published_date": getattr(item, "published_date", None) or getattr(item, "date", None),
                        "domain": getattr(item, "domain", None),
                        "score": getattr(item, "score", None),
                    })
            return results
        except Exception as e:
            logger.error(f"TinyFish Search API error: {e}")
            return []

    def fetch_page_content(
        self,
        urls: List[str],
        purpose: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Renders complex SPAs via TinyFish Fetch API and extracts clean content.
        """
        if not self._client or not urls:
            return []

        try:
            response = self._client.fetch.get_contents(
                urls=urls[:5],  # Limit to top 5 articles for efficiency
                purpose=purpose or "Extract factual news events and dates corroborating satellite change",
            )
            
            extracted = []
            if hasattr(response, "results") and response.results:
                for res in response.results:
                    extracted.append({
                        "url": getattr(res, "url", ""),
                        "title": getattr(res, "title", ""),
                        "content": getattr(res, "content", "") or getattr(res, "markdown", "") or getattr(res, "text", ""),
                        "status": getattr(res, "status", 200),
                    })
            return extracted
        except Exception as e:
            logger.error(f"TinyFish Fetch API error: {e}")
            return []

    def run_browser_agent(
        self,
        url: str,
        goal: str,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a goal-driven autonomous browser action via TinyFish Web Agent API.
        Useful for navigating portals (e.g. disaster portals, FIRMS, dam gauges).
        """
        if not self._client:
            return {"status": "error", "message": "TinyFish client not configured"}

        try:
            response = self._client.agent.run(
                url=url,
                goal=goal,
                output_schema=output_schema,
            )
            return {
                "status": "success",
                "run_id": getattr(response, "run_id", None) or getattr(response, "id", None),
                "data": getattr(response, "data", None) or getattr(response, "output", None),
                "summary": getattr(response, "summary", "") or str(response),
            }
        except Exception as e:
            logger.error(f"TinyFish Agent Run error: {e}")
            return {"status": "error", "error": str(e)}


# Global singleton instance
tinyfish_service = TinyFishClientService()
