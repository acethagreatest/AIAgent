#!/usr/bin/env python3
"""
Selenium-based scraper for JavaScript-heavy betting sites.
Handles dynamic content loading and complex interactions.
"""

import asyncio
import time
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import structlog

logger = structlog.get_logger()


class BrowserType(Enum):
    """Supported browser types."""
    CHROME = "chrome"
    FIREFOX = "firefox"
    EDGE = "edge"


@dataclass
class SeleniumConfig:
    """Configuration for Selenium scraper."""
    headless: bool = True
    window_size: tuple[int, int] = (1920, 1080)
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    implicit_wait: int = 10
    page_load_timeout: int = 30
    script_timeout: int = 30
    disable_images: bool = True
    disable_css: bool = False
    disable_javascript: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.window_size[0] <= 0 or self.window_size[1] <= 0:
            self.window_size = (1920, 1080)
        if self.implicit_wait <= 0:
            self.implicit_wait = 10
        if self.page_load_timeout <= 0:
            self.page_load_timeout = 30
        if self.script_timeout <= 0:
            self.script_timeout = 30


class SeleniumScraper:
    """Selenium-based scraper for JavaScript-heavy sites."""
    
    def __init__(self, config: SeleniumConfig = None):
        """
        Initialize Selenium scraper.
        
        Args:
            config: Selenium configuration
        """
        self.config = config or SeleniumConfig()
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None
    
    def __enter__(self):
        """Context manager entry."""
        self._setup_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup()
    
    def _setup_driver(self):
        """Setup Chrome WebDriver with configuration."""
        try:
            chrome_options = Options()
            
            # Basic options
            if self.config.headless:
                chrome_options.add_argument("--headless")
            
            chrome_options.add_argument(f"--window-size={self.config.window_size[0]},{self.config.window_size[1]}")
            chrome_options.add_argument(f"--user-agent={self.config.user_agent}")
            
            # Performance options
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            
            if self.config.disable_images:
                chrome_options.add_argument("--disable-images")
            if self.config.disable_css:
                chrome_options.add_argument("--disable-css")
            
            # Privacy options
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Disable JavaScript if needed
            if self.config.disable_javascript:
                prefs = {"profile.managed_default_content_settings.javascript": 2}
                chrome_options.add_experimental_option("prefs", prefs)
            
            # Setup service
            try:
                service = Service(ChromeDriverManager().install())
                print(f"ChromeDriver path: {service.path}")
            except Exception as e:
                logger.error(f"Failed to setup ChromeDriver service: {e}")
                raise
            
            # Create driver
            try:
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            except Exception as e:
                logger.error(f"Failed to create Chrome WebDriver: {e}")
                raise
            
            # Configure timeouts
            self.driver.implicitly_wait(self.config.implicit_wait)
            self.driver.set_page_load_timeout(self.config.page_load_timeout)
            self.driver.set_script_timeout(self.config.script_timeout)
            
            # Create wait object
            self.wait = WebDriverWait(self.driver, self.config.implicit_wait)
            
            # Execute stealth script
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("Selenium WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Selenium WebDriver: {e}")
            raise
    
    def _cleanup(self):
        """Cleanup WebDriver resources."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Selenium WebDriver cleaned up")
            except Exception as e:
                logger.warning(f"Error during WebDriver cleanup: {e}")
            finally:
                self.driver = None
                self.wait = None
    
    def navigate_to(self, url: str, wait_for_element: str = None) -> bool:
        """
        Navigate to URL and optionally wait for specific element.
        
        Args:
            url: URL to navigate to
            wait_for_element: CSS selector to wait for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.debug(f"Navigating to: {url}")
            self.driver.get(url)
            
            if wait_for_element:
                self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element)))
                logger.debug(f"Element found: {wait_for_element}")
            
            # Wait for page to stabilize
            time.sleep(2)
            return True
            
        except TimeoutException:
            logger.warning(f"Timeout waiting for element {wait_for_element} on {url}")
            return False
        except WebDriverException as e:
            logger.error(f"WebDriver error navigating to {url}: {e}")
            return False
    
    def wait_for_element(self, selector: str, timeout: int = 10) -> bool:
        """
        Wait for element to be present.
        
        Args:
            selector: CSS selector
            timeout: Timeout in seconds
            
        Returns:
            True if element found, False otherwise
        """
        try:
            wait = WebDriverWait(self.driver, timeout)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            return True
        except TimeoutException:
            logger.warning(f"Element not found: {selector}")
            return False
    
    def find_elements(self, selector: str) -> List[Any]:
        """
        Find elements by CSS selector.
        
        Args:
            selector: CSS selector
            
        Returns:
            List of WebElements
        """
        try:
            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            logger.debug(f"Found {len(elements)} elements with selector: {selector}")
            return elements
        except Exception as e:
            logger.warning(f"Error finding elements with selector {selector}: {e}")
            return []
    
    def find_element(self, selector: str) -> Optional[Any]:
        """
        Find single element by CSS selector.
        
        Args:
            selector: CSS selector
            
        Returns:
            WebElement or None
        """
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            return element
        except NoSuchElementException:
            logger.debug(f"Element not found: {selector}")
            return None
        except Exception as e:
            logger.warning(f"Error finding element with selector {selector}: {e}")
            return None
    
    def get_page_source(self) -> str:
        """Get page source HTML."""
        try:
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Error getting page source: {e}")
            return ""
    
    def execute_script(self, script: str) -> Any:
        """
        Execute JavaScript in browser.
        
        Args:
            script: JavaScript code to execute
            
        Returns:
            Script result
        """
        try:
            return self.driver.execute_script(script)
        except Exception as e:
            logger.warning(f"Error executing script: {e}")
            return None
    
    def scroll_to_element(self, element: Any) -> bool:
        """
        Scroll to element.
        
        Args:
            element: WebElement to scroll to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
            time.sleep(1)  # Wait for scroll to complete
            return True
        except Exception as e:
            logger.warning(f"Error scrolling to element: {e}")
            return False
    
    def scroll_to_bottom(self) -> bool:
        """Scroll to bottom of page."""
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for content to load
            return True
        except Exception as e:
            logger.warning(f"Error scrolling to bottom: {e}")
            return False
    
    def wait_for_ajax(self, timeout: int = 10) -> bool:
        """
        Wait for AJAX requests to complete.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if AJAX complete, False otherwise
        """
        try:
            wait = WebDriverWait(self.driver, timeout)
            wait.until(lambda driver: driver.execute_script("return jQuery.active == 0"))
            return True
        except TimeoutException:
            logger.warning("AJAX requests did not complete within timeout")
            return False
        except Exception as e:
            logger.warning(f"Error waiting for AJAX: {e}")
            return False
    
    def extract_json_data(self, script_selector: str = "script[type='application/json']") -> List[Dict[str, Any]]:
        """
        Extract JSON data from script tags.
        
        Args:
            script_selector: CSS selector for script tags
            
        Returns:
            List of parsed JSON objects
        """
        json_data = []
        
        try:
            script_elements = self.find_elements(script_selector)
            
            for script in script_elements:
                try:
                    script_content = script.get_attribute('innerHTML')
                    if script_content:
                        data = json.loads(script_content)
                        json_data.append(data)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Error parsing script JSON: {e}")
                    continue
            
            logger.debug(f"Extracted {len(json_data)} JSON objects")
            
        except Exception as e:
            logger.warning(f"Error extracting JSON data: {e}")
        
        return json_data
    
    def take_screenshot(self, filename: str = None) -> str:
        """
        Take screenshot of current page.
        
        Args:
            filename: Optional filename for screenshot
            
        Returns:
            Screenshot filename
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"selenium_screenshot_{timestamp}.png"
            
            self.driver.save_screenshot(filename)
            logger.info(f"Screenshot saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return ""
    
    def get_console_logs(self) -> List[Dict[str, Any]]:
        """
        Get browser console logs.
        
        Returns:
            List of console log entries
        """
        try:
            logs = self.driver.get_log('browser')
            return logs
        except Exception as e:
            logger.warning(f"Error getting console logs: {e}")
            return []
    
    def wait_for_text(self, text: str, timeout: int = 10) -> bool:
        """
        Wait for specific text to appear on page.
        
        Args:
            text: Text to wait for
            timeout: Timeout in seconds
            
        Returns:
            True if text found, False otherwise
        """
        try:
            wait = WebDriverWait(self.driver, timeout)
            wait.until(EC.text_to_be_present_in_element((By.TAG_NAME, "body"), text))
            return True
        except TimeoutException:
            logger.warning(f"Text not found: {text}")
            return False
        except Exception as e:
            logger.warning(f"Error waiting for text: {e}")
            return False


class BettingSiteSeleniumScraper(SeleniumScraper):
    """Specialized Selenium scraper for betting sites."""
    
    def __init__(self, config: SeleniumConfig = None):
        """Initialize betting site scraper."""
        super().__init__(config)
        self.betting_site_selectors = {
            'fanduel': {
                'prop_containers': [
                    '[data-testid*="prop"]',
                    '[class*="prop"]',
                    '[class*="market"]',
                    '.player-prop',
                    '.market-group',
                    '[class*="betting-market"]'
                ],
                'player_name': [
                    '[class*="player"] [class*="name"]',
                    '[class*="player-name"]',
                    '[class*="participant"]',
                    'h3, h4, h5'
                ],
                'stat_line': [
                    '[class*="stat"]',
                    '[class*="line"]',
                    '[class*="market"]',
                    '[class*="type"]'
                ],
                'odds': [
                    '[class*="odds"]',
                    '[class*="price"]',
                    '[class*="bet"]',
                    '[class*="line"]'
                ]
            },
            'draftkings': {
                'prop_containers': [
                    '[data-testid*="market"]',
                    '[class*="market"]',
                    '[class*="prop"]',
                    '[class*="bet"]',
                    '.sportsbook-market',
                    '.market-group',
                    '[class*="sportsbook"]',
                    '[class*="outcome"]'
                ],
                'player_name': [
                    '[class*="player"] [class*="name"]',
                    '[class*="player-name"]',
                    '[class*="participant"]',
                    '[class*="athlete"]',
                    '[class*="market-title"]',
                    'h3, h4, h5'
                ],
                'stat_line': [
                    '[class*="stat"]',
                    '[class*="line"]',
                    '[class*="market"]',
                    '[class*="type"]',
                    '[class*="outcome"]'
                ],
                'odds': [
                    '[class*="odds"]',
                    '[class*="price"]',
                    '[class*="bet"]',
                    '[class*="line"]',
                    '[class*="decimal"]'
                ]
            },
            'betmgm': {
                'prop_containers': [
                    '[class*="market"]',
                    '[class*="prop"]',
                    '[class*="bet"]',
                    '[class*="outcome"]',
                    '.betting-market',
                    '.market-group',
                    '.sportsbook-market',
                    '[class*="sportsbook"]',
                    '[data-testid*="market"]'
                ],
                'player_name': [
                    '[class*="player"] [class*="name"]',
                    '[class*="player-name"]',
                    '[class*="participant"]',
                    '[class*="athlete"]',
                    '[class*="competitor"]',
                    '[class*="market-title"]',
                    '[class*="outcome-title"]',
                    'h3, h4, h5'
                ],
                'stat_line': [
                    '[class*="stat"]',
                    '[class*="line"]',
                    '[class*="market"]',
                    '[class*="type"]',
                    '[class*="outcome"]',
                    '[class*="selection"]'
                ],
                'odds': [
                    '[class*="odds"]',
                    '[class*="price"]',
                    '[class*="bet"]',
                    '[class*="line"]',
                    '[class*="decimal"]'
                ]
            }
        }
    
    def scrape_betting_site(self, site: str, url: str, sport: str) -> List[Dict[str, Any]]:
        """
        Scrape prop bets from a betting site.
        
        Args:
            site: Site name (fanduel, draftkings, etc.)
            url: URL to scrape
            sport: Sport name
            
        Returns:
            List of prop bet data
        """
        logger.info(f"Scraping {site} for {sport} prop bets")
        
        try:
            # Navigate to site
            if not self.navigate_to(url):
                logger.error(f"Failed to navigate to {url}")
                return []
            
            # Wait for content to load
            time.sleep(5)
            
            # Try to find prop bet containers
            selectors = self.betting_site_selectors.get(site, {})
            prop_containers = []
            
            for selector in selectors.get('prop_containers', []):
                containers = self.find_elements(selector)
                if containers:
                    prop_containers.extend(containers)
                    logger.debug(f"Found {len(containers)} containers with selector: {selector}")
            
            if not prop_containers:
                logger.warning(f"No prop containers found on {site}")
                return []
            
            # Extract prop bet data
            prop_bets = []
            for container in prop_containers:
                prop_data = self._extract_prop_data(container, selectors, site, sport)
                if prop_data:
                    prop_bets.append(prop_data)
            
            logger.info(f"Scraped {len(prop_bets)} prop bets from {site}")
            return prop_bets
            
        except Exception as e:
            logger.error(f"Error scraping {site}: {e}")
            return []
    
    def _extract_prop_data(self, container: Any, selectors: Dict[str, List[str]], site: str, sport: str) -> Optional[Dict[str, Any]]:
        """Extract prop bet data from container."""
        try:
            # Extract player name
            player_name = self._extract_text(container, selectors.get('player_name', []))
            if not player_name:
                return None
            
            # Extract stat line
            stat_line = self._extract_text(container, selectors.get('stat_line', []))
            if not stat_line:
                return None
            
            # Parse stat type and line value
            stat_type, line_value = self._parse_stat_line(stat_line)
            if not stat_type or not line_value:
                return None
            
            # Extract odds
            odds_text = self._extract_text(container, selectors.get('odds', []))
            over_odds, under_odds = self._parse_odds(odds_text)
            
            return {
                'player_name': player_name,
                'stat_type': stat_type,
                'line_value': line_value,
                'over_odds': over_odds,
                'under_odds': under_odds,
                'betting_app': site,
                'game_info': f"{sport.upper()} Game",
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.warning(f"Error extracting prop data: {e}")
            return None
    
    def _extract_text(self, container: Any, selectors: List[str]) -> Optional[str]:
        """Extract text using multiple selectors."""
        for selector in selectors:
            try:
                element = container.find_element(By.CSS_SELECTOR, selector)
                text = element.text.strip()
                if text:
                    return text
            except NoSuchElementException:
                continue
        return None
    
    def _parse_stat_line(self, text: str) -> tuple[Optional[str], Optional[float]]:
        """Parse stat type and line value from text."""
        patterns = [
            r'(\w+)\s+(Over|Under)\s+(\d+\.?\d*)',
            r'(\w+)\s+(\d+\.?\d*)\s+(Over|Under)',
            r'(\w+)\s+(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                stat_type = match.group(1).lower()
                line_value = float(match.group(3) if len(match.groups()) == 3 else match.group(2))
                return stat_type, line_value
        
        return None, None
    
    def _parse_odds(self, text: str) -> tuple[Optional[float], Optional[float]]:
        """Parse odds from text."""
        if not text:
            return None, None
        
        # Look for over/under odds patterns
        over_match = re.search(r'Over\s*([+-]?\d+)', text, re.I)
        under_match = re.search(r'Under\s*([+-]?\d+)', text, re.I)
        
        over_odds = float(over_match.group(1)) if over_match else None
        under_odds = float(under_match.group(1)) if under_match else None
        
        return over_odds, under_odds
