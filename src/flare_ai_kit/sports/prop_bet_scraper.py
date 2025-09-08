#!/usr/bin/env python3
"""
Prop Bet Scraper for Major Sports Betting Sites
Scrapes prop bet data from FanDuel, Prize Picks, Underdog, DraftKings, and BetMGM
"""

import asyncio
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import structlog

from .selenium_scraper import SeleniumScraper, BettingSiteSeleniumScraper, SeleniumConfig
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

logger = structlog.get_logger()


class BettingApp(Enum):
    """Supported betting applications."""
    FANDUEL = "fanduel"
    PRIZE_PICKS = "prize_picks"
    UNDERDOG = "underdog"
    DRAFTKINGS = "draftkings"
    BETMGM = "betmgm"


@dataclass
class PropBet:
    """Represents a prop bet from a betting site."""
    player_name: str
    stat_type: str  # e.g., "points", "rebounds", "assists"
    line_value: float
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None
    betting_app: BettingApp = BettingApp.FANDUEL
    game_info: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PropBetScraper:
    """Base class for prop bet scraping."""
    
    def __init__(self):
        self.ua = UserAgent()
        self.session = None
        self.config = {
            'REQUEST_TIMEOUT': 30,
            'MAX_RETRIES': 3,
            'RETRY_DELAY': 2,
            'RATE_LIMIT_DELAY': 1
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config['REQUEST_TIMEOUT'])
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, url: str, headers: Dict[str, str] = None) -> Optional[str]:
        """Make HTTP request with retry logic."""
        if headers is None:
            headers = {
                'User-Agent': self.ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        
        for attempt in range(self.config['MAX_RETRIES']):
            try:
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.debug(f"Successfully scraped {url}")
                        return content
                    elif response.status == 429:  # Rate limited
                        wait_time = self.config['RETRY_DELAY'] * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config['MAX_RETRIES'] - 1:
                    await asyncio.sleep(self.config['RETRY_DELAY'] * (2 ** attempt))
        
        logger.error(f"Failed to scrape {url} after {self.config['MAX_RETRIES']} attempts")
        return None


class FanDuelScraper(PropBetScraper):
    """Scraper for FanDuel prop bets."""
    
    def __init__(self, use_selenium: bool = True):
        """Initialize FanDuel scraper."""
        super().__init__()
        self.use_selenium = use_selenium
        self.selenium_config = SeleniumConfig(
            headless=True,
            disable_images=True,
            implicit_wait=15
        )
    
    async def scrape_prop_bets(self, sport: str) -> List[PropBet]:
        """Scrape prop bets from FanDuel."""
        logger.info(f"Scraping FanDuel prop bets for {sport}")
        
        # FanDuel URLs for different sports
        urls = {
            'nfl': 'https://sportsbook.fanduel.com/nfl-player-props',
            'nba': 'https://sportsbook.fanduel.com/nba-player-props',
            'mlb': 'https://sportsbook.fanduel.com/mlb-player-props',
        }
        
        url = urls.get(sport.lower())
        if not url:
            logger.warning(f"No FanDuel URL found for sport: {sport}")
            return []
        
        if self.use_selenium:
            return await self._scrape_with_selenium(url, sport)
        else:
            html = await self._make_request(url)
            if not html:
                return []
            return self._parse_fanduel_html(html, sport)
    
    async def _scrape_with_selenium(self, url: str, sport: str) -> List[PropBet]:
        """Scrape using Selenium for JavaScript-heavy content."""
        try:
            # Run Selenium in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._selenium_scrape, url, sport)
        except Exception as e:
            logger.warning(f"Selenium scraping failed, falling back to HTTP: {e}")
            # Fallback to HTTP scraping
            html = await self._make_request(url)
            if not html:
                return []
            return self._parse_fanduel_html(html, sport)
    
    def _selenium_scrape(self, url: str, sport: str) -> List[PropBet]:
        """Selenium scraping implementation."""
        with BettingSiteSeleniumScraper(self.selenium_config) as scraper:
            try:
                # Navigate to page
                if not scraper.navigate_to(url, wait_for_element="body"):
                    logger.warning(f"Failed to navigate to {url}")
                    return []
                
                # Wait for content to load
                time.sleep(5)
                
                # Try to find prop bet containers
                prop_containers = []
                selectors = [
                    '[data-testid*="prop"]',
                    '[class*="prop"]',
                    '[class*="market"]',
                    '[class*="bet"]',
                    '.player-prop',
                    '.market-group',
                    '.betting-market'
                ]
                
                for selector in selectors:
                    containers = scraper.find_elements(selector)
                    if containers:
                        prop_containers.extend(containers)
                        logger.debug(f"Found {len(containers)} containers with selector: {selector}")
                
                if not prop_containers:
                    logger.warning("No prop containers found with Selenium")
                    return []
                
                # Extract prop bet data
                prop_bets = []
                for container in prop_containers:
                    prop_data = self._extract_selenium_prop_data(container, sport)
                    if prop_data:
                        prop_bets.append(prop_data)
                
                logger.info(f"Selenium scraped {len(prop_bets)} prop bets from FanDuel")
                return prop_bets
                
            except Exception as e:
                logger.error(f"Selenium scraping error: {e}")
                return []
    
    def _extract_selenium_prop_data(self, container: Any, sport: str) -> Optional[PropBet]:
        """Extract prop bet data from Selenium container."""
        try:
            # Extract player name
            player_name = self._extract_selenium_text(container, [
                '[class*="player"] [class*="name"]',
                '[class*="player-name"]',
                'h3, h4, h5',
                '[class*="participant"]'
            ])
            
            if not player_name:
                return None
            
            # Extract stat line
            stat_line = self._extract_selenium_text(container, [
                '[class*="stat"]',
                '[class*="line"]',
                '[class*="market"]',
                '[class*="type"]'
            ])
            
            if not stat_line:
                return None
            
            # Parse stat type and line value
            stat_type, line_value = self._parse_stat_line(stat_line)
            if not stat_type or not line_value:
                return None
            
            # Extract odds
            odds_text = self._extract_selenium_text(container, [
                '[class*="odds"]',
                '[class*="price"]',
                '[class*="bet"]'
            ])
            
            over_odds, under_odds = self._parse_odds(odds_text)
            
            return PropBet(
                player_name=player_name,
                stat_type=stat_type,
                line_value=line_value,
                over_odds=over_odds,
                under_odds=under_odds,
                betting_app=BettingApp.FANDUEL,
                game_info=f"{sport.upper()} Game"
            )
            
        except Exception as e:
            logger.warning(f"Error extracting Selenium prop data: {e}")
            return None
    
    def _extract_selenium_text(self, container: Any, selectors: List[str]) -> Optional[str]:
        """Extract text using multiple selectors from Selenium container."""
        for selector in selectors:
            try:
                element = container.find_element(By.CSS_SELECTOR, selector)
                text = element.text.strip()
                if text:
                    return text
            except NoSuchElementException:
                continue
        return None
    
    def _parse_fanduel_html(self, html: str, sport: str) -> List[PropBet]:
        """Parse FanDuel HTML for prop bets."""
        soup = BeautifulSoup(html, 'html.parser')
        prop_bets = []
        
        try:
            # Look for JSON data in script tags first
            script_tags = soup.find_all('script', type='application/json')
            for script in script_tags:
                try:
                    data = json.loads(script.string)
                    if 'props' in data or 'markets' in data:
                        prop_bets.extend(self._extract_fanduel_json_props(data, sport))
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # Fallback to HTML parsing
            if not prop_bets:
                # Look for prop bet containers with more specific selectors
                selectors = [
                    '[data-testid*="prop"]',
                    '[class*="prop"]',
                    '[class*="market"]',
                    '[class*="bet"]',
                    '.player-prop',
                    '.market-group',
                    '.betting-market'
                ]
                
                for selector in selectors:
                    containers = soup.select(selector)
                    for container in containers:
                        prop_bet = self._parse_fanduel_container(container, sport)
                        if prop_bet:
                            prop_bets.append(prop_bet)
            
            # If still no props found, try to find any text that looks like prop bets
            if not prop_bets:
                prop_bets = self._extract_prop_bets_from_text(soup, sport, BettingApp.FANDUEL)
                    
        except Exception as e:
            logger.warning(f"Error parsing FanDuel HTML: {e}")
        
        logger.info(f"Scraped {len(prop_bets)} prop bets from FanDuel")
        return prop_bets
    
    def _extract_fanduel_json_props(self, data: Dict[str, Any], sport: str) -> List[PropBet]:
        """Extract prop bets from FanDuel JSON data."""
        prop_bets = []
        
        try:
            # Navigate through JSON structure to find prop data
            if 'props' in data:
                props_data = data['props']
            elif 'markets' in data:
                props_data = data['markets']
            else:
                return prop_bets
            
            # Extract prop bets from the data structure
            for prop in props_data:
                if isinstance(prop, dict):
                    player_name = prop.get('player', {}).get('name', '')
                    stat_type = prop.get('stat', '')
                    line_value = prop.get('line', 0)
                    over_odds = prop.get('overOdds')
                    under_odds = prop.get('underOdds')
                    
                    if player_name and stat_type and line_value:
                        prop_bet = PropBet(
                            player_name=player_name,
                            stat_type=stat_type,
                            line_value=float(line_value),
                            over_odds=over_odds,
                            under_odds=under_odds,
                            betting_app=BettingApp.FANDUEL,
                            game_info=f"{sport.upper()} Game"
                        )
                        prop_bets.append(prop_bet)
                        
        except Exception as e:
            logger.warning(f"Error extracting FanDuel JSON props: {e}")
        
        return prop_bets
    
    def _parse_fanduel_container(self, container, sport: str) -> Optional[PropBet]:
        """Parse a single FanDuel container for prop bet data."""
        try:
            # Extract player name
            player_elem = container.find(['span', 'div', 'h3', 'h4'], class_=re.compile(r'player|name', re.I))
            if not player_elem:
                player_elem = container.find(['span', 'div'], string=re.compile(r'[A-Z][a-z]+ [A-Z][a-z]+'))
            
            if not player_elem:
                return None
            
            player_name = player_elem.get_text(strip=True)
            
            # Extract stat type and line
            stat_elem = container.find(['span', 'div'], class_=re.compile(r'stat|type|market|line', re.I))
            if not stat_elem:
                # Look for text patterns that might contain stat info
                text_elements = container.find_all(['span', 'div'], string=re.compile(r'\d+\.?\d*'))
                for elem in text_elements:
                    text = elem.get_text(strip=True)
                    if re.search(r'\d+\.?\d*', text):
                        stat_elem = elem
                        break
            
            if not stat_elem:
                return None
            
            stat_text = stat_elem.get_text(strip=True)
            stat_type, line_value = self._parse_stat_line(stat_text)
            
            if stat_type and line_value:
                # Extract odds
                over_odds, under_odds = self._extract_odds(container)
                
                return PropBet(
                    player_name=player_name,
                    stat_type=stat_type,
                    line_value=line_value,
                    over_odds=over_odds,
                    under_odds=under_odds,
                    betting_app=BettingApp.FANDUEL,
                    game_info=f"{sport.upper()} Game"
                )
                
        except Exception as e:
            logger.warning(f"Error parsing FanDuel container: {e}")
        
        return None
    
    def _extract_prop_bets_from_text(self, soup: BeautifulSoup, sport: str, app: BettingApp) -> List[PropBet]:
        """Extract prop bets by looking for text patterns in the HTML."""
        prop_bets = []
        
        try:
            # Look for text that matches prop bet patterns
            text_elements = soup.find_all(['span', 'div', 'p'], string=re.compile(r'[A-Z][a-z]+ [A-Z][a-z]+.*\d+\.?\d*'))
            
            for elem in text_elements:
                text = elem.get_text(strip=True)
                
                # Check if this looks like a prop bet
                if self._looks_like_prop_bet(text):
                    player_name, stat_type, line_value = self._extract_prop_data_from_text(text)
                    
                    if player_name and stat_type and line_value:
                        prop_bet = PropBet(
                            player_name=player_name,
                            stat_type=stat_type,
                            line_value=line_value,
                            betting_app=app,
                            game_info=f"{sport.upper()} Game"
                        )
                        prop_bets.append(prop_bet)
                        
        except Exception as e:
            logger.warning(f"Error extracting prop bets from text: {e}")
        
        return prop_bets
    
    def _looks_like_prop_bet(self, text: str) -> bool:
        """Check if text looks like a prop bet."""
        # Look for patterns like "Player Name Points 25.5" or "Player Name Over 25.5"
        patterns = [
            r'[A-Z][a-z]+ [A-Z][a-z]+.*\d+\.?\d*',
            r'[A-Z][a-z]+ [A-Z][a-z]+.*(Over|Under).*\d+\.?\d*',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _extract_prop_data_from_text(self, text: str) -> tuple[Optional[str], Optional[str], Optional[float]]:
        """Extract player name, stat type, and line value from text."""
        # Pattern: "Player Name Stat Type Line Value"
        patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+)\s+(\w+)\s+(\d+\.?\d*)',
            r'([A-Z][a-z]+ [A-Z][a-z]+)\s+(\w+)\s+(Over|Under)\s+(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                player_name = match.group(1)
                stat_type = match.group(2).lower()
                line_value = float(match.group(3) if len(match.groups()) == 3 else match.group(4))
                return player_name, stat_type, line_value
        
        return None, None, None
    
    def _parse_stat_line(self, text: str) -> tuple[Optional[str], Optional[float]]:
        """Parse stat type and line value from text."""
        # Common patterns: "Points Over 25.5", "Rebounds Under 8.5", etc.
        patterns = [
            r'(\w+)\s+(Over|Under)\s+(\d+\.?\d*)',
            r'(\w+)\s+(\d+\.?\d*)\s+(Over|Under)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                stat_type = match.group(1).lower()
                line_value = float(match.group(3) if len(match.groups()) == 3 else match.group(2))
                return stat_type, line_value
        
        return None, None
    
    def _extract_odds(self, container) -> tuple[Optional[float], Optional[float]]:
        """Extract over/under odds from container."""
        over_odds = None
        under_odds = None
        
        # Look for odds elements
        odds_elements = container.find_all(['span', 'div'], class_=re.compile(r'odds|price', re.I))
        
        for elem in odds_elements:
            text = elem.get_text(strip=True)
            if '+' in text or '-' in text:
                try:
                    odds = float(text.replace('+', '').replace('-', ''))
                    if 'over' in elem.get('class', []) or 'o' in text.lower():
                        over_odds = odds
                    elif 'under' in elem.get('class', []) or 'u' in text.lower():
                        under_odds = odds
                except ValueError:
                    continue
        
        return over_odds, under_odds


class PrizePicksScraper(PropBetScraper):
    """Scraper for Prize Picks prop bets."""
    
    async def scrape_prop_bets(self, sport: str) -> List[PropBet]:
        """Scrape prop bets from Prize Picks."""
        logger.info(f"Scraping Prize Picks prop bets for {sport}")
        
        # Prize Picks URLs
        urls = {
            'nfl': 'https://prizepicks.com/nfl',
            'nba': 'https://prizepicks.com/nba',
            'mlb': 'https://prizepicks.com/mlb',
        }
        
        url = urls.get(sport.lower())
        if not url:
            logger.warning(f"No Prize Picks URL found for sport: {sport}")
            return []
        
        html = await self._make_request(url)
        if not html:
            return []
        
        return self._parse_prizepicks_html(html, sport)
    
    def _parse_prizepicks_html(self, html: str, sport: str) -> List[PropBet]:
        """Parse Prize Picks HTML for prop bets."""
        soup = BeautifulSoup(html, 'html.parser')
        prop_bets = []
        
        try:
            # Look for player prop cards
            prop_cards = soup.find_all(['div', 'section'], class_=re.compile(r'card|prop|player', re.I))
            
            for card in prop_cards:
                # Extract player name
                player_elem = card.find(['h3', 'h4', 'span'], class_=re.compile(r'player|name', re.I))
                if not player_elem:
                    continue
                
                player_name = player_elem.get_text(strip=True)
                
                # Extract stat and line
                stat_elem = card.find(['span', 'div'], class_=re.compile(r'stat|line|projection', re.I))
                if not stat_elem:
                    continue
                
                stat_text = stat_elem.get_text(strip=True)
                stat_type, line_value = self._parse_prizepicks_stat(stat_text)
                
                if stat_type and line_value:
                    prop_bet = PropBet(
                        player_name=player_name,
                        stat_type=stat_type,
                        line_value=line_value,
                        betting_app=BettingApp.PRIZE_PICKS,
                        game_info=f"{sport.upper()} Game"
                    )
                    prop_bets.append(prop_bet)
                    
        except Exception as e:
            logger.warning(f"Error parsing Prize Picks HTML: {e}")
        
        logger.info(f"Scraped {len(prop_bets)} prop bets from Prize Picks")
        return prop_bets
    
    def _parse_prizepicks_stat(self, text: str) -> tuple[Optional[str], Optional[float]]:
        """Parse Prize Picks stat format."""
        # Prize Picks format: "Points 25.5", "Rebounds 8.5", etc.
        patterns = [
            r'(\w+)\s+(\d+\.?\d*)',
            r'(\w+)\s+Over\s+(\d+\.?\d*)',
            r'(\w+)\s+Under\s+(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                stat_type = match.group(1).lower()
                line_value = float(match.group(2))
                return stat_type, line_value
        
        return None, None


class UnderdogScraper(PropBetScraper):
    """Scraper for Underdog prop bets."""
    
    async def scrape_prop_bets(self, sport: str) -> List[PropBet]:
        """Scrape prop bets from Underdog."""
        logger.info(f"Scraping Underdog prop bets for {sport}")
        
        # Underdog URLs
        urls = {
            'nfl': 'https://underdogfantasy.com/pickem/nfl',
            'nba': 'https://underdogfantasy.com/pickem/nba',
            'mlb': 'https://underdogfantasy.com/pickem/mlb',
        }
        
        url = urls.get(sport.lower())
        if not url:
            logger.warning(f"No Underdog URL found for sport: {sport}")
            return []
        
        html = await self._make_request(url)
        if not html:
            return []
        
        return self._parse_underdog_html(html, sport)
    
    def _parse_underdog_html(self, html: str, sport: str) -> List[PropBet]:
        """Parse Underdog HTML for prop bets."""
        soup = BeautifulSoup(html, 'html.parser')
        prop_bets = []
        
        try:
            # Look for pick'em cards
            pick_cards = soup.find_all(['div', 'section'], class_=re.compile(r'pick|card|prop', re.I))
            
            for card in pick_cards:
                # Extract player name
                player_elem = card.find(['h3', 'h4', 'span'], class_=re.compile(r'player|name', re.I))
                if not player_elem:
                    continue
                
                player_name = player_elem.get_text(strip=True)
                
                # Extract stat and line
                stat_elem = card.find(['span', 'div'], class_=re.compile(r'stat|line|projection', re.I))
                if not stat_elem:
                    continue
                
                stat_text = stat_elem.get_text(strip=True)
                stat_type, line_value = self._parse_underdog_stat(stat_text)
                
                if stat_type and line_value:
                    prop_bet = PropBet(
                        player_name=player_name,
                        stat_type=stat_type,
                        line_value=line_value,
                        betting_app=BettingApp.UNDERDOG,
                        game_info=f"{sport.upper()} Game"
                    )
                    prop_bets.append(prop_bet)
                    
        except Exception as e:
            logger.warning(f"Error parsing Underdog HTML: {e}")
        
        logger.info(f"Scraped {len(prop_bets)} prop bets from Underdog")
        return prop_bets
    
    def _parse_underdog_stat(self, text: str) -> tuple[Optional[str], Optional[float]]:
        """Parse Underdog stat format."""
        # Underdog format: "Points 25.5", "Rebounds 8.5", etc.
        patterns = [
            r'(\w+)\s+(\d+\.?\d*)',
            r'(\w+)\s+Over\s+(\d+\.?\d*)',
            r'(\w+)\s+Under\s+(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                stat_type = match.group(1).lower()
                line_value = float(match.group(2))
                return stat_type, line_value
        
        return None, None


class DraftKingsScraper(PropBetScraper):
    """Scraper for DraftKings prop bets."""
    
    def __init__(self, use_selenium: bool = True):
        """Initialize DraftKings scraper."""
        super().__init__()
        self.use_selenium = use_selenium
        self.selenium_config = SeleniumConfig(
            headless=True,
            disable_images=True,
            implicit_wait=15
        )
    
    async def scrape_prop_bets(self, sport: str) -> List[PropBet]:
        """Scrape prop bets from DraftKings."""
        logger.info(f"Scraping DraftKings prop bets for {sport}")
        
        # DraftKings URLs
        urls = {
            'nfl': 'https://sportsbook.draftkings.com/nfl-player-props',
            'nba': 'https://sportsbook.draftkings.com/nba-player-props',
            'mlb': 'https://sportsbook.draftkings.com/mlb-player-props',
        }
        
        url = urls.get(sport.lower())
        if not url:
            logger.warning(f"No DraftKings URL found for sport: {sport}")
            return []
        
        if self.use_selenium:
            return await self._scrape_with_selenium(url, sport)
        else:
            html = await self._make_request(url)
            if not html:
                return []
            return self._parse_draftkings_html(html, sport)
    
    async def _scrape_with_selenium(self, url: str, sport: str) -> List[PropBet]:
        """Scrape using Selenium for JavaScript-heavy content."""
        try:
            # Run Selenium in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._selenium_scrape, url, sport)
        except Exception as e:
            logger.warning(f"Selenium scraping failed, falling back to HTTP: {e}")
            # Fallback to HTTP scraping
            html = await self._make_request(url)
            if not html:
                return []
            return self._parse_draftkings_html(html, sport)
    
    def _selenium_scrape(self, url: str, sport: str) -> List[PropBet]:
        """Selenium scraping implementation for DraftKings."""
        with BettingSiteSeleniumScraper(self.selenium_config) as scraper:
            try:
                # Navigate to page
                if not scraper.navigate_to(url, wait_for_element="body"):
                    logger.warning(f"Failed to navigate to {url}")
                    return []
                
                # Wait for content to load
                time.sleep(5)
                
                # Try to find prop bet containers
                prop_containers = []
                selectors = [
                    '[data-testid*="market"]',
                    '[class*="market"]',
                    '[class*="prop"]',
                    '[class*="bet"]',
                    '.sportsbook-market',
                    '.market-group',
                    '.betting-market',
                    '[class*="sportsbook"]'
                ]
                
                for selector in selectors:
                    containers = scraper.find_elements(selector)
                    if containers:
                        prop_containers.extend(containers)
                        logger.debug(f"Found {len(containers)} containers with selector: {selector}")
                
                if not prop_containers:
                    logger.warning("No prop containers found with Selenium")
                    return []
                
                # Extract prop bet data
                prop_bets = []
                for container in prop_containers:
                    prop_data = self._extract_selenium_prop_data(container, sport)
                    if prop_data:
                        prop_bets.append(prop_data)
                
                logger.info(f"Selenium scraped {len(prop_bets)} prop bets from DraftKings")
                return prop_bets
                
            except Exception as e:
                logger.error(f"Selenium scraping error: {e}")
                return []
    
    def _extract_selenium_prop_data(self, container: Any, sport: str) -> Optional[PropBet]:
        """Extract prop bet data from Selenium container for DraftKings."""
        try:
            # Extract player name
            player_name = self._extract_selenium_text(container, [
                '[class*="player"] [class*="name"]',
                '[class*="player-name"]',
                '[class*="participant"]',
                '[class*="athlete"]',
                'h3, h4, h5',
                '[class*="market-title"]'
            ])
            
            if not player_name:
                return None
            
            # Extract stat line
            stat_line = self._extract_selenium_text(container, [
                '[class*="stat"]',
                '[class*="line"]',
                '[class*="market"]',
                '[class*="type"]',
                '[class*="outcome"]'
            ])
            
            if not stat_line:
                return None
            
            # Parse stat type and line value
            stat_type, line_value = self._parse_stat_line(stat_line)
            if not stat_type or not line_value:
                return None
            
            # Extract odds
            odds_text = self._extract_selenium_text(container, [
                '[class*="odds"]',
                '[class*="price"]',
                '[class*="bet"]',
                '[class*="line"]'
            ])
            
            over_odds, under_odds = self._parse_odds(odds_text)
            
            return PropBet(
                player_name=player_name,
                stat_type=stat_type,
                line_value=line_value,
                over_odds=over_odds,
                under_odds=under_odds,
                betting_app=BettingApp.DRAFTKINGS,
                game_info=f"{sport.upper()} Game"
            )
            
        except Exception as e:
            logger.warning(f"Error extracting Selenium prop data: {e}")
            return None
    
    def _extract_selenium_text(self, container: Any, selectors: List[str]) -> Optional[str]:
        """Extract text using multiple selectors from Selenium container."""
        for selector in selectors:
            try:
                element = container.find_element(By.CSS_SELECTOR, selector)
                text = element.text.strip()
                if text:
                    return text
            except NoSuchElementException:
                continue
        return None
    
    def _parse_draftkings_html(self, html: str, sport: str) -> List[PropBet]:
        """Parse DraftKings HTML for prop bets."""
        soup = BeautifulSoup(html, 'html.parser')
        prop_bets = []
        
        try:
            # Look for prop bet markets
            markets = soup.find_all(['div', 'section'], class_=re.compile(r'market|prop|bet', re.I))
            
            for market in markets:
                # Extract player name
                player_elem = market.find(['span', 'div'], class_=re.compile(r'player|name', re.I))
                if not player_elem:
                    continue
                
                player_name = player_elem.get_text(strip=True)
                
                # Extract stat and line
                stat_elem = market.find(['span', 'div'], class_=re.compile(r'stat|line|market', re.I))
                if not stat_elem:
                    continue
                
                stat_text = stat_elem.get_text(strip=True)
                stat_type, line_value = self._parse_draftkings_stat(stat_text)
                
                if stat_type and line_value:
                    # Extract odds
                    over_odds, under_odds = self._extract_odds(market)
                    
                    prop_bet = PropBet(
                        player_name=player_name,
                        stat_type=stat_type,
                        line_value=line_value,
                        over_odds=over_odds,
                        under_odds=under_odds,
                        betting_app=BettingApp.DRAFTKINGS,
                        game_info=f"{sport.upper()} Game"
                    )
                    prop_bets.append(prop_bet)
                    
        except Exception as e:
            logger.warning(f"Error parsing DraftKings HTML: {e}")
        
        logger.info(f"Scraped {len(prop_bets)} prop bets from DraftKings")
        return prop_bets
    
    def _parse_draftkings_stat(self, text: str) -> tuple[Optional[str], Optional[float]]:
        """Parse DraftKings stat format."""
        # DraftKings format: "Points Over 25.5", "Rebounds Under 8.5", etc.
        patterns = [
            r'(\w+)\s+(Over|Under)\s+(\d+\.?\d*)',
            r'(\w+)\s+(\d+\.?\d*)\s+(Over|Under)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                stat_type = match.group(1).lower()
                line_value = float(match.group(3) if len(match.groups()) == 3 else match.group(2))
                return stat_type, line_value
        
        return None, None


class BetMGMScraper(PropBetScraper):
    """Scraper for BetMGM prop bets."""
    
    def __init__(self, use_selenium: bool = True):
        """Initialize BetMGM scraper."""
        super().__init__()
        self.use_selenium = use_selenium
        self.selenium_config = SeleniumConfig(
            headless=True,
            disable_images=True,
            implicit_wait=15
        )
    
    async def scrape_prop_bets(self, sport: str) -> List[PropBet]:
        """Scrape prop bets from BetMGM."""
        logger.info(f"Scraping BetMGM prop bets for {sport}")
        
        # BetMGM URLs
        urls = {
            'nfl': 'https://sports.betmgm.com/nfl-player-props',
            'nba': 'https://sports.betmgm.com/nba-player-props',
            'mlb': 'https://sports.betmgm.com/mlb-player-props',
        }
        
        url = urls.get(sport.lower())
        if not url:
            logger.warning(f"No BetMGM URL found for sport: {sport}")
            return []
        
        if self.use_selenium:
            return await self._scrape_with_selenium(url, sport)
        else:
            html = await self._make_request(url)
            if not html:
                return []
            return self._parse_betmgm_html(html, sport)
    
    async def _scrape_with_selenium(self, url: str, sport: str) -> List[PropBet]:
        """Scrape using Selenium for JavaScript-heavy content."""
        try:
            # Run Selenium in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._selenium_scrape, url, sport)
        except Exception as e:
            logger.warning(f"Selenium scraping failed, falling back to HTTP: {e}")
            # Fallback to HTTP scraping
            html = await self._make_request(url)
            if not html:
                return []
            return self._parse_betmgm_html(html, sport)
    
    def _selenium_scrape(self, url: str, sport: str) -> List[PropBet]:
        """Selenium scraping implementation for BetMGM."""
        with BettingSiteSeleniumScraper(self.selenium_config) as scraper:
            try:
                # Navigate to page
                if not scraper.navigate_to(url, wait_for_element="body"):
                    logger.warning(f"Failed to navigate to {url}")
                    return []
                
                # Wait for content to load
                time.sleep(5)
                
                # Try to find prop bet containers
                prop_containers = []
                selectors = [
                    '[class*="market"]',
                    '[class*="prop"]',
                    '[class*="bet"]',
                    '[class*="outcome"]',
                    '.betting-market',
                    '.market-group',
                    '.sportsbook-market',
                    '[class*="sportsbook"]',
                    '[data-testid*="market"]'
                ]
                
                for selector in selectors:
                    containers = scraper.find_elements(selector)
                    if containers:
                        prop_containers.extend(containers)
                        logger.debug(f"Found {len(containers)} containers with selector: {selector}")
                
                if not prop_containers:
                    logger.warning("No prop containers found with Selenium")
                    return []
                
                # Extract prop bet data
                prop_bets = []
                for container in prop_containers:
                    prop_data = self._extract_selenium_prop_data(container, sport)
                    if prop_data:
                        prop_bets.append(prop_data)
                
                logger.info(f"Selenium scraped {len(prop_bets)} prop bets from BetMGM")
                return prop_bets
                
            except Exception as e:
                logger.error(f"Selenium scraping error: {e}")
                return []
    
    def _extract_selenium_prop_data(self, container: Any, sport: str) -> Optional[PropBet]:
        """Extract prop bet data from Selenium container for BetMGM."""
        try:
            # Extract player name
            player_name = self._extract_selenium_text(container, [
                '[class*="player"] [class*="name"]',
                '[class*="player-name"]',
                '[class*="participant"]',
                '[class*="athlete"]',
                '[class*="competitor"]',
                'h3, h4, h5',
                '[class*="market-title"]',
                '[class*="outcome-title"]'
            ])
            
            if not player_name:
                return None
            
            # Extract stat line
            stat_line = self._extract_selenium_text(container, [
                '[class*="stat"]',
                '[class*="line"]',
                '[class*="market"]',
                '[class*="type"]',
                '[class*="outcome"]',
                '[class*="selection"]'
            ])
            
            if not stat_line:
                return None
            
            # Parse stat type and line value
            stat_type, line_value = self._parse_stat_line(stat_line)
            if not stat_type or not line_value:
                return None
            
            # Extract odds
            odds_text = self._extract_selenium_text(container, [
                '[class*="odds"]',
                '[class*="price"]',
                '[class*="bet"]',
                '[class*="line"]',
                '[class*="decimal"]'
            ])
            
            over_odds, under_odds = self._parse_odds(odds_text)
            
            return PropBet(
                player_name=player_name,
                stat_type=stat_type,
                line_value=line_value,
                over_odds=over_odds,
                under_odds=under_odds,
                betting_app=BettingApp.BETMGM,
                game_info=f"{sport.upper()} Game"
            )
            
        except Exception as e:
            logger.warning(f"Error extracting Selenium prop data: {e}")
            return None
    
    def _extract_selenium_text(self, container: Any, selectors: List[str]) -> Optional[str]:
        """Extract text using multiple selectors from Selenium container."""
        for selector in selectors:
            try:
                element = container.find_element(By.CSS_SELECTOR, selector)
                text = element.text.strip()
                if text:
                    return text
            except NoSuchElementException:
                continue
        return None
    
    def _parse_betmgm_html(self, html: str, sport: str) -> List[PropBet]:
        """Parse BetMGM HTML for prop bets."""
        soup = BeautifulSoup(html, 'html.parser')
        prop_bets = []
        
        try:
            # Look for prop bet markets
            markets = soup.find_all(['div', 'section'], class_=re.compile(r'market|prop|bet', re.I))
            
            for market in markets:
                # Extract player name
                player_elem = market.find(['span', 'div'], class_=re.compile(r'player|name', re.I))
                if not player_elem:
                    continue
                
                player_name = player_elem.get_text(strip=True)
                
                # Extract stat and line
                stat_elem = market.find(['span', 'div'], class_=re.compile(r'stat|line|market', re.I))
                if not stat_elem:
                    continue
                
                stat_text = stat_elem.get_text(strip=True)
                stat_type, line_value = self._parse_betmgm_stat(stat_text)
                
                if stat_type and line_value:
                    # Extract odds
                    over_odds, under_odds = self._extract_odds(market)
                    
                    prop_bet = PropBet(
                        player_name=player_name,
                        stat_type=stat_type,
                        line_value=line_value,
                        over_odds=over_odds,
                        under_odds=under_odds,
                        betting_app=BettingApp.BETMGM,
                        game_info=f"{sport.upper()} Game"
                    )
                    prop_bets.append(prop_bet)
                    
        except Exception as e:
            logger.warning(f"Error parsing BetMGM HTML: {e}")
        
        logger.info(f"Scraped {len(prop_bets)} prop bets from BetMGM")
        return prop_bets
    
    def _parse_betmgm_stat(self, text: str) -> tuple[Optional[str], Optional[float]]:
        """Parse BetMGM stat format."""
        # BetMGM format: "Points Over 25.5", "Rebounds Under 8.5", etc.
        patterns = [
            r'(\w+)\s+(Over|Under)\s+(\d+\.?\d*)',
            r'(\w+)\s+(\d+\.?\d*)\s+(Over|Under)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                stat_type = match.group(1).lower()
                line_value = float(match.group(3) if len(match.groups()) == 3 else match.group(2))
                return stat_type, line_value
        
        return None, None


class PropBetAggregator:
    """Aggregates prop bets from multiple betting sites."""
    
    def __init__(self, use_selenium: bool = True):
        """
        Initialize prop bet aggregator.
        
        Args:
            use_selenium: Whether to use Selenium for JavaScript-heavy sites
        """
        self.use_selenium = use_selenium
        self.scrapers = {
            BettingApp.FANDUEL: FanDuelScraper(use_selenium=use_selenium),
            BettingApp.PRIZE_PICKS: PrizePicksScraper(),
            BettingApp.UNDERDOG: UnderdogScraper(),
            BettingApp.DRAFTKINGS: DraftKingsScraper(use_selenium=use_selenium),
            BettingApp.BETMGM: BetMGMScraper(use_selenium=use_selenium),
        }
    
    async def scrape_all_prop_bets(self, sport: str) -> List[PropBet]:
        """Scrape prop bets from all supported betting sites."""
        logger.info(f"Scraping prop bets from all sites for {sport}")
        
        all_prop_bets = []
        
        # Scrape from each site concurrently
        tasks = []
        for app, scraper in self.scrapers.items():
            task = self._scrape_site_prop_bets(scraper, sport, app)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for result in results:
            if isinstance(result, list):
                all_prop_bets.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Scraping failed: {result}")
        
        logger.info(f"Total prop bets scraped: {len(all_prop_bets)}")
        return all_prop_bets
    
    async def _scrape_site_prop_bets(self, scraper: PropBetScraper, sport: str, app: BettingApp) -> List[PropBet]:
        """Scrape prop bets from a single site."""
        try:
            async with scraper:
                return await scraper.scrape_prop_bets(sport)
        except Exception as e:
            logger.warning(f"Failed to scrape {app.value}: {e}")
            return []
    
    def find_best_odds(self, prop_bets: List[PropBet], stat_type: str, player_name: str) -> Optional[PropBet]:
        """Find the best odds for a specific prop bet."""
        matching_bets = [
            bet for bet in prop_bets
            if bet.stat_type.lower() == stat_type.lower() and 
               bet.player_name.lower() == player_name.lower()
        ]
        
        if not matching_bets:
            return None
        
        # Find best over odds
        over_bets = [bet for bet in matching_bets if bet.over_odds is not None]
        if over_bets:
            best_over = max(over_bets, key=lambda x: x.over_odds)
            return best_over
        
        # Find best under odds
        under_bets = [bet for bet in matching_bets if bet.under_odds is not None]
        if under_bets:
            best_under = max(under_bets, key=lambda x: x.under_odds)
            return best_under
        
        return matching_bets[0]  # Return first match if no odds available
