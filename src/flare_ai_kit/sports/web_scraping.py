"""Web scraping utilities for sports data collection."""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import structlog
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

logger = structlog.get_logger(__name__)


class WebScrapingConfig:
    """Configuration for web scraping."""
    
    # Request settings
    REQUEST_TIMEOUT = 10
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    # Rate limiting
    REQUEST_DELAY = 1  # seconds between requests
    
    # Headers
    HEADERS = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }


class SportsWebScraper:
    """Web scraper for sports data from various sources."""
    
    def __init__(self):
        """Initialize the web scraper."""
        self.ua = UserAgent()
        self.session = None
        self.config = WebScrapingConfig()
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.REQUEST_TIMEOUT)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, url: str, retries: int = None) -> Optional[str]:
        """
        Make a web request with retries and error handling.
        
        Args:
            url: URL to request
            retries: Number of retries (uses config default if None)
            
        Returns:
            HTML content or None if failed
        """
        if retries is None:
            retries = self.config.MAX_RETRIES
            
        headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        for attempt in range(retries + 1):
            try:
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.debug(f"Successfully scraped {url}")
                        return content
                    elif response.status == 429:  # Rate limited
                        wait_time = self.config.RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < retries:
                    await asyncio.sleep(self.config.RETRY_DELAY * (2 ** attempt))
                    
        logger.error(f"Failed to scrape {url} after {retries + 1} attempts")
        return None
    
    async def scrape_espn_scores(self, sport: str, date: datetime = None) -> List[Dict[str, Any]]:
        """
        Scrape ESPN scores for a specific sport and date.
        
        Args:
            sport: Sport name (nba, nfl, mlb, etc.)
            date: Date to scrape (defaults to today)
            
        Returns:
            List of game data
        """
        if date is None:
            date = datetime.now()
            
        # ESPN URL format
        date_str = date.strftime("%Y%m%d")
        url = f"https://www.espn.com/{sport}/scoreboard/_/date/{date_str}"
        
        html = await self._make_request(url)
        if not html:
            return []
            
        soup = BeautifulSoup(html, 'html.parser')
        games = []
        
        try:
            # Try multiple selectors for different ESPN layouts
            game_selectors = [
                'div[class*="ScoreCell"]',
                'div[class*="ScoreboardGameCell"]',
                'div[class*="GameCell"]',
                'div[class*="game"]'
            ]
            
            game_containers = []
            for selector in game_selectors:
                containers = soup.select(selector)
                if containers:
                    game_containers = containers
                    logger.debug(f"Found {len(containers)} games using selector: {selector}")
                    break
            
            # If no games found with selectors, try to find any game-like elements
            if not game_containers:
                game_containers = soup.find_all('div', class_=lambda x: x and 'game' in x.lower())
                logger.debug(f"Found {len(game_containers)} games using fallback selector")
            
            for container in game_containers:
                game_data = self._parse_espn_game(container)
                if game_data:
                    games.append(game_data)
                    
        except Exception as e:
            logger.error(f"Error parsing ESPN scores: {e}")
            
        logger.info(f"Scraped {len(games)} games from ESPN {sport}")
        return games
    
    def _parse_espn_game(self, container) -> Optional[Dict[str, Any]]:
        """Parse a single ESPN game container with multiple layout support."""
        try:
            # Try multiple selectors for team names
            team_selectors = [
                'div[class*="TeamName"]',
                'div[class*="team-name"]',
                'span[class*="team"]',
                'a[class*="team"]',
                'h3', 'h4', 'h5'  # Fallback to heading elements
            ]
            
            teams = []
            for selector in team_selectors:
                team_elements = container.select(selector)
                if len(team_elements) >= 2:
                    teams = team_elements
                    break
            
            if len(teams) < 2:
                # Try to find any text that looks like team names
                all_text = container.get_text()
                # Look for patterns like "Team A vs Team B" or "Team A @ Team B"
                import re
                team_pattern = r'([A-Za-z\s]+?)\s+(?:vs|@|at)\s+([A-Za-z\s]+)'
                match = re.search(team_pattern, all_text)
                if match:
                    away_team = match.group(1).strip()
                    home_team = match.group(2).strip()
                else:
                    return None
            else:
                home_team = teams[1].get_text(strip=True)
                away_team = teams[0].get_text(strip=True)
            
            # Clean team names
            home_team = self._clean_team_name(home_team)
            away_team = self._clean_team_name(away_team)
            
            # Try multiple selectors for scores
            score_selectors = [
                'div[class*="Score"]',
                'span[class*="score"]',
                'div[class*="points"]',
                'span[class*="points"]'
            ]
            
            scores = []
            for selector in score_selectors:
                score_elements = container.select(selector)
                if len(score_elements) >= 2:
                    scores = score_elements
                    break
            
            home_score = None
            away_score = None
            
            if len(scores) >= 2:
                try:
                    home_score = int(scores[1].get_text(strip=True))
                    away_score = int(scores[0].get_text(strip=True))
                except ValueError:
                    # Try to extract numbers from text
                    import re
                    all_text = container.get_text()
                    numbers = re.findall(r'\d+', all_text)
                    if len(numbers) >= 2:
                        home_score = int(numbers[-1])  # Last number is usually home score
                        away_score = int(numbers[-2])  # Second to last is away score
            else:
                # Try to extract scores from any text
                import re
                all_text = container.get_text()
                numbers = re.findall(r'\d+', all_text)
                if len(numbers) >= 2:
                    home_score = int(numbers[-1])
                    away_score = int(numbers[-2])
            
            # Extract game status and time
            status_selectors = [
                'div[class*="Time"]',
                'div[class*="Status"]',
                'span[class*="time"]',
                'span[class*="status"]'
            ]
            
            status = "Final"
            game_time = None
            
            for selector in status_selectors:
                status_elem = container.select_one(selector)
                if status_elem:
                    status_text = status_elem.get_text(strip=True)
                    if status_text:
                        if any(word in status_text.lower() for word in ['final', 'ft', 'ot', 'end']):
                            status = "Final"
                        elif any(word in status_text.lower() for word in ['live', 'q', 'quarter', 'half']):
                            status = "Live"
                        else:
                            status = status_text
                        game_time = status_text
                        break
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'status': status,
                'game_time': game_time,
                'source': 'espn'
            }
            
        except Exception as e:
            logger.warning(f"Error parsing ESPN game: {e}")
            return None
    
    def _clean_team_name(self, name: str) -> str:
        """Clean and standardize team names."""
        if not name:
            return ""
        
        # Remove common prefixes/suffixes
        name = name.strip()
        name = re.sub(r'^\d+\s*', '', name)  # Remove leading numbers
        name = re.sub(r'\s*\(\d+\)$', '', name)  # Remove trailing numbers in parentheses
        name = re.sub(r'\s*#\d+$', '', name)  # Remove trailing #numbers
        
        # Common team name mappings
        team_mappings = {
            'LAL': 'Lakers',
            'GSW': 'Warriors',
            'BOS': 'Celtics',
            'MIA': 'Heat',
            'DEN': 'Nuggets',
            'KC': 'Chiefs',
            'BUF': 'Bills',
            'TB': 'Buccaneers',
            'DAL': 'Cowboys'
        }
        
        return team_mappings.get(name, name)
    
    async def scrape_espn_players(self, sport: str, team: str = None) -> List[Dict[str, Any]]:
        """
        Scrape ESPN player rosters for a sport and optionally specific team.
        
        Args:
            sport: Sport name (nba, nfl, mlb, etc.)
            team: Optional team name to filter by
            
        Returns:
            List of player data
        """
        try:
            # ESPN teams page URL
            if team:
                # Search for specific team
                team_slug = team.lower().replace(' ', '-')
                url = f"https://www.espn.com/{sport}/team/roster/_/name/{team_slug}"
            else:
                # Get all teams for the sport
                url = f"https://www.espn.com/{sport}/teams"
            
            html = await self._make_request(url)
            if not html:
                return []
                
            soup = BeautifulSoup(html, 'html.parser')
            players = []
            
            if team:
                # Parse single team roster
                players = self._parse_team_roster(soup, sport, team)
            else:
                # Parse all teams
                team_links = soup.find_all('a', href=re.compile(r'/{sport}/team/roster'.format(sport=sport)))
                for link in team_links[:5]:  # Limit to first 5 teams for demo
                    team_url = urljoin(url, link['href'])
                    team_html = await self._make_request(team_url)
                    if team_html:
                        team_soup = BeautifulSoup(team_html, 'html.parser')
                        team_name = link.get_text(strip=True)
                        team_players = self._parse_team_roster(team_soup, sport, team_name)
                        players.extend(team_players)
            
            logger.info(f"Scraped {len(players)} players from ESPN {sport}")
            return players
            
        except Exception as e:
            logger.error(f"Error scraping ESPN players: {e}")
            return []
    
    def _parse_team_roster(self, soup: BeautifulSoup, sport: str, team: str) -> List[Dict[str, Any]]:
        """Parse a team roster page with enhanced JSON extraction."""
        players = []
        
        try:
            # First try to extract JSON data from script tags
            script_tags = soup.find_all('script', type='application/json')
            for script in script_tags:
                try:
                    data = json.loads(script.string)
                    if 'roster' in data or 'athletes' in data:
                        players = self._extract_players_from_json(data, sport, team)
                        if players:
                            logger.debug(f"Extracted {len(players)} players from JSON data")
                            return players
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # Try to find JSON in window.__INITIAL_STATE__ or similar
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string and ('roster' in script.string or 'athletes' in script.string):
                    try:
                        # Extract JSON from script content
                        import re
                        json_match = re.search(r'window\.__INITIAL_STATE__\s*=\s*({.*?});', script.string)
                        if json_match:
                            data = json.loads(json_match.group(1))
                            players = self._extract_players_from_json(data, sport, team)
                            if players:
                                logger.debug(f"Extracted {len(players)} players from window.__INITIAL_STATE__")
                                return players
                    except (json.JSONDecodeError, AttributeError):
                        continue
            
            # Look for ESPN's specific roster structure
            roster_section = soup.find('section', class_='Roster')
            if roster_section:
                players = self._parse_espn_roster_section(roster_section, sport, team)
                if players:
                    return players
            
            # Fallback to HTML table parsing
            roster_table = soup.find('table', class_=re.compile(r'roster|players|table'))
            if not roster_table:
                roster_table = soup.find('div', class_=re.compile(r'roster|players'))
            
            if roster_table:
                rows = roster_table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    player_data = self._parse_player_row(row, sport, team)
                    if player_data:
                        players.append(player_data)
            else:
                # Final fallback: look for any player-like elements
                player_elements = soup.find_all(['div', 'span'], class_=re.compile(r'player|name'))
                for elem in player_elements[:10]:  # Limit to 10 players
                    player_name = elem.get_text(strip=True)
                    if player_name and len(player_name) > 2:
                        players.append({
                            'name': player_name,
                            'team': team,
                            'sport': sport,
                            'position': 'Unknown',
                            'source': 'espn'
                        })
                        
        except Exception as e:
            logger.warning(f"Error parsing team roster for {team}: {e}")
            
        return players
    
    def _parse_espn_roster_section(self, roster_section, sport: str, team: str) -> List[Dict[str, Any]]:
        """Parse ESPN's specific roster section structure."""
        players = []
        
        try:
            # Look for roster tables within the section
            tables = roster_section.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows:
                    # Skip header rows
                    if row.find('th'):
                        continue
                    
                    cells = row.find_all('td')
                    if len(cells) < 3:  # Need at least name, position, and some other data
                        continue
                    
                    # Extract player data from cells
                    player_data = self._parse_espn_player_cells(cells, sport, team)
                    if player_data:
                        players.append(player_data)
                        
        except Exception as e:
            logger.warning(f"Error parsing ESPN roster section: {e}")
            
        return players
    
    def _parse_espn_player_cells(self, cells, sport: str, team: str) -> Optional[Dict[str, Any]]:
        """Parse player data from ESPN table cells."""
        try:
            if len(cells) < 3:
                return None
            
            # Extract name (usually first cell with a link)
            name_cell = cells[0]
            name_link = name_cell.find('a')
            if name_link:
                name = name_link.get_text(strip=True)
            else:
                name = name_cell.get_text(strip=True)
            
            if not name or len(name) < 2:
                return None
            
            # Extract position (usually second cell)
            position = cells[1].get_text(strip=True) if len(cells) > 1 else 'Unknown'
            
            # Extract additional data
            height = ''
            weight = ''
            age = 0
            experience = 0
            college = ''
            jersey = ''
            
            # Try to extract from remaining cells
            for i, cell in enumerate(cells[2:], 2):
                cell_text = cell.get_text(strip=True)
                
                # Height (e.g., "6' 5\"")
                if "'" in cell_text and '"' in cell_text:
                    height = cell_text
                # Weight (e.g., "237 lbs")
                elif 'lbs' in cell_text.lower():
                    weight = cell_text
                # Age (numeric)
                elif cell_text.isdigit() and 18 <= int(cell_text) <= 50:
                    age = int(cell_text)
                # Experience (numeric, usually smaller)
                elif cell_text.isdigit() and 0 <= int(cell_text) <= 20:
                    experience = int(cell_text)
                # Jersey number (numeric, usually 1-2 digits)
                elif cell_text.isdigit() and 0 <= int(cell_text) <= 99:
                    jersey = cell_text
                # College (text, usually longer)
                elif len(cell_text) > 3 and not cell_text.isdigit():
                    college = cell_text
            
            return {
                'name': name,
                'team': team,
                'sport': sport,
                'position': position,
                'height': height,
                'weight': weight,
                'age': age,
                'experience': experience,
                'college': college,
                'jersey': jersey,
                'source': 'espn'
            }
            
        except Exception as e:
            logger.warning(f"Error parsing ESPN player cells: {e}")
            return None
    
    def _extract_players_from_json(self, data: Dict[str, Any], sport: str, team: str) -> List[Dict[str, Any]]:
        """Extract player data from ESPN's JSON structure."""
        players = []
        
        try:
            # Navigate through the JSON structure to find roster data
            roster_data = None
            
            # Try different possible paths in the JSON
            possible_paths = [
                ['roster', 'groups'],
                ['roster', 'athletes'],
                ['groups'],
                ['athletes'],
                ['data', 'roster', 'groups'],
                ['data', 'roster', 'athletes']
            ]
            
            for path in possible_paths:
                current = data
                for key in path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        current = None
                        break
                
                if current and isinstance(current, list):
                    roster_data = current
                    break
            
            if not roster_data:
                return players
            
            # Extract players from the roster data
            for group in roster_data:
                if isinstance(group, dict) and 'athletes' in group:
                    for athlete in group['athletes']:
                        if isinstance(athlete, dict):
                            player_data = self._parse_athlete_json(athlete, sport, team)
                            if player_data:
                                players.append(player_data)
                elif isinstance(group, dict) and 'name' in group:
                    # Direct athlete data
                    player_data = self._parse_athlete_json(group, sport, team)
                    if player_data:
                        players.append(player_data)
                        
        except Exception as e:
            logger.warning(f"Error extracting players from JSON: {e}")
            
        return players
    
    def _parse_athlete_json(self, athlete: Dict[str, Any], sport: str, team: str) -> Optional[Dict[str, Any]]:
        """Parse individual athlete data from JSON."""
        try:
            # Extract basic player information
            name = athlete.get('name', '')
            if not name:
                return None
                
            position = athlete.get('position', 'Unknown')
            height = athlete.get('height', '')
            weight = athlete.get('weight', '')
            age = athlete.get('age', 0)
            experience = athlete.get('experience', 0)
            college = athlete.get('college', '')
            jersey = athlete.get('jersey', '')
            
            # Extract additional stats if available
            stats = {}
            for key, value in athlete.items():
                if key not in ['name', 'position', 'height', 'weight', 'age', 'experience', 'college', 'jersey', 'href', 'uid', 'id']:
                    if isinstance(value, (int, float)):
                        stats[key] = value
                    elif isinstance(value, str) and value.isdigit():
                        stats[key] = int(value)
            
            return {
                'name': name,
                'team': team,
                'sport': sport,
                'position': position,
                'height': height,
                'weight': weight,
                'age': age,
                'experience': experience,
                'college': college,
                'jersey': jersey,
                'stats': stats,
                'source': 'espn'
            }
            
        except Exception as e:
            logger.warning(f"Error parsing athlete JSON: {e}")
            return None
    
    def _parse_player_row(self, row, sport: str, team: str) -> Optional[Dict[str, Any]]:
        """Parse a single player row from roster table."""
        try:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:
                return None
                
            # Extract player name (usually first cell)
            name_cell = cells[0]
            name = name_cell.get_text(strip=True)
            
            # Extract position (usually second cell)
            position = "Unknown"
            if len(cells) > 1:
                position = cells[1].get_text(strip=True)
            
            # Extract additional stats if available
            stats = {}
            for i, cell in enumerate(cells[2:], 2):
                cell_text = cell.get_text(strip=True)
                if cell_text and cell_text.isdigit():
                    stats[f'stat_{i}'] = int(cell_text)
                elif cell_text:
                    stats[f'info_{i}'] = cell_text
            
            return {
                'name': name,
                'team': team,
                'sport': sport,
                'position': position,
                'stats': stats,
                'source': 'espn'
            }
            
        except Exception as e:
            logger.warning(f"Error parsing player row: {e}")
            return None
    
    async def scrape_espn_player_stats(self, sport: str, player_name: str) -> Optional[Dict[str, Any]]:
        """
        Scrape ESPN player statistics.
        
        Args:
            sport: Sport name
            player_name: Player name to search for
            
        Returns:
            Player statistics or None
        """
        # ESPN player search URL
        search_url = f"https://www.espn.com/{sport}/players/_/name/{player_name.replace(' ', '-').lower()}"
        
        html = await self._make_request(search_url)
        if not html:
            return None
            
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            # Find player stats table
            stats_table = soup.find('table', class_='Table')
            if not stats_table:
                return None
                
            # Parse stats
            stats = {}
            rows = stats_table.find_all('tr')
            
            for row in rows[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    
                    # Try to convert to number
                    try:
                        if '.' in value:
                            stats[key] = float(value)
                        else:
                            stats[key] = int(value)
                    except ValueError:
                        stats[key] = value
                        
            return {
                'name': player_name,
                'sport': sport,
                'stats': stats,
                'source': 'espn'
            }
            
        except Exception as e:
            logger.error(f"Error parsing ESPN player stats: {e}")
            return None
    
    async def scrape_odds_data(self, sport: str, date: datetime = None) -> List[Dict[str, Any]]:
        """
        Scrape odds data from various sources.
        
        Args:
            sport: Sport name
            date: Date to scrape
            
        Returns:
            List of odds data
        """
        if date is None:
            date = datetime.now()
            
        odds_data = []
        
        # Scrape from multiple sources
        sources = [
            self._scrape_vegas_insider_odds,
            self._scrape_oddsshark_odds,
        ]
        
        for source_func in sources:
            try:
                data = await source_func(sport, date)
                odds_data.extend(data)
            except Exception as e:
                logger.warning(f"Error scraping odds from {source_func.__name__}: {e}")
                
        return odds_data
    
    async def _scrape_vegas_insider_odds(self, sport: str, date: datetime) -> List[Dict[str, Any]]:
        """Scrape odds from Vegas Insider."""
        # Vegas Insider URL format
        date_str = date.strftime("%Y%m%d")
        url = f"https://www.vegasinsider.com/{sport}/odds/las-vegas/"
        
        html = await self._make_request(url)
        if not html:
            return []
            
        soup = BeautifulSoup(html, 'html.parser')
        odds_data = []
        
        try:
            # Find odds table
            odds_table = soup.find('table', class_='frodds-data-tbl')
            if not odds_table:
                return []
                
            rows = odds_table.find_all('tr')
            for row in rows[1:]:  # Skip header
                cells = row.find_all('td')
                if len(cells) >= 4:
                    team1 = cells[0].get_text(strip=True)
                    team2 = cells[1].get_text(strip=True)
                    spread = cells[2].get_text(strip=True)
                    total = cells[3].get_text(strip=True)
                    
                    odds_data.append({
                        'team1': team1,
                        'team2': team2,
                        'spread': spread,
                        'total': total,
                        'source': 'vegas_insider'
                    })
                    
        except Exception as e:
            logger.error(f"Error parsing Vegas Insider odds: {e}")
            
        return odds_data
    
    async def _scrape_oddsshark_odds(self, sport: str, date: datetime) -> List[Dict[str, Any]]:
        """Scrape odds from OddsShark."""
        # OddsShark URL format
        date_str = date.strftime("%Y-%m-%d")
        url = f"https://www.oddsshark.com/{sport}/odds?date={date_str}"
        
        html = await self._make_request(url)
        if not html:
            return []
            
        soup = BeautifulSoup(html, 'html.parser')
        odds_data = []
        
        try:
            # Find odds containers
            game_containers = soup.find_all('div', class_='odds-table-row')
            
            for container in game_containers:
                teams = container.find_all('div', class_='team-name')
                if len(teams) >= 2:
                    team1 = teams[0].get_text(strip=True)
                    team2 = teams[1].get_text(strip=True)
                    
                    # Extract spread and total
                    spread_elem = container.find('div', class_='spread')
                    total_elem = container.find('div', class_='total')
                    
                    spread = spread_elem.get_text(strip=True) if spread_elem else None
                    total = total_elem.get_text(strip=True) if total_elem else None
                    
                    odds_data.append({
                        'team1': team1,
                        'team2': team2,
                        'spread': spread,
                        'total': total,
                        'source': 'oddsshark'
                    })
                    
        except Exception as e:
            logger.error(f"Error parsing OddsShark odds: {e}")
            
        return odds_data
    
    async def scrape_weather_data(self, venue: str, date: datetime) -> Optional[Dict[str, Any]]:
        """
        Scrape weather data for a venue.
        
        Args:
            venue: Venue name
            date: Date for weather
            
        Returns:
            Weather data or None
        """
        # Use OpenWeatherMap or similar service
        # For now, return mock data
        return {
            'venue': venue,
            'date': date.isoformat(),
            'temperature': 72,
            'humidity': 65,
            'wind_speed': 8,
            'conditions': 'Clear',
            'source': 'mock'
        }


class SportsDataScraper:
    """High-level sports data scraper that coordinates multiple sources."""
    
    def __init__(self):
        """Initialize the sports data scraper."""
        self.scraper = SportsWebScraper()
        
    async def get_games_data(self, sport: str, date: datetime = None) -> List[Dict[str, Any]]:
        """
        Get comprehensive games data for a sport.
        
        Args:
            sport: Sport name
            date: Date to scrape
            
        Returns:
            List of game data
        """
        async with self.scraper as scraper:
            # Scrape from multiple sources
            games_data = []
            
            # ESPN scores
            espn_games = await scraper.scrape_espn_scores(sport, date)
            games_data.extend(espn_games)
            
            # Add odds data
            odds_data = await scraper.scrape_odds_data(sport, date)
            
            # Merge odds with games
            for game in games_data:
                game['odds'] = self._find_matching_odds(game, odds_data)
                
            return games_data
    
    def _find_matching_odds(self, game: Dict[str, Any], odds_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find matching odds data for a game."""
        home_team = game.get('home_team', '').lower()
        away_team = game.get('away_team', '').lower()
        
        for odds in odds_data:
            team1 = odds.get('team1', '').lower()
            team2 = odds.get('team2', '').lower()
            
            # Simple matching logic
            if (home_team in team1 and away_team in team2) or (home_team in team2 and away_team in team1):
                return odds
                
        return None
    
    async def get_player_stats(self, sport: str, player_name: str) -> Optional[Dict[str, Any]]:
        """
        Get player statistics.
        
        Args:
            sport: Sport name
            player_name: Player name
            
        Returns:
            Player statistics
        """
        async with self.scraper as scraper:
            return await scraper.scrape_espn_player_stats(sport, player_name)
    
    async def get_weather_data(self, venue: str, date: datetime) -> Optional[Dict[str, Any]]:
        """
        Get weather data for a venue.
        
        Args:
            venue: Venue name
            date: Date for weather
            
        Returns:
            Weather data
        """
        async with self.scraper as scraper:
            return await scraper.scrape_weather_data(venue, date)
    
    async def get_players_via_scraping(self, sport: str) -> List[Dict[str, Any]]:
        """
        Get players using web scraping.
        
        Args:
            sport: Sport name
            
        Returns:
            List of player data
        """
        async with self.scraper as scraper:
            return await scraper.scrape_espn_players(sport)
