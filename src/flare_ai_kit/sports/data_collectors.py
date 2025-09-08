"""Data collectors for sports betting aggregator."""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import structlog

from .models import (
    Player, Game, PropBet, Sport, League, PropType, BettingApp,
    SportsDataRequest
)
from .web_scraping import SportsDataScraper
from .prop_bet_scraper import PropBetAggregator

logger = structlog.get_logger(__name__)


class SportsDataCollector:
    """Collects sports data from various APIs and web scraping."""
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize sports data collector.
        
        Args:
            api_keys: Dictionary of API keys for different services
        """
        self.api_keys = api_keys
        self.session: Optional[aiohttp.ClientSession] = None
        self.web_scraper = SportsDataScraper()
        self.prop_bet_aggregator = PropBetAggregator()
        
        # API endpoints
        self.espn_base = "https://site.api.espn.com/apis/site/v2/sports"
        self.sportsradar_base = "https://api.sportradar.com"
        self.odds_api_base = "https://api.the-odds-api.com/v4"
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "Flare-AI-Kit/1.0"}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_players(self, sport: Sport, league: League) -> List[Player]:
        """
        Get players for a specific sport and league.
        
        Args:
            sport: Sport to get players for
            league: League to get players for
            
        Returns:
            List of Player objects
        """
        if not self.session:
            raise RuntimeError("Sports data collector not initialized")
            
        try:
            # Map sport/league to ESPN API path
            sport_path = self._get_espn_sport_path(sport, league)
            url = f"{self.espn_base}/{sport_path}/players"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    players = self._parse_players(data, sport, league)
                    logger.info(f"Retrieved {len(players)} players for {sport.value}")
                    return players
                else:
                    logger.error(f"Failed to get players: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting players: {e}")
            return []
    
    async def get_games(
        self, 
        sport: Sport, 
        league: League, 
        date: Optional[datetime] = None
    ) -> List[Game]:
        """
        Get games for a specific sport, league, and date.
        
        Args:
            sport: Sport to get games for
            league: League to get games for
            date: Date to get games for (defaults to today)
            
        Returns:
            List of Game objects
        """
        if not self.session:
            raise RuntimeError("Sports data collector not initialized")
            
        if date is None:
            date = datetime.now()
            
        try:
            sport_path = self._get_espn_sport_path(sport, league)
            date_str = date.strftime("%Y%m%d")
            url = f"{self.espn_base}/{sport_path}/scoreboard"
            
            params = {"dates": date_str}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    games = self._parse_games(data, sport, league)
                    logger.info(f"Retrieved {len(games)} games for {sport.value} on {date_str}")
                    return games
                else:
                    logger.error(f"Failed to get games: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting games: {e}")
            return []
    
    async def get_player_stats(
        self, 
        player_id: str, 
        sport: Sport, 
        league: League,
        season: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed player statistics.
        
        Args:
            player_id: Player identifier
            sport: Sport
            league: League
            season: Season (defaults to current)
            
        Returns:
            Dictionary of player statistics
        """
        if not self.session:
            raise RuntimeError("Sports data collector not initialized")
            
        try:
            sport_path = self._get_espn_sport_path(sport, league)
            url = f"{self.espn_base}/{sport_path}/players/{player_id}/stats"
            
            params = {}
            if season:
                params["season"] = season
                
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    stats = self._parse_player_stats(data, sport)
                    logger.info(f"Retrieved stats for player {player_id}")
                    return stats
                else:
                    logger.error(f"Failed to get player stats: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting player stats: {e}")
            return {}
    
    def _get_espn_sport_path(self, sport: Sport, league: League) -> str:
        """Get ESPN API path for sport/league combination."""
        mapping = {
            (Sport.NFL, League.NFL): "football/nfl",
            (Sport.NBA, League.NBA): "basketball/nba",
            (Sport.WNBA, League.WNBA): "basketball/wnba",
            (Sport.MLB, League.MLB): "baseball/mlb",
            (Sport.MLS, League.MLS): "soccer/usa.1",
            (Sport.ESPORTS, League.LCS): "esports/league-of-legends",
            (Sport.ESPORTS, League.CSGO): "esports/counter-strike",
            (Sport.ESPORTS, League.VALORANT): "esports/valorant",
            (Sport.ESPORTS, League.DOTA2): "esports/dota-2",
        }
        
        return mapping.get((sport, league), "basketball/nba")
    
    def _parse_players(self, data: Dict[str, Any], sport: Sport, league: League) -> List[Player]:
        """Parse players from ESPN API response."""
        players = []
        
        try:
            for player_data in data.get("athletes", []):
                player = Player(
                    id=player_data.get("id", ""),
                    name=player_data.get("displayName", ""),
                    position=player_data.get("position", {}).get("abbreviation", ""),
                    team=player_data.get("team", {}).get("displayName", ""),
                    sport=sport,
                    league=league,
                    height=player_data.get("height"),
                    weight=player_data.get("weight"),
                    age=player_data.get("age"),
                    injury_status=player_data.get("injuries", [{}])[0].get("status") if player_data.get("injuries") else None,
                    is_active=player_data.get("active", True)
                )
                players.append(player)
                
        except Exception as e:
            logger.error(f"Error parsing players: {e}")
            
        return players
    
    def _parse_games(self, data: Dict[str, Any], sport: Sport, league: League) -> List[Game]:
        """Parse games from ESPN API response."""
        games = []
        
        try:
            for event in data.get("events", []):
                game = Game(
                    id=event.get("id", ""),
                    home_team=event.get("competitions", [{}])[0].get("competitors", [{}])[0].get("team", {}).get("displayName", ""),
                    away_team=event.get("competitions", [{}])[0].get("competitors", [{}])[1].get("team", {}).get("displayName", ""),
                    sport=sport,
                    league=league,
                    game_date=datetime.fromisoformat(event.get("date", "").replace("Z", "+00:00")),
                    venue=event.get("competitions", [{}])[0].get("venue", {}).get("fullName"),
                    status=event.get("status", {}).get("type", {}).get("name", "scheduled"),
                    home_score=int(event.get("competitions", [{}])[0].get("competitors", [{}])[0].get("score", 0)),
                    away_score=int(event.get("competitions", [{}])[0].get("competitors", [{}])[1].get("score", 0))
                )
                games.append(game)
                
        except Exception as e:
            logger.error(f"Error parsing games: {e}")
            
        return games
    
    def _parse_player_stats(self, data: Dict[str, Any], sport: Sport) -> Dict[str, Any]:
        """Parse player statistics from ESPN API response."""
        stats = {}
        
        try:
            # Extract season stats
            for stat_category in data.get("stats", []):
                category = stat_category.get("label", "")
                for stat in stat_category.get("stats", []):
                    stat_name = stat.get("label", "")
                    stat_value = stat.get("value", 0)
                    stats[f"{category}_{stat_name}"] = float(stat_value) if stat_value else 0.0
                    
        except Exception as e:
            logger.error(f"Error parsing player stats: {e}")
            
        return stats
    
    async def get_players_via_scraping(self, sport: Sport, league: League) -> List[Player]:
        """
        Get players using web scraping as fallback.
        
        Args:
            sport: Sport type
            league: League type
            
        Returns:
            List of Player objects
        """
        try:
            # Use web scraper to get real player data from ESPN
            sport_name = sport.value.lower()
            
            # Scrape players from ESPN
            scraped_players = await self.web_scraper.get_players_via_scraping(sport_name)
            
            # Convert scraped data to Player objects
            players = []
            for player_data in scraped_players:
                try:
                    player = Player(
                        id=player_data['name'].lower().replace(' ', '_'),
                        name=player_data['name'],
                        position=player_data.get('position', 'Unknown'),
                        team=player_data.get('team', 'Unknown'),
                        sport=sport,
                        league=league,
                        season_averages=player_data.get('stats', {}),
                        recent_stats=player_data.get('stats', {}),
                        source=player_data.get('source', 'espn_scraping')
                    )
                    players.append(player)
                except Exception as e:
                    logger.warning(f"Error creating player from scraped data: {e}")
                    continue
            
            # If no players scraped, fall back to sample data
            if not players:
                players = self._get_sample_players(sport, league)
                logger.info(f"Using sample data for {sport.value} - no real data scraped")
            else:
                logger.info(f"Retrieved {len(players)} players via ESPN scraping for {sport.value}")
            
            return players
            
        except Exception as e:
            logger.error(f"Web scraping failed for players: {e}")
            # Fall back to sample data
            return self._get_sample_players(sport, league)
    
    def _get_sample_players(self, sport: Sport, league: League) -> List[Player]:
        """Get sample players for testing when APIs are unavailable."""
        if sport == Sport.NBA:
            return [
                Player(
                    id="lebron_james",
                    name="LeBron James",
                    position="SF",
                    team="Lakers",
                    sport=sport,
                    league=league,
                    season_averages={"points": 25.0, "rebounds": 7.0, "assists": 8.0},
                    recent_stats={"points": 28.0, "rebounds": 8.0, "assists": 9.0}
                ),
                Player(
                    id="stephen_curry",
                    name="Stephen Curry",
                    position="PG",
                    team="Warriors",
                    sport=sport,
                    league=league,
                    season_averages={"points": 26.0, "rebounds": 4.0, "assists": 6.0},
                    recent_stats={"points": 30.0, "rebounds": 5.0, "assists": 7.0}
                )
            ]
        elif sport == Sport.NFL:
            return [
                Player(
                    id="patrick_mahomes",
                    name="Patrick Mahomes",
                    position="QB",
                    team="Chiefs",
                    sport=sport,
                    league=league,
                    season_averages={"passing_yards": 300.0, "touchdowns": 2.5, "interceptions": 0.8},
                    recent_stats={"passing_yards": 350.0, "touchdowns": 3.0, "interceptions": 1.0}
                ),
                Player(
                    id="josh_allen",
                    name="Josh Allen",
                    position="QB",
                    team="Bills",
                    sport=sport,
                    league=league,
                    season_averages={"passing_yards": 280.0, "touchdowns": 2.2, "interceptions": 1.1},
                    recent_stats={"passing_yards": 320.0, "touchdowns": 2.0, "interceptions": 0.0}
                )
            ]
        else:
            return []


class BettingDataCollector:
    """Collects betting data from various betting apps."""
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize betting data collector.
        
        Args:
            api_keys: Dictionary of API keys for betting services
        """
        self.api_keys = api_keys
        self.session: Optional[aiohttp.ClientSession] = None
        self.prop_bet_aggregator = PropBetAggregator()
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "Flare-AI-Kit/1.0"}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_prop_bets(
        self, 
        sport: Sport, 
        league: League,
        date: Optional[datetime] = None
    ) -> List[PropBet]:
        """
        Get prop bets from all supported betting apps using the new scraper.
        
        Args:
            sport: Sport to get prop bets for
            league: League to get prop bets for
            date: Date to get prop bets for
            
        Returns:
            List of PropBet objects
        """
        logger.info(f"Collecting prop bets for {sport.value} {league.value}")
        
        try:
            # Use the new prop bet aggregator
            scraped_prop_bets = await self.prop_bet_aggregator.scrape_all_prop_bets(sport.value)
            
            # Convert scraped prop bets to our PropBet model
            all_prop_bets = []
            for scraped_bet in scraped_prop_bets:
                prop_bet = PropBet(
                    player_name=scraped_bet.player_name,
                    stat_type=scraped_bet.stat_type,
                    line_value=scraped_bet.line_value,
                    over_odds=scraped_bet.over_odds,
                    under_odds=scraped_bet.under_odds,
                    betting_app=scraped_bet.betting_app,
                    game_info=scraped_bet.game_info,
                    timestamp=scraped_bet.timestamp
                )
                all_prop_bets.append(prop_bet)
            
            logger.info(f"Collected {len(all_prop_bets)} prop bets for {sport.value}")
            return all_prop_bets
            
        except Exception as e:
            logger.warning(f"Failed to collect prop bets: {e}")
            return []
    
    async def _get_fanduel_props(
        self, 
        sport: Sport, 
        league: League, 
        date: Optional[datetime]
    ) -> List[PropBet]:
        """Get prop bets from FanDuel."""
        # This would integrate with FanDuel's API
        # For now, return mock data
        logger.info("Collecting FanDuel prop bets...")
        return []
    
    async def _get_prize_picks_props(
        self, 
        sport: Sport, 
        league: League, 
        date: Optional[datetime]
    ) -> List[PropBet]:
        """Get prop bets from Prize Picks."""
        # This would integrate with Prize Picks' API
        # For now, return mock data
        logger.info("Collecting Prize Picks prop bets...")
        return []
    
    async def _get_underdog_props(
        self, 
        sport: Sport, 
        league: League, 
        date: Optional[datetime]
    ) -> List[PropBet]:
        """Get prop bets from Underdog."""
        # This would integrate with Underdog's API
        # For now, return mock data
        logger.info("Collecting Underdog prop bets...")
        return []


class WeatherDataCollector:
    """Collects weather data for outdoor sports."""
    
    def __init__(self, api_key: str):
        """
        Initialize weather data collector.
        
        Args:
            api_key: OpenWeatherMap API key
        """
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_weather_for_game(
        self, 
        venue: str, 
        game_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Get weather data for a specific game venue and date.
        
        Args:
            venue: Game venue
            game_date: Game date and time
            
        Returns:
            Weather data dictionary or None
        """
        if not self.session:
            raise RuntimeError("Weather data collector not initialized")
            
        try:
            # This would integrate with OpenWeatherMap API
            # For now, return mock data
            logger.info(f"Getting weather for {venue} on {game_date}")
            return {
                "temperature": 72.0,
                "humidity": 65.0,
                "wind_speed": 8.0,
                "precipitation": 0.0,
                "conditions": "Clear"
            }
            
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
            return None
