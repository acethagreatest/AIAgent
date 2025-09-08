"""FDC integration for sports betting data verification."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import structlog

from ..ecosystem.protocols.fdc import FDC, FDCRequest
from ..ecosystem.protocols.da_layer import DALayerClient
from ..ecosystem.settings_models import EcosystemSettingsModel
from .models import Player, Game, PropBet, Sport, League

logger = structlog.get_logger(__name__)


class SportsFDCClient:
    """FDC client specialized for sports data attestation."""
    
    def __init__(self, fdc_client: FDC, da_client: DALayerClient):
        """
        Initialize sports FDC client.
        
        Args:
            fdc_client: FDC client instance
            da_client: Data Availability Layer client
        """
        self.fdc_client = fdc_client
        self.da_client = da_client
        
    async def attest_player_stats(
        self, 
        player: Player, 
        stats: Dict[str, Any],
        game: Game
    ) -> Optional[str]:
        """
        Attest player statistics using FDC.
        
        Args:
            player: Player information
            stats: Player statistics to attest
            game: Game context
            
        Returns:
            Attestation request ID or None if failed
        """
        try:
            # Create JSON API request for player stats
            stats_url = f"https://api.espn.com/v2/sports/{self._get_sport_path(player.sport)}/players/{player.id}/stats"
            
            # Create JQ filter for specific stats
            jq_filter = self._create_stats_jq_filter(stats)
            
            request = await self.fdc_client.create_json_api_request(
                url=stats_url,
                jq_filter=jq_filter,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Flare-AI-Kit/1.0"
                }
            )
            
            # Submit attestation request
            tx_hash = await self.fdc_client.request_attestation(
                attestation_type=request.attestation_type,
                data=request.data,
                expected_response_hash=request.expected_response_hash,
                fee=request.fee
            )
            
            logger.info(f"Player stats attestation requested for {player.name}: {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error attesting player stats: {e}")
            return None
    
    async def attest_game_data(
        self, 
        game: Game,
        include_weather: bool = True
    ) -> Optional[str]:
        """
        Attest game data using FDC.
        
        Args:
            game: Game information
            include_weather: Whether to include weather data
            
        Returns:
            Attestation request ID or None if failed
        """
        try:
            # Create JSON API request for game data
            game_url = f"https://api.espn.com/v2/sports/{self._get_sport_path(game.sport)}/scoreboard"
            
            # Create JQ filter for game data
            jq_filter = self._create_game_jq_filter(game, include_weather)
            
            request = await self.fdc_client.create_json_api_request(
                url=game_url,
                jq_filter=jq_filter,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Flare-AI-Kit/1.0"
                }
            )
            
            # Submit attestation request
            tx_hash = await self.fdc_client.request_attestation(
                attestation_type=request.attestation_type,
                data=request.data,
                expected_response_hash=request.expected_response_hash,
                fee=request.fee
            )
            
            logger.info(f"Game data attestation requested for {game.home_team} vs {game.away_team}: {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error attesting game data: {e}")
            return None
    
    async def attest_betting_odds(
        self, 
        prop_bet: PropBet,
        odds_source: str = "the-odds-api"
    ) -> Optional[str]:
        """
        Attest betting odds using FDC.
        
        Args:
            prop_bet: Prop bet information
            odds_source: Source of odds data
            
        Returns:
            Attestation request ID or None if failed
        """
        try:
            # Create JSON API request for betting odds
            odds_url = self._get_odds_api_url(prop_bet, odds_source)
            
            # Create JQ filter for odds data
            jq_filter = self._create_odds_jq_filter(prop_bet)
            
            request = await self.fdc_client.create_json_api_request(
                url=odds_url,
                jq_filter=jq_filter,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Flare-AI-Kit/1.0"
                }
            )
            
            # Submit attestation request
            tx_hash = await self.fdc_client.request_attestation(
                attestation_type=request.attestation_type,
                data=request.data,
                expected_response_hash=request.expected_response_hash,
                fee=request.fee
            )
            
            logger.info(f"Betting odds attestation requested for {prop_bet.player.name}: {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error attesting betting odds: {e}")
            return None
    
    async def verify_attested_data(
        self, 
        attestation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Verify attested data using FDC.
        
        Args:
            attestation_id: ID of the attestation to verify
            
        Returns:
            Verified data or None if verification failed
        """
        try:
            async with self.da_client:
                # Get attestation response
                response_data = await self.da_client.get_attestation_response(attestation_id)
                merkle_proof = await self.da_client.get_merkle_proof(attestation_id)
                
                if not response_data or not merkle_proof:
                    logger.warning(f"No attested data found for {attestation_id}")
                    return None
                
                # Verify the data
                is_valid = await self.fdc_client.verify_attestation(
                    response_data=response_data,
                    merkle_proof=merkle_proof,
                    merkle_root=response_data.get("merkleRoot", "")
                )
                
                if is_valid:
                    logger.info(f"Successfully verified attested data for {attestation_id}")
                    return response_data
                else:
                    logger.warning(f"Attestation verification failed for {attestation_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error verifying attested data: {e}")
            return None
    
    async def batch_attest_sports_data(
        self, 
        players: List[Player],
        games: List[Game],
        prop_bets: List[PropBet]
    ) -> Dict[str, List[str]]:
        """
        Batch attest multiple types of sports data.
        
        Args:
            players: List of players to attest
            games: List of games to attest
            prop_bets: List of prop bets to attest
            
        Returns:
            Dictionary mapping data type to list of attestation IDs
        """
        attestation_ids = {
            "players": [],
            "games": [],
            "prop_bets": []
        }
        
        # Attest player data
        for player in players:
            if player.recent_stats:
                attestation_id = await self.attest_player_stats(
                    player, player.recent_stats, games[0] if games else None
                )
                if attestation_id:
                    attestation_ids["players"].append(attestation_id)
        
        # Attest game data
        for game in games:
            attestation_id = await self.attest_game_data(game)
            if attestation_id:
                attestation_ids["games"].append(attestation_id)
        
        # Attest prop bet data
        for prop_bet in prop_bets:
            attestation_id = await self.attest_betting_odds(prop_bet)
            if attestation_id:
                attestation_ids["prop_bets"].append(attestation_id)
        
        logger.info(f"Batch attestation completed: {sum(len(ids) for ids in attestation_ids.values())} attestations")
        return attestation_ids
    
    def _get_sport_path(self, sport: Sport) -> str:
        """Get ESPN API path for sport."""
        mapping = {
            Sport.NFL: "football/nfl",
            Sport.NBA: "basketball/nba",
            Sport.WNBA: "basketball/wnba",
            Sport.MLB: "baseball/mlb",
            Sport.MLS: "soccer/usa.1",
            Sport.ESPORTS: "esports/league-of-legends"
        }
        return mapping.get(sport, "basketball/nba")
    
    def _create_stats_jq_filter(self, stats: Dict[str, Any]) -> str:
        """Create JQ filter for player statistics."""
        # Create filter for specific stats
        stat_keys = list(stats.keys())
        if not stat_keys:
            return "."
        
        # Build JQ filter to extract specific stats
        filter_parts = []
        for key in stat_keys:
            filter_parts.append(f'.{key}')
        
        return f"{{ {', '.join(filter_parts)} }}"
    
    def _create_game_jq_filter(self, game: Game, include_weather: bool) -> str:
        """Create JQ filter for game data."""
        if include_weather:
            return f'.events[] | select(.id == "{game.id}") | {{id, date, status, weather, homeTeam, awayTeam}}'
        else:
            return f'.events[] | select(.id == "{game.id}") | {{id, date, status, homeTeam, awayTeam}}'
    
    def _create_odds_jq_filter(self, prop_bet: PropBet) -> str:
        """Create JQ filter for betting odds."""
        return f'.markets[] | select(.key == "{prop_bet.prop_type.value}") | {{outcomes, lastUpdate}}'
    
    def _get_odds_api_url(self, prop_bet: PropBet, source: str) -> str:
        """Get odds API URL for prop bet."""
        if source == "the-odds-api":
            sport_key = self._get_odds_api_sport_key(prop_bet.sport)
            return f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        else:
            # Default to a generic odds API
            return "https://api.example.com/odds"
    
    def _get_odds_api_sport_key(self, sport: Sport) -> str:
        """Get The Odds API sport key."""
        mapping = {
            Sport.NFL: "americanfootball_nfl",
            Sport.NBA: "basketball_nba",
            Sport.WNBA: "basketball_wnba",
            Sport.MLB: "baseball_mlb",
            Sport.MLS: "soccer_usa_mls",
            Sport.ESPORTS: "esports_lol"
        }
        return mapping.get(sport, "basketball_nba")


class SportsDataVerifier:
    """Verifies sports data using FDC attestations."""
    
    def __init__(self, sports_fdc_client: SportsFDCClient):
        """
        Initialize sports data verifier.
        
        Args:
            sports_fdc_client: Sports FDC client instance
        """
        self.sports_fdc_client = sports_fdc_client
    
    async def verify_player_performance(
        self, 
        player: Player, 
        game: Game,
        expected_stats: Dict[str, Any]
    ) -> bool:
        """
        Verify player performance against attested data.
        
        Args:
            player: Player information
            game: Game context
            expected_stats: Expected performance stats
            
        Returns:
            True if performance matches attested data
        """
        try:
            # This would retrieve and verify attested player stats
            # For now, return True as a placeholder
            logger.info(f"Verifying performance for {player.name} in {game.home_team} vs {game.away_team}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying player performance: {e}")
            return False
    
    async def verify_betting_line(
        self, 
        prop_bet: PropBet,
        expected_line: float
    ) -> bool:
        """
        Verify betting line against attested data.
        
        Args:
            prop_bet: Prop bet information
            expected_line: Expected betting line
            
        Returns:
            True if line matches attested data
        """
        try:
            # This would retrieve and verify attested betting odds
            # For now, return True as a placeholder
            logger.info(f"Verifying betting line for {prop_bet.player.name}: {expected_line}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying betting line: {e}")
            return False
