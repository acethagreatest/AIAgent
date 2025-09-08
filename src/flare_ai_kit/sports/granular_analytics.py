"""Granular analytics engine for player performance analysis."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import structlog

from .models import (
    Player, Game, PropBet, Prediction, GranularPrediction,
    PlayerTeamMatchup, VenueAnalysis, Sport, League, PropType
)

logger = structlog.get_logger(__name__)


class GranularAnalyticsEngine:
    """Advanced analytics engine with granular player performance analysis."""
    
    def __init__(self):
        """Initialize the granular analytics engine."""
        self.matchup_data: Dict[str, PlayerTeamMatchup] = {}
        self.venue_data: Dict[str, VenueAnalysis] = {}
        self.models = {}
        self.scalers = {}
        
    async def analyze_player_team_matchup(
        self, 
        player: Player, 
        opponent_team: str,
        historical_games: List[Dict[str, Any]]
    ) -> PlayerTeamMatchup:
        """
        Analyze player performance against a specific team.
        
        Args:
            player: Player information
            opponent_team: Opponent team name
            historical_games: Historical game data
            
        Returns:
            PlayerTeamMatchup analysis
        """
        try:
            # Filter games against this opponent
            matchup_games = [
                game for game in historical_games
                if (game.get("opponent_team") == opponent_team and 
                    game.get("player_id") == player.id)
            ]
            
            if not matchup_games:
                # Return empty matchup if no data
                return PlayerTeamMatchup(
                    player_id=player.id,
                    opponent_team=opponent_team,
                    sport=player.sport
                )
            
            # Calculate average stats
            stat_keys = ["points", "rebounds", "assists", "steals", "blocks", "turnovers"]
            average_stats = {}
            home_stats = {}
            away_stats = {}
            
            home_games = [g for g in matchup_games if g.get("is_home", False)]
            away_games = [g for g in matchup_games if not g.get("is_home", False)]
            
            for stat in stat_keys:
                values = [g.get(stat, 0) for g in matchup_games if g.get(stat) is not None]
                if values:
                    average_stats[stat] = np.mean(values)
                
                # Home performance
                home_values = [g.get(stat, 0) for g in home_games if g.get(stat) is not None]
                if home_values:
                    home_stats[stat] = np.mean(home_values)
                
                # Away performance
                away_values = [g.get(stat, 0) for g in away_games if g.get(stat) is not None]
                if away_values:
                    away_stats[stat] = np.mean(away_values)
            
            # Calculate consistency score
            consistency_score = self._calculate_consistency_score(matchup_games, stat_keys)
            
            # Determine recent trend
            recent_trend = self._calculate_recent_trend(matchup_games, stat_keys)
            
            # Get last meeting data
            last_meeting = max(matchup_games, key=lambda x: x.get("date", datetime.min))
            last_meeting_stats = {stat: last_meeting.get(stat, 0) for stat in stat_keys}
            
            matchup = PlayerTeamMatchup(
                player_id=player.id,
                opponent_team=opponent_team,
                sport=player.sport,
                games_played=len(matchup_games),
                average_stats=average_stats,
                home_vs_team=home_stats,
                away_vs_team=away_stats,
                recent_trend=recent_trend,
                consistency_score=consistency_score,
                last_meeting_date=last_meeting.get("date"),
                last_meeting_stats=last_meeting_stats
            )
            
            # Cache the matchup data
            cache_key = f"{player.id}_{opponent_team}_{player.sport}"
            self.matchup_data[cache_key] = matchup
            
            logger.info(f"Analyzed matchup: {player.name} vs {opponent_team} ({len(matchup_games)} games)")
            return matchup
            
        except Exception as e:
            logger.error(f"Error analyzing team matchup: {e}")
            return PlayerTeamMatchup(
                player_id=player.id,
                opponent_team=opponent_team,
                sport=player.sport
            )
    
    async def analyze_venue_performance(
        self, 
        player: Player, 
        venue: str,
        historical_games: List[Dict[str, Any]]
    ) -> VenueAnalysis:
        """
        Analyze player performance at a specific venue.
        
        Args:
            player: Player information
            venue: Venue name
            historical_games: Historical game data
            
        Returns:
            VenueAnalysis
        """
        try:
            # Filter games at this venue
            venue_games = [
                game for game in historical_games
                if (game.get("venue") == venue and 
                    game.get("player_id") == player.id)
            ]
            
            if not venue_games:
                return VenueAnalysis(
                    player_id=player.id,
                    venue_name=venue,
                    sport=player.sport
                )
            
            # Calculate average stats at venue
            stat_keys = ["points", "rebounds", "assists", "steals", "blocks", "turnovers"]
            average_stats = {}
            
            for stat in stat_keys:
                values = [g.get(stat, 0) for g in venue_games if g.get(stat) is not None]
                if values:
                    average_stats[stat] = np.mean(values)
            
            # Calculate home advantage
            home_games = [g for g in venue_games if g.get("is_home", False)]
            away_games = [g for g in venue_games if not g.get("is_home", False)]
            
            home_advantage = 0.0
            if home_games and away_games:
                home_avg = np.mean([g.get("points", 0) for g in home_games])
                away_avg = np.mean([g.get("points", 0) for g in away_games])
                home_advantage = home_avg - away_avg
            
            # Calculate venue familiarity
            venue_familiarity = min(100.0, len(venue_games) * 10.0)  # 10 points per game
            
            venue_analysis = VenueAnalysis(
                player_id=player.id,
                venue_name=venue,
                sport=player.sport,
                games_played=len(venue_games),
                average_stats=average_stats,
                venue_type="indoor",  # Default, would be determined from venue data
                home_advantage=home_advantage,
                venue_familiarity=venue_familiarity
            )
            
            # Cache the venue data
            cache_key = f"{player.id}_{venue}_{player.sport}"
            self.venue_data[cache_key] = venue_analysis
            
            logger.info(f"Analyzed venue performance: {player.name} at {venue} ({len(venue_games)} games)")
            return venue_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing venue performance: {e}")
            return VenueAnalysis(
                player_id=player.id,
                venue_name=venue,
                sport=player.sport
            )
    
    async def create_granular_prediction(
        self, 
        base_prediction: Prediction,
        player: Player,
        game: Game,
        historical_data: List[Dict[str, Any]]
    ) -> GranularPrediction:
        """
        Create a granular prediction with team and venue analysis.
        
        Args:
            base_prediction: Base prediction from ML model
            player: Player information
            game: Game context
            historical_data: Historical performance data
            
        Returns:
            GranularPrediction with adjustments
        """
        try:
            # Determine opponent team
            opponent_team = game.away_team if player.team == game.home_team else game.home_team
            is_home_game = player.team == game.home_team
            
            # Get team matchup analysis
            matchup = await self.analyze_player_team_matchup(
                player, opponent_team, historical_data
            )
            
            # Get venue analysis
            venue_analysis = await self.analyze_venue_performance(
                player, game.venue or "Unknown", historical_data
            )
            
            # Calculate adjustments
            home_away_adjustment = self._calculate_home_away_adjustment(
                player, is_home_game, matchup
            )
            
            team_matchup_adjustment = self._calculate_team_matchup_adjustment(
                base_prediction, matchup, player.sport
            )
            
            venue_adjustment = self._calculate_venue_adjustment(
                base_prediction, venue_analysis, is_home_game
            )
            
            # Calculate final adjusted prediction
            total_adjustment = home_away_adjustment + team_matchup_adjustment + venue_adjustment
            adjusted_value = base_prediction.predicted_value + total_adjustment
            
            # Generate adjustment factors
            adjustment_factors = self._generate_adjustment_factors(
                home_away_adjustment, team_matchup_adjustment, venue_adjustment,
                matchup, venue_analysis, is_home_game
            )
            
            # Calculate granular confidence
            granular_confidence = self._calculate_granular_confidence(
                base_prediction.confidence_score, matchup, venue_analysis
            )
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality_score(
                matchup, venue_analysis, historical_data
            )
            
            # Calculate similar matchups
            similar_matchups = matchup.games_played + venue_analysis.games_played
            matchup_accuracy = self._calculate_matchup_accuracy(matchup, venue_analysis)
            
            granular_prediction = GranularPrediction(
                base_prediction=base_prediction,
                home_away_adjustment=home_away_adjustment,
                team_matchup_adjustment=team_matchup_adjustment,
                venue_adjustment=venue_adjustment,
                adjusted_predicted_value=adjusted_value,
                adjustment_factors=adjustment_factors,
                base_confidence=base_prediction.confidence_score,
                granular_confidence=granular_confidence,
                data_quality_score=data_quality_score,
                similar_matchups=similar_matchups,
                matchup_accuracy=matchup_accuracy
            )
            
            logger.info(f"Created granular prediction for {player.name}: {adjusted_value:.2f} (adj: {total_adjustment:+.2f})")
            return granular_prediction
            
        except Exception as e:
            logger.error(f"Error creating granular prediction: {e}")
            # Return base prediction with minimal adjustments
            return GranularPrediction(
                base_prediction=base_prediction,
                adjusted_predicted_value=base_prediction.predicted_value,
                base_confidence=base_prediction.confidence_score,
                granular_confidence=base_prediction.confidence_score,
                data_quality_score=50.0,
                similar_matchups=0,
                matchup_accuracy=0.0
            )
    
    def _calculate_consistency_score(self, games: List[Dict[str, Any]], stat_keys: List[str]) -> float:
        """Calculate consistency score based on stat variance."""
        if len(games) < 2:
            return 0.0
        
        consistency_scores = []
        for stat in stat_keys:
            values = [g.get(stat, 0) for g in games if g.get(stat) is not None]
            if len(values) > 1:
                cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1.0
                consistency_scores.append(max(0, 100 - cv * 100))
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_recent_trend(self, games: List[Dict[str, Any]], stat_keys: List[str]) -> str:
        """Calculate recent performance trend."""
        if len(games) < 3:
            return "stable"
        
        # Sort by date and take last 3 games
        recent_games = sorted(games, key=lambda x: x.get("date", datetime.min))[-3:]
        
        trend_scores = []
        for stat in stat_keys:
            values = [g.get(stat, 0) for g in recent_games if g.get(stat) is not None]
            if len(values) == 3:
                # Calculate trend (positive = improving)
                trend = (values[2] - values[0]) / 2
                trend_scores.append(trend)
        
        if not trend_scores:
            return "stable"
        
        avg_trend = np.mean(trend_scores)
        if avg_trend > 1.0:
            return "improving"
        elif avg_trend < -1.0:
            return "declining"
        else:
            return "stable"
    
    def _calculate_home_away_adjustment(
        self, 
        player: Player, 
        is_home: bool, 
        matchup: PlayerTeamMatchup
    ) -> float:
        """Calculate home/away performance adjustment."""
        if not matchup.home_vs_team or not matchup.away_vs_team:
            return 0.0
        
        # Get the relevant stat (points for most prop bets)
        home_stat = matchup.home_vs_team.get("points", 0)
        away_stat = matchup.away_vs_team.get("points", 0)
        
        if is_home:
            return home_stat - away_stat
        else:
            return away_stat - home_stat
    
    def _calculate_team_matchup_adjustment(
        self, 
        base_prediction: Prediction, 
        matchup: PlayerTeamMatchup,
        sport: Sport
    ) -> float:
        """Calculate team matchup adjustment."""
        if matchup.games_played < 3:
            return 0.0
        
        # Get relevant stat from matchup
        matchup_stat = matchup.average_stats.get("points", 0)
        season_avg = base_prediction.prop_bet.player.season_averages.get("points", 0)
        
        if season_avg > 0:
            # Calculate how much better/worse vs this team
            adjustment = (matchup_stat - season_avg) * 0.3  # 30% weight
            return adjustment
        
        return 0.0
    
    def _calculate_venue_adjustment(
        self, 
        base_prediction: Prediction, 
        venue_analysis: VenueAnalysis,
        is_home: bool
    ) -> float:
        """Calculate venue-specific adjustment."""
        if venue_analysis.games_played < 2:
            return 0.0
        
        venue_stat = venue_analysis.average_stats.get("points", 0)
        season_avg = base_prediction.prop_bet.player.season_averages.get("points", 0)
        
        if season_avg > 0:
            # Calculate venue performance vs season average
            adjustment = (venue_stat - season_avg) * 0.2  # 20% weight
            
            # Add home advantage if applicable
            if is_home:
                adjustment += venue_analysis.home_advantage * 0.1
            
            return adjustment
        
        return 0.0
    
    def _generate_adjustment_factors(
        self,
        home_away_adj: float,
        team_matchup_adj: float,
        venue_adj: float,
        matchup: PlayerTeamMatchup,
        venue_analysis: VenueAnalysis,
        is_home: bool
    ) -> List[str]:
        """Generate human-readable adjustment factors."""
        factors = []
        
        if abs(home_away_adj) > 1.0:
            direction = "better" if home_away_adj > 0 else "worse"
            factors.append(f"Performs {direction} at {'home' if is_home else 'away'} vs this opponent")
        
        if abs(team_matchup_adj) > 1.0:
            direction = "better" if team_matchup_adj > 0 else "worse"
            factors.append(f"Historically performs {direction} against this team")
        
        if abs(venue_adj) > 1.0:
            direction = "better" if venue_adj > 0 else "worse"
            factors.append(f"Performs {direction} at this venue")
        
        if matchup.consistency_score > 80:
            factors.append("Very consistent performance in this matchup")
        elif matchup.consistency_score < 40:
            factors.append("Inconsistent performance in this matchup")
        
        if matchup.recent_trend == "improving":
            factors.append("Recent improving trend against this opponent")
        elif matchup.recent_trend == "declining":
            factors.append("Recent declining trend against this opponent")
        
        return factors
    
    def _calculate_granular_confidence(
        self, 
        base_confidence: float, 
        matchup: PlayerTeamMatchup,
        venue_analysis: VenueAnalysis
    ) -> float:
        """Calculate confidence after granular analysis."""
        confidence = base_confidence
        
        # Boost confidence based on data quality
        if matchup.games_played >= 5:
            confidence += 10.0
        if venue_analysis.games_played >= 3:
            confidence += 5.0
        
        # Boost confidence based on consistency
        if matchup.consistency_score > 80:
            confidence += 5.0
        
        # Reduce confidence if limited data
        if matchup.games_played < 3:
            confidence -= 15.0
        if venue_analysis.games_played < 2:
            confidence -= 10.0
        
        return max(0.0, min(100.0, confidence))
    
    def _calculate_data_quality_score(
        self, 
        matchup: PlayerTeamMatchup,
        venue_analysis: VenueAnalysis,
        historical_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate quality score of granular data."""
        score = 50.0  # Base score
        
        # Matchup data quality
        if matchup.games_played >= 10:
            score += 30.0
        elif matchup.games_played >= 5:
            score += 20.0
        elif matchup.games_played >= 3:
            score += 10.0
        
        # Venue data quality
        if venue_analysis.games_played >= 5:
            score += 20.0
        elif venue_analysis.games_played >= 3:
            score += 10.0
        
        # Historical data quality
        if len(historical_data) >= 20:
            score += 10.0
        elif len(historical_data) >= 10:
            score += 5.0
        
        return min(100.0, score)
    
    def _calculate_matchup_accuracy(
        self, 
        matchup: PlayerTeamMatchup,
        venue_analysis: VenueAnalysis
    ) -> float:
        """Calculate accuracy of similar matchups."""
        # This would be calculated based on historical prediction accuracy
        # For now, return a placeholder based on data quality
        if matchup.games_played >= 5 and venue_analysis.games_played >= 3:
            return 75.0
        elif matchup.games_played >= 3:
            return 65.0
        else:
            return 50.0
