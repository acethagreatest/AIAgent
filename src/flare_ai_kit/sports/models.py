"""Data models for sports betting aggregator."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal

from pydantic import BaseModel, Field


class Sport(str, Enum):
    """Supported sports."""
    NFL = "NFL"
    NBA = "NBA"
    WNBA = "WNBA"
    MLB = "MLB"
    MLS = "MLS"
    ESPORTS = "ESPORTS"


class League(str, Enum):
    """Supported leagues."""
    NFL = "NFL"
    NBA = "NBA"
    WNBA = "WNBA"
    MLB = "MLB"
    MLS = "MLS"
    LCS = "LCS"  # League of Legends
    CSGO = "CSGO"
    VALORANT = "VALORANT"
    DOTA2 = "DOTA2"


class BettingApp(str, Enum):
    """Supported betting applications."""
    FANDUEL = "FANDUEL"
    PRIZE_PICKS = "PRIZE_PICKS"
    UNDERDOG = "UNDERDOG"
    DRAFTKINGS = "DRAFTKINGS"
    BETMGM = "BETMGM"


class PropType(str, Enum):
    """Types of prop bets."""
    POINTS = "POINTS"
    REBOUNDS = "REBOUNDS"
    ASSISTS = "ASSISTS"
    YARDS = "YARDS"
    TOUCHDOWNS = "TOUCHDOWNS"
    STRIKEOUTS = "STRIKEOUTS"
    HITS = "HITS"
    GOALS = "GOALS"
    SAVES = "SAVES"
    KILLS = "KILLS"  # Esports
    DEATHS = "DEATHS"  # Esports
    ASSISTS_ESPORTS = "ASSISTS_ESPORTS"  # Esports


class Player(BaseModel):
    """Player information and statistics."""
    
    id: str = Field(..., description="Unique player identifier")
    name: str = Field(..., description="Player full name")
    position: str = Field(..., description="Player position")
    team: str = Field(..., description="Current team")
    sport: Sport = Field(..., description="Sport the player plays")
    league: League = Field(..., description="League the player is in")
    
    # Physical attributes
    height: Optional[str] = Field(None, description="Player height")
    weight: Optional[int] = Field(None, description="Player weight in lbs")
    age: Optional[int] = Field(None, description="Player age")
    
    # Status
    injury_status: Optional[str] = Field(None, description="Current injury status")
    is_active: bool = Field(True, description="Whether player is currently active")
    
    # Performance metrics (last 10 games)
    recent_stats: Dict[str, float] = Field(default_factory=dict, description="Recent performance stats")
    season_averages: Dict[str, float] = Field(default_factory=dict, description="Season averages")
    
    # Advanced metrics
    advanced_stats: Dict[str, float] = Field(default_factory=dict, description="Advanced analytics")
    
    # Granular performance data
    home_stats: Dict[str, float] = Field(default_factory=dict, description="Home game performance")
    away_stats: Dict[str, float] = Field(default_factory=dict, description="Away game performance")
    team_matchup_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Performance vs specific teams")
    venue_specific_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Performance at specific venues")
    
    class Config:
        use_enum_values = True


class Game(BaseModel):
    """Game information and context."""
    
    id: str = Field(..., description="Unique game identifier")
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    sport: Sport = Field(..., description="Sport")
    league: League = Field(..., description="League")
    
    # Game details
    game_date: datetime = Field(..., description="Game date and time")
    venue: Optional[str] = Field(None, description="Game venue")
    weather: Optional[Dict[str, Any]] = Field(None, description="Weather conditions")
    
    # Game status
    status: str = Field("scheduled", description="Game status (scheduled, live, finished)")
    quarter: Optional[str] = Field(None, description="Current quarter/period")
    home_score: Optional[int] = Field(None, description="Home team score")
    away_score: Optional[int] = Field(None, description="Away team score")
    
    # Context
    is_playoff: bool = Field(False, description="Whether this is a playoff game")
    importance: str = Field("regular", description="Game importance (regular, playoff, championship)")


class PropBet(BaseModel):
    """Prop bet information from betting apps."""
    
    id: str = Field(..., description="Unique prop bet identifier")
    player: Player = Field(..., description="Player the bet is about")
    game: Game = Field(..., description="Game context")
    
    # Bet details
    prop_type: PropType = Field(..., description="Type of prop bet")
    line: Decimal = Field(..., description="Betting line (over/under value)")
    over_odds: Decimal = Field(..., description="Odds for over")
    under_odds: Decimal = Field(..., description="Odds for under")
    
    # App information
    betting_app: BettingApp = Field(..., description="Betting application")
    app_bet_id: str = Field(..., description="Bet ID on the betting app")
    url: Optional[str] = Field(None, description="Direct link to the bet")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="When bet was created")
    expires_at: Optional[datetime] = Field(None, description="When bet expires")
    is_live: bool = Field(True, description="Whether bet is currently available")
    
    # FDC verification
    is_verified: bool = Field(False, description="Whether data is FDC verified")
    attestation_id: Optional[str] = Field(None, description="FDC attestation ID")
    
    class Config:
        use_enum_values = True


class Prediction(BaseModel):
    """AI prediction for a prop bet."""
    
    prop_bet: PropBet = Field(..., description="The prop bet being predicted")
    
    # Prediction details
    predicted_value: float = Field(..., description="Predicted statistical value")
    recommendation: str = Field(..., description="OVER or UNDER recommendation")
    confidence_score: float = Field(..., description="Confidence score (0-100)")
    
    # Analysis breakdown
    factors: List[str] = Field(default_factory=list, description="Key factors in prediction")
    player_form: str = Field(..., description="Player's recent form analysis")
    matchup_analysis: str = Field(..., description="Matchup-specific analysis")
    weather_impact: Optional[str] = Field(None, description="Weather impact analysis")
    
    # Historical context
    historical_accuracy: float = Field(..., description="Historical accuracy of similar predictions")
    sample_size: int = Field(..., description="Number of similar historical cases")
    
    # Risk assessment
    risk_level: str = Field(..., description="Risk level (LOW, MEDIUM, HIGH)")
    edge_percentage: float = Field(..., description="Expected edge percentage")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="When prediction was made")
    model_version: str = Field("1.0", description="ML model version used")
    
    class Config:
        use_enum_values = True


class ConfidenceScore(BaseModel):
    """Detailed confidence breakdown."""
    
    overall_confidence: float = Field(..., description="Overall confidence (0-100)")
    
    # Component scores
    player_form_score: float = Field(..., description="Player form confidence (0-100)")
    matchup_score: float = Field(..., description="Matchup confidence (0-100)")
    historical_score: float = Field(..., description="Historical data confidence (0-100)")
    data_quality_score: float = Field(..., description="Data quality confidence (0-100)")
    model_confidence: float = Field(..., description="ML model confidence (0-100)")
    
    # Risk factors
    injury_risk: float = Field(..., description="Injury risk factor (0-100)")
    weather_risk: float = Field(..., description="Weather risk factor (0-100)")
    sample_size_risk: float = Field(..., description="Sample size risk factor (0-100)")
    
    # Explanation
    explanation: str = Field(..., description="Human-readable confidence explanation")


class PlayerTeamMatchup(BaseModel):
    """Player performance against specific teams."""
    
    player_id: str = Field(..., description="Player identifier")
    opponent_team: str = Field(..., description="Opponent team name")
    sport: Sport = Field(..., description="Sport")
    
    # Performance metrics
    games_played: int = Field(0, description="Number of games played against this team")
    average_stats: Dict[str, float] = Field(default_factory=dict, description="Average stats vs this team")
    home_vs_team: Dict[str, float] = Field(default_factory=dict, description="Home performance vs this team")
    away_vs_team: Dict[str, float] = Field(default_factory=dict, description="Away performance vs this team")
    
    # Trend analysis
    recent_trend: str = Field("stable", description="Recent trend (improving, declining, stable)")
    consistency_score: float = Field(0.0, description="Consistency score (0-100)")
    
    # Last meeting
    last_meeting_date: Optional[datetime] = Field(None, description="Date of last meeting")
    last_meeting_stats: Dict[str, float] = Field(default_factory=dict, description="Stats from last meeting")
    
    class Config:
        use_enum_values = True


class VenueAnalysis(BaseModel):
    """Player performance at specific venues."""
    
    player_id: str = Field(..., description="Player identifier")
    venue_name: str = Field(..., description="Venue name")
    sport: Sport = Field(..., description="Sport")
    
    # Performance metrics
    games_played: int = Field(0, description="Number of games played at this venue")
    average_stats: Dict[str, float] = Field(default_factory=dict, description="Average stats at this venue")
    
    # Venue-specific factors
    venue_type: str = Field("indoor", description="Venue type (indoor, outdoor, dome)")
    altitude: Optional[float] = Field(None, description="Venue altitude")
    capacity: Optional[int] = Field(None, description="Venue capacity")
    
    # Performance trends
    home_advantage: float = Field(0.0, description="Home advantage factor")
    venue_familiarity: float = Field(0.0, description="Familiarity with venue (0-100)")
    
    class Config:
        use_enum_values = True


class GranularPrediction(BaseModel):
    """Enhanced prediction with granular analysis."""
    
    base_prediction: Prediction = Field(..., description="Base prediction")
    
    # Granular analysis
    home_away_adjustment: float = Field(0.0, description="Home/away performance adjustment")
    team_matchup_adjustment: float = Field(0.0, description="Team matchup adjustment")
    venue_adjustment: float = Field(0.0, description="Venue-specific adjustment")
    
    # Adjusted prediction
    adjusted_predicted_value: float = Field(..., description="Final adjusted prediction")
    adjustment_factors: List[str] = Field(default_factory=list, description="Factors that influenced adjustment")
    
    # Confidence breakdown
    base_confidence: float = Field(..., description="Base confidence score")
    granular_confidence: float = Field(..., description="Confidence after granular analysis")
    data_quality_score: float = Field(..., description="Quality of granular data")
    
    # Historical context
    similar_matchups: int = Field(0, description="Number of similar historical matchups")
    matchup_accuracy: float = Field(0.0, description="Accuracy of similar matchups")
    
    class Config:
        use_enum_values = True


class BettingRecommendation(BaseModel):
    """Complete betting recommendation."""
    
    prediction: Prediction = Field(..., description="The prediction")
    confidence: ConfidenceScore = Field(..., description="Confidence breakdown")
    
    # Best app recommendation
    recommended_app: BettingApp = Field(..., description="Best app for this bet")
    best_odds: Decimal = Field(..., description="Best available odds")
    value_rating: str = Field(..., description="Value rating (EXCELLENT, GOOD, FAIR, POOR)")
    
    # Alternative options
    alternative_apps: List[Dict[str, Any]] = Field(default_factory=list, description="Other app options")
    
    # Action items
    action_required: str = Field(..., description="What user should do")
    urgency: str = Field(..., description="Urgency level (LOW, MEDIUM, HIGH)")
    
    # Granular analysis (optional)
    granular_prediction: Optional[GranularPrediction] = Field(None, description="Granular analysis if available")
    
    class Config:
        use_enum_values = True


class SportsDataRequest(BaseModel):
    """Request for sports data collection."""
    
    sport: Sport = Field(..., description="Sport to collect data for")
    league: League = Field(..., description="League to collect data for")
    date_range: Optional[tuple[datetime, datetime]] = Field(None, description="Date range for data")
    teams: Optional[List[str]] = Field(None, description="Specific teams to focus on")
    players: Optional[List[str]] = Field(None, description="Specific players to focus on")
    
    # Data types
    include_player_stats: bool = Field(True, description="Include player statistics")
    include_game_data: bool = Field(True, description="Include game data")
    include_injury_reports: bool = Field(True, description="Include injury reports")
    include_weather: bool = Field(True, description="Include weather data")
    
    class Config:
        use_enum_values = True
