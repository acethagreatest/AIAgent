#!/usr/bin/env python3
"""
Sports Betting Data Aggregator

A comprehensive sports betting data aggregator that:
- Collects data from multiple sports (NFL, NBA, WNBA, MLB, MLS, Esports)
- Analyzes player performance and prop bets
- Provides AI-powered predictions with confidence scores
- Verifies data using Flare Data Connector (FDC)
- Recommends best betting apps and odds
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.flare_ai_kit.sports import (
    SportsDataCollector, BettingDataCollector, WeatherDataCollector,
    PredictionEngine, ConfidenceCalculator, PropBetAnalyzer,
    SportsFDCClient, Sport, League, BettingApp
)
from src.flare_ai_kit.sports.models import Player
from src.flare_ai_kit.ecosystem.protocols.fdc import FDC
from src.flare_ai_kit.ecosystem.protocols.da_layer import DALayerClient
from src.flare_ai_kit.ecosystem.settings_models import EcosystemSettingsModel


class SportsBettingAggregator:
    """Main sports betting data aggregator."""
    
    def __init__(self):
        """Initialize the sports betting aggregator."""
        self.api_keys = {
            "espn": os.getenv("ESPN_API_KEY", ""),
            "sportsradar": os.getenv("SPORTSRADAR_API_KEY", ""),
            "odds_api": os.getenv("ODDS_API_KEY", ""),
            "openweather": os.getenv("OPENWEATHER_API_KEY", ""),
            "fanduel": os.getenv("FANDUEL_API_KEY", ""),
            "prize_picks": os.getenv("PRIZE_PICKS_API_KEY", ""),
            "underdog": os.getenv("UNDERDOG_API_KEY", "")
        }
        
        # Initialize components
        self.sports_collector = None
        self.betting_collector = None
        self.weather_collector = None
        self.prediction_engine = PredictionEngine()
        self.confidence_calculator = ConfidenceCalculator()
        self.prop_bet_analyzer = PropBetAnalyzer()
        
        # FDC components
        self.fdc_client = None
        self.da_client = None
        self.sports_fdc_client = None
    
    async def initialize(self):
        """Initialize all components."""
        print("🚀 Initializing Sports Betting Aggregator...")
        
        # Initialize data collectors
        self.sports_collector = SportsDataCollector(self.api_keys)
        self.betting_collector = BettingDataCollector(self.api_keys)
        self.weather_collector = WeatherDataCollector(self.api_keys.get("openweather", ""))
        
        # Initialize FDC
        ecosystem_settings = EcosystemSettingsModel(
            web3_provider_url=os.getenv("FLARE_RPC_URL"),
            account_address=os.getenv("WALLET_ADDRESS"),
            account_private_key=os.getenv("WALLET_PRIVATE_KEY"),
            is_testnet=True
        )
        
        self.fdc_client = await FDC.create(ecosystem_settings)
        self.da_client = DALayerClient("https://da-layer.flare.network")
        self.sports_fdc_client = SportsFDCClient(self.fdc_client, self.da_client)
        
        print("✅ Initialization complete!")
    
    async def collect_sports_data(
        self, 
        sports: List[Sport], 
        date: Optional[datetime] = None
    ) -> Dict[str, List]:
        """
        Collect sports data for specified sports and date.
        
        Args:
            sports: List of sports to collect data for
            date: Date to collect data for (defaults to today)
            
        Returns:
            Dictionary containing collected data
        """
        if not self.sports_collector:
            raise RuntimeError("Sports collector not initialized")
        
        print(f"📊 Collecting sports data for {[s.value for s in sports]}...")
        
        all_data = {
            "players": [],
            "games": [],
            "prop_bets": []
        }
        
        async with self.sports_collector, self.betting_collector, self.weather_collector:
            for sport in sports:
                # Get league for sport
                league = self._get_league_for_sport(sport)
                
                # Collect players
                players = await self.sports_collector.get_players(sport, league)
                
                # If no players from API, try web scraping
                if not players:
                    players = await self.sports_collector.get_players_via_scraping(sport, league)
                    
                all_data["players"].extend(players)
                
                # Collect games
                games = await self.sports_collector.get_games(sport, league, date)
                all_data["games"].extend(games)
                
                # Collect prop bets
                prop_bets = await self.betting_collector.get_prop_bets(sport, league, date)
                all_data["prop_bets"].extend(prop_bets)
                
                # Add weather data to games
                for game in games:
                    if game.venue and sport in [Sport.NFL, Sport.MLB, Sport.MLS]:
                        weather = await self.weather_collector.get_weather_for_game(game.venue, game.game_date)
                        if weather:
                            game.weather = weather
        
        print(f"✅ Collected {len(all_data['players'])} players, {len(all_data['games'])} games, {len(all_data['prop_bets'])} prop bets")
        return all_data
    
    async def analyze_and_predict(
        self, 
        data: Dict[str, List],
        historical_data: List[Dict[str, Any]] = None,
        use_granular_analysis: bool = True
    ) -> List:
        """
        Analyze data and generate predictions with granular analysis.
        
        Args:
            data: Collected sports data
            historical_data: Historical performance data for granular analysis
            use_granular_analysis: Whether to use team/venue specific analysis
            
        Returns:
            List of betting recommendations
        """
        print("🤖 Analyzing data and generating predictions...")
        
        if use_granular_analysis:
            print("🔍 Using granular team/venue analysis...")
        
        # Generate predictions using the enhanced analyzer
        recommendations = await self.prop_bet_analyzer.analyze_prop_bets(
            prop_bets=data["prop_bets"],
            players=data["players"],
            games=data["games"],
            historical_data=historical_data,
            use_granular_analysis=use_granular_analysis
        )
        
        print(f"✅ Generated {len(recommendations)} betting recommendations")
        return recommendations
    
    async def attest_data(self, data: Dict[str, List]) -> Dict[str, List[str]]:
        """
        Attest data using FDC for verification.
        
        Args:
            data: Collected sports data
            
        Returns:
            Dictionary of attestation IDs by data type
        """
        if not self.sports_fdc_client:
            print("⚠️ FDC not available, skipping data attestation")
            return {}
        
        print("🔐 Attesting data using Flare Data Connector...")
        
        attestation_ids = await self.sports_fdc_client.batch_attest_sports_data(
            players=data["players"],
            games=data["games"],
            prop_bets=data["prop_bets"]
        )
        
        print(f"✅ Attested {sum(len(ids) for ids in attestation_ids.values())} data points")
        return attestation_ids
    
    async def display_recommendations(
        self, 
        recommendations: List,
        top_n: int = 10
    ):
        """
        Display top betting recommendations with granular analysis.
        
        Args:
            recommendations: List of betting recommendations
            top_n: Number of top recommendations to display
        """
        print(f"\n🏆 TOP {min(top_n, len(recommendations))} BETTING RECOMMENDATIONS")
        print("=" * 80)
        
        for i, rec in enumerate(recommendations[:top_n], 1):
            pred = rec.prediction
            conf = rec.confidence
            
            print(f"\n{i}. {pred.prop_bet.player.name} ({pred.prop_bet.sport.value})")
            print(f"   Prop: {pred.prop_bet.prop_type.value} {pred.recommendation} {pred.prop_bet.line}")
            print(f"   Predicted Value: {pred.predicted_value:.2f}")
            print(f"   Confidence: {conf.overall_confidence:.1f}%")
            print(f"   Value Rating: {rec.value_rating}")
            print(f"   Recommended App: {rec.recommended_app}")
            print(f"   Best Odds: {rec.best_odds}")
            print(f"   Action: {rec.action_required}")
            print(f"   Urgency: {rec.urgency}")
            
            # Display granular analysis if available
            if rec.granular_prediction:
                granular = rec.granular_prediction
                print(f"   🔍 Granular Analysis:")
                print(f"      Data Quality: {granular.data_quality_score:.1f}%")
                print(f"      Similar Matchups: {granular.similar_matchups}")
                print(f"      Matchup Accuracy: {granular.matchup_accuracy:.1f}%")
                
                if granular.adjustment_factors:
                    print(f"      Key Factors:")
                    for factor in granular.adjustment_factors[:3]:  # Show top 3 factors
                        print(f"        • {factor}")
                
                if abs(granular.home_away_adjustment) > 0.5:
                    print(f"      Home/Away Adjustment: {granular.home_away_adjustment:+.2f}")
                if abs(granular.team_matchup_adjustment) > 0.5:
                    print(f"      Team Matchup Adjustment: {granular.team_matchup_adjustment:+.2f}")
                if abs(granular.venue_adjustment) > 0.5:
                    print(f"      Venue Adjustment: {granular.venue_adjustment:+.2f}")
            
            if conf.explanation:
                print(f"   Analysis: {conf.explanation}")
    
    def _get_league_for_sport(self, sport: Sport) -> League:
        """Get the league for a sport."""
        mapping = {
            Sport.NFL: League.NFL,
            Sport.NBA: League.NBA,
            Sport.WNBA: League.WNBA,
            Sport.MLB: League.MLB,
            Sport.MLS: League.MLS,
            Sport.ESPORTS: League.LCS
        }
        return mapping.get(sport, League.NBA)
    
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
    
    async def run_daily_analysis(self, sports: List[Sport] = None):
        """
        Run daily analysis for all sports.
        
        Args:
            sports: List of sports to analyze (defaults to all)
        """
        if sports is None:
            sports = [Sport.NFL, Sport.NBA, Sport.WNBA, Sport.MLB, Sport.MLS, Sport.ESPORTS]
        
        print("🌅 Starting daily sports betting analysis...")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"🏈 Sports: {[s.value for s in sports]}")
        
        try:
            # Initialize
            await self.initialize()
            
            # Collect data
            data = await self.collect_sports_data(sports)
            
            # Attest data (optional)
            attestation_ids = await self.attest_data(data)
            
            # Analyze and predict
            recommendations = await self.analyze_and_predict(data)
            
            # Display results
            await self.display_recommendations(recommendations)
            
            print(f"\n✅ Daily analysis complete!")
            print(f"📊 Total recommendations: {len(recommendations)}")
            if attestation_ids:
                print(f"🔐 Data attestations: {sum(len(ids) for ids in attestation_ids.values())}")
            
        except Exception as e:
            print(f"❌ Error during daily analysis: {e}")
            raise


async def main():
    """Main function to run the sports betting aggregator."""
    aggregator = SportsBettingAggregator()
    
    # Run analysis for today
    await aggregator.run_daily_analysis()


if __name__ == "__main__":
    print("🎯 Sports Betting Data Aggregator")
    print("Powered by Flare AI Kit & Flare Data Connector")
    print("=" * 60)
    
    asyncio.run(main())
