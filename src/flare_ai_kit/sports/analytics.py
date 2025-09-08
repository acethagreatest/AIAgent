"""Analytics engine for sports betting predictions."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import structlog

from .models import (
    Player, Game, PropBet, Prediction, ConfidenceScore, 
    BettingRecommendation, GranularPrediction, Sport, League, PropType
)
from .granular_analytics import GranularAnalyticsEngine

logger = structlog.get_logger(__name__)


class PredictionEngine:
    """ML-powered prediction engine for prop bets."""
    
    def __init__(self):
        """Initialize the prediction engine."""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    async def predict_prop_bet(
        self, 
        prop_bet: PropBet, 
        player: Player, 
        game: Game,
        historical_data: List[Dict[str, Any]]
    ) -> Prediction:
        """
        Predict the outcome of a prop bet.
        
        Args:
            prop_bet: The prop bet to predict
            player: Player information
            game: Game context
            historical_data: Historical performance data
            
        Returns:
            Prediction object with recommendation and confidence
        """
        try:
            # Extract features
            features = self._extract_features(prop_bet, player, game, historical_data)
            
            # Get model for this prop type
            model = self._get_model(prop_bet.prop_type)
            scaler = self._get_scaler(prop_bet.prop_type)
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Make prediction
            predicted_value = model.predict(features_scaled)[0]
            
            # Determine recommendation
            recommendation = "OVER" if predicted_value > float(prop_bet.line) else "UNDER"
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                prop_bet, player, game, predicted_value, features
            )
            
            # Generate analysis
            analysis = self._generate_analysis(
                prop_bet, player, game, predicted_value, features
            )
            
            prediction = Prediction(
                prop_bet=prop_bet,
                predicted_value=predicted_value,
                recommendation=recommendation,
                confidence_score=confidence,
                factors=analysis["factors"],
                player_form=analysis["player_form"],
                matchup_analysis=analysis["matchup_analysis"],
                weather_impact=analysis.get("weather_impact"),
                historical_accuracy=analysis["historical_accuracy"],
                sample_size=analysis["sample_size"],
                risk_level=analysis["risk_level"],
                edge_percentage=analysis["edge_percentage"]
            )
            
            logger.info(f"Generated prediction for {player.name}: {recommendation} {prop_bet.line}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            # Return a default prediction
            return self._create_default_prediction(prop_bet, player, game)
    
    def _extract_features(
        self, 
        prop_bet: PropBet, 
        player: Player, 
        game: Game,
        historical_data: List[Dict[str, Any]]
    ) -> List[float]:
        """Extract features for ML model."""
        features = []
        
        # Player performance features
        features.extend([
            player.season_averages.get("points", 0.0),
            player.season_averages.get("rebounds", 0.0),
            player.season_averages.get("assists", 0.0),
            player.season_averages.get("minutes", 0.0),
        ])
        
        # Recent form (last 5 games)
        recent_games = historical_data[-5:] if len(historical_data) >= 5 else historical_data
        if recent_games:
            recent_avg = np.mean([game.get(prop_bet.prop_type.value.lower(), 0) for game in recent_games])
            features.append(recent_avg)
        else:
            features.append(0.0)
        
        # Game context features
        features.extend([
            1.0 if game.is_playoff else 0.0,
            1.0 if game.importance == "championship" else 0.0,
        ])
        
        # Weather features (if available)
        if game.weather:
            features.extend([
                game.weather.get("temperature", 72.0),
                game.weather.get("humidity", 50.0),
                game.weather.get("wind_speed", 0.0),
            ])
        else:
            features.extend([72.0, 50.0, 0.0])
        
        # Player status features
        features.extend([
            1.0 if player.is_active else 0.0,
            1.0 if player.injury_status == "healthy" else 0.0,
        ])
        
        # Time-based features
        game_hour = game.game_date.hour
        features.extend([
            np.sin(2 * np.pi * game_hour / 24),  # Time of day
            np.cos(2 * np.pi * game_hour / 24),
        ])
        
        # Ensure we have exactly 15 features
        while len(features) < 15:
            features.append(0.0)
        features = features[:15]  # Truncate if too many
        
        return features
    
    def _get_model(self, prop_type: PropType) -> Any:
        """Get or create model for prop type."""
        if prop_type not in self.models:
            # Create a new model for this prop type
            self.models[prop_type] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Train with dummy data (in production, this would use historical data)
            X_dummy = np.random.rand(100, 15)  # 15 features
            y_dummy = np.random.rand(100) * 50  # Random targets
            self.models[prop_type].fit(X_dummy, y_dummy)
            
        return self.models[prop_type]
    
    def _get_scaler(self, prop_type: PropType) -> Any:
        """Get or create scaler for prop type."""
        if prop_type not in self.scalers:
            self.scalers[prop_type] = StandardScaler()
            
            # Fit with dummy data (in production, this would use historical data)
            X_dummy = np.random.rand(100, 15)
            self.scalers[prop_type].fit(X_dummy)
            
        return self.scalers[prop_type]
    
    def _calculate_confidence(
        self, 
        prop_bet: PropBet, 
        player: Player, 
        game: Game,
        predicted_value: float,
        features: List[float]
    ) -> float:
        """Calculate confidence score for prediction."""
        confidence = 50.0  # Base confidence
        
        # Adjust based on data quality
        if len(player.season_averages) > 5:
            confidence += 10.0
        
        # Adjust based on recent form consistency
        recent_stats = list(player.recent_stats.values())
        if recent_stats and np.std(recent_stats) < 5.0:
            confidence += 15.0
        
        # Adjust based on sample size
        if len(player.recent_stats) >= 10:
            confidence += 10.0
        
        # Adjust based on injury status
        if player.injury_status == "healthy":
            confidence += 5.0
        elif player.injury_status:
            confidence -= 20.0
        
        # Adjust based on game importance
        if game.is_playoff:
            confidence += 5.0
        
        # Ensure confidence is between 0 and 100
        return max(0.0, min(100.0, confidence))
    
    def _generate_analysis(
        self, 
        prop_bet: PropBet, 
        player: Player, 
        game: Game,
        predicted_value: float,
        features: List[float]
    ) -> Dict[str, Any]:
        """Generate detailed analysis for prediction."""
        factors = []
        
        # Player form analysis
        if player.recent_stats:
            recent_avg = np.mean(list(player.recent_stats.values()))
            if recent_avg > player.season_averages.get(prop_bet.prop_type.value.lower(), 0):
                factors.append("Player in good form")
            else:
                factors.append("Player below season average")
        
        # Matchup analysis
        if game.is_playoff:
            factors.append("Playoff game - higher intensity")
        
        # Weather impact
        if game.weather and prop_bet.sport in [Sport.NFL, Sport.MLB, Sport.MLS]:
            temp = game.weather.get("temperature", 72)
            if temp < 40 or temp > 90:
                factors.append("Extreme weather conditions")
        
        # Historical accuracy (mock data)
        historical_accuracy = 65.0 + np.random.normal(0, 5)
        sample_size = len(player.recent_stats) + np.random.randint(10, 50)
        
        # Risk assessment
        risk_level = "MEDIUM"
        if player.injury_status and player.injury_status != "healthy":
            risk_level = "HIGH"
        elif len(player.recent_stats) < 5:
            risk_level = "HIGH"
        elif historical_accuracy > 75:
            risk_level = "LOW"
        
        # Edge calculation - more sophisticated for over/under
        line_value = float(prop_bet.line)
        if predicted_value > line_value:
            # OVER bet - positive edge if prediction is higher than line
            edge_percentage = (predicted_value - line_value) / line_value * 100
        else:
            # UNDER bet - positive edge if prediction is lower than line
            edge_percentage = (line_value - predicted_value) / line_value * 100
        
        return {
            "factors": factors,
            "player_form": f"Recent form: {player.recent_stats.get(prop_bet.prop_type.value.lower(), 'N/A')}",
            "matchup_analysis": f"Playing {game.away_team if player.team == game.home_team else game.home_team}",
            "weather_impact": f"Weather: {game.weather.get('conditions', 'Unknown')}" if game.weather else None,
            "historical_accuracy": max(0, min(100, historical_accuracy)),
            "sample_size": max(1, sample_size),
            "risk_level": risk_level,
            "edge_percentage": max(0, edge_percentage)
        }
    
    def _create_default_prediction(
        self, 
        prop_bet: PropBet, 
        player: Player, 
        game: Game
    ) -> Prediction:
        """Create a default prediction when ML fails."""
        return Prediction(
            prop_bet=prop_bet,
            predicted_value=float(prop_bet.line),
            recommendation="OVER",
            confidence_score=30.0,
            factors=["Insufficient data for accurate prediction"],
            player_form="Unknown",
            matchup_analysis="Unknown",
            historical_accuracy=30.0,
            sample_size=1,
            risk_level="HIGH",
            edge_percentage=0.0
        )


class ConfidenceCalculator:
    """Calculates detailed confidence scores for predictions."""
    
    def calculate_confidence(self, prediction: Prediction) -> ConfidenceScore:
        """
        Calculate detailed confidence breakdown.
        
        Args:
            prediction: The prediction to analyze
            
        Returns:
            ConfidenceScore with detailed breakdown
        """
        # Base confidence from prediction
        overall_confidence = prediction.confidence_score
        
        # Calculate component scores
        player_form_score = self._calculate_player_form_score(prediction)
        matchup_score = self._calculate_matchup_score(prediction)
        historical_score = self._calculate_historical_score(prediction)
        data_quality_score = self._calculate_data_quality_score(prediction)
        model_confidence = self._calculate_model_confidence(prediction)
        
        # Calculate risk factors
        injury_risk = self._calculate_injury_risk(prediction)
        weather_risk = self._calculate_weather_risk(prediction)
        sample_size_risk = self._calculate_sample_size_risk(prediction)
        
        # Generate explanation
        explanation = self._generate_confidence_explanation(
            overall_confidence, player_form_score, matchup_score, 
            historical_score, data_quality_score, model_confidence
        )
        
        return ConfidenceScore(
            overall_confidence=overall_confidence,
            player_form_score=player_form_score,
            matchup_score=matchup_score,
            historical_score=historical_score,
            data_quality_score=data_quality_score,
            model_confidence=model_confidence,
            injury_risk=injury_risk,
            weather_risk=weather_risk,
            sample_size_risk=sample_size_risk,
            explanation=explanation
        )
    
    def _calculate_player_form_score(self, prediction: Prediction) -> float:
        """Calculate player form confidence score."""
        # This would analyze recent performance trends
        return min(100.0, prediction.confidence_score + 10.0)
    
    def _calculate_matchup_score(self, prediction: Prediction) -> float:
        """Calculate matchup confidence score."""
        # This would analyze head-to-head and team matchups
        return min(100.0, prediction.confidence_score + 5.0)
    
    def _calculate_historical_score(self, prediction: Prediction) -> float:
        """Calculate historical data confidence score."""
        return prediction.historical_accuracy
    
    def _calculate_data_quality_score(self, prediction: Prediction) -> float:
        """Calculate data quality confidence score."""
        # Based on sample size and data completeness
        if prediction.sample_size > 20:
            return 90.0
        elif prediction.sample_size > 10:
            return 70.0
        else:
            return 50.0
    
    def _calculate_model_confidence(self, prediction: Prediction) -> float:
        """Calculate ML model confidence score."""
        return min(100.0, prediction.confidence_score + 15.0)
    
    def _calculate_injury_risk(self, prediction: Prediction) -> float:
        """Calculate injury risk factor."""
        player = prediction.prop_bet.player
        if player.injury_status == "healthy":
            return 10.0
        elif player.injury_status:
            return 80.0
        else:
            return 30.0
    
    def _calculate_weather_risk(self, prediction: Prediction) -> float:
        """Calculate weather risk factor."""
        game = prediction.prop_bet.game
        if game.weather:
            temp = game.weather.get("temperature", 72)
            if temp < 32 or temp > 100:
                return 70.0
            elif temp < 40 or temp > 90:
                return 40.0
        return 20.0
    
    def _calculate_sample_size_risk(self, prediction: Prediction) -> float:
        """Calculate sample size risk factor."""
        if prediction.sample_size < 5:
            return 80.0
        elif prediction.sample_size < 10:
            return 50.0
        else:
            return 20.0
    
    def _generate_confidence_explanation(
        self, 
        overall: float, 
        player_form: float, 
        matchup: float,
        historical: float, 
        data_quality: float, 
        model: float
    ) -> str:
        """Generate human-readable confidence explanation."""
        if overall >= 80:
            return f"High confidence prediction ({overall:.1f}%) based on strong player form ({player_form:.1f}%) and reliable historical data ({historical:.1f}%)"
        elif overall >= 60:
            return f"Moderate confidence prediction ({overall:.1f}%) with some uncertainty in data quality ({data_quality:.1f}%)"
        else:
            return f"Low confidence prediction ({overall:.1f}%) due to limited data or high risk factors"


class PropBetAnalyzer:
    """Analyzes and compares prop bets across different apps."""
    
    def __init__(self):
        """Initialize the prop bet analyzer."""
        self.prediction_engine = PredictionEngine()
        self.confidence_calculator = ConfidenceCalculator()
        self.granular_engine = GranularAnalyticsEngine()
    
    async def analyze_prop_bets(
        self, 
        prop_bets: List[PropBet],
        players: List[Player],
        games: List[Game],
        historical_data: List[Dict[str, Any]] = None,
        use_granular_analysis: bool = True
    ) -> List[BettingRecommendation]:
        """
        Analyze a list of prop bets and generate recommendations.
        
        Args:
            prop_bets: List of prop bets to analyze
            players: List of players
            games: List of games
            historical_data: Historical performance data for granular analysis
            use_granular_analysis: Whether to use granular team/venue analysis
            
        Returns:
            List of betting recommendations
        """
        recommendations = []
        
        # Create lookup dictionaries
        players_dict = {p.id: p for p in players}
        games_dict = {g.id: g for g in games}
        
        # Use empty historical data if not provided
        if historical_data is None:
            historical_data = []
        
        for prop_bet in prop_bets:
            try:
                # Get player and game data
                player = players_dict.get(prop_bet.player.id)
                game = games_dict.get(prop_bet.game.id)
                
                if not player or not game:
                    continue
                
                # Generate base prediction
                prediction = await self.prediction_engine.predict_prop_bet(
                    prop_bet, player, game, historical_data
                )
                
                # Create granular prediction if enabled and data available
                granular_prediction = None
                if use_granular_analysis and historical_data:
                    try:
                        granular_prediction = await self.granular_engine.create_granular_prediction(
                            prediction, player, game, historical_data
                        )
                        
                        # Update prediction with granular adjustments
                        if granular_prediction:
                            prediction.predicted_value = granular_prediction.adjusted_predicted_value
                            prediction.confidence_score = granular_prediction.granular_confidence
                            
                            # Add granular factors to prediction factors
                            prediction.factors.extend(granular_prediction.adjustment_factors)
                            
                    except Exception as e:
                        logger.warning(f"Granular analysis failed for {player.name}: {e}")
                
                # Calculate confidence
                confidence = self.confidence_calculator.calculate_confidence(prediction)
                
                # Find best app for this bet
                best_app, best_odds = self._find_best_app(prop_bets, prop_bet, prediction)
                
                # Create recommendation
                recommendation = BettingRecommendation(
                    prediction=prediction,
                    confidence=confidence,
                    recommended_app=best_app,
                    best_odds=best_odds,
                    value_rating=self._calculate_value_rating(prediction, best_odds),
                    alternative_apps=self._get_alternative_apps(prop_bets, prop_bet, prediction),
                    action_required=self._generate_action_required(prediction, confidence),
                    urgency=self._calculate_urgency(prediction, confidence),
                    granular_prediction=granular_prediction
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                logger.error(f"Error analyzing prop bet {prop_bet.id}: {e}")
                continue
        
        # Sort by confidence and value
        recommendations.sort(key=lambda x: (x.confidence.overall_confidence, x.value_rating), reverse=True)
        
        logger.info(f"Generated {len(recommendations)} betting recommendations")
        return recommendations
    
    def _find_best_app(self, all_prop_bets: List[PropBet], target_prop: PropBet, prediction: Prediction = None) -> Tuple[str, float]:
        """Find the best app and odds for a prop bet."""
        # Find similar prop bets across apps
        similar_bets = [
            bet for bet in all_prop_bets
            if (bet.player.id == target_prop.player.id and 
                bet.prop_type == target_prop.prop_type and
                bet.game.id == target_prop.game.id)
        ]
        
        if not similar_bets:
            # Return appropriate odds based on recommendation
            if prediction and prediction.recommendation == "UNDER":
                return str(target_prop.betting_app), float(target_prop.under_odds)
            else:
                return str(target_prop.betting_app), float(target_prop.over_odds)
        
        # Find best odds based on recommendation direction
        if prediction and prediction.recommendation == "UNDER":
            # For UNDER bets, we want the best under odds (highest positive value)
            best_bet = max(similar_bets, key=lambda x: float(x.under_odds))
            return str(best_bet.betting_app), float(best_bet.under_odds)
        else:
            # For OVER bets, we want the best over odds (highest positive value)
            best_bet = max(similar_bets, key=lambda x: float(x.over_odds))
            return str(best_bet.betting_app), float(best_bet.over_odds)
    
    def _calculate_value_rating(self, prediction: Prediction, odds: float) -> str:
        """Calculate value rating for a bet."""
        edge = prediction.edge_percentage
        if edge > 10:
            return "EXCELLENT"
        elif edge > 5:
            return "GOOD"
        elif edge > 2:
            return "FAIR"
        else:
            return "POOR"
    
    def _get_alternative_apps(self, all_prop_bets: List[PropBet], target_prop: PropBet, prediction: Prediction = None) -> List[Dict[str, Any]]:
        """Get alternative app options for a prop bet."""
        alternatives = []
        
        similar_bets = [
            bet for bet in all_prop_bets
            if (bet.player.id == target_prop.player.id and 
                bet.prop_type == target_prop.prop_type and
                bet.game.id == target_prop.game.id and
                bet.betting_app != target_prop.betting_app)
        ]
        
        for bet in similar_bets:
            # Choose appropriate odds based on recommendation
            if prediction and prediction.recommendation == "UNDER":
                odds = float(bet.under_odds)
            else:
                odds = float(bet.over_odds)
                
            alternatives.append({
                "app": str(bet.betting_app),
                "odds": odds,
                "url": bet.url,
                "recommendation": prediction.recommendation if prediction else "OVER"
            })
        
        return alternatives
    
    def _generate_action_required(self, prediction: Prediction, confidence: ConfidenceScore) -> str:
        """Generate action required message."""
        if confidence.overall_confidence >= 80:
            return f"Strong bet - {prediction.recommendation} {prediction.prop_bet.line}"
        elif confidence.overall_confidence >= 60:
            return f"Moderate bet - {prediction.recommendation} {prediction.prop_bet.line}"
        else:
            return f"Consider carefully - {prediction.recommendation} {prediction.prop_bet.line}"
    
    def _calculate_urgency(self, prediction: Prediction, confidence: ConfidenceScore) -> str:
        """Calculate urgency level."""
        if confidence.overall_confidence >= 80 and prediction.edge_percentage > 5:
            return "HIGH"
        elif confidence.overall_confidence >= 60:
            return "MEDIUM"
        else:
            return "LOW"
