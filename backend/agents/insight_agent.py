"""
Insight Agent - Uses LLM to interpret forecast results and generate insights
"""
import pandas as pd
from typing import Dict, List
import logging
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightAgent:
    """Agent responsible for generating AI-powered insights from forecasts"""
    
    def __init__(self):
        self.llm = None
        if config.GOOGLE_API_KEY and config.GOOGLE_API_KEY != "your_google_api_key_here":
            self.llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                temperature=0.7,
                google_api_key=config.GOOGLE_API_KEY
            )
        
    def analyze_forecast(self, predictions: pd.DataFrame, metrics: Dict) -> str:
        """
        Analyze forecast results and generate insights
        
        Args:
            predictions: DataFrame with predictions
            metrics: Dictionary with model metrics
            
        Returns:
            Text analysis of the forecast
        """
        logger.info("Analyzing forecast results...")
        
        # Calculate statistics
        stats = {
            "avg_actual": predictions['energy_output_kWh'].mean(),
            "avg_predicted": predictions['predicted_output_kWh'].mean(),
            "max_actual": predictions['energy_output_kWh'].max(),
            "max_predicted": predictions['predicted_output_kWh'].max(),
            "min_actual": predictions['energy_output_kWh'].min(),
            "min_predicted": predictions['predicted_output_kWh'].min(),
            "mae": metrics.get('mae', 0),
            "rmse": metrics.get('rmse', 0)
        }
        
        # Generate insights using LLM if available
        if self.llm:
            prompt = ChatPromptTemplate.from_template(
                """You are an expert renewable energy analyst. Analyze the following forecast data and provide insights.
                
                Forecast Statistics:
                - Average Actual Output: {avg_actual:.2f} kWh
                - Average Predicted Output: {avg_predicted:.2f} kWh
                - Maximum Output: {max_actual:.2f} kWh (actual), {max_predicted:.2f} kWh (predicted)
                - Minimum Output: {min_actual:.2f} kWh (actual), {min_predicted:.2f} kWh (predicted)
                - Model Accuracy: MAE = {mae:.2f}, RMSE = {rmse:.2f}
                
                Provide a concise analysis covering:
                1. Overall forecast accuracy
                2. Key patterns or trends
                3. Potential concerns or opportunities
                4. Recommendations for optimization
                
                Keep the response professional and actionable."""
            )
            
            try:
                response = self.llm.invoke(
                    prompt.format(**stats)
                )
                return response.content
            except Exception as e:
                logger.error(f"Error generating LLM insights: {e}")
                return self._generate_rule_based_insights(stats)
        else:
            return self._generate_rule_based_insights(stats)
    
    def _generate_rule_based_insights(self, stats: Dict) -> str:
        """Generate insights using rule-based approach"""
        insights = []
        
        # Accuracy assessment
        accuracy_pct = (1 - stats['mae'] / stats['avg_actual']) * 100
        if accuracy_pct > 90:
            insights.append(f"✓ Excellent forecast accuracy ({accuracy_pct:.1f}%)")
        elif accuracy_pct > 80:
            insights.append(f"✓ Good forecast accuracy ({accuracy_pct:.1f}%)")
        else:
            insights.append(f"⚠ Moderate forecast accuracy ({accuracy_pct:.1f}%) - model may need retraining")
        
        # Output comparison
        diff_pct = ((stats['avg_predicted'] - stats['avg_actual']) / stats['avg_actual']) * 100
        if abs(diff_pct) < 5:
            insights.append(f"✓ Predicted output closely matches actual ({diff_pct:+.1f}%)")
        elif diff_pct > 0:
            insights.append(f"⚠ Model overestimates by {diff_pct:.1f}%")
        else:
            insights.append(f"⚠ Model underestimates by {abs(diff_pct):.1f}%")
        
        # Recommendations
        insights.append("\nRecommendations:")
        insights.append("• Monitor forecast accuracy daily and retrain if MAE exceeds threshold")
        insights.append("• Consider energy storage during high-output periods")
        insights.append("• Plan maintenance during predicted low-output periods")
        
        return "\n".join(insights)
    
    def run(self, forecast_result: Dict) -> Dict:
        """
        Execute the insight agent
        
        Args:
            forecast_result: Results from ForecastAgent
            
        Returns:
            Dictionary with insights
        """
        logger.info("InsightAgent: Generating insights...")
        
        predictions = forecast_result['predictions']
        metrics = forecast_result['metrics']
        
        # Generate analysis
        analysis = self.analyze_forecast(predictions, metrics)
        
        return {
            "status": "success",
            "analysis": analysis,
            "metrics": metrics,
            "summary": {
                "total_predictions": len(predictions),
                "avg_output": float(predictions['energy_output_kWh'].mean()),
                "avg_predicted": float(predictions['predicted_output_kWh'].mean())
            }
        }
