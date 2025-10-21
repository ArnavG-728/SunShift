"""
Chat Agent - Conversational interface for user queries about forecasts
"""
import pandas as pd
from typing import Dict, List, Optional
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatAgent:
    """Agent responsible for handling user conversations about energy forecasts"""
    
    def __init__(self):
        self.llm = None
        self.memory = ConversationBufferMemory()
        self.context = {}
        
        if config.GOOGLE_API_KEY and config.GOOGLE_API_KEY != "your_google_api_key_here":
            self.llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                temperature=0.7,
                google_api_key=config.GOOGLE_API_KEY
            )
    
    def set_context(self, forecast_data: Dict, insights: Dict):
        """Set context from forecast and insights"""
        self.context = {
            "forecast_data": forecast_data,
            "insights": insights
        }
    
    def answer_query(self, query: str) -> str:
        """
        Answer user query about forecasts
        
        Args:
            query: User question
            
        Returns:
            AI-generated response
        """
        logger.info(f"Processing query: {query}")
        
        if not self.llm:
            return self._rule_based_response(query)
        
        # Build context for LLM
        context_str = self._build_context_string()
        
        prompt = ChatPromptTemplate.from_template(
            """You are GreenCast AI, an expert assistant for renewable energy forecasting and optimization.
            
            You have access to the following forecast data and context:
            {context}
            
            User Question: {query}
            
            Instructions:
            1. Provide accurate, data-driven responses based on the forecast context
            2. Reference specific metrics, values, and trends when available
            3. Explain technical concepts in clear, accessible language
            4. Offer actionable recommendations for energy optimization
            5. If data is missing, acknowledge it and suggest how to obtain it
            6. Be concise but comprehensive - aim for 2-4 sentences
            7. Use specific numbers and percentages when discussing accuracy or predictions
            
            Response:"""
        )
        
        try:
            response = self.llm.invoke(
                prompt.format(context=context_str, query=query)
            )
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._rule_based_response(query)
    
    def _build_context_string(self) -> str:
        """Build comprehensive context string from available data"""
        parts = []
        parts.append("=== FORECAST CONTEXT ===\n")
        
        # Try to load latest predictions and insights
        try:
            import pandas as pd
            from pathlib import Path
            import json
            
            # Load 24h predictions
            pred_24h = config.DATA_DIR / 'predictions_24h.csv'
            pred_7d = config.DATA_DIR / 'predictions_7d.csv'
            
            if pred_24h.exists():
                df = pd.read_csv(pred_24h)
                parts.append(f"\n📊 24-Hour Forecast ({len(df)} hours):")
                parts.append(f"  - Predicted Range: {df['predicted_output_kWh'].min():.2f} - {df['predicted_output_kWh'].max():.2f} kWh")
                parts.append(f"  - Average: {df['predicted_output_kWh'].mean():.2f} kWh/hour")
                parts.append(f"  - Total: {df['predicted_output_kWh'].sum():.1f} kWh")
                parts.append(f"  - Peak Hour: {df.loc[df['predicted_output_kWh'].idxmax(), 'timestamp']}")
                
                # Weather context
                if 'temperature' in df.columns:
                    parts.append(f"\n🌡️ Weather Conditions:")
                    parts.append(f"  - Avg Temperature: {df['temperature'].mean():.1f}°C")
                    parts.append(f"  - Avg Wind Speed: {df['wind_speed'].mean():.1f} m/s")
                    if 'solar_irradiance' in df.columns:
                        parts.append(f"  - Avg Solar: {df['solar_irradiance'].mean():.0f} W/m²")
            
            if pred_7d.exists():
                df7 = pd.read_csv(pred_7d)
                parts.append(f"\n📅 7-Day Forecast:")
                parts.append(f"  - Daily Average: {df7['total_kwh'].mean():.1f} kWh/day")
                parts.append(f"  - Weekly Total: {df7['total_kwh'].sum():.1f} kWh")
                parts.append(f"  - Best Day: {df7.loc[df7['total_kwh'].idxmax(), 'date']} ({df7['total_kwh'].max():.1f} kWh)")
                
        except Exception as e:
            logger.warning(f"Could not load predictions: {e}")
        
        if not self.context:
            parts.append("\nNote: Run forecast for detailed analysis.")
            return "\n".join(parts)
        
        # Add insights and analysis
        if 'insights' in self.context:
            insights = self.context['insights']
            if 'analysis' in insights:
                parts.append(f"AI Analysis:\n{insights.get('analysis', 'N/A')}\n")
            
            if 'metrics' in insights:
                metrics = insights['metrics']
                parts.append("Model Performance Metrics:")
                parts.append(f"  - Mean Absolute Error (MAE): {metrics.get('mae', 0):.2f} kWh")
                parts.append(f"  - Root Mean Square Error (RMSE): {metrics.get('rmse', 0):.2f} kWh")
                
                # Calculate accuracy percentage based on actual data
                if metrics.get('mae', 0) > 0:
                    # Calculate accuracy based on average predicted output
                    try:
                        avg_output = df['predicted_output_kWh'].mean() if 'predicted_output_kWh' in df.columns else 10.0
                        if avg_output > 0:
                            accuracy = max(0, (1 - metrics.get('mae', 0) / avg_output) * 100)
                        else:
                            accuracy = 0
                    except:
                        accuracy = max(0, (1 - metrics.get('mae', 0) / 10) * 100)
                    parts.append(f"  - Estimated Accuracy: {accuracy:.1f}%\n")
        
        # Add forecast data summary
        if 'forecast_data' in self.context:
            forecast = self.context['forecast_data']
            parts.append(f"Model Type: {forecast.get('model_type', 'LSTM')}")
            
            if 'predictions' in forecast:
                preds = forecast['predictions']
                if isinstance(preds, list) and len(preds) > 0:
                    parts.append(f"Forecast Horizon: {len(preds)} hours")
                    
                    # Calculate summary statistics
                    try:
                        import pandas as pd
                        df = pd.DataFrame(preds)
                        if 'predicted_output_kWh' in df.columns:
                            parts.append(f"\nForecast Summary:")
                            parts.append(f"  - Average Predicted Output: {df['predicted_output_kWh'].mean():.2f} kWh")
                            parts.append(f"  - Peak Predicted Output: {df['predicted_output_kWh'].max():.2f} kWh")
                            parts.append(f"  - Minimum Predicted Output: {df['predicted_output_kWh'].min():.2f} kWh")
                    except:
                        pass
        
        parts.append("\n=== END CONTEXT ===")
        return "\n".join(parts)
    
    def _rule_based_response(self, query: str) -> str:
        """Generate enhanced rule-based response when LLM unavailable"""
        query_lower = query.lower()
        
        # Accuracy and error questions
        if 'accuracy' in query_lower or 'error' in query_lower or 'performance' in query_lower:
            if 'insights' in self.context and 'metrics' in self.context['insights']:
                metrics = self.context['insights']['metrics']
                mae = metrics.get('mae', 0)
                rmse = metrics.get('rmse', 0)
                # Calculate accuracy based on actual average output
                try:
                    pred_24h = config.DATA_DIR / 'predictions_24h.csv'
                    if pred_24h.exists():
                        df = pd.read_csv(pred_24h)
                        avg_output = df['predicted_output_kWh'].mean() if 'predicted_output_kWh' in df.columns else 10.0
                    else:
                        avg_output = 10.0
                    accuracy = max(0, (1 - mae / avg_output) * 100) if avg_output > 0 else 0
                except:
                    accuracy = max(0, (1 - mae / 10) * 100)
                return f"The forecast model achieves {accuracy:.1f}% accuracy with a Mean Absolute Error (MAE) of {mae:.2f} kWh and RMSE of {rmse:.2f} kWh. This indicates {'excellent' if accuracy > 90 else 'good' if accuracy > 80 else 'moderate'} prediction reliability."
            return "Forecast accuracy metrics are not available yet. Please run the forecast pipeline first to generate predictions and calculate performance metrics."
        
        # Tomorrow/future predictions
        elif 'tomorrow' in query_lower or 'next' in query_lower or 'future' in query_lower:
            if 'forecast_data' in self.context and 'predictions' in self.context['forecast_data']:
                try:
                    import pandas as pd
                    preds = self.context['forecast_data']['predictions']
                    df = pd.DataFrame(preds)
                    if 'predicted_output_kWh' in df.columns:
                        avg = df['predicted_output_kWh'].mean()
                        peak = df['predicted_output_kWh'].max()
                        return f"The forecast predicts an average output of {avg:.2f} kWh with a peak of {peak:.2f} kWh over the next 24 hours. Run the forecast pipeline to get the most up-to-date predictions."
                except:
                    pass
            return "To get tomorrow's forecast, please run the forecasting pipeline with the latest weather data."
        
        # Why/explanation questions
        elif 'why' in query_lower or 'reason' in query_lower or 'cause' in query_lower:
            return "Energy output variations are primarily driven by weather conditions: solar irradiance (affected by cloud cover and time of day), wind speed (cubic relationship with wind power), temperature, and humidity. The LSTM model analyzes these patterns to predict future generation."
        
        # Low/high generation questions
        elif 'low' in query_lower and 'generation' in query_lower:
            return "Low generation periods are typically caused by: (1) reduced solar irradiance from cloudy weather or nighttime, (2) low wind speeds, (3) seasonal variations. Consider scheduling maintenance or grid purchases during these periods."
        
        elif 'high' in query_lower and 'generation' in query_lower:
            return "High generation periods occur during: (1) peak solar hours (10am-2pm) with clear skies, (2) high wind speeds, (3) optimal temperature conditions. These are ideal times for energy storage or grid sales."
        
        # Model/how questions
        elif 'model' in query_lower or 'how' in query_lower and 'work' in query_lower:
            return "GreenCast uses a Bidirectional LSTM (Long Short-Term Memory) neural network with 3 layers, batch normalization, and dropout regularization. It analyzes 24-hour sequences of weather data and historical generation to predict future output with high accuracy."
        
        # Optimization questions
        elif 'optimize' in query_lower or 'improve' in query_lower or 'better' in query_lower:
            return "To optimize energy generation: (1) Schedule maintenance during predicted low-output periods, (2) Plan energy storage during high-output forecasts, (3) Adjust grid participation based on predictions, (4) Monitor forecast accuracy and retrain the model with new data monthly."
        
        # Data questions
        elif 'data' in query_lower or 'weather' in query_lower:
            return "The model uses weather data including temperature, humidity, wind speed, and solar irradiance. You can switch between synthetic data (for testing) and real OpenWeather API data by setting USE_SYNTHETIC_DATA in the .env file."
        
        # Default response with suggestions
        else:
            return "I can help you with: forecast accuracy, tomorrow's predictions, generation patterns, model explanations, optimization strategies, and weather data. Try asking: 'What's the forecast accuracy?', 'Why is generation low?', or 'How can I optimize energy production?'"
    
    def run(self, query: str, forecast_result: Optional[Dict] = None, 
            insight_result: Optional[Dict] = None) -> Dict:
        """
        Execute the chat agent
        
        Args:
            query: User query
            forecast_result: Optional forecast data
            insight_result: Optional insights
            
        Returns:
            Dictionary with response
        """
        logger.info("ChatAgent: Processing user query...")
        
        # Update context if provided
        if forecast_result:
            self.context['forecast_data'] = forecast_result
        if insight_result:
            self.context['insights'] = insight_result
        
        # Generate response
        response = self.answer_query(query)
        
        return {
            "status": "success",
            "query": query,
            "response": response
        }
