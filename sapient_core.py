"""
Sapient Core - Strategic Intelligence Layer for Autonomous Trading System
Philosophy: Replaces generic orchestrator with strategic intelligence that determines
WHAT to research, WHY to trade, and HOW to evolve based on market conditions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Firebase imports with error handling
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, exceptions
    FIREBASE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Firebase not available: {e}")
    FIREBASE_AVAILABLE = False


class MarketRegime(Enum):
    """Market regime classification for adaptive strategy selection"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CRASH = "crash"
    UNKNOWN = "unknown"


@dataclass
class StrategicDecision:
    """Structured decision output from Sapient Core"""
    timestamp: datetime
    market_regime: MarketRegime
    recommended_actions: List[str]
    confidence_score: float
    reasoning: str
    risk_level: str  # LOW, MEDIUM, HIGH


class SapientCore:
    """
    The brain of the autonomous trading system. Makes strategic decisions about
    what to research, when to trade, and how to evolve strategies.
    
    Key Responsibilities:
    1. Analyze market conditions and classify regimes
    2. Decide research priorities
    3. Evaluate strategy performance
    4. Evolve system parameters
    5. Maintain strategic memory in persistent storage
    """
    
    def __init__(self, firebase_config_path: Optional[str] = None):
        """
        Initialize Sapient Core with optional Firebase persistence.
        
        Args:
            firebase_config_path: Path to Firebase service account JSON (optional)
        """
        self.logger = self._setup_logging()
        self.logger.info("Initializing Sapient Core...")
        
        # Initialize state variables
        self.market_regime: MarketRegime = MarketRegime.UNKNOWN
        self.strategic_memory: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
        
        # Initialize Firebase if available and configured
        self.db = None
        if FIREBASE_AVAILABLE and firebase_config_path:
            self.db = self._init_firebase(firebase_config_path)
        else:
            self.logger.warning("Firebase not initialized. Using in-memory storage only.")
        
        # Initialize ML models for anomaly detection
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        # Initialize research priorities
        self.research_priorities = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": []
        }
        
        self.logger.info("Sapient Core initialization complete")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure structured logging for the Sapient Core"""
        logger = logging.getLogger('SapientCore')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_firebase(self, config_path: str):
        """
        Initialize Firebase Firestore for persistent strategic memory.
        
        Args:
            config_path: Path to Firebase service account JSON
            
        Returns:
            firestore.Client: Firestore database client
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If Firebase initialization fails
        """
        try:
            if not config_path or not isinstance(config_path, str):
                raise ValueError("Invalid Firebase config path")
            
            self.logger.info(f"Initializing Firebase with config: {config_path}")
            
            # Check if Firebase app already exists
            if not firebase_admin._apps:
                cred = credentials.Certificate(config_path)
                firebase_admin.initialize_app(cred)
            
            db = firestore.client()
            
            # Test connection
            test_ref = db.collection('system_health').document('connection_test')
            test_ref.set({
                'test_timestamp': datetime.utcnow(),
                'status': 'connected'
            })
            
            self.logger.info("Firebase Firestore initialized successfully")
            return db
            
        except FileNotFoundError as e:
            self.logger.error(f"Firebase config file not found: {e}")
            raise
        except exceptions.FirebaseError as e:
            self.logger.error(f"Firebase initialization error: {e}")
            raise ValueError(f"Firebase initialization failed: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error initializing Firebase: {e}")
            raise
    
    def analyze_market_regime(self, price_data: pd.DataFrame) -> MarketRegime:
        """
        Analyze price data to determine current market regime.
        
        Args:
            price_data: DataFrame with 'close' prices and datetime index
            
        Returns:
            MarketRegime: Classified market regime
        """
        self.logger.info("Analyzing market regime...")
        
        if price_data.empty or 'close' not in price_data.columns:
            self.logger.warning("Invalid price data for regime analysis")
            return MarketRegime.UNKNOWN
        
        try:
            closes = price_data['close'].values
            
            # Calculate key metrics
            returns = np.diff(np.log(closes))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Simple trend detection
            sma_short = pd.Series(closes).rolling(window=20).mean().iloc[-1]
            sma_long = pd.Series(closes).rolling(window=50).mean().iloc[-1]
            
            # Regime classification logic
            if volatility > 0.4:  # 40% annualized volatility threshold
                regime = MarketRegime.CRASH
            elif volatility > 0.25:
                regime = MarketRegime.VOLATILE
            elif sma_short > sma_long * 1.02:  # 2% above long SMA
                regime = MarketRegime.TRENDING_BULL
            elif sma_short < sma_long * 0.98:  # 2% below long SMA
                regime = MarketRegime.TRENDING_BEAR
            else:
                regime = MarketRegime.RANGING
            
            self.market_regime = regime
            self.logger.info(f"Detected market regime: {regime.value}")
            
            # Store in strategic memory
            self._update_strategic_memory('market_regime', {
                'regime': regime.value,
                'timestamp': datetime.utcnow(),
                'volatility': float(volatility),
                'trend_strength': float((sma_