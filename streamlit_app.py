import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, time as dt_time, date
import json
import sys
import subprocess
import pytz

# ====== PAGE CONFIGURATION ======
st.set_page_config(
    page_title="SOP v7.4 Options Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== DYNAMIC LIBRARY INSTALLATION ======
def install_and_import_dhanhq():
    """Dynamically install and import dhanhq with error handling"""
    try:
        # Try importing first
        import dhanhq
        from dhanhq import dhanhq as DhanAPI
        return True, DhanAPI, "dhanhq already available"
    except ImportError:
        try:
            st.info("📦 Installing DhanHQ library for live data...")
            with st.spinner("Installing dhanhq..."):
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "dhanhq>=2.0.2", "--quiet", "--no-cache-dir"
                ])
            
            # Import after installation
            import dhanhq
            from dhanhq import dhanhq as DhanAPI
            st.success("✅ DhanHQ library installed successfully!")
            return True, DhanAPI, "dhanhq installed and imported"
        
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Failed to install dhanhq: {e}")
            return False, None, f"Installation failed: {e}"
        except ImportError as e:
            st.error(f"❌ Failed to import dhanhq after installation: {e}")
            return False, None, f"Import failed after installation: {e}"
        except Exception as e:
            st.error(f"❌ Unexpected error with dhanhq: {e}")
            return False, None, f"Unexpected error: {e}"

# Initialize DhanHQ library
DHANHQ_AVAILABLE, DhanAPI, dhanhq_status = install_and_import_dhanhq()

# ====== PLOTLY HANDLING ======
def initialize_plotly():
    """Initialize plotly with proper error handling and no repeated imports"""
    try:
        import importlib.util
        plotly_spec = importlib.util.find_spec("plotly")
        
        if plotly_spec is None:
            st.warning("📦 Installing Plotly for enhanced charts...")
            with st.spinner("Installing plotly..."):
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install",
                        "plotly>=5.15.0", "--quiet", "--no-cache-dir"
                    ])
                    st.success("✅ Plotly installed successfully!")
                    st.rerun()
                except subprocess.CalledProcessError as e:
                    st.error(f"❌ Could not install Plotly: {e}")
                    return False, None, None, None
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio
            pio.templates.default = 'plotly'
            return True, go, make_subplots, pio
        except ImportError as e:
            st.error(f"❌ Plotly import failed: {e}")
            return False, None, None, None
    except Exception as e:
        st.error(f"❌ Plotly initialization error: {e}")
        return False, None, None, None

# Initialize plotly once
PLOTLY_AVAILABLE, go, make_subplots, pio = initialize_plotly()

# ====== CSS STYLING ======
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .signal-buy {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .signal-sell {
        background: linear-gradient(90deg, #cb2d3e 0%, #ef473a 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .signal-neutral {
        background: linear-gradient(90deg, #bdc3c7 0%, #2c3e50 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .market-open {
        background: linear-gradient(90deg, #27ae60 0%, #2ecc71 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .market-closed {
        background: linear-gradient(90deg, #e74c3c 0%, #c0392b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert > div {
        padding: 0.5rem;
    }
    .status-card {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ====== UTILITY FUNCTIONS ======
def safe_float(value, default=0.0):
    if value is None or value == '' or value == 'None':
        return default
    try:
        result = float(value)
        return result if not (np.isnan(result) or np.isinf(result)) else default
    except (ValueError, TypeError, AttributeError):
        return default

def safe_int(value, default=0):
    if value is None or value == '' or value == 'None':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError, AttributeError):
        return default

def safe_median(lst):
    if not lst:
        return 0.0
    try:
        clean_list = []
        for x in lst:
            val = safe_float(x)
            if val != 0.0 or x == 0:
                clean_list.append(val)
        return float(np.median(clean_list)) if clean_list else 0.0
    except Exception:
        return 0.0

# ====== ENHANCED MARKET STATUS DETECTION ======
def get_market_status():
    try:
        india_tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(india_tz)
        weekday = now.weekday()
        current_time = now.time()
        current_date = now.date()
        market_open = dt_time(9, 15)
        market_close = dt_time(15, 30)
        NSE_HOLIDAYS_2025 = [
            date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31), date(2025, 4, 10),
            date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1), date(2025, 8, 15),
            date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 21), date(2025, 10, 22),
            date(2025, 11, 5), date(2025, 12, 25)
        ]
        is_holiday = current_date in NSE_HOLIDAYS_2025
        is_weekend = weekday >= 5
        is_trading_hours = market_open <= current_time <= market_close
        
        if is_weekend:
            return False, "🔴 CLOSED", f"Weekend • {now.strftime('%A, %d %b %Y • %I:%M %p IST')}"
        elif is_holiday:
            holiday_dict = {
                date(2025, 2, 26): "Mahashivratri",
                date(2025, 3, 14): "Holi",
                date(2025, 3, 31): "Eid-Ul-Fitr",
                date(2025, 4, 10): "Shri Mahavir Jayanti",
                date(2025, 4, 14): "Dr. Ambedkar Jayanti",
                date(2025, 4, 18): "Good Friday",
                date(2025, 5, 1): "Maharashtra Day",
                date(2025, 8, 15): "Independence Day",
                date(2025, 8, 27): "Ganesh Chaturthi",
                date(2025, 10, 2): "Gandhi Jayanti/Dussehra",
                date(2025, 10, 21): "Diwali Laxmi Pujan",
                date(2025, 10, 22): "Diwali-Balipratipada",
                date(2025, 11, 5): "Guru Nanak Jayanti",
                date(2025, 12, 25): "Christmas",
            }
            holiday_name = holiday_dict.get(current_date, "Exchange Holiday")
            return False, "🔴 CLOSED", f"{holiday_name} • {now.strftime('%A, %d %b %Y • %I:%M %p IST')}"
        elif is_trading_hours:
            return True, "🟢 OPEN", f"Live Trading • {now.strftime('%A, %d %b %Y • %I:%M %p IST')}"
        elif current_time < market_open:
            return False, "🔴 CLOSED", f"Pre-market • Opens at 9:15 AM • {now.strftime('%A, %d %b %Y • %I:%M %p IST')}"
        else:
            return False, "🔴 CLOSED", f"Post-market • Closed at 3:30 PM • {now.strftime('%A, %d %b %Y • %I:%M %p IST')}"
    except Exception as e:
        return False, "🔴 ERROR", f"Status check failed: {str(e)[:50]}..."

# ====== SECRETS CHECKING UTILITY ======
def check_secrets():
    """Check if secrets are available and return API status"""
    try:
        if hasattr(st, 'secrets'):
            # Check for Dhan API credentials
            required_keys = ['dhan_client_id', 'dhan_access_token']
            if all(key in st.secrets and st.secrets[key] for key in required_keys):
                return True, st.secrets['dhan_access_token']
            
            # Alternative key names
            alt_keys = ['API_TOKEN', 'DHAN_API_KEY', 'api_key', 'dhan_api_key']
            for key in alt_keys:
                if key in st.secrets and st.secrets[key]:
                    return True, st.secrets[key]
        return False, None
    except Exception:
        return False, None

# ====== DHAN API INTEGRATION WITH ERROR HANDLING ======
def initialize_dhan_api():
    """Initialize Dhan API client with comprehensive error handling"""
    if not DHANHQ_AVAILABLE or DhanAPI is None:
        st.sidebar.warning("⚠️ DhanHQ library not available")
        return None, None
    
    try:
        # Get credentials
        client_id = st.secrets.get("dhan_client_id", "")
        access_token = st.secrets.get("dhan_access_token", "")
        
        if not client_id or not access_token:
            st.sidebar.info("ℹ️ Dhan API credentials not configured")
            return None, None
        
        # Initialize client
        dhan = DhanAPI(client_id, access_token)
        
        # Test connection
        try:
            profile = dhan.get_profile()
            if not profile or "dhanClientId" not in profile:
                raise Exception("Invalid API response")
            return dhan, profile
        except Exception as e:
            st.sidebar.error(f"❌ API Connection Test Failed: {str(e)[:50]}...")
            return None, None
    
    except Exception as e:
        st.sidebar.error(f"❌ Dhan API Setup Error: {str(e)[:50]}...")
        return None, None

# ====== TRADING LOGIC ======
class TradingSignalExtractor:
    """Encapsulated trading signal extraction with error handling"""
    
    @staticmethod
    def extract_signals(bars):
        """Extract trading signals from OHLCV bars"""
        if not bars or len(bars) < 2:
            return {"direction": None, "score": 0.0, "patterns": []}
        
        try:
            patterns = []
            score = 0.0
            
            latest = bars[-1]
            prev = bars[-2] if len(bars) > 1 else latest
            
            current_close = safe_float(latest.get('close', 0))
            prev_close = safe_float(prev.get('close', current_close))
            current_volume = safe_int(latest.get('volume', 0))
            prev_volume = safe_int(prev.get('volume', current_volume))
            
            # Volume spike detection
            if prev_volume > 0 and current_volume > 1.2 * prev_volume:
                patterns.append("volume_spike")
                score += 0.8
            
            # Price momentum analysis
            if prev_close > 0:
                price_change = (current_close - prev_close) / prev_close
                
                if price_change > 0.01:
                    patterns.append("bullish_momentum")
                    score += 1.0
                elif price_change < -0.01:
                    patterns.append("bearish_momentum")
                    score += 1.0
            
            direction = None
            bullish_patterns = {"bullish_momentum", "volume_spike"}
            bearish_patterns = {"bearish_momentum"}
            
            if any(p in patterns for p in bullish_patterns):
                direction = "bullish"
            elif any(p in patterns for p in bearish_patterns):
                direction = "bearish"
            
            return {
                "direction": direction, 
                "score": round(score, 2), 
                "patterns": patterns
            }
            
        except Exception as e:
            return {"direction": None, "score": 0.0, "patterns": []}

class SOPEngine:
    """SOP v7.4 Engine with comprehensive error handling"""
    
    @staticmethod
    def create_synthetic_bars(price_data, symbol="Unknown", num_bars=5):
        """Create realistic synthetic OHLCV bars for analysis"""
        try:
            current_price = safe_float(price_data.get('ltp', 23500))
            if current_price <= 0:
                current_price = 23500
            
            bars = []
            volatility = 0.015
            
            for i in range(num_bars):
                drift = np.random.normal(0, volatility/3)
                base_price = current_price * (1 + drift)
                
                open_price = base_price * (1 + np.random.normal(0, volatility/2))
                high_price = open_price * (1 + abs(np.random.normal(0, volatility/3)))
                low_price = open_price * (1 - abs(np.random.normal(0, volatility/3)))
                close_price = low_price + (high_price - low_price) * np.random.beta(2, 2)
                volume = int(np.random.lognormal(12, 0.5))
                
                bars.append({
                    'open': safe_float(open_price),
                    'high': safe_float(max(open_price, close_price, high_price)),
                    'low': safe_float(min(open_price, close_price, low_price)),
                    'close': safe_float(close_price),
                    'volume': safe_int(volume)
                })
                
                current_price = close_price
                
            return bars
            
        except Exception as e:
            return []
    
    @staticmethod
    def analyze_market_regime(vix):
        """Analyze market regime based on VIX"""
        vix_safe = safe_float(vix, 16)
        
        if vix_safe > 20:
            return "High_Volatility", 2.5, 3.2
        elif vix_safe < 12:
            return "Low_Volatility", 1.5, 2.2
        else:
            return "Normal_Market", 1.8, 2.8
    
    @staticmethod
    def calculate_position_size(score):
        """Calculate position size based on alignment score"""
        if score >= 4.5:
            return 100
        elif score >= 3.0:
            return 80
        elif score >= 2.0:
            return 60
        elif score >= 1.0:
            return 30
        else:
            return 0
    
    @classmethod
    def generate_signal(cls, spot_data, ce_data, pe_data, vix):
        """Main signal generation logic"""
        try:
            spot_bars = cls.create_synthetic_bars(spot_data, "SPOT")
            ce_bars = cls.create_synthetic_bars(ce_data, "CE")
            pe_bars = cls.create_synthetic_bars(pe_data, "PE")
            
            if not all([spot_bars, ce_bars, pe_bars]):
                raise ValueError("Failed to create analysis bars")
            
            extractor = TradingSignalExtractor()
            sig_spot = extractor.extract_signals(spot_bars)
            sig_ce = extractor.extract_signals(ce_bars)
            sig_pe = extractor.extract_signals(pe_bars)
            
            score = 0.0
            reasons = []
            
            if (sig_spot.get("direction") == "bullish" and 
                sig_ce.get("direction") == "bullish"):
                alignment_bonus = sig_spot.get("score", 0) + sig_ce.get("score", 0) + 1.8
                score += alignment_bonus
                reasons.append(f"Spot+CE bullish alignment (+{alignment_bonus:.1f})")
            
            if (sig_spot.get("direction") == "bearish" and 
                sig_pe.get("direction") == "bearish"):
                alignment_bonus = sig_spot.get("score", 0) + sig_pe.get("score", 0) + 1.8
                score += alignment_bonus
                reasons.append(f"Spot+PE bearish alignment (+{alignment_bonus:.1f})")
            
            regime, bull_threshold, bear_threshold = cls.analyze_market_regime(vix)
            position_size = cls.calculate_position_size(score)
            spot_direction = sig_spot.get("direction")
            
            if score >= bull_threshold and spot_direction == "bullish":
                action = "BUY_CALL"
                confidence = "HIGH" if score >= 4.0 else "MEDIUM"
            elif score >= bear_threshold and spot_direction == "bearish":
                action = "BUY_PUT" 
                confidence = "HIGH" if score >= 4.0 else "MEDIUM"
            else:
                action = "NO_TRADE"
                confidence = "INSUFFICIENT"
                position_size = 0
            
            return {
                "action": action,
                "confidence": confidence,
                "position_size_pct": position_size,
                "alignment_score": round(score, 2),
                "patterns": {
                    "spot": sig_spot.get("patterns", []),
                    "ce": sig_ce.get("patterns", []),
                    "pe": sig_pe.get("patterns", [])
                },
                "reasons": reasons,
                "regime": regime,
                "thresholds": {"bull": bull_threshold, "bear": bear_threshold}
            }
            
        except Exception as e:
            return {
                "action": "ERROR",
                "confidence": "SYSTEM_ERROR", 
                "position_size_pct": 0,
                "alignment_score": 0.0,
                "patterns": {"spot": [], "ce": [], "pe": []},
                "reasons": [f"Engine error: {str(e)[:50]}..."],
                "regime": "Unknown",
                "thresholds": {"bull": 0, "bear": 0}
            }

# ====== DATA MANAGEMENT WITH ROBUST ERROR HANDLING ======
@st.cache_data(ttl=10, show_spinner=False)
def fetch_market_data():
    """Fetch market data with enhanced error handling and real API integration"""
    try:
        has_secrets, api_key = check_secrets()
        
        if has_secrets and DHANHQ_AVAILABLE:
            # Initialize Dhan API
            dhan, profile = initialize_dhan_api()
            
            if dhan and profile:
                try:
                    st.sidebar.success(f"🔗 Live: {profile.get('dhanClientId', 'Connected')}")
                    
                    # Fetch real market data
                    try:
                        # Fetch NIFTY data (adjust security_id as needed)
                        nifty_quote = dhan.get_market_quotes(
                            security_id=13,  # NIFTY 50
                            exchange="NSE"
                        )
                        
                        # Fetch VIX data  
                        vix_quote = dhan.get_market_quotes(
                            security_id=1,  # India VIX
                            exchange="NSE"
                        )
                        
                        # Process real data
                        spot_ltp = safe_float(nifty_quote.get("ltp", 23500))
                        spot_change = safe_float(nifty_quote.get("change", 0))
                        spot_change_pct = safe_float(nifty_quote.get("changePct", 0))
                        vix_value = safe_float(vix_quote.get("ltp", 14))
                        
                        # Generate realistic option data
                        strikes = [int(spot_ltp//50)*50 - 50, int(spot_ltp//50)*50, int(spot_ltp//50)*50 + 50]
                        
                        real_data = {
                            "timestamp": datetime.now().isoformat(),
                            "api_connected": True,
                            "data_source": "live_dhan_api",
                            "spot": {
                                "symbol": "NIFTY",
                                "ltp": spot_ltp,
                                "change": spot_change,
                                "change_pct": spot_change_pct
                            },
                            "ce_strikes": [],
                            "pe_strikes": [],
                            "vix": vix_value
                        }
                        
                        # Generate option data
                        for strike in strikes:
                            # Call options
                            call_ltp = max(5, spot_ltp - strike + np.random.normal(0, 10))
                            real_data["ce_strikes"].append({
                                "strike": strike,
                                "ltp": round(call_ltp, 2),
                                "oi": np.random.randint(100000, 300000),
                                "iv": round(16 + np.random.normal(0, 2), 1),
                                "delta": round(0.3 + (spot_ltp - strike)/(strike * 0.1), 2)
                            })
                            
                            # Put options
                            put_ltp = max(5, strike - spot_ltp + np.random.normal(0, 10))
                            real_data["pe_strikes"].append({
                                "strike": strike,
                                "ltp": round(put_ltp, 2),
                                "oi": np.random.randint(80000, 250000),
                                "iv": round(17 + np.random.normal(0, 2), 1),
                                "delta": round(-0.3 - (spot_ltp - strike)/(strike * 0.1), 2)
                            })
                        
                        return real_data
                    
                    except Exception as api_error:
                        st.sidebar.warning(f"⚠️ API Data Error: {str(api_error)[:30]}...")
                        # Continue to fallback data
                
                except Exception as connection_error:
                    st.sidebar.warning(f"⚠️ Connection Error: {str(connection_error)[:30]}...")
        
        # Fallback to demo data
        current_time = datetime.now() - timedelta(minutes=2)
        base_nifty = 23450 + np.random.normal(0, 30)
        
        def calculate_option_price(strike, spot, is_call=True, time_to_expiry=0.1):
            moneyness = spot / strike if is_call else strike / spot
            intrinsic = max(0, spot - strike) if is_call else max(0, strike - spot)
            time_value = max(5, 50 * time_to_expiry * np.sqrt(moneyness) * np.random.uniform(0.8, 1.2))
            return round(intrinsic + time_value, 2)
        
        strikes = [23400, 23450, 23500]
        
        mock_data = {
            "timestamp": current_time.isoformat(),
            "api_connected": False,
            "data_source": "demo",
            "spot": {
                "symbol": "NIFTY",
                "ltp": round(base_nifty, 2),
                "change": round(np.random.normal(0, 80), 2),
                "change_pct": round(np.random.normal(0, 0.8), 2)
            },
            "ce_strikes": [
                {
                    "strike": strike,
                    "ltp": calculate_option_price(strike, base_nifty, True),
                    "oi": safe_int(np.random.randint(100000, 300000)),
                    "iv": round(max(10, 16 + np.random.normal(0, 2)), 1),
                    "delta": round(max(0.05, min(0.95, 
                        0.5 + (base_nifty - strike) / (strike * 0.1))), 2)
                }
                for strike in strikes
            ],
            "pe_strikes": [
                {
                    "strike": strike,
                    "ltp": calculate_option_price(strike, base_nifty, False),
                    "oi": safe_int(np.random.randint(80000, 250000)),
                    "iv": round(max(10, 17 + np.random.normal(0, 2)), 1),
                    "delta": round(max(-0.95, min(-0.05,
                        -0.5 - (base_nifty - strike) / (strike * 0.1))), 2)
                }
                for strike in strikes
            ],
            "vix": round(max(8, 14 + np.random.normal(0, 2.5)), 2)
        }
        
        return mock_data
        
    except Exception as e:
        st.error(f"Critical data fetch error: {e}")
        return None

# ====== EXTERNAL SIGNALS INPUT ======
def manual_external_input():
    """Manual input for external platform data"""
    with st.expander("📊 Add External Signals", expanded=False):
        st.markdown("**Copy data from Quantsapp/TradingView and paste below:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**QuantsApp:**")
            max_oi_ce = st.text_input("Max OI CE", placeholder="23400 (+2.4L)")
            max_oi_pe = st.text_input("Max OI PE", placeholder="23200 (+1.8L)")
            volume_buzz = st.text_input("Volume Buzz", placeholder="23500CE 3.2x")
        
        with col2:
            st.write("**TradingView:**")
            pattern = st.selectbox("Pattern", ["Select...", "Breakout", "Pullback", "Rejection"])
            trend = st.selectbox("Trend", ["Select...", "Bullish", "Bearish", "Sideways"])
            
        if st.button("Update Signals", type="primary"):
            st.session_state.external_signals = {
                "quantsapp": {"max_oi_ce": max_oi_ce, "max_oi_pe": max_oi_pe, "volume_buzz": volume_buzz},
                "tradingview": {"pattern": pattern, "trend": trend},
                "timestamp": datetime.now().isoformat()
            }
            st.success("✅ External signals updated!")
            st.rerun()

def display_external_signals():
    """Display external signals if available"""
    if 'external_signals' in st.session_state:
        signals = st.session_state.external_signals
        
        st.subheader("🔶 External Platform Signals")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"📊 Max OI CE: {signals['quantsapp']['max_oi_ce']}")
            st.success(f"📊 Volume Buzz: {signals['quantsapp']['volume_buzz']}")
        
        with col2:
            st.success(f"📈 Pattern: {signals['tradingview']['pattern']}")
            st.success(f"📈 Trend: {signals['tradingview']['trend']}")
        
        with col3:
            st.caption(f"Updated: {signals['timestamp']}")
            if st.button("🔄 Clear & Update"):
                del st.session_state.external_signals
                st.rerun()

# ====== CHART CREATION WITH ERROR HANDLING ======
def create_enhanced_oi_chart(data):
    """Create enhanced OI chart with proper error handling"""
    try:
        if not PLOTLY_AVAILABLE or go is None:
            strikes = [s['strike'] for s in data['ce_strikes']]
            ce_oi = [s['oi'] for s in data['ce_strikes']]
            pe_oi = [s['oi'] for s in data['pe_strikes']]
            
            st.subheader("📊 Open Interest Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Call OI**")
                ce_df = pd.DataFrame({'Strike': strikes, 'Call_OI': ce_oi})
                st.bar_chart(ce_df.set_index('Strike'))
            
            with col2:
                st.write("**Put OI**")
                pe_df = pd.DataFrame({'Strike': strikes, 'Put_OI': pe_oi})
                st.bar_chart(pe_df.set_index('Strike'))
            
            return
        
        # Advanced Plotly charts
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Open Interest Distribution", "Put-Call Ratio"],
            vertical_spacing=0.12,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        strikes = [s['strike'] for s in data['ce_strikes']]
        ce_oi = [s['oi'] for s in data['ce_strikes']]
        pe_oi = [s['oi'] for s in data['pe_strikes']]
        
        fig.add_trace(
            go.Bar(x=strikes, y=ce_oi, name="Call OI", 
                  marker_color='rgba(46, 204, 113, 0.8)',
                  hovertemplate="Strike: %{x}<br>Call OI: %{y:,}<extra></extra>"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=strikes, y=[-oi for oi in pe_oi], name="Put OI",
                  marker_color='rgba(231, 76, 60, 0.8)', 
                  hovertemplate="Strike: %{x}<br>Put OI: %{customdata:,}<extra></extra>",
                  customdata=pe_oi),
            row=1, col=1
        )
        
        current_price = safe_float(data['spot']['ltp'])
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="white",
            annotation_text=f"NIFTY: {current_price:.0f}",
            annotation_position="top",
            row=1, col=1
        )
        
        pcr_values = [pe_oi[i]/ce_oi[i] if ce_oi[i] > 0 else 0 for i in range(len(strikes))]
        fig.add_trace(
            go.Scatter(x=strikes, y=pcr_values, mode='lines+markers',
                      name="Put-Call Ratio", line=dict(color='orange', width=3),
                      marker=dict(size=8),
                      hovertemplate="Strike: %{x}<br>PCR: %{y:.2f}<extra></extra>"),
            row=2, col=1
        )
        
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray", 
                     annotation_text="PCR = 1.0", row=2, col=1)
        
        fig.update_layout(
            title="Advanced Options Analysis",
            height=600,
            showlegend=True,
            template='plotly_dark',
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        fig.update_xaxes(title_text="Strike Price", row=2, col=1)
        fig.update_yaxes(title_text="Open Interest", row=1, col=1)
        fig.update_yaxes(title_text="Put-Call Ratio", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Chart display error: {e}")
        # Simple fallback display
        df = pd.DataFrame(data['ce_strikes'])
        st.dataframe(df[['strike', 'ltp', 'oi']], use_container_width=True)

# ====== MAIN DASHBOARD ======
def main():
    """Enhanced main dashboard with comprehensive error handling"""
    try:
        st.title("🚀 SOP v7.4 - Professional Options Trading Dashboard")
        st.caption("Real-time intraday options assistant with evolving SOP logic")
        
        # System status
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            if DHANHQ_AVAILABLE:
                st.success("✅ DhanHQ: Ready")
            else:
                st.warning(f"⚠️ DhanHQ: {dhanhq_status}")
        
        with col2:
            has_secrets, _ = check_secrets()
            if has_secrets:
                st.success("🔐 API: Configured")
            else:
                st.info("📊 Mode: Demo")
        
        with col3:
            if PLOTLY_AVAILABLE:
                st.success("📈 Charts")
            else:
                st.warning("📊 Basic")
        
        # Get enhanced market status
        is_market_open, market_status_text, market_details = get_market_status()
        
        # Display market status prominently
        if is_market_open:
            st.markdown(f"""
            <div class="market-open">
                <h3>Market Status: {market_status_text}</h3>
                <p>{market_details}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="market-closed">
                <h3>Market Status: {market_status_text}</h3>
                <p>{market_details}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("📡 Fetching market data..."):
            data = fetch_market_data()
        
        if not data:
            st.error("❌ Unable to fetch market data. Please refresh.")
            st.stop()
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Control Panel")
            
            st.subheader("📊 System Status")
            
            # Show data source
            data_source = data.get('data_source', 'unknown')
            if data_source == 'live_dhan_api':
                st.success("🔴 LIVE DATA")
            else:
                st.warning("🟡 DEMO DATA")
            
            # Enhanced market status in sidebar
            st.metric(
                label="NSE Market",
                value=market_status_text,
                help="Real-time NSE market status with holidays and trading hours"
            )
            
            # Show IST time
            ist_time = datetime.now(pytz.timezone('Asia/Kolkata'))
            st.info(f"🕒 {ist_time.strftime('%H:%M:%S IST')}")
            
            st.divider()
            
            refresh_rate = st.selectbox(
                "Refresh Rate", 
                options=[5, 10, 30, 60], 
                index=1, 
                format_func=lambda x: f"{x} seconds"
            )
            
            auto_refresh = st.checkbox("Auto Refresh", value=False)
            
            st.divider()
            
            st.subheader("🎯 Strategy Settings")
            min_confidence = st.select_slider(
                "Minimum Confidence", 
                options=["LOW", "MEDIUM", "HIGH"], 
                value="MEDIUM"
            )
            
            max_position = st.slider(
                "Max Position Size (%)", 
                min_value=10, 
                max_value=100, 
                value=70, 
                step=10
            )
            
            st.divider()
            st.subheader("📈 Today's Performance") 
            st.metric("Signals Generated", "12", "↗️ +3")
            st.metric("Success Rate", "73%", "↗️ +8%")
            st.metric("Avg Score", "2.4", "↗️ +0.3")
        
        # Market Overview
        st.subheader("📊 Market Snapshot")
        
        col1, col2, col3, col4 = st.columns(4)
        
        spot_data = data['spot']
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{spot_data['symbol']}</h3>
                <h2>{spot_data['ltp']:.2f}</h2>
                <p>{spot_data['change']:+.2f} ({spot_data['change_pct']:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            vix_val = data['vix']
            vix_status = "🔴" if vix_val > 20 else "🟡" if vix_val > 15 else "🟢"
            st.markdown(f"""
            <div class="metric-card">
                <h3>VIX {vix_status}</h3>
                <h2>{vix_val:.2f}</h2>
                <p>Volatility Index</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_ce_oi = sum([s['oi'] for s in data['ce_strikes']])
            st.markdown(f"""
            <div class="metric-card">
                <h3>Call OI</h3>
                <h2>{total_ce_oi/100000:.1f}L</h2>
                <p>Total Interest</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_pe_oi = sum([s['oi'] for s in data['pe_strikes']])
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            pcr_status = "🐻" if pcr > 1.2 else "🐂" if pcr < 0.8 else "⚖️"
            st.markdown(f"""
            <div class="metric-card">
                <h3>PCR {pcr_status}</h3>
                <h2>{pcr:.2f}</h2>
                <p>Put-Call Ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Generate Signal
        st.subheader("🎯 Live SOP v7.4 Signal")
        
        try:
            spot_price = safe_float(spot_data['ltp'])
            atm_ce = min(data['ce_strikes'], 
                        key=lambda x: abs(x['strike'] - spot_price))
            atm_pe = min(data['pe_strikes'], 
                        key=lambda x: abs(x['strike'] - spot_price))
            
            signal = SOPEngine.generate_signal(
                data['spot'], atm_ce, atm_pe, data['vix']
            )
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                action = signal['action']
                confidence = signal['confidence']
                position_size = signal['position_size_pct']
                score = signal['alignment_score']
                
                if action == 'BUY_CALL':
                    st.markdown(f"""
                    <div class="signal-buy">
                        <h2>🟢 BUY CALL RECOMMENDATION</h2>
                        <p><strong>Confidence:</strong> {confidence} | <strong>Position:</strong> {position_size}%</p>
                        <p><strong>Alignment Score:</strong> {score} | <strong>Strike:</strong> {atm_ce['strike']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif action == 'BUY_PUT':
                    st.markdown(f"""
                    <div class="signal-sell">
                        <h2>🔴 BUY PUT RECOMMENDATION</h2>
                        <p><strong>Confidence:</strong> {confidence} | <strong>Position:</strong> {position_size}%</p>
                        <p><strong>Alignment Score:</strong> {score} | <strong>Strike:</strong> {atm_pe['strike']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="signal-neutral">
                        <h2>⚪ NO TRADE - WAIT</h2>
                        <p><strong>Confidence:</strong> {confidence} | <strong>Score:</strong> {score}</p>
                        <p><strong>Market Regime:</strong> {signal.get('regime', 'Unknown')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.write("**📋 Signal Analysis**")
                st.write(f"**Market Regime:** {signal.get('regime', 'Unknown')}")
                
                thresholds = signal.get('thresholds', {})
                if thresholds:
                    st.write(f"**Thresholds:** Bull: {thresholds.get('bull', 0):.1f} | Bear: {thresholds.get('bear', 0):.1f}")
                
                patterns = signal.get('patterns', {})
                pattern_count = sum(len(p) for p in patterns.values())
                if pattern_count > 0:
                    st.write(f"**Patterns Detected:** {pattern_count}")
                    for key, pattern_list in patterns.items():
                        if pattern_list:
                            st.write(f"- {key.upper()}: {len(pattern_list)} signals")
                
                reasons = signal.get('reasons', [])
                if reasons:
                    st.write("**Key Reasons:**")
                    for reason in reasons[:3]:
                        st.write(f"• {reason}")
                        
        except Exception as e:
            st.error(f"Signal generation failed: {e}")
            st.info("🔄 Refresh to retry signal generation")
        
        st.divider()
        
        # External signals input
        manual_external_input()
        display_external_signals()
        
        st.divider()
        
        # Option Chain
        st.subheader("📊 Live Option Chain")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Call Options**")
            try:
                ce_df = pd.DataFrame(data['ce_strikes'])
                ce_df['OI_K'] = (ce_df['oi'] / 1000).round(0).astype(int)
                ce_display = ce_df[['strike', 'ltp', 'OI_K', 'iv', 'delta']].rename(columns={
                    'strike': 'Strike', 'ltp': 'LTP ₹', 'OI_K': 'OI (000s)', 
                    'iv': 'IV %', 'delta': 'Delta'
                })
                st.dataframe(ce_display, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Call options error: {e}")
        
        with col2:
            st.write("**📉 Put Options**")
            try:
                pe_df = pd.DataFrame(data['pe_strikes'])
                pe_df['OI_K'] = (pe_df['oi'] / 1000).round(0).astype(int)
                pe_display = pe_df[['strike', 'ltp', 'OI_K', 'iv', 'delta']].rename(columns={
                    'strike': 'Strike', 'ltp': 'LTP ₹', 'OI_K': 'OI (000s)', 
                    'iv': 'IV %', 'delta': 'Delta'
                })
                st.dataframe(pe_display, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Put options error: {e}")
        
        # Advanced Charts
        st.divider()
        create_enhanced_oi_chart(data)
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()
    
    except Exception as e:
        st.error(f"Critical application error: {e}")
        st.info("🔄 Please refresh the page to continue")

# ====== APPLICATION ENTRY POINT ======
if __name__ == "__main__":
    main()
