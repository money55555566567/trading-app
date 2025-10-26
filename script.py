import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'fxreplay.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "")
    STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
    STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    START_CAPITAL = float(os.environ.get("START_CAPITAL", "10000"))
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_subscribed = db.Column(db.Boolean, default=False)
    stripe_customer_id = db.Column(db.String(255), nullable=True)
    stripe_subscription_id = db.Column(db.String(255), nullable=True)

class SavedSession(db.Model):
    __tablename__ = "sessions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    name = db.Column(db.String(255))
    data_json = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
from flask import Blueprint, request, render_template, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, LoginManager
from .models import db, User

auth_bp = Blueprint("auth", __name__)

login_manager = LoginManager()
login_manager.login_view = "auth.login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"].lower().strip()
        password = request.form["password"]
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "danger")
            return redirect(url_for("auth.register"))
        user = User(email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Registered and logged in", "success")
        return redirect(url_for("main_dashboard"))
    return render_template("register.html")

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].lower().strip()
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("main_dashboard"))
        flash("Invalid credentials", "danger")
        return redirect(url_for("auth.login"))
    return render_template("login.html")

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out", "info")
    return redirect(url_for("auth.login"))
import pandas as pd
import ccxt
import yfinance as yf
import time

def fetch_yfinance(ticker, start, end, interval="1h"):
    """
    Fetch OHLCV from yfinance. Returns DataFrame index=timestamp columns: open, high, low, close, volume
    """
    df = yf.download(tickers=ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data from yfinance for {ticker}")
    df = df.rename(columns=str.lower).reset_index()
    if "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"])
    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
    else:
        df["timestamp"] = pd.to_datetime(df["index"], errors="coerce")
    df = df.set_index("timestamp")
    # keep canonical columns
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["close"])
    return df

def fetch_ccxt(symbol, start, end, timeframe="1h", exchange_name="binance"):
    """
    Fetch OHLCV using ccxt. symbol example: 'BTC/USDT'
    """
    exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
    since = int(pd.to_datetime(start).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end).timestamp() * 1000)
    ohlcv = []
    limit = 1000
    while since < end_ts:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        ohlcv.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < limit:
            break
        time.sleep(exchange.rateLimit / 1000.0)
    if not ohlcv:
        raise ValueError(f"No data from ccxt for {symbol}")
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    return df
import pandas as pd
import numpy as np

def sma_signals(df, fast=10, slow=30):
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(fast).mean()
    df["sma_slow"] = df["close"].rolling(slow).mean()
    df["signal"] = 0
    df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
    df["signal"] = df["signal"].astype(int)
    return df

def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).fillna(0)
    down = -1 * delta.clip(upper=0).fillna(0)
    ema_up = up.ewm(com=window - 1, adjust=False).mean()
    ema_down = down.ewm(com=window - 1, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-9)
    return 100 - (100 / (1 + rs))

def rsi_signals(df, rsi_period=14, buy_thr=30, sell_thr=70):
    df = df.copy()
    df["rsi"] = rsi(df["close"], window=rsi_period)
    df["signal"] = 0
    df.loc[df["rsi"] < buy_thr, "signal"] = 1
    df.loc[df["rsi"] > sell_thr, "signal"] = 0
    return df

def infer_periods_per_year(df):
    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        return 252
    median_days = diffs.dt.total_seconds().median() / 86400.0
    if median_days <= 0:
        return 252
    return max(1, int(round(365.0 / median_days)))

def run_backtest(df, start_capital=10000, strategy="sma", params=None):
    params = params or {}
    if strategy == "sma":
        fast = int(params.get("fast", 10))
        slow = int(params.get("slow", 30))
        df = sma_signals(df, fast=fast, slow=slow)
    elif strategy == "rsi":
        rsi_p = int(params.get("rsi_period", 14))
        buy = float(params.get("buy_thr", 30))
        sell = float(params.get("sell_thr", 70))
        df = rsi_signals(df, rsi_period=rsi_p, buy_thr=buy, sell_thr=sell)
    else:
        raise ValueError("Unknown strategy")

    df = df.copy()
    df["position"] = df["signal"].shift(1).fillna(0)
    df["pct_ret"] = df["close"].pct_change().fillna(0)
    df["strat_ret"] = df["pct_ret"] * df["position"]
    equity = (1 + df["strat_ret"]).cumprod() * start_capital
    equity = equity.fillna(method="ffill").fillna(start_capital)

    trades = []
    in_trade = False
    entry_price = None
    entry_idx = None
    for idx, row in df.iterrows():
        pos = int(row["position"])
        price = float(row["close"])
        if not in_trade and pos == 1:
            in_trade = True
            entry_price = price
            entry_idx = idx
        elif in_trade and pos == 0:
            exit_price = price
            pnl = exit_price - entry_price
            rtn = pnl / entry_price
            trades.append({
                "entry_time": entry_idx.isoformat(),
                "exit_time": idx.isoformat(),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "pnl": float(pnl),
                "return": float(rtn)
            })
            in_trade = False
            entry_price = None
            entry_idx = None
    if in_trade:
        last_idx = df.index[-1]
        exit_price = float(df.iloc[-1]["close"])
        pnl = exit_price - entry_price
        rtn = pnl / entry_price
        trades.append({
            "entry_time": entry_idx.isoformat(),
            "exit_time": last_idx.isoformat(),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "pnl": float(pnl),
            "return": float(rtn)
        })

    periods_per_year = infer_periods_per_year(df)
    total_days = (df.index[-1] - df.index[0]).days if len(df.index) > 1 else 1
    years = total_days / 365.0 if total_days > 0 else (len(df.index) / periods_per_year)
    start_equity = start_capital
    end_equity = float(equity.iloc[-1])
    total_return = (end_equity / start_equity) - 1.0
    cagr = (end_equity / start_equity) ** (1.0 / max(years, 1e-9)) - 1.0
    periodic_rets = df["strat_ret"].dropna()
    ann_vol = periodic_rets.std() * (periods_per_year ** 0.5) if len(periodic_rets) > 1 else 0.0
    sharpe = (cagr / ann_vol) if ann_vol > 0 else 0.0
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = float(drawdown.min())
    dd_durations = []
    last_peak_equity = -1
    current_dd_start = None
    for t, val in equity.items():
        if val >= last_peak_equity:
            last_peak_equity = val
            if current_dd_start is not None:
                dd_durations.append((t - current_dd_start).days)
                current_dd_start = None
        else:
            if current_dd_start is None:
                current_dd_start = t
    max_dd_duration = max(dd_durations) if dd_durations else 0

    total_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    avg_win = (np.mean([t["pnl"] for t in wins]) if wins else 0.0)
    avg_loss = (np.mean([t["pnl"] for t in losses]) if losses else 0.0)

    metrics = {
        "start_equity": float(start_equity),
        "end_equity": float(end_equity),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "max_dd_duration_days": int(max_dd_duration),
        "total_trades": int(total_trades),
        "win_rate_percent": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "periods_per_year": int(periods_per_year),
        "years": float(years)
    }

    equity_series = [{"timestamp": t.isoformat(), "equity": float(v)} for t, v in equity.items()]

    return metrics, equity_series, trades, df.reset_index().to_dict(orient="records")
import stripe
from flask import request, Blueprint, current_app, jsonify
from .models import db, User

stripe_bp = Blueprint("stripe", __name__)

@stripe_bp.route("/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", None)
    secret = current_app.config.get("STRIPE_WEBHOOK_SECRET")
    stripe.api_key = current_app.config.get("STRIPE_API_KEY")

    # Development convenience: if no webhook secret provided, attempt to construct event directly (less secure)
    if not secret:
        try:
            event = stripe.Event.construct_from(request.get_json(force=True), stripe.api_key)
        except Exception:
            return jsonify({"error": "invalid webhook"}), 400
    else:
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, secret)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    # Handle subscription/payments events
    if event.type in ("checkout.session.completed", "invoice.payment_succeeded", "customer.subscription.created"):
        data = event.data.object
        customer_id = data.get("customer")
        if customer_id:
            user = User.query.filter_by(stripe_customer_id=customer_id).first()
            if user:
                user.is_subscribed = True
                db.session.commit()

    if event.type == "customer.subscription.deleted":
        customer_id = event.data.object.get("customer")
        user = User.query.filter_by(stripe_customer_id=customer_id).first()
        if user:
            user.is_subscribed = False
            db.session.commit()

    return jsonify({"status": "ok"})
from flask import Flask, render_template, request, jsonify, redirect, url_for, current_app
from flask_migrate import Migrate
from flask_login import login_required, current_user
from .config import Config
from .models import db, User
from .auth import auth_bp, login_manager
from .backtester import run_backtest
from .data_fetch import fetch_ccxt, fetch_yfinance
from .stripe_webhook import stripe_bp
import stripe

def create_app():
    app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")
    app.config.from_object(Config)
    db.init_app(app)
    migrate = Migrate(app, db)
    login_manager.init_app(app)
    app.register_blueprint(auth_bp)
    app.register_blueprint(stripe_bp, url_prefix="/stripe")

    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for("main_dashboard"))
        return render_template("index.html")

    @app.route("/dashboard")
    @login_required
    def main_dashboard():
        return render_template("dashboard.html", user=current_user, stripe_public_key=app.config.get("STRIPE_PUBLISHABLE_KEY"))

    @app.route("/api/data", methods=["POST"])
    @login_required
    def api_data():
        payload = request.get_json() or {}
        symbol = payload.get("symbol", "AAPL")
        market = payload.get("market", "equity")
        start = payload.get("start", "2024-01-01")
        end = payload.get("end", "2024-06-01")
        interval = payload.get("interval", "1h")
        try:
            if market == "crypto":
                df = fetch_ccxt(symbol, start, end, timeframe=interval)
            else:
                df = fetch_yfinance(symbol, start, end, interval=interval)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        out = df.reset_index().to_dict(orient="records")
        return jsonify(out)

    @app.route("/api/backtest", methods=["POST"])
    @login_required
    def api_backtest():
        if not current_user.is_subscribed:
            return jsonify({"error": "subscription_required"}), 402
        body = request.get_json() or {}
        symbol = body.get("symbol", "AAPL")
        market = body.get("market", "equity")
        start = body.get("start", "2024-01-01")
        end = body.get("end", "2024-06-01")
        interval = body.get("interval", "1h")
        strategy = body.get("strategy", "sma")
        params = body.get("params", {})
        try:
            if market == "crypto":
                df = fetch_ccxt(symbol, start, end, timeframe=interval)
            else:
                df = fetch_yfinance(symbol, start, end, interval=interval)
            metrics, equity_series, trades, annotated = run_backtest(
                df, start_capital=current_app.config["START_CAPITAL"], strategy=strategy, params=params
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"metrics": metrics, "equity": equity_series, "trades": trades, "annotated": annotated})

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/dashboard/replay")
    @login_required
    def replay_page():
        return render_template("replay.html")

    @app.route("/create-checkout-session", methods=["POST"])
    @login_required
    def create_checkout_session():
        stripe.api_key = app.config.get("STRIPE_API_KEY")
        data = request.get_json() or {}
        price_id = data.get("price_id", "price_XXXXXXXXXXXXXX")
        domain = data.get("domain", request.host_url)
        try:
            session = stripe.checkout.Session.create(
                customer_email=current_user.email,
                payment_method_types=["card"],
                mode="subscription",
                line_items=[{"price": price_id, "quantity": 1}],
                success_url=domain + "dashboard?session_id={CHECKOUT_SESSION_ID}",
                cancel_url=domain + "dashboard",
            )
            if session.get("customer"):
                current_user.stripe_customer_id = session["customer"]
                from .models import db
                db.session.commit()
            return jsonify({"id": session["id"], "checkout_url": session.url})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
"""
Run this script to create a test user and optionally toggle subscription flag.
Usage:
  python admin_create_user.py email@example.com password --subscribe
"""
import sys
import os
from getpass import getpass

# set env var FLASK_APP or ensure backend on pythonpath
from backend.app import create_app
from backend.models import db, User
from werkzeug.security import generate_password_hash

def main():
    if len(sys.argv) < 3:
        print("Usage: python admin_create_user.py email password [--subscribe]")
        return
    email = sys.argv[1].lower()
    password = sys.argv[2]
    subscribe = "--subscribe" in sys.argv
    app = create_app()
    with app.app_context():
        db.create_all()
        if User.query.filter_by(email=email).first():
            u = User.query.filter_by(email=email).first()
            print("User already exists:", u.email)
            if subscribe:
                u.is_subscribed = True
                db.session.commit()
                print("Subscribed flag set.")
            return
        user = User(email=email, password_hash=generate_password_hash(password), is_subscribed=subscribe)
        db.session.add(user)
        db.session.commit()
        print("Created user:", user.email)
        if subscribe:
            print("User subscribed (dev flag).")

if __name__ == "__main__":
    main()


python -m venv venv
source venv/bin/activate        # macOS / Linux
# or
venv\Scripts\activate           # Windows

pip install -r backend/requirements.txt

# from project root
python -c "from backend.app import create_app; app=create_app(); ctx=app.app_context(); ctx.push(); from backend.models import db; db.create_all(); print('DB created')"

python backend/admin_create_user.py test@example.com mypassword --subscribe

export FLASK_APP=backend.app:create_app
export FLASK_ENV=development
# optionally set Stripe test keys:
export STRIPE_API_KEY=""
export STRIPE_PUBLISHABLE_KEY=""
export STRIPE_WEBHOOK_SECRET=""
python -m backend.app
# or run with gunicorn (prod-ish)
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:create_app()


