from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo



import sys
sys.path.append("/workspaces/finance-/build")

import finance


NYC = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def ensure_date(d) :
    return d.date() if isinstance(d, datetime) else d


def yearfrac_act365(asof, expiry) :
    a = datetime.combine(ensure_date(asof), datetime.min.time(), tzinfo=UTC)
    e = datetime.combine(ensure_date(expiry), datetime.min.time(), tzinfo=NYC)
    e = e.replace(hour=16, minute=0).astimezone(UTC)
    T = max((e - a).total_seconds() / (365.0 * 24 * 3600), 1.0 / 365.0)
    return float(T)



def to_asset_table(asset_id, symbol, asset_class) :
    return pd.DataFrame([
        {"asset_id": int(asset_id), "symbol": symbol, "asset_class": asset_class}
    ], columns=["asset_id", "symbol", "asset_class"])  


def get_close_on_or_before(ticker, asof) :
    h = ticker.history(period="3mo", interval="1d").copy()
    if h.empty:
        return None
    h = h.reset_index()
    h["Date"] = h["Date"].dt.date
    h = h[h["Date"] <= asof]
    if h.empty:
        return None
    return float(h.iloc[-1]["Close"])



class pour_bid_ask : 
    def __init__(self, S0,delta, volume) :
        self.volume = volume
        self.S0 = S0
        self.delta = delta

"""def simulate_bid_ask(pour_ask,
                     base_pct=0.01,     
                     spread_min=0.05,    
                     vol_high=1000,      
                     vol_low=100):       
    mid = pour_ask.S0 if pour_ask.S0 > 0 else np.nan
    if np.isnan(mid): 
        return np.nan, np.nan

    delta = pour_ask.delta
    volume = pour_ask.volume

    moneyness_factor = 1 + max(0, 0.5 - min(delta, 0.5)) * 2  

    if volume > vol_high:
        illiquidity_factor = 0.5
    elif volume < vol_low:
        illiquidity_factor = 2.0
    else:
        illiquidity_factor = 1.0

    spread = max(base_pct * mid * illiquidity_factor * moneyness_factor, spread_min)
    
    spread *= (1 + np.random.uniform(-0.05, 0.05))

    bid = max(mid - spread / 2, 0.01)
    ask = mid + spread / 2
    return bid, ask

"""


def to_prices_df(asset_id, symbol, asof) :
    t = yf.Ticker(symbol)
    start = ensure_date(asof) - timedelta(days=10)
    end = ensure_date(asof) + timedelta(days=1)
    h = t.history(start=start, end=end, interval="1d", auto_adjust=False)
    if h.empty:
        raise RuntimeError(f"Aucun historique daily pour {symbol} entre {start} et {end}")


    df = h.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df[df["Date"] <= ensure_date(asof)]
    if df.empty:
        raise RuntimeError(f"Pas de barre daily ≤ {asof} pour {symbol}")
    row = df.iloc[-1]
    rec = {
    "asset_id": asset_id,
    "asof": row["Date"],
    "open": float(row.get("Open", np.nan)),
    "close": float(row.get("Close", np.nan)),
    "high": float(row.get("High", np.nan)),
    "low": float(row.get("Low", np.nan)),
    "volume": float(row.get("Volume", np.nan)),
    }
    return pd.DataFrame([rec], columns=["asset_id", "asof", "open", "close", "high", "low", "volume"])

@dataclass
class OptionQuote:
    asset_id: int
    asof: date
    contract_symbol: str
    expiry: date
    strike: float
    volume: Optional[float]
    iv: Optional[float]  
    openInterest: float
    bid : float
    ask : float


OPTION_QUOTES_COLS = [
    "asset_id", "asof", "contract_symbol", "expiry", "strike",  "volume", "iv" , "openInterest", "bid" , "ask"
]


def _extract_side(df, asset_id, asof):
    rows: list[OptionQuote] = []
    if df is None or df.empty:
        return rows

    for _, r in df.iterrows():
        rows.append(
            OptionQuote(
                asset_id=asset_id,
                asof=ensure_date(asof),
                contract_symbol=str(r.get("contractSymbol", "")),
                expiry=date.fromisoformat(str(r.get("contractSymbol", "")).split(str(asset_id))[-1][:10])
                if False else None,  # on ne parse pas expiry depuis le symbol (peu fiable)
                strike=float(r.get("strike", np.nan)),
                volume=float(r.get("volume", np.nan)) if not pd.isna(r.get("volume")) else None,
                iv=float(r.get("impliedVolatility", np.nan)) if not pd.isna(r.get("impliedVolatility")) else None,
                openInterest = float(r.get("openInterest", np.nan)) if not pd.isna(r.get("openInterest"))  else None , 
                bid =float(r.get("bid", np.nan)) if not pd.isna(r.get("bid")) else None,
                ask =float(r.get("ask", np.nan)) if not pd.isna(r.get("ask")) else None
                
            )
        )
    return rows


"""def build_option_quotes_snapshot_df(asset_id, symbol, asof) :
    
    t = yf.Ticker(symbol)
    rows = []
    expiries = t.options or []  
    for exp_str in expiries:
        try:
            exp_date = date.fromisoformat(exp_str)
        except Exception:
            continue
        oc = t.option_chain(exp_str)
        rows += _extract_side(oc.calls, asset_id, asof)
       # rows += _extract_side(oc.puts, asset_id, asof)
        for r in rows if (isinstance(oc.calls, pd.DataFrame)) else []:
            r.expiry = exp_date


    df = pd.DataFrame([r.__dict__ for r in rows], columns=OPTION_QUOTES_COLS)

    return df"""

def build_option_quotes_snapshot_df(asset_id, symbol, asof):
    t = yf.Ticker(symbol)
    rows = []
    expiries = t.options or []  

    for exp_str in expiries:
        try:
            exp_date = date.fromisoformat(exp_str)
        except Exception:
            continue

        try:
            oc = t.option_chain(exp_str)
        except Exception as e:
       
            continue

        new_rows = []
        if isinstance(oc.calls, pd.DataFrame):
            new_rows = _extract_side(oc.calls, asset_id, asof)
            
      
        for r in new_rows:
            try:
                setattr(r, "expiry", exp_date)
            except Exception:
                # sinon si r est un dict
                try:
                    r.expiry = exp_date
                except Exception:
                    pass

        rows.extend(new_rows)


    if len(rows) == 0:
        return pd.DataFrame(columns=OPTION_QUOTES_COLS)

    if hasattr(rows[0], "__dict__"):
        df = pd.DataFrame([r.__dict__ for r in rows], columns=OPTION_QUOTES_COLS)
    else:
        df = pd.DataFrame(rows, columns=OPTION_QUOTES_COLS)

    return df



VOL_SURF_COLS = ["asset_id", "asof", "tenor", "moneyness", "iv"]


def to_vol_surface(quotes_df, asset_id, symbol, asof) :
    t = yf.Ticker(symbol)
    S = get_close_on_or_before(t, asof)
    if S is None or math.isnan(S):
        raise RuntimeError(f"Spot introuvable pour {symbol} au {asof}")

    rows = []
    for _, r in quotes_df.iterrows():
        if pd.isna(r.get("expiry")) or pd.isna(r.get("strike")):
            continue
        if pd.isna(r.get("iv")):
            continue
        T = yearfrac_act365(asof, r["expiry"])  
        m = float(r["strike"]) / float(S)
        rows.append({
            "asset_id": asset_id,
            "asof": ensure_date(asof),
            "tenor": round(float(T), 6),
            "moneyness": round(float(m), 6),
            "iv": round(float(r["iv"]), 6),
        })

    return pd.DataFrame(rows, columns=VOL_SURF_COLS)

def interpolate_sigma(vol_surface_df, asof, T, K, S, k: int = 7) :
    
    df = vol_surface_df.copy()
    if df.empty:
        return float("nan")
    m0 = float(K) / float(S)
    df = df[df["asof"] == ensure_date(asof)].copy()
    if df.empty:
        return float("nan")

    eps = 1e-8
    dT = (df["tenor"].astype(float) - float(T)) / max(float(T), eps)
    dM = (df["moneyness"].astype(float) - float(m0)) / max(float(m0), eps)
    dist = np.hypot(dT, dM)
    df = df.assign(dist=dist)

    exact = df[df["dist"] <= eps]
    if not exact.empty:
        return float(exact.iloc[0]["iv"])  

    k = min(k, len(df))
    nn = df.nsmallest(k, "dist").copy()
    nn.loc[:, "w"] = 1.0 / np.maximum(nn["dist"], eps)
    # Normalisation des poids
    w = nn["w"].to_numpy()
    ivs = nn["iv"].astype(float).to_numpy()
    if np.all(np.isnan(ivs)):
        return float("nan")
    w = w / w.sum()
    return float(np.nansum(w * ivs))


SIM_PARAMS_COLS = [
    "asset_id", "asof", "contract_symbol", "expiry", "strike",
    "S0", "T", "r", "sigma","delta" ,"bid", "ask" , "openInterest" , "volume"
]


def build_simulation_params_df(asset_id,symbol,asof,r_const,quotes_df,vol_surface_df,contracts = None) :
   
    t = yf.Ticker(symbol)
    S0 = get_close_on_or_before(t, asof)
    if S0 is None:
        raise RuntimeError(f"Spot introuvable pour {symbol} au {asof}")

    rows = []

    if contracts is None:
        g = quotes_df.groupby(["contract_symbol", "expiry", "strike", "volume", "openInterest", "ask" , "bid"], dropna=False).size().reset_index()[
            ["contract_symbol", "expiry", "strike", "volume", "openInterest" , "ask" , "bid"]
        ]
        contracts = [(
            str(r.contract_symbol), ensure_date(r.expiry), float(r.strike), float(r.volume) , float(r.openInterest) , float(r.bid) , float(r.ask)
        ) for r in g.itertuples(index=False)]

    for cs, exp, K, volume, oi , bd, ak in contracts:
        T = yearfrac_act365(asof, exp)
        iv_row = quotes_df[
            (quotes_df["contract_symbol"] == cs)
            & (quotes_df["expiry"] == exp)
            & (quotes_df["strike"].astype(float) == float(K))
        ]
        sigma = None
        if not iv_row.empty and not pd.isna(iv_row.iloc[0]["iv"]):
            sigma = float(iv_row.iloc[0]["iv"])  # IV Yahoo
            iv_source = "option_quotes"
            iv_method = "yahoo_iv"
        else:
            # Interpoler depuis la surface
            sigma = interpolate_sigma(vol_surface_df, asof, T, K, S0)
            iv_source = "vol_surfaces"
            iv_method = "knn7_norm"

        BS_pa = finance.BS_parametres(S0, K,T,r_const,sigma)
        delta = finance.call_delta(BS_pa)

   #     bid_ask_params = pour_bid_ask(S0,delta, volume)

        rows.append({
            "asset_id": asset_id,
            "asof": ensure_date(asof),
            "contract_symbol": cs,
            "expiry": ensure_date(exp),
            "strike": float(K),
            "S0": round(float(S0), 6),
            "T": round(float(T), 5),
            "r": float(r_const),
            "sigma": round(float(sigma), 9) if sigma is not None and not math.isnan(sigma) else np.nan,
            "delta" : delta,
            "bid" : round(float(bd), 5) , 
            "ask" : round(float(ak) ,5),
            "openInterest"  :  round(float(oi), 5), 
            "volume" : round(volume,5)
            
            
        })

    return pd.DataFrame(rows, columns=SIM_PARAMS_COLS)



def daily_snapshot_for_symbol(asset_id, symbol, asof, r_flat = 0.04):
    prices_df = to_prices_df(asset_id, symbol, asof)
    quotes_df = build_option_quotes_snapshot_df(asset_id, symbol, asof)
    vol_surface_df = to_vol_surface(quotes_df, asset_id, symbol, asof)
    sim_params_df = build_simulation_params_df(asset_id, symbol, asof, r_flat, quotes_df, vol_surface_df)
    return prices_df, quotes_df, vol_surface_df, sim_params_df




def main() : 
    asof_today = datetime.now(UTC).date()
    asset_id = 10101  
    symbol = "AAPL"

    prices_df, quotes_df, vol_surface_df, sim_params_df = daily_snapshot_for_symbol(asset_id, symbol, asof_today, r_flat=0.04)
    global df_simu
    df_simu = sim_params_df.copy()
"""    print(df_simu)

    print("prices : ",prices_df.head())
    print("quotes :" ,quotes_df.head())
    print("vol",vol_surface_df["iv"])
    print("sim",sim_params_df["bid"].head())


    """

main()
