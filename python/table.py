import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import uuid
from zoneinfo import ZoneInfo  
#on importe le module C++ compilé pour pouvoir calculer les prix des options et les différents paramètres voulues
import sys
sys.path.append("/workspaces/finance-/build")   
import finance as fn




def yearfrac_act365(expiry_str) :
    now = datetime.now(ZoneInfo("UTC"))
    nyc = ZoneInfo("America/New_York")
    expiry_local = datetime.fromisoformat(expiry_str).replace(
        hour=16, minute=0, second=0, microsecond=0, tzinfo=nyc
    )
    expiry_utc = expiry_local.astimezone(ZoneInfo("UTC"))
    T = (expiry_utc - now).total_seconds() / (365.0 * 24 * 3600)
    return (max(T, 1/365))



def to_assets_table():  
    df_assets = pd.DataFrame(columns =["id", "symbol", "asset_class"] )
    nom = input("Nom de l'actif (ex: AAPL): ")
    type = input("Type d'action (EQ, FX, INDEX , CRYPTO): ")
    octal = "".join(format(ord(t),"03o") for t in nom)
    df_assets.loc[len(df_assets)] = {"id": int(octal), "symbol": nom, "asset_class": type}
    return df_assets
    

def prices_to_table(df_assets):

    df_prices = pd.DataFrame(columns =["assets_id","asof", "open", "close", "high", "low", "volume"] )
    
    ticket = yf.Ticker(df_assets["symbol"].iloc[0])
    asof = datetime.now().date
    hist= ticket.history(period="1d")
    close = round(hist["Close"].iloc[0],3)
    ope = round(hist["Open"].iloc[0],3)
    low = round(hist["Low"].iloc[0],3)
    high = round(hist["High"].iloc[0],3)
    vol = round(hist["Volume"].iloc[0],3)
    df_prices.loc[len(df_prices)] = {"assets_id": df_assets["id"][0], "asof": asof, "open": ope, "close" : close, "high" : high, "low": low, "volume" : volume}
    return df_prices



def to_vol_surfaces_table(df_assets):
    df_vol_surfaces = pd.DataFrame(columns =["assets_id", "asof", "tenor", "moneyness", "ivol"] )
    asset_id = df_assets["id"].iloc[0]
    ticker = yf.Ticker(df_assets["symbol"].iloc[0])
    asof = datetime.now().date()
    for expiry in ticker.options:
        parametre = ticker.option_chain(expiry)
        tenor = yearfrac_act365(expiry)
        for i in range(len(parametre.calls)):
            moneyness = parametre.calls["strike"][i]/ticker.history(period="1d")["Close"].iloc[-1]
            ivol = parametre.calls["impliedVolatility"][i]
            df_vol_surfaces.loc[len(df_vol_surfaces)] = {"assets_id": asset_id, "asof": asof, "tenor": round(tenor,3), "moneyness": round(moneyness,3), "ivol": round(float(ivol),3)}
    return df_vol_surfaces
    
    

def to_finale_ivol(df_vol_surfaces, date, T,K,S0):
    df_vol = df_vol_surfaces[df_vol_surfaces["asof"] == date].copy()
    tenor_base = T
    moneyness_base = K / S0

    df_vol["distance"] = np.sqrt(
        (((df_vol["tenor"] - tenor_base)/tenor_base)**2) + 
        (((df_vol["moneyness"] - moneyness_base)/moneyness_base)**2)
    )

    df_interpolation = df_vol.nsmallest(5, "distance")

    poids = 1 / df_interpolation["distance"]
    ivol_finale = np.average(df_interpolation["ivol"], weights=poids)

    return ivol_finale


"""
def parametres_title(df_assets, date):  # pour faire fonctionner nos algorithmes de valorisation, nous avons besion de S0, K, T, r, sigma 
    params = {}
    ticker = yf.Ticker(df_assets["symbol"].iloc[0])
    S0 = ticker.history("Date" == date)["Close"].iloc[0]
    parametre = ticker.option_chain(ticker.options.iloc[0])
    K = parametre.calls["strike"].iloc[0]
    r = 0.04   #taux de la fe en ce moment, changer plus tard pour avoir les données exactes
    expiry = ticker.options[0]
    exp = yearfrac_act365(expiry)
    params.update({"S0": round(float(S0),3)})
    params.update({"K": round(float(K),3)})
    params.update({"T": round(exp,3)})
    params.update({"r": r})
    sigma = round(to_finale_ivol(to_vol_surfaces_table(df_assets),exp,K,S0),3)
    params.update({"sigma": round(float(sigma),3)})
    return params
    

    """

def to_params_table(df_assets) : 
    df_params = pd.DataFrame(columns= ["asset_id",  "contract_symbol", "asof", "expiry", "strike", "S0", "r", "sigma"])
    asset_id = df_assets["id"].iloc[0]
    asof = datetime.now().date
    expiry = yf.Ticker("AAPL").options[0]
    T = (expiry - asof)/365
    df_expiry = yf.Ticker("AAPL").option_chain(expiry).calls[0]
    strike = df_expiry["Strike"]
    S0 = yf.Ticker("AAPL").history["Close"].iloc[0]
    r  = 0.04
    sigma = to_finale_ivol(df_assets, asof, T,strike,S0)


def to_backtest_table(df_assets, params_start_ts, params_end_ts):  
    df_backtest = pd.DataFrame(columns =["id", "name", "asset_id" , "asof_start", "asof_end", "params_start_ts","params_end_ts"] )
    val = {}
    val.update({"id": uuid.uuid4()})
    val.update({"name": input("Nom du backtest: ")})
    val.update({"asset_id": df_assets["id"].iloc[0]})
    asof_start = datetime.strptime(input("Date de début (DD-MM-YYYY ): "), "%d-%m-%Y")
    val.update({"asof_start": asof_start})
    asof_end = datetime.now().date()
    val.update({"asof_end": asof_end})
    param_start = params_start_ts(df_assets,asof_start)
    
    
    df_backtest.loc[len(df_backtest)] = val
    return df_backtest




def to_trades_table(df_backtest, df_prices, params):
    df_trades = pd.DataFrame(columns =[ "backtest_id", "ts", "position", "price", "pnl"] )
    backtest_id = df_backtest["id"][0]
    ts = df_prices["ts"][0]
    position = int(input("Position (1 pour achat, -1 pour vente): "))
    price = fn.call_price(params["S0"], params["K"], params["T"], params["r"], params["sigma"])
    pnl = position * (df_prices["close"][0] - price)
    df_trades.loc[len(df_trades)] = {"backtest_id": backtest_id, "ts": ts, "position": position, "price": round(price,3), "pnl": round(pnl,3)}
    return df_trades


def to_metrics_table(df_backtest, df_trades): 
    df_metrics = pd.DataFrame(columns =["backtest_id", "metric", "value"] )
    backtest_id = df_backtest["id"][0]
    metric = "total_pnl"
    value = df_trades["pnl"].sum()
    df_metrics.loc[len(df_metrics)] = {"backtest_id": backtest_id, "metric": metric, "value": round(value,3)}
    return df_metrics





ticket = yf.Ticker("AAPL")
expiry = ticket.options[0]
hist= ticket.option_chain(expiry).calls

print(hist)
