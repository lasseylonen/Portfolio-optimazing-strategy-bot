import os
import pandas as pd
import numpy as np
import datetime
import time
from binance.client import Client
from dotenv import load_dotenv

# Lataa .env-tiedosto
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("API-avaimia ei löydy ympäristömuuttujista.")

client = Client(API_KEY, API_SECRET, testnet=False)

symbols = ['BNBUSDT', 'BTCUSDT', 'ETHUSDT']
interval = '5m'

# ========================
# Asetukset
# ========================
TAKER_FEE = 0.001
slippage = 0.001
MIN_COOLDOWN = 5
risk_per_trade = 0.05
INITIAL_BALANCE = 1000.0
ATR_WINDOW = 14
MIN_AVG_VOLUME = 50
VOLATILITY_MIN = 1e-6

# ATR logiikan multipliers
atr_tp_multipliers = [1.5, 2.0, 2.5, 3.0, 3.5]
atr_sl_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]

# Kiinteät TP/SL prosentit (0% -> 8%, step 0.5%)
fixed_tp = [x * 0.005 for x in range(0, 15)]
fixed_sl = [x * 0.005 for x in range(0, 15)]

# ========================
# Apufunktiot
# ========================

def get_klines_full(symbol, interval, start_str, end_str=None):
    df = pd.DataFrame()
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else int(datetime.datetime.now().timestamp() * 1000)
    while True:
        try:
            data = client.get_klines(symbol=symbol, interval=interval, limit=1000, startTime=start_ts, endTime=end_ts)
            if not data:
                break
            temp = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "trades",
                "taker_base", "taker_quote", "ignore"
            ])
            temp["open_time"] = pd.to_datetime(temp["open_time"], unit="ms")
            temp[["open","high","low","close","volume"]] = temp[["open","high","low","close","volume"]].astype(float)
            df = pd.concat([df, temp[["open_time","open","high","low","close","volume"]]], ignore_index=True)
            start_ts = data[-1][0] + 1
            if start_ts >= end_ts:
                break
            time.sleep(0.2)
        except Exception as e:
            print(f"Virhe haettaessa dataa {symbol}: {e}")
            break
    return df

def calculate_atr(df, window=ATR_WINDOW):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean().bfill()

def calculate_rsi(prices, period=14):
    series = pd.Series(prices)
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values

def prepare_symbol_data(df):
    df = df.copy()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['close'].values)
    df['ATR'] = calculate_atr(df)
    df['avg_vol_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
    return df

# ========================
# Strategiat
# ========================
def signal_strategies(df):
    last_close = df['close'].iloc[-1]
    if len(df) < 2:
        prev_high = last_close
        prev_low = last_close
    else:
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        
    signals = []

    # Strategy 1: EMA50>EMA200 + RSI>50
    if not pd.isna(df['EMA50'].iloc[-1]) and not pd.isna(df['EMA200'].iloc[-1]):
        if df['EMA50'].iloc[-1] > df['EMA200'].iloc[-1] and df['RSI'].iloc[-1] > 50 and last_close > prev_high:
            signals.append('Long')
        elif df['EMA50'].iloc[-1] < df['EMA200'].iloc[-1] and df['RSI'].iloc[-1] < 50 and last_close < prev_low:
            signals.append('Short')
        else:
            signals.append('')
    else:
        signals.append('')

    # Strategy 2: EMA20>EMA50 + RSI>50
    if not pd.isna(df['EMA20'].iloc[-1]) and not pd.isna(df['EMA50'].iloc[-1]):
        if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] and df['RSI'].iloc[-1] > 50 and last_close > prev_high:
            signals.append('Long')
        elif df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1] and df['RSI'].iloc[-1] < 50 and last_close < prev_low:
            signals.append('Short')
        else:
            signals.append('')
    else:
        signals.append('')

    # Strategy 3: EMA crossover 20/50
    if not pd.isna(df['EMA20'].iloc[-1]) and not pd.isna(df['EMA50'].iloc[-1]):
        if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] and last_close > prev_high:
            signals.append('Long')
        elif df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1] and last_close < prev_low:
            signals.append('Short')
        else:
            signals.append('')
    else:
        signals.append('')

    # Strategy 4: RSI momentum
    if not pd.isna(df['RSI'].iloc[-1]):
        if df['RSI'].iloc[-1] > 60:
            signals.append('Long')
        elif df['RSI'].iloc[-1] < 40:
            signals.append('Short')
        else:
            signals.append('')
    else:
        signals.append('')

    # Strategy 5: ATR breakout
    if not pd.isna(df['ATR'].iloc[-1]):
        if last_close > prev_high and df['ATR'].iloc[-1] > VOLATILITY_MIN:
            signals.append('Long')
        elif last_close < prev_low and df['ATR'].iloc[-1] > VOLATILITY_MIN:
            signals.append('Short')
        else:
            signals.append('')
    else:
        signals.append('')

    return signals

# ========================
# Backtest kronologisesti
# ========================
def run_backtest(symbol_dfs):
    balance = INITIAL_BALANCE
    open_trades = []
    trades = []
    equity_curve = []
    last_signal = {sym: None for sym in symbol_dfs.keys()}

    # Seuraa paras strategia/signal + TP/SL symbolikohtaisesti
    best_per_symbol = {sym: {'strategy': None, 'signal': None, 'tp_price': None, 'sl_price': None, 'reward': -np.inf} 
                       for sym in symbol_dfs.keys()}

    all_times = sorted(set().union(*[df['open_time'] for df in symbol_dfs.values()]))

    for t in all_times:
        for sym, df in symbol_dfs.items():
            if t not in set(df['open_time']):
                continue
            idx = df.index[df['open_time']==t][0]
            window = df.iloc[:idx+1]

            # Päivitä avoimet treidit
            still_open = []
            for trade in open_trades:
                if trade['symbol'] != sym:
                    still_open.append(trade)
                    continue
                row = df.iloc[idx]
                if trade['type']=='Long':
                    if row['low'] <= trade['sl_price']:
                        balance -= trade['risk_amount']
                        trade['exit_price']=trade['sl_price']
                        trade['exit_time']=row['open_time']
                        trade['net_pnl']=-trade['risk_amount']
                        trades.append(trade)
                    elif row['high'] >= trade['tp_price']:
                        balance += trade['reward_amount']
                        trade['exit_price']=trade['tp_price']
                        trade['exit_time']=row['open_time']
                        trade['net_pnl']=trade['reward_amount']
                        trades.append(trade)
                    else:
                        still_open.append(trade)
                else:
                    if row['high']>=trade['sl_price']:
                        balance -= trade['risk_amount']
                        trade['exit_price']=trade['sl_price']
                        trade['exit_time']=row['open_time']
                        trade['net_pnl']=-trade['risk_amount']
                        trades.append(trade)
                    elif row['low']<=trade['tp_price']:
                        balance += trade['reward_amount']
                        trade['exit_price']=trade['tp_price']
                        trade['exit_time']=row['open_time']
                        trade['net_pnl']=trade['reward_amount']
                        trades.append(trade)
                    else:
                        still_open.append(trade)
            open_trades = still_open

            # Uusi signaali
            strat_signals = signal_strategies(window)
            best_signal = None
            best_tp_sl = None
            best_profit = -np.inf
            best_entry = None
            best_pos_size = None
            best_risk_amt = None
            best_reward_amt = None
            best_strategy_index = None

            # Testaa kaikki strategiat
            for strat_idx, sig in enumerate(strat_signals, start=1):
                if sig=='':
                    continue

                # ATR logiikka
                atr_val = window['ATR'].iloc[-1]
                for tp_mult in atr_tp_multipliers:
                    for sl_mult in atr_sl_multipliers:
                        entry = df['open'].iloc[idx]*(1+slippage if sig=='Long' else 1-slippage)
                        stop = atr_val*sl_mult
                        if stop <= 0:
                            continue
                        risk_amt = balance*risk_per_trade
                        pos_size = risk_amt/stop
                        tp_price = entry+atr_val*tp_mult if sig=='Long' else entry-atr_val*tp_mult
                        sl_price = entry-atr_val*sl_mult if sig=='Long' else entry+atr_val*sl_mult
                        reward_amt = pos_size*(tp_price-entry if sig=='Long' else entry-tp_price)
                        if reward_amt>best_profit:
                            best_profit=reward_amt
                            best_signal=sig
                            best_tp_sl=(tp_price, sl_price)
                            best_entry=entry
                            best_pos_size=pos_size
                            best_risk_amt=risk_amt
                            best_reward_amt=reward_amt
                            best_strategy_index=strat_idx

                # Kiinteä TP/SL logiikka
                entry = df['open'].iloc[idx]*(1+slippage if sig=='Long' else 1-slippage)
                for tp_p in fixed_tp:
                    for sl_p in fixed_sl:
                        tp_price = entry*(1+tp_p if sig=='Long' else 1-tp_p)
                        sl_price = entry*(1-sl_p if sig=='Long' else 1+sl_p)
                        price_diff = abs(entry - sl_price)
                        if price_diff <= 0:
                            continue
                        risk_amt = balance*risk_per_trade
                        pos_size = risk_amt/price_diff
                        reward_amt = pos_size*abs(tp_price-entry)
                        if reward_amt>best_profit:
                            best_profit=reward_amt
                            best_signal=sig
                            best_tp_sl=(tp_price, sl_price)
                            best_entry=entry
                            best_pos_size=pos_size
                            best_risk_amt=risk_amt
                            best_reward_amt=reward_amt
                            best_strategy_index=strat_idx

            # Lisää avoin treidi
            if best_signal and best_signal!=last_signal[sym]:
                open_trades.append({
                    'symbol': sym,
                    'type': best_signal,
                    'entry_time': df['open_time'].iloc[idx],
                    'entry_price': best_entry,
                    'tp_price': best_tp_sl[0],
                    'sl_price': best_tp_sl[1],
                    'position_size': best_pos_size,
                    'risk_amount': best_risk_amt,
                    'reward_amount': best_reward_amt,
                    'strategy_index': best_strategy_index
                })
                last_signal[sym]=best_signal

                # Päivitä paras strategia per symbol
                if best_reward_amt > best_per_symbol[sym]['reward']:
                    best_per_symbol[sym].update({
                        'strategy': best_strategy_index,
                        'signal': best_signal,
                        'tp_price': best_tp_sl[0],
                        'sl_price': best_tp_sl[1],
                        'reward': best_reward_amt
                    })

        equity_curve.append((t,balance))
        for sym in symbol_dfs.keys():
            if not any(tr['symbol']==sym for tr in open_trades):
                last_signal[sym]=None

    return trades, equity_curve, balance, best_per_symbol


# ========================
# Pääohjelma
# ========================
def main():
    symbol_dfs={}
    print("Haetaan dataa symboleille...")
    for sym in symbols:
        print(f"Hakee {sym}...")
        df = get_klines_full(sym, interval, "2024-10-01","2025-10-01")
        if len(df)>0:
            df_prepared = prepare_symbol_data(df)
            symbol_dfs[sym]=df_prepared
            print(f"{sym} riviä: {len(df_prepared)}")
        else:
            print(f"{sym}: Ei dataa")

    print("Ajetaan kronologinen backtest 5 strategialla + ATR ja kiinteät TP/SL...")
    # KORJAA TÄMÄ RIVI - LISÄÄ NELJÄS MUUTTUJA
    trades, equity_curve, final_balance, best_strategies = run_backtest(symbol_dfs)
    
    print(f"Loppusaldo: {final_balance:.2f}")
    print(f"Kauppoja yhteensä: {len(trades)}")
    wins = sum(1 for t in trades if t['net_pnl']>0)
    win_rate = wins/len(trades)*100 if trades else 0
    print(f"Voittoprosentti: {win_rate:.2f}%")
    
    # Voit myös tulostaa parhaat strategiat
    print("\nParhaat strategiat symboleittain:")
    for sym, best in best_strategies.items():
        if best['strategy']:
            print(f"{sym}: Strategia {best['strategy']}, Signaali {best['signal']}")

if __name__=="__main__":
    main()

