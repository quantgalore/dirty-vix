# -*- coding: utf-8 -*-
"""
Created in 2024

@author: Quant Galore
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import math

from scipy.stats import norm
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

def black_scholes(option_type, S, K, t, r, q, sigma):
    """
    Calculate the Black-Scholes option price.

    :param option_type: 'call' for call option, 'put' for put option.
    :param S: Current stock price.
    :param K: Strike price.
    :param t: Time to expiration (in years).
    :param r: Risk-free interest rate (annualized).
    :param q: Dividend yield (annualized).
    :param sigma: Stock price volatility (annualized).

    :return: Option price.
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    if option_type == 'call':
        return S * math.exp(-q * t) * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    elif option_type == 'put':
        return K * math.exp(-r * t) * norm.cdf(-d2) - S * math.exp(-q * t) * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be either 'call' or 'put'.")
        
def call_implied_vol(S, K, t, r, option_price):
    q = 0.01
    option_type = "call"

    def f_call(sigma):
        return black_scholes(option_type, S, K, t, r, q, sigma) - option_price

    call_newton_vol = optimize.newton(f_call, x0=0.50, tol=0.05, maxiter=50)
    return call_newton_vol

def put_implied_vol(S, K, t, r, option_price):
    q = 0.01
    option_type = "put"

    def f_put(sigma):
        return black_scholes(option_type, S, K, t, r, q, sigma) - option_price

    put_newton_vol = optimize.newton(f_put, x0=0.50, tol=0.05, maxiter=50)
    return put_newton_vol


polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

trading_dates = calendar.schedule(start_date = "2023-01-01", end_date = (datetime.today()-timedelta(days = 1))).index.strftime("%Y-%m-%d").values

ticker = "TSLA"

index_data_list = []
times = []

# date = trading_dates[1:][-1]
for date in trading_dates[1:]:
    
    try:
        
        start_time = datetime.now()

        underlying_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        underlying_data.index = pd.to_datetime(underlying_data.index, unit="ms", utc=True).tz_convert("America/New_York")
        
        price = underlying_data["c"].iloc[0]
        
        quote_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours = 15, minutes = 55)).value
        close_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours = 16, minutes = 0)).value
    
        ''' Vol Calculation'''
        
        calls = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&contract_type=call&as_of={date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
        calls["days_to_exp"] = (pd.to_datetime(calls["expiration_date"]) - pd.to_datetime(date)).dt.days
        calls = calls[calls["days_to_exp"] >= 5].copy()
        
        nearest_exp_date = calls["expiration_date"].iloc[0]
        
        calls = calls[calls["expiration_date"] == nearest_exp_date].copy()
        calls["distance_from_price"] = abs(round(((calls["strike_price"] - price) / price)*100, 2))
        
        atm_call = calls.nsmallest(1, "distance_from_price")
        
        call_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{atm_call['ticker'].iloc[0]}?timestamp.gte={quote_timestamp}&timestamp.lt={close_timestamp}&order=desc&limit=100&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
        call_quotes.index = pd.to_datetime(call_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
        call_quotes["mid_price"] = round((call_quotes["bid_price"] + call_quotes["ask_price"]) / 2, 2)
        
        # 604800 seconds -> 7 days
        time_to_expiration = (604800 / 86400) / 252
        
        atm_call_vol = call_implied_vol(S=price, K=atm_call["strike_price"].iloc[0], t=time_to_expiration, r=.045, option_price=call_quotes["mid_price"].iloc[0])

        ###  Puts ###
        
        puts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&contract_type=put&as_of={date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
        puts["days_to_exp"] = (pd.to_datetime(puts["expiration_date"]) - pd.to_datetime(date)).dt.days
        
        puts = puts[puts["expiration_date"] == nearest_exp_date].copy()
        puts["distance_from_price"] = abs(round(((price - puts["strike_price"]) / puts["strike_price"])*100, 2))
        
        atm_put = puts.nsmallest(1, "distance_from_price")
        
        put_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{atm_put['ticker'].iloc[0]}?timestamp.gte={quote_timestamp}&timestamp.lt={close_timestamp}&order=desc&limit=100&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
        put_quotes.index = pd.to_datetime(put_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
        put_quotes["mid_price"] = round((put_quotes["bid_price"] + put_quotes["ask_price"]) / 2, 2)
        
        atm_put_vol = put_implied_vol(S=price, K=atm_put["strike_price"].iloc[0], t=time_to_expiration, r=.045, option_price=put_quotes["mid_price"].iloc[0])
                
        atm_vol = round(((atm_call_vol + atm_put_vol) / 2)*100, 2)

        expected_move = (round((atm_vol / np.sqrt(252)), 2))

        index_data = pd.DataFrame([{"date": date, "v_index": expected_move}])
        index_data_list.append(index_data)
         
        end_time = datetime.now()
        seconds_to_complete = (end_time - start_time).total_seconds()
        times.append(seconds_to_complete)
        iteration = round((np.where(trading_dates==date)[0][0]/len(trading_dates))*100,2)
        iterations_remaining = len(trading_dates) - np.where(trading_dates==date)[0][0]
        average_time_to_complete = np.mean(times)
        estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
        time_remaining = estimated_completion_time - datetime.now()
        print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")

    except Exception as data_error:
        print(data_error)
        continue

full_index_data = pd.concat(index_data_list)

vix_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/I:VIX/range/1/day/{full_index_data['date'].iloc[0]}/{full_index_data['date'].iloc[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
vix_data.index = pd.to_datetime(vix_data.index, unit="ms", utc=True).tz_convert("America/New_York")
vix_data["date"] = vix_data.index.strftime("%Y-%m-%d")
vix_data["expected_move"] = round(vix_data["c"] / np.sqrt(252), 2)

index_and_vix = pd.merge(full_index_data, vix_data[["date","expected_move"]], on = "date")

# Min-Max Calculation

index_min = index_and_vix['v_index'].min()
index_max = index_and_vix['v_index'].max()
index_and_vix['v_index_normalized'] = (index_and_vix['v_index'] - index_min) / (index_max - index_min)

vix_min = index_and_vix['expected_move'].min()
vix_max = index_and_vix['expected_move'].max()
index_and_vix['vix_normalized'] = (index_and_vix['expected_move'] - vix_min) / (vix_max - vix_min)

plt.figure(dpi=200)
plt.xticks(rotation=45)
plt.plot(pd.to_datetime(index_and_vix["date"]), index_and_vix["v_index_normalized"])
plt.plot(pd.to_datetime(index_and_vix["date"]), index_and_vix["vix_normalized"])
plt.legend(['TSLA Expected Move', "VIX Expected Move"])
plt.xlabel("Date")
plt.ylabel("Expected Move")
plt.show()

underlying_stock_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{full_index_data['date'].iloc[0]}/{full_index_data['date'].iloc[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
underlying_stock_data.index = pd.to_datetime(underlying_stock_data.index, unit="ms", utc=True).tz_convert("America/New_York")
underlying_stock_data["date"] = underlying_stock_data.index.strftime("%Y-%m-%d")

index_and_underlying = pd.merge(index_and_vix[["v_index_normalized", "vix_normalized", "date"]], underlying_stock_data[["date","c"]], on = "date")

index_and_underlying["underlying_returns"] = round(index_and_underlying["c"].pct_change()*100, 2)
index_and_underlying["abs_underlying_returns"] = abs(round(index_and_underlying["c"].pct_change()*100, 2))
index_and_underlying["next_underlying_returns"] = index_and_underlying["underlying_returns"].shift(-1)
index_and_underlying["abs_next_underlying_returns"] = index_and_underlying["abs_underlying_returns"].shift(-1)
index_and_underlying = index_and_underlying.dropna()

index_and_underlying["spread"] = index_and_underlying["v_index_normalized"] - index_and_underlying["vix_normalized"]

v_index = index_and_underlying["spread"]
abs_next_underlying_returns = index_and_underlying["abs_next_underlying_returns"]

# Calculate the line of best fit
coefficients = np.polyfit(v_index, abs_next_underlying_returns, 1)
polynomial = np.poly1d(coefficients)
line_of_best_fit = polynomial(v_index)

plt.figure(dpi=200)
plt.xticks(rotation=45)
plt.scatter(index_and_underlying["spread"], index_and_underlying["abs_next_underlying_returns"])
plt.plot(v_index, line_of_best_fit, color='red', label='Line of best fit')
plt.xlabel(f"{ticker} Expectations - Market Expectations")
plt.ylabel("Realized Next Day Move")
plt.show()

###

# Binning into averages to make the crowded scatterplot more intuitive

num_bins = 10
bins = np.linspace(index_and_underlying["spread"].min(), index_and_underlying["spread"].max(), num_bins + 1)

# Calculate the average abs_next_underlying_returns for each bin
index_and_underlying['binned_spread'] = pd.cut(index_and_underlying['spread'], bins)
binned_data = index_and_underlying.groupby('binned_spread')['abs_next_underlying_returns'].mean().reset_index()

# Compute bin centers for plotting
bin_centers = [interval.mid for interval in binned_data['binned_spread']]

# Plotting
plt.figure(dpi=200)
plt.xticks(rotation=45)
# plt.scatter(index_and_underlying["spread"], index_and_underlying["abs_next_underlying_returns"], alpha=0.5, label='Data Points')
plt.plot(v_index, line_of_best_fit, color='red', label='Line of best fit')

# Plot the binned averages
plt.scatter(bin_centers, binned_data['abs_next_underlying_returns'], color='green', label='Binned Averages', s=50)

plt.xlabel(f"{ticker} Expectations - Market Expectations")
plt.ylabel("Realized Next Day Move")
plt.legend()
plt.show()