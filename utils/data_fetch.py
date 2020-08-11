## Imports
import os
import sys
import json
import requests
from argparse import ArgumentParser
from .stock_file import get_stocks_from_file
from .nasdaq import get_nasdaq_traded_stocks, get_nasdaq_listed_stocks
from .fmp import FmpCompany
from .yahoo import YahooCompany
from .graham import graham_filter

## Constants

DATA_FOLDER = "../data/"


## Data Fetch

## finnhub
fin_token = "bsomk7frh5r8ktik10l0"
symb = "AAPL"
start_datetime = "1572651390"
end_datetime = "1572910590"

def finnhub_hist(sym=symb, start_datetime=start_datetime, end_datetime=end_datetime):
    r = requests.get("".join(["https://finnhub.io/api/v1/stock/candle?symbol=", sym, "&resolution=1&from=", start_datetime , "&to=", end_datetime , "&token=", fin_token]))
    return r.json()

## Stocklist

def fetch_symbol_data(symbol):
    """
    Retrieve the data for the given symbol from Yahoo and FMP.
    """
    fmp_company = FmpCompany(symbol)
    yahoo_company = YahooCompany(symbol)
    return {'symbol': symbol,
            'rating': fmp_company.rating,
            'share-price': yahoo_company.share_price,
            'total-debt': yahoo_company.total_debt,
            'total-debt-equity': yahoo_company.total_debt_equity,
            'pe-trailing': yahoo_company.pe_trailing,
            'pe-forward': yahoo_company.pe_forward,
            'p-bv': yahoo_company.p_bv,
            'dividend-forward': yahoo_company.dividend_forward,
            'current-ratio': yahoo_company.current_ratio,
            'latest-net-income': yahoo_company.net_income,
            'net-income': yahoo_company.get_net_income_series(),
            'total-revenue': yahoo_company.revenue,
            'gross-profit': yahoo_company.gross_profit,
            'total-assets': yahoo_company.total_assets}

def pull(symbol):
    """
    Like fetch(), but also stores the result on the file system.
    """
    company = fetch_symbol_data(symbol)
    filename = os.path.join(DATA_FOLDER, symbol + '.json')
    with open(filename, 'w') as fp:
        json.dump(company, fp)
    return company

def load(symbol):
    """
    Like pull(), but uses already stored version from file system,
    if available.
    """
    filename = os.path.join(DATA_FOLDER, symbol + '.json')
    if not os.path.isfile(filename):
        return pull(symbol)
    with open(filename) as fp:
        return json.load(fp)

def dir(source):
    if source == 'nasdaq-traded':
        stock_list = get_nasdaq_traded_stocks()
    elif source == 'nasdaq-listed':
        stock_list = get_nasdaq_listed_stocks()
    else:
        print('unknown source: ' + repr(source))
    for l in stock_list:
        print(l)
    return stock_list

def pull_action(symbols, filenames, force = False):
    for filename in filenames:
        try:
            symbols += get_stocks_from_file(filename)
        except OSError as e:
            parser.error(e)
    for symbol in symbols:
        if force:
            print(" Fetching fundamental data for " + symbol)
            pull(symbol)
        else:
            print(" Fundamental data for {} is cached".format(symbol))
            load(symbol)


def graham(symbols, filenames, force = False, verbose = 0):
    dump_successful = True if verbose >= 1 else False
    dump_failed = True if verbose >= 2 else False

    symbols = symbols
    for filename in filenames:
        try:
            symbols += get_stocks_from_file(filename)
        except OSError as e:
            print(e)

    for symbol in symbols:
        if force:
            company = pull(symbol)
        else:
            company = load(symbol)
        graham_filter(company,
                      dump_successful=dump_successful,
                      dump_failed=dump_failed)