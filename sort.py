import os
import ssl
import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from selenium import webdriver

ssl._create_default_https_context = ssl._create_unverified_context
dir_path = os.path.dirname(os.path.realpath(__file__))

# Constents

MIN_VOLUME = 1000000
Y5_SLOPE = 10
MAXY_SLOPE = 0
MAX_TRAILINGPE = 30
MAX_PEG = 3
MIN_EPS = 5

def dt64_to_float(dt64):
    """Converts numpy.datetime64 to year as float.

    Rounded to days

    Parameters
    ----------
    dt64 : np.datetime64 or np.ndarray(dtype='datetime64[X]')
        date data

    Returns
    -------
    float or np.ndarray(dtype=float)
        Year in floating point representation
    """

    year = dt64.astype('M8[Y]')
    # print('year:', year)
    days = (dt64 - year).astype('timedelta64[D]')
    # print('days:', days)
    year_next = year + np.timedelta64(1, 'Y')
    # print('year_next:', year_next)
    days_of_year = (year_next.astype('M8[D]') - year.astype('M8[D]')
                    ).astype('timedelta64[D]')
    # print('days_of_year:', days_of_year)
    dt_float = 1970 + year.astype(float) + days / (days_of_year)
    # print('dt_float:', dt_float)
    return dt_float

def sortInfo(tickers):

    output_symbols = []

    for tick in tickers:
        try:
            print("Evaluating: " + tick)
            # Get the ticker and history
            ticker = yf.Ticker(tick)
            eps = ticker.info['trailingEps']
            peg = ticker.info['pegRatio']
            volume = ticker.info['volume']
            history5Y = ticker.history("5y")
            historyMax = ticker.history("max")

            # Split the list into an x and y to use later
            sf5 = history5Y["Close"]
            sfMax = historyMax["Close"]
            df5 = pd.DataFrame({'Date':sf5.index, 'Values':sf5.values})
            dfMax = pd.DataFrame({'Date':sfMax.index, 'Values':sfMax.values})

            x5 = df5['Date'].tolist()
            y5 = df5['Values'].tolist()
            xMax = dfMax['Date'].tolist()
            yMax = dfMax['Values'].tolist()

            x5Format = dt64_to_float(df5['Date'].to_numpy())
            xMaxFormat = dt64_to_float(dfMax['Date'].to_numpy())
            trailingPE = ticker.info['trailingPE']
            forwardPE = ticker.info['forwardPE']

            # Create a line of best fit
            a, b = np.polyfit(x5Format, y5, 1)
            aMax, bMax = np.polyfit(xMaxFormat, yMax, 1)

            # If the forward PE is lower than the trailing PE, that means the analysts are expecting earnings to increase
            if a > Y5_SLOPE and aMax > MAXY_SLOPE and (trailingPE < MAX_TRAILINGPE and trailingPE > 0) and forwardPE < trailingPE and eps > MIN_EPS and (peg < MAX_PEG and peg > 0) and volume > MIN_VOLUME:
                #output_symbols.append(tick + " TrailingPE: " + str(trailingPE) + " ForwardPE: " + str(forwardPE) + " eps: " + str(eps) + " peg: " + str(peg))
                output_symbols.append(tick)

            # Plot the data
            #plt.plot(x5, y5)
            #plt.plot(x5, a*x5Format+b)
            #plt.title("Slope 5Y: " + str(round(a)))
            #plt.ylabel('Price($)')
            #plt.xlabel('Date', rotation=0)
            #plt.savefig(dir_path + r'\\Outputs\\' + tick + '.png')
            #plt.clf()
        except:
            continue
    
    return output_symbols


# There are 2 tables on the Wikipedia page
# we want the first table
payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
first_table = payload[0]
second_table = payload[1]

df = first_table
df.head()

# Get all the symbols
symbols = df['Symbol'].values.tolist()
final_output = []

# Split the list into equal parts
symbol_chunck_one = symbols[0:50]
symbol_chunck_two = symbols[50:100]
symbol_chunck_three = symbols[100:150]
symbol_chunck_four = symbols[150:200]
symbol_chunck_five = symbols[200:250]
symbol_chunck_six = symbols[250:300]
symbol_chunck_seven = symbols[300:350]
symbol_chunck_eight = symbols[350:400]
symbol_chunck_nine = symbols[400:450]
symbol_chunck_ten = symbols[450:505]

pool = ThreadPool(processes=10)

async_result_one = pool.apply_async(sortInfo, (symbol_chunck_one,))
async_result_two = pool.apply_async(sortInfo, (symbol_chunck_two,))
async_result_three = pool.apply_async(sortInfo, (symbol_chunck_three,))
async_result_four = pool.apply_async(sortInfo, (symbol_chunck_four,))
async_result_five = pool.apply_async(sortInfo, (symbol_chunck_five,))
async_result_six = pool.apply_async(sortInfo, (symbol_chunck_six,))
async_result_seven = pool.apply_async(sortInfo, (symbol_chunck_seven,))
async_result_eight = pool.apply_async(sortInfo, (symbol_chunck_eight,))
async_result_nine = pool.apply_async(sortInfo, (symbol_chunck_nine,))
async_result_ten = pool.apply_async(sortInfo, (symbol_chunck_ten,))

return_val_one = async_result_one.get()
return_val_two = async_result_two.get()
return_val_three = async_result_three.get()
return_val_four = async_result_four.get()
return_val_five = async_result_five.get()
return_val_six = async_result_six.get()
return_val_seven = async_result_seven.get()
return_val_eight = async_result_eight.get()
return_val_nine = async_result_nine.get()
return_val_ten = async_result_ten.get()

# Combine the lists into one
found_tickers = return_val_one + return_val_two + return_val_three + return_val_four + return_val_five + return_val_six + return_val_seven + return_val_eight + return_val_nine + return_val_ten

print("\nFound Stocks Meeting Criteria:")
browser = webdriver.Firefox(executable_path=r'C:\Users\jones\AppData\Roaming\Python\Python37\geckodriver.exe')

tab_count = 1
for x in found_tickers:
    print(x)

    if tab_count == 1:
        browser.get(r"https://finance.yahoo.com/quote/" + x + "/")
    else:
        browser.execute_script("window.open('about:blank', " + "'tab" + str(tab_count) + "');")
        browser.switch_to.window(browser.window_handles[-1])
        browser.get(r"https://finance.yahoo.com/quote/" + x + "/")
    tab_count = tab_count + 1
    time.sleep(5)