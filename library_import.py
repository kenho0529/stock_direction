import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import yfinance as yf
from yahoofinancials import YahooFinancials
import talib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier