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



class stock_direction_rf():
    
    def movingaverage(x, window):
    x[f'SMA'+str(window)]=x['Close'].rolling(window=window).mean()
    return x
    
    def momentum(df,stock, day):
        df[str(stock)+'_momentum']=df['daily_close_change'].apply(lambda x:1 if x>=0 else 0)
        if day>1:
            df[str(stock)+str(day)+'_day_momentum']=df[str(stock)+'_momentum'].rolling(window=day).sum()
        elif day<1:
            print('The window cannot be negative')
        return df

    def __init__(self, stock, start, end=date.today()
                 , test=None):
        self.stock = stock
        self.start = start
        self.end= end
        self.test=test
        
    def get_data(self):
        df= yf.download(self.stock, 
                start=self.start, 
                end=self.end, 
                progress=False)
        
        return df
        
    
    
    def clean_and_feature(self, df):
        
        
        #generate the percentage change of each day close price
        df['daily_close_change']=df['Close'].diff()/df['Close']
        #generate the direction of the stock move to each day
        momentum(df,self.stock,1)
        #shift the momentum in a negative direction for predicting the next day direction with today information
        df[str(self.stock)+'_momentum']=df[str(self.stock)+'_momentum'].shift(-1)
        #generate the SMA rate each day
        movingaverage(df, 10)
        movingaverage(df, 20)
        movingaverage(df, 50)
        #for the test day in the next day
        if self.test==None:
            next_day_test= df.iloc[-1].dropna().values.reshape(1,len(df.values[-1])-1)
        else:
            next_day_test=self.test
        df.dropna(inplace=True)
        return df, next_day_test
        
    def split_train_data(self, df):
        #Splitting the data into X and y, where df is the X data
        y=df[str(self.stock)+'_momentum']
        X=df.drop([str(self.stock)+'_momentum'], axis=1)
        #Scale the data with Min Max Scaler
        scaler=preprocessing.MinMaxScaler()
        scaler.fit(X)
        X=scaler.transform(X)
        
        return X, y
    
    def time_series_CV(self,X,y):
        tscv = TimeSeriesSplit(n_splits=5)
        i = 1
        score = []
        #cross validation for random forest to get the optimal hyperparameter
        for train_index, test_index in tscv.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test =  X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            criterion=['gini', 'entropy']
            for ne in np.linspace(50, 100, 6):
                for md in np.linspace(20, 40, 5):
                    for msl in np.linspace(30, 100, 8):
                        for c in criterion:
                            rfc = RandomForestClassifier(
                                criterion=c,
                                n_estimators=int(ne),
                                max_depth=int(md),
                                min_samples_leaf=int(msl))
                            rfc.fit(X_train, y_train)
                            score.append([i,
                                            c, 
                                            ne,
                                            md, 
                                            msl, 
                                            rfc.score(X_test, y_test)])
            i += 1
        
        #find the best parameter
        max_score=0
        parameter=[]
        for i in range(len(score)):
            if score[i][5]>=max_score:
                max_score=score[i][5]
                parameter=score[i]
        return score, parameter
        
    
    def predict(self, X, y, parameter, test):
        #Use the optimal hyperparameter obatained and train the all previous data except today data and use today data to
        #predict the stock direction in next trading date
        
        rfc = RandomForestClassifier(
                                criterion=parameter[1],
                                n_estimators=int(parameter[2]),
                                max_depth=int(parameter[3]),
                                min_samples_leaf=int(parameter[4]))
        
        rfc=rfc.fit(X,y)
        y_pred=rfc.predict(test)
        result=None
        if int(y_pred) ==1:
            result=print(str(self.stock)+ ' will close at a higher price in the next trading day')
        else:
            result=print(str(self.stock)+' will close at a lower price in the next trading dday')
            
        return y_pred, result
            
    def run_all(self,train=False):
        #setting a default parameter if don't want to train for a long period of time, which allows to skip the tunning part
        default_parameter=[5, 'entropy', 60.0, 40.0, 50.0, 0.615]
        
        df=self.get_data()
        df, test_data=self.clean_and_feature(df)
        X,y=self.split_train_data(df)
        
        if train==False:
            parameter=default_parameter
            y_pred, result = self.predict(X,y, parameter, test_data)
            return result
        else:
            score, parameter=self.time_series_CV(X,y)
            y_pred, result=self.predict(X,y, parameter, test_data)
            return parameter , result