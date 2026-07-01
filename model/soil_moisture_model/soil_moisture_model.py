import pandas as pd
from sklearn.linear_model import  SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib
import os

class soily:
    
    def __init__(self,data_path='soil_moist.csv',save_path='soily.pkl'):
        
        self.save_path = save_path
        
        if os.path.exists(self.save_path):
            self.load_model()
        else: 
            self.scaler = StandardScaler()
            self.model = SGDRegressor(
                loss='squared_error',
                penalty=None,
                learning_rate='constant',
                eta0=0.01,
                max_iter=50000,
                tol=1e-8,
                random_state=42
            )
            
            self._initalize_base_model(data_path)
    
    def _initalize_base_model(self, data_path):
        
        df = pd.read_csv(data_path)

        X = df[['hour_sin','hour_cos','soil_temp','rain_fall','prev_5','prev_4','prev_3','prev_2','prev_1']]
        y = df['future_moisture']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        
        # y_pred = self.model.predict(X_test_scaled)
        # # self.initial_r2 = r2_score(y_test, y_pred)
        # # self.initial_rmse = root_mean_squared_error(y_test, y_pred)

    def predict(self, inp):
        x = self.scaler.transform(inp)
        return self.model.predict(x)
    
    def train(self,df):
        
        x = df.drop(columns='future_moisture')
        y = df['future_moisture']
        
        x_scaled = self.scaler.transform(x)
    
        self.model.partial_fit(x_scaled,y)
        
        print("partial_fit done. Saving updated model...")
        
        self.save_model()
        
        
    def save_model(self):
        
        state = {
            'model':self.model,
            'scaler':self.scaler
        }
        
        joblib.dump(state, self.save_path)
        
        print(f"Model state successfully saved to {self.save_path}")
    
    def load_model(self):
        state = joblib.load(self.save_path)
        self.model = state['model']
        self.scaler = state['scaler']
        print(f"Successfully loaded previous model state from {self.save_path}")
        
    def current_performance(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        return r2, rmse