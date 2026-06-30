import pandas as pd
from sklearn.linear_model import  SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

class sais_model:
    
    def __init__(self,data_path='soil_moist.csv',save_path='sais_model_state.pkl'):
        
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

        X = df.drop(columns='future_moisture')
        y = df['future_moisture']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        self.initial_r2 = r2_score(y_test, y_pred)
        self.initial_rmse = root_mean_squared_error(y_test, y_pred)

    def predict(self, inp):
        x = self.scaler.transform(inp)
        return self.model.predict(x)
    
    def train(self,df):
        
        x = self.scalar.transform(df.drop(columns='future_moisture'))
        y = df['future_moisture']
        
        self.model.partial_fit(x,y)
        
        print("partial_fit done")