import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ferti:

    _MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, save_path=None, data_path=None):
        self.save_path = save_path or os.path.join(self._MODEL_DIR, 'ferti_model.pkl')
        self.data_path = data_path or os.path.join(self._MODEL_DIR, 'soil_fertilization.csv')
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self._initialize_model()

    def _initialize_model(self):
        if os.path.exists(self.save_path):
            self._load()                  
        else:
            self._train_and_save()        

    def _train_and_save(self):
        df = pd.read_csv(self.data_path)
        df = df[df['Output'].isin([0, 1])]

        X = df.drop(columns='Output')
        y = df['Output']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        self.model.fit(X_train, y_train)
        self._save()

    def _save(self):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, self.save_path)

    def _load(self):
        state = joblib.load(self.save_path)
        self.model  = state['model']
        self.scaler = state['scaler']

    def predict(self, inp):
        return self.model.predict(self.scaler.transform(inp))