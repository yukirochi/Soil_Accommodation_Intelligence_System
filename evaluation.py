import pandas as pd

class Evaluation:
    
    def __init__(self):
        pass
    def evaluate_result(self, model_name, result):
        if model_name == 'soil_moisture':
            result = result[0]
            if result < 0 or result > 100:
                raise ValueError(f"Predicted soil moisture value {result} is out of expected range (0-100).")
            return result
         
        elif model_name == 'soil_fertility':
            result = int(result[0])   # ferti.predict returns a numpy array
            if result not in [0, 1]:
                raise ValueError(f"Predicted soil fertility value {result} is not valid. Expected 0 or 1.")
            return result
        else:
            raise ValueError(f"Model '{model_name}' not recognized.")
    
        