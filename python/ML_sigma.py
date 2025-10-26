import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import supabase
from sklearn.neighbors import NearestNeighbors

# Dans ce programme, nous allons à partir d'un jeu de données CSV, entraîner un modèle de classification pour calculer le sigma des options
# Sigma dépend de plusieurs facteurs comme la maturity et le strike price. 

import os
from dotenv import load_dotenv
load_dotenv()
supabase_url  = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")


supabase_client = supabase.create_client(supabase_url, supabase_key)


df = pd.DataFrame(supabase_client.table("vol_surfaces").select("*").execute().data)



def volatility_calibre(df_volatility) : 
    X = df_volatility[['tenor', 'moneyness']]  
    y = df_volatility['iv']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)  #Il nous faudra prendre un autre modèle plus adapté à la régression
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    return model  

model_volatility = volatility_calibre(df)

def predict_sigma(model, T, strike, S0, r, q) :
    input_data = np.array([[T, strike, S0, r, q]])
    predicted_sigma = model.predict(input_data)
    return predicted_sigma[0]

T_example = 0.5  
strike_example = 100  
S0_example = 105  
r_example = 0.04  
predicted_sigma = predict_sigma(model_volatility, T_example, strike_example, S0_example, r_example)
print(f'Predicted Sigma: {predicted_sigma}')



