import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import supabase
from sklearn.neighbors import NearestNeighbors

# Dans ce programme, nous allons à partir d'un jeu de données CSV, entraîner un modèle de classification pour calculer le sigma des options
# Sigma dépend de plusieurs facteurs comme la maturity et le strike price. 


supabase_url: str = "https://wehzchguwwpopqpzyvpc.supabase.co"
supabase_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndlaHpjaGd1d3dwb3BxcHp5dnBjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc3MTE1OTQsImV4cCI6MjA3MzI4NzU5NH0.hK5fX9YowK83jx8MAzzNm5enBdvgU2XC4shZreACO2s"



supabase_client = supabase.create_client(supabase_url, supabase_key)


df = pd.DataFrame(supabase_client.table("vol_surfaces").select("*").execute().data)



def volatility_calibre(df_volatility) : 
    # Préparation des données
    X = df_volatility[['tenor', 'moneyness']]  
    y = df_volatility['iv']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Évaluation du modèle
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    return model  

model_volatility = volatility_calibre(df)

def predict_sigma(model, T, strike, S0, r, q) :
    input_data = np.array([[T, strike, S0, r, q]])
    predicted_sigma = model.predict(input_data)
    return predicted_sigma[0]

# Exemple d'utilisation
T_example = 0.5  # Maturity
strike_example = 100  # Strike price
S0_example = 105  # Current stock price
r_example = 0.04  # Risk-free rate
predicted_sigma = predict_sigma(model_volatility, T_example, strike_example, S0_example, r_example)
print(f'Predicted Sigma: {predicted_sigma}')



