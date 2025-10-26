import openai
import supabase
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import base64
from dotenv import load_dotenv
load_dotenv()
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("La clé d'API OpenAI n'est pas définie.")
except Exception as e:
    print(f"Erreur de configuration : {e}")
    exit()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_API_KEY")
if supabase_key is None:
    print("Erreur : La clé d'API Supabase n'est pas définie.")
    exit()
supabase_client = supabase.create_client(supabase_url, supabase_key)


# --- 4. Préparation des données ---
try:
    portfolio_options = supabase_client.table("portfolio_options").select("*").execute().data
    df_options = pd.DataFrame(portfolio_options)

    portfolio_pnl = supabase_client.table("daily_portfolio_pnl").select("*").execute().data
    df_portfolio = pd.DataFrame(portfolio_pnl)


except Exception as e:
    print(f"Erreur lors de la récupération des données : {e}")
    exit()

# --- 5. Appel à l'API GPT-4 Vision ---
# On construit le message avec le texte ET les images
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """Mets-toi dans la peau d'un expert en reporting pour une table de trading sur dérivés actions.
                
                Rédige un rapport d'analyse hebdomadaire en français, au format **Markdown**. Le rapport doit être structuré, clair et professionnel.
                
                **Partie 1 : Analyse de la Performance des Modèles de Pricing**
                Analyse les deux graphiques ci-joints :
                1.  `greeks_vs_mid_price.png` : Valide la cohérence de nos calculs de Greeks. Commente chaque sous-graphique.
                2.  `error_boxplot_by_maturity.png` : Analyse l'erreur de nos modèles (BS, MC, CRR) en fonction de la maturité. Identifie les biais et les forces/faiblesses de chaque modèle.
                
                **Partie 2 : Suivi du Portefeuille de Delta-Hedging**
                Voici l'état actuel de notre portefeuille d'options et sa performance. Analyse ces données pour commenter la stratégie de couverture, le P&L et les risques actuels.
                
                Données des positions d'options :
                {}
                
                Données de performance du portefeuille :
                {}
                
                Conclus le rapport avec des recommandations pour la semaine à venir.
                """.format(df_options.to_string(), df_portfolio.to_string())
            }
         
            
           
        ]
    }
]

try:
    print("Envoi de la requête à l'API OpenAI...")
    client = openai.ChatCompletion.create(
        model="gpt-4-vision-preview", 
        messages=messages,
        max_tokens=4096 
    )
    report_content_md = client.choices[0].message.content
    print("Rapport généré avec succès !")
    print(report_content_md)

    with open("rapport_hebdomadaire.md", "w", encoding="utf-8") as f:
        f.write(report_content_md)
    print("Rapport sauvegardé en rapport_hebdomadaire.md")

except Exception as e:
    print(f"Une erreur est survenue lors de l'appel à l'API : {e}")