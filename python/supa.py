import supabase
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

print("Variables d'environnement chargées:")
print(f"SUPABASE_KEY: {os.environ.get('SUPABASE_KEY')}")

# Utiliser les variables d'environnement
supabase_url: str = os.environ.get("SUPABASE_URL")
supabase_key: str = os.environ.get("SUPABASE_KEY")

# Vérifier si les variables sont bien définies
if not supabase_key:
    raise ValueError("SUPABASE_KEY n'est pas définie dans les variables d'environnement")

try:
    client = supabase.create_client(supabase_url, supabase_key)
    response = client.table("assets").select("*").execute()
    print("Connexion réussie!")
    print("Données:", response)

except Exception as e:
    print("Erreur de connexion:", str(e))