import openai


picture = "/png/greeks_vs_mid_price.png"


openai.api_key = "sk-..."
openai.Model.list()
client = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": f"Mets-toi dans la peau d'un expert en finance de marché et reporting, tu dois m'aider à "
        "analyser des résultats de pricing d'options. Je vais te fournir des métriques d'erreur pour trois modèles de pricing "
        "(Black-Scholes, Monte Carlo, Cox-Ross-Rubinstein) ainsi que des visualisations. Aide moi à interpréter ces résultats "
        "et à identifier les points forts et faibles de chaque modèle. Je vais également te donner une série d'options d AAPL "
        "avec toutes leurs caractéristiques (Greeks) de manière journalière il faudra que tu me dises ce que cela veut dire du "
        "point de vue du marché. Génère moi un rapport d'analyse détaillé en français sous forme de pdf avec les graphiques que "
        "je vais te joindre. Les graphiques que tu dois analyser sont dans les fichiers png dans le dossier ainsi que le csv "
        "data.csv de 15 lignes, appuie toi sur {picture} pour analyser les résultats."},
    ]
)   


if __name__== "__main__":
    print(client.choices[0].message.content)
