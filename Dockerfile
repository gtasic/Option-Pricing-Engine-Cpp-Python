FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y build-essential cmake python3-dev

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN cmake -S . -B build && cmake --build build

# Étape 7: Définir la commande par défaut à exécuter quand on lance un conteneur
# Par exemple, on peut lancer le script qui analyse la performance des modèles.
CMD ["python", "python/analyse_performance.py"]