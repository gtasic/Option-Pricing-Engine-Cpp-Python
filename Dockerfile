# === Étape 1: Image de base ===
FROM python:3.12-slim-bookworm

# === Étape 2: Installation des Outils Système ===
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    gfortran \
    libblas-dev \
    liblapack-dev

# === Étape 3: Configuration de l'Environnement ===
WORKDIR /app

# === Étape 4: Installation des Dépendances Python ===
# On copie d'abord le fichier des dépendances.
COPY requirements.txt .

# On installe TORCH en premier, en utilisant son dépôt officiel.

# Ensuite, on installe TOUTES les autres librairies du fichier.
# pip ignorera intelligemment la ligne de torch car il est déjà installé.
RUN pip install --no-cache-dir -r requirements.txt

# === Étape 5: Copie du Code du Projet ===
COPY . .

# === Étape 6: Compilation du Module C++ ===
RUN cmake -S . -B build && cmake --build build

# === Étape 7: Commande d'Exécution par Défaut ===
CMD ["python", "python/analyse_performance.py"]