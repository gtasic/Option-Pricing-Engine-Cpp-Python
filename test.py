import sys
sys.path.insert(0, "build")  # on ajoute le dossier build au chemin Python

import projet
print("Doc:", projet.__doc__)
print("2+3 =", projet.add(2, 3)) 