
import plotly.graph_objects as go
import numpy as np

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Création du graphique 3D
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.update_layout(scene=dict(zaxis=dict(range=[-1, 1])))

# Affichage
fig.show()
#Sauver le fichier en format HTML
fig.write_html("surface_plot.html")

