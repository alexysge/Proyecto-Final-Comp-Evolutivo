import numpy as np
from CS import cuckoo_search

# Definición de datos específicos

np.random.seed(7) # Semilla para repetir experimentos
n_assets = 3  # Número de activos en el portafolio
returns = np.array([0.10, 0.15, 0.12])  # Rentabilidad esperada de cada activo
cov_matrix = np.array([
    [0.05, 0.02, 0.01],
    [0.02, 0.06, 0.03],
    [0.01, 0.03, 0.04]
])  # Matriz de covarianza

# Parámetros de Cuckoo Search
n_nidos = 10
n_gen = 100
pa = 0.30  # Tasa de descubrimiento
lambda_risk = 0.5  # Parámetro de aversión al riesgo

# Función de utilidad (objetivo a maximizar)
def utilidad(weights):
    rend = np.dot(weights, returns)  # Calcula la rentabilidad esperada
    riesgo = np.dot(weights.T, np.dot(cov_matrix, weights))  # Calcula el riesgo 
    return rend - lambda_risk * riesgo  # Función de utilidad

# Inicialización dinámica de nidos (portafolios iniciales)
nidos = np.random.dirichlet(np.ones(n_assets), size=n_nidos)

# Ejecución de Cuckoo Search
mejor_portafolio_global, mejor_utilidad_global, mejor_portafolio_absoluto, mejor_utilidad_absoluta, historia_utilidades = cuckoo_search(nidos, n_gen, pa, utilidad)

# Resultados
print("------Parámetros utilizados------")
print("Número de nidos:", n_nidos)
print("Generaciones:", n_gen)
print("Tasa de descubrimiento:", pa)
print("Mejor asignación de pesos (global):", mejor_portafolio_global)
print("Utilidad del portafolio óptimo (global):", mejor_utilidad_global)
print("Mejor asignación de pesos (absoluto):", mejor_portafolio_absoluto)
print("Utilidad del portafolio óptimo (absoluto):", mejor_utilidad_absoluta)
print("------------")
print("Evolución de utilidades por generación:")
for gen, utilidades in enumerate(historia_utilidades):
    print(f"Generación {gen + 1}: {utilidades}")
