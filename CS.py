import numpy as np

def cuckoo_search(nidos, n_gen, pa, objetivo):
    """
    Implementación del algoritmo Cuckoo Search con seguimiento del mejor nido absoluto.
    
    Retorna:
        - El mejor nido global encontrado.
        - El mejor nido absoluto encontrado durante todo el proceso.
        - La evolución de las utilidades por generación.
    """
    nidos = np.array(nidos, dtype=float)
    n_nidos, n_dimensiones = nidos.shape

    # Inicializar el mejor nido global y absoluto
    mejor_nido_global = nidos[0]
    mejor_utilidad_global = objetivo(mejor_nido_global)
    mejor_nido_absoluto = mejor_nido_global
    mejor_utilidad_absoluta = mejor_utilidad_global

    historia_utilidades = []

    for gen in range(n_gen):
        utilidades_actuales = []

        for i in range(n_nidos):
            step_size = np.random.standard_normal(n_dimensiones) * 0.1
            nuevo_nido = nidos[i] + step_size
            nuevo_nido = np.clip(nuevo_nido, 0, 1)
            nuevo_nido /= nuevo_nido.sum()

            if objetivo(nuevo_nido) > objetivo(nidos[i]):
                nidos[i] = nuevo_nido

        for nido in nidos:
            utilidad_actual = objetivo(nido)
            utilidades_actuales.append(utilidad_actual)
            if utilidad_actual > mejor_utilidad_absoluta:
                mejor_utilidad_absoluta = utilidad_actual
                mejor_nido_absoluto = nido

        historia_utilidades.append(utilidades_actuales)

        eliminar_indices = np.random.rand(n_nidos) < pa
        nuevos_nidos = np.random.dirichlet(np.ones(n_dimensiones), sum(eliminar_indices))
        nidos[eliminar_indices] = nuevos_nidos

        mejor_nido_generacion = max(nidos, key=objetivo)
        mejor_utilidad_generacion = objetivo(mejor_nido_generacion)
        if mejor_utilidad_generacion > mejor_utilidad_global:
            mejor_utilidad_global = mejor_utilidad_generacion
            mejor_nido_global = mejor_nido_generacion

    return mejor_nido_global, mejor_utilidad_global, mejor_nido_absoluto, mejor_utilidad_absoluta, historia_utilidades