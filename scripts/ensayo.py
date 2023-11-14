

mi_diccionario = {'b': 2, 'a': 1, 'c': 3, 'd': 5, 'e': 4}

# Paso 1: Ordenar el diccionario por valores en orden ascendente
diccionario_ordenado_por_valor = dict(sorted(mi_diccionario.items(), key=lambda item: item[1]))

# Paso 2: Obtener los dos primeros y los dos últimos elementos
dos_primeros_elementos = list(diccionario_ordenado_por_valor.items())[:2]
dos_ultimos_elementos = list(diccionario_ordenado_por_valor.items())[-2:]

# Paso 3: Encontrar los dos elementos consecutivos con la menor diferencia
items_ordenados = list(diccionario_ordenado_por_valor.items())
diferencias = [(items_ordenados[i], items_ordenados[i + 1]) for i in range(len(items_ordenados) - 1)]
min_diferencia = min(diferencias, key=lambda x: abs(x[0][1] - x[1][1]))

print("Dos primeros elementos:", dict(dos_primeros_elementos))
print("Dos últimos elementos:", dict(dos_ultimos_elementos))
print(min_diferencia)
print("Dos elementos consecutivos con la menor diferencia:", dict(min_diferencia))

