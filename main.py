import numpy as np
from pycparser import parse_file
import re
import os
import random

# Función para seleccionar un archivo aleatorio de una carpeta
def seleccionar_archivo_aleatorio(carpeta):
    archivos = os.listdir(carpeta)
    archivo_aleatorio = random.choice(archivos)
    return os.path.join(carpeta, archivo_aleatorio)

# Ejemplo de uso de la función para seleccionar archivos aleatorios
# carpeta = r"C:\Users\jordi\Escritorio\DetectorPlagio\PlagiarismDetector\Dataset_C\A2016\Z1\Z1"
# archivo_seleccionado_1 = seleccionar_archivo_aleatorio(carpeta)
# archivo_seleccionado_2 = seleccionar_archivo_aleatorio(carpeta)


# Rutas de archivos seleccionados manualmente (Marcadas como plagio en el dataset)
# student2821,student8295
archivo_seleccionado_1 = r"C:\Users\jordi\Escritorio\DetectorPlagio\PlagiarismDetector\Dataset_C\A2016\Z1\Z1\student2821.c"
archivo_seleccionado_2 = r"C:\Users\jordi\Escritorio\DetectorPlagio\PlagiarismDetector\Dataset_C\A2016\Z1\Z1\student8295.c"

# student4934,student6617
# archivo_seleccionado_1 = r"C:\Users\jordi\Escritorio\DetectorPlagio\PlagiarismDetector\Dataset_C\A2016\Z1\Z1\student4934.c"
# archivo_seleccionado_2 = r"C:\Users\jordi\Escritorio\DetectorPlagio\PlagiarismDetector\Dataset_C\A2016\Z1\Z1\student6617.c"

# student4934,student6617 modificado
# archivo_seleccionado_1 = r"C:\Users\jordi\Escritorio\DetectorPlagio\PlagiarismDetector\Dataset_C\changedCodes\student4934modified.c"
# archivo_seleccionado_2 = r"C:\Users\jordi\Escritorio\DetectorPlagio\PlagiarismDetector\Dataset_C\changedCodes\student6617modified.c"

# Programas con funciones y sin funciones
# archivo_seleccionado_1 = r"C:\Users\jordi\Escritorio\DetectorPlagio\PlagiarismDetector\Dataset_C\changedCodes\sum.c"
# archivo_seleccionado_2 = r"C:\Users\jordi\Escritorio\DetectorPlagio\PlagiarismDetector\Dataset_C\changedCodes\sum2.c"


print(archivo_seleccionado_1)
print(archivo_seleccionado_2)


# Función para eliminar líneas de inclusión, comentarios y 'namespace' de un archivo C
def eliminar_includes(input_file, output_file):
    try:
        with open(input_file, 'r') as file_in:
            lines = file_in.readlines()

        pattern = re.compile(r'^\s*(#include+.*)|(/\*[\s\S]*?\*/)|(//.*)')
        filtered_lines = [line for line in lines if not pattern.match(line)]
        filtered_lines = [line for line in filtered_lines if 'namespace' not in line]

        with open(output_file, 'w') as file_out:
            while filtered_lines[0] == '\n':
              filtered_lines.pop(0)
            file_out.writelines(filtered_lines)

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{input_file}'.")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

# Archivo de salida (código sin líneas de inclusión)
archivo_salida_1 = "exit_1.c"
archivo_salida_2 = "exit_2.c"

# Llamada a la función para eliminar las líneas de inclusión de los archivos seleccionados
eliminar_includes(archivo_seleccionado_1, archivo_salida_1)
eliminar_includes(archivo_seleccionado_2, archivo_salida_2)

# Parsear el código C en un AST
ast_1 = parse_file(archivo_salida_1, use_cpp=True, cpp_path=r'C:\MinGW\bin\cpp.exe')
ast_2 = parse_file(archivo_salida_2, use_cpp=True, cpp_path=r'C:\MinGW\bin\cpp.exe')

# Función para convertir el AST a una representación de cadena
def ast_to_string(node, indent=0):
    if node is None:
        return ""
    node_type = node.__class__.__name__
    node_str = f"{' ' * indent}{node_type}\n"
    for attr_name, attr_value in node.children():
        child_str = ast_to_string(attr_value, indent + 4)
        node_str += f"{' ' * (indent + 2)}{attr_name}:\n{child_str}"
    return node_str

# Convertir el AST a una representación de cadena
ast_string_1 = ast_to_string(ast_1)
ast_string_2 = ast_to_string(ast_2)

# Guardar el AST como texto en un archivo
with open("ast_1.txt", "w") as file:
    file.write(ast_string_1)

with open("ast_2.txt", "w") as file:
    file.write(ast_string_2)

# Abrir los archivos AST en formato de texto
with open ("ast_1.txt", "r") as myfile:
    data1 = myfile.read().replace('\n', '')

with open ("ast_2.txt", "r") as myfile2:
    data2 = myfile2.read().replace('\n', '')

def generate_dictionary(string, dictionary):
    # Divide la cadena en palabras individuales
    my_arr = string.split()
    index = 0
    # Para cada palabra en la cadena
    for word in my_arr:
        # Si la palabra no está en el diccionario
        if word not in dictionary:
            # Asigna un índice único a la palabra en el diccionario
            dictionary[word] = index
            index += 1

def generate_following_tokens(string):
    # Divide la cadena en palabras individuales
    my_arr = string.split()
    following_tokens = {}
    # Para cada índice i en el rango de la longitud de la lista menos 1
    for i in range(len(my_arr) - 1):
        current_token = my_arr[i]
        next_token = my_arr[i + 1]
        # Si el token actual no está en el diccionario de tokens siguientes
        if current_token not in following_tokens:
            # Crea una lista vacía para almacenar los siguientes tokens
            following_tokens[current_token] = []
        # Agrega el siguiente token a la lista de tokens siguientes del token actual
        following_tokens[current_token].append(next_token)

    return following_tokens

def initialize_matrix(size):
    # Inicializa una matriz cuadrada de tamaño 'size' con ceros
    return np.zeros((size, size))

def transition_matrix_from_string(string, max_size):
    dictionary = {}
    # Genera un diccionario de tokens a partir de la cadena
    generate_dictionary(string, dictionary)
    # Genera un diccionario de tokens siguientes a partir de la cadena
    following_tokens = generate_following_tokens(string)

    size = max_size
    # Inicializa una matriz de transición con ceros
    matrix = initialize_matrix(size)

    # Para cada token y su lista de tokens siguientes en el diccionario de tokens siguientes
    for token, following in following_tokens.items():
        # Calcula la probabilidad de transición de un token a sus tokens siguientes
        prob = 1 / len(following)
        i = dictionary[token]

        # Para cada token siguiente
        for x in following:
            j = dictionary[x]
            # Incrementa la entrada correspondiente en la matriz de transición
            matrix[i][j] += prob

    return matrix

def cosine_similarity(A, B):
    # Calcula la similitud coseno entre dos matrices A y B
    BT = np.transpose(B)
    C = np.dot(BT, A)
    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)
    similarity = np.trace(C) / (normA * normB)

    return similarity

# Calcula el tamaño máximo del diccionario
max_size = max(len(set(data1.split())), len(set(data2.split())))

# Genera las matrices de transición para las cadenas de texto
matrix1 = transition_matrix_from_string(data1, max_size)
matrix2 = transition_matrix_from_string(data2, max_size)

# Calcula la similitud coseno entre las matrices
similarity = cosine_similarity(matrix1, matrix2)

# Definir umbral de similitud para considerar como plagio
umbral_similitud = 70

# Calcular la similitud como porcentaje
similitud_porcentaje = similarity * 100

# Verificar si la similitud es mayor al umbral
if similitud_porcentaje > umbral_similitud:
    print(f"\n¡Plagio detectado! Nivel de similitud: {similitud_porcentaje:.2f}%\n")
else:
    print(f"\nNo es plagio. Nivel de similitud: {similitud_porcentaje:.2f}%\n")
