import numpy as np
from pycparser import parse_file
import re
import os
import random

def seleccionar_archivo_aleatorio(carpeta):
    # Obtener la lista de archivos en la carpeta
    archivos = os.listdir(carpeta)
    # Seleccionar un archivo aleatorio
    archivo_aleatorio = random.choice(archivos)
    # Devolver el nombre completo del archivo seleccionado
    return os.path.join(carpeta, archivo_aleatorio)

# Ejemplo de uso
carpeta = r"C:\Users\jordi\Escritorio\DetectorPlagio\PlagiarismDetector\1"
archivo_seleccionado_1 = seleccionar_archivo_aleatorio(carpeta)
archivo_seleccionado_2 = seleccionar_archivo_aleatorio(carpeta)

print(archivo_seleccionado_1)
print(archivo_seleccionado_2)

def eliminar_includes(input_file, output_file):
    try:
        # Abrir el archivo de entrada para lectura
        with open(input_file, 'r') as file_in:
            # Leer todas las líneas del archivo
            lines = file_in.readlines()

        # Expresión regular para buscar líneas de inclusión (#include ...) o comentarios (// ...) o comentarios multilinea (/* ... */)
        pattern = re.compile(r'^\s*(#include+.*)|(/\*[\s\S]*?\*/)|(//.*)')

        # Filtrar las líneas que no son líneas de inclusión ni comentarios
        filtered_lines = [line for line in lines if not pattern.match(line)]

        # Filtrar las líneas que contienen la palabra 'namespace'
        filtered_lines = [line for line in filtered_lines if 'namespace' not in line]

        # Abrir el archivo de salida para escritura
        with open(output_file, 'w') as file_out:
            # Escribir las líneas filtradas en el archivo de salida
            while filtered_lines[0] == '\n':
              filtered_lines.pop(0)
            file_out.writelines(filtered_lines)

        print(f"Se eliminaron las líneas de inclusión, comentarios, 'namespace' y comentarios multilinea del archivo '{input_file}'.")
        print(f"El resultado se ha guardado en '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{input_file}'.")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

# Archivo de salida (código sin líneas de inclusión)
archivo_salida_1 = "exit_1.c"
archivo_salida_2 = "exit_2.c"

# Llama a la función para eliminar las líneas de inclusión
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


#Abrimos el ast en .txt

with open ("ast_1.txt", "r") as myfile:
    data1 = myfile.read().replace('\n', '')

with open ("ast_2.txt", "r") as myfile2:
    data2 = myfile2.read().replace('\n', '')


def generate_dictionary(string, dictionary):
    my_arr = string.split()
    index = 0
    for word in my_arr:
        if word not in dictionary:
            dictionary[word] = index
            index += 1

def generate_following_tokens(string):
    my_arr = string.split()
    following_tokens = {}
    for i in range(len(my_arr) - 1):
        current_token = my_arr[i]
        next_token = my_arr[i + 1]

        if current_token not in following_tokens:
            following_tokens[current_token] = []

        following_tokens[current_token].append(next_token)

    return following_tokens

def initialize_matrix(size):
    return np.zeros((size, size))

def transition_matrix_from_string(string, max_size):
    dictionary = {}
    generate_dictionary(string, dictionary)

    following_tokens = generate_following_tokens(string)

    size = max_size
    matrix = initialize_matrix(size)

    for token, following in following_tokens.items():
        prob = 1 / len(following)
        i = dictionary[token]

        for x in following:
            j = dictionary[x]
            matrix[i][j] += prob

    return matrix

def cosine_similarity(A, B):
  
    # Calcular la similitud coseno
    BT = np.transpose(B)
    C = np.dot(BT, A)
    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)
    similarity = np.trace(C) / (normA * normB)

    return similarity

# Calcular el tamaño máximo del diccionario
max_size = max(len(set(data1.split())), len(set(data2.split())))

# Generar matrices de transición
matrix1 = transition_matrix_from_string(data1, max_size)
matrix2 = transition_matrix_from_string(data2, max_size)

# Calcular la similitud coseno
similarity = cosine_similarity(matrix1, matrix2)
print("Coseno (cosine similarity):", similarity)
