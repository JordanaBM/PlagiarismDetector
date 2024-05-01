import numpy as np
from pycparser import parse_file
import re


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

def detectar_plagio(archivos):
    # Inicializar variables para contadores de aciertos y errores
    aciertos = 0
    errores = 0

    # Para cada par de archivos en la lista de archivos
    for par in archivos:
        archivo_seleccionado_1 = par[0]
        archivo_seleccionado_2 = par[1]

        # Llamada a la función para eliminar las líneas de inclusión de los archivos seleccionados
        eliminar_includes(archivo_seleccionado_1, "exit_1.c")
        eliminar_includes(archivo_seleccionado_2, "exit_2.c")

        # Parsear el código C en un AST
        ast_1 = parse_file("exit_1.c", use_cpp=True, cpp_path=r'C:\MinGW\bin\cpp.exe')
        ast_2 = parse_file("exit_2.c", use_cpp=True, cpp_path=r'C:\MinGW\bin\cpp.exe')

        # Convertir el AST a una representación de cadena
        ast_string_1 = ast_to_string(ast_1)
        ast_string_2 = ast_to_string(ast_2)

        # Calcular max_size
        max_size = max(len(set(ast_string_1.split())), len(set(ast_string_2.split())))

        # Generar las matrices de transición para las cadenas de texto
        matrix1 = transition_matrix_from_string(ast_string_1, max_size)
        matrix2 = transition_matrix_from_string(ast_string_2, max_size)

        # Calcular la similitud coseno entre las matrices
        similarity = cosine_similarity(matrix1, matrix2)

        # Definir umbral de similitud para considerar como plagio
        umbral_similitud = 70

        # Calcular la similitud como porcentaje
        similitud_porcentaje = similarity * 100

        # Mostrar información de similitud y archivos
        print(f"Archivos: {archivo_seleccionado_1}, {archivo_seleccionado_2}")
        print(f"Porcentaje de similitud: {similitud_porcentaje:.2f}%\n")

        # Verificar si la similitud es mayor al umbral
        if similitud_porcentaje > umbral_similitud:
            aciertos += 1
        else:
            errores += 1

    # Calcular el accuracy
    total_archivos = len(archivos)
    accuracy = (aciertos / total_archivos) * 100

    print(f"\nTotal de archivos analizados: {total_archivos}")
    print(f"Aciertos: {aciertos}")
    print(f"Errores: {errores}")
    print(f"Accuracy: {accuracy:.2f}%")

# Lista de pares de archivos a analizar
carpeta = r"C:\Users\jordi\Escritorio\DetectorPlagio\PlagiarismDetector\Dataset_C\A2016\Z1\Z1"

archivos = [
    (f"{carpeta}\student7386.c", f"{carpeta}\student5378.c"),
    (f"{carpeta}\student2821.c", f"{carpeta}\student8295.c"),
    (f"{carpeta}\student4934.c", f"{carpeta}\student6617.c"),
    (f"{carpeta}\student8598.c", f"{carpeta}\student3331.c"),
    (f"{carpeta}\student7888.c", f"{carpeta}\student7704.c"),
    # (f"{carpeta}\student7386.c", f"{carpeta}\student5378.c"),
    # (f"{carpeta}\student9358.c", f"{carpeta}\student2953.c"),
    # (f"{carpeta}\student7386.c", f"{carpeta}\student5378.c"),
    # Agrega más pares de archivos según sea necesario
]


# Llamar a la función para detectar plagio y calcular el accuracy
detectar_plagio(archivos)
