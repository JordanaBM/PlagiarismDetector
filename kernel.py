import os
import re
import math
import shutil
import numpy as np
import subprocess
from pycparser import parse_file, c_ast

# Cadenas de Markov -----------------------------------------------------------
def generate_dictionary(string, dictionary):
    index = 0
    for char in string:
        if char not in dictionary:
            dictionary[char] = index
            index += 1

def generate_following_characters(string):
    following_chars = {}
    for i in range(len(string) - 1):
        current_char = string[i]
        next_char = string[i + 1]

        if current_char not in following_chars:
            following_chars[current_char] = []

        following_chars[current_char].append(next_char)

    return following_chars

def initialize_matrix(matrix, size):
    for i in range(size):
        matrix.append(np.zeros(size))

def transition_matrix_from_string(string):
    dictionary = {}
    generate_dictionary(string, dictionary)

    following_chars = generate_following_characters(string)

    matrix = []
    initialize_matrix(matrix, len(dictionary))

    for char, following in following_chars.items():
        prob = 1 / len(following)
        i = dictionary[char]

        for x in following:
            j = dictionary[x]
            matrix[i][j] += prob

    return matrix, len(dictionary)

def cosine_similarity(A, B):
    BT = np.transpose(B)
    C = np.dot(BT, A)
    prod_int = np.trace(C)
    normA = np.sqrt(np.trace(np.dot(np.transpose(A), A)))
    normB = np.sqrt(np.trace(np.dot(np.transpose(B), B)))
    similarity = prod_int / (normA * normB)

    return similarity

# Grafos de llamada de funciones -----------------------------------------------------------
def generate_call_graph_c(code_file, output_dot_file):
    # Ejecutar cflow para generar el archivo DOT
    try:
        subprocess.run(['cflow2dot', '-i', code_file, '-o', output_dot_file, '-f', 'dot'], check=True)
        print("DOT file generated:", output_dot_file + "0.dot")
    except subprocess.CalledProcessError as e:
        print("Error al generar el call graph:", e)

def dot_to_adjacency_matrix(dot_file):
    # Leer el archivo DOT
    with open(dot_file, 'r') as f:
        dot_content = f.read()

    # Buscar las conexiones entre nodos en el archivo DOT
    connections = re.findall(r'\b(\w+)\s*->\s*(\w+)\b', dot_content)

    # Obtener todos los nodos únicos en el grafo
    nodes = set()
    for connection in connections:
        nodes.add(connection[0])
        nodes.add(connection[1])

    # Crear un diccionario para mapear índices de nodos a nombres de nodos en el grafo
    node_names = {i: node for i, node in enumerate(nodes)}

    # Crear un diccionario inverso para mapear nombres de nodos a índices de nodos en el grafo
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # Inicializar una matriz de adyacencia llena de ceros
    num_nodes = len(nodes)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # Actualizar la matriz de adyacencia con las conexiones entre nodos
    for connection in connections:
        source = node_to_index[connection[0]]
        target = node_to_index[connection[1]]
        adjacency_matrix[source, target] = 1  # Marcar la conexión entre los nodos

    return adjacency_matrix, node_names

# Obtener el nombre del nodo correspondiente al índice dado
def get_node_name(index, node_names):
    return node_names[index]

def node_kernel(node1, node2):
    # Comparación entre nodos mediante cadenas de Markov
    matrix1, size1 = transition_matrix_from_string(node1)
    matrix2, size2 = transition_matrix_from_string(node2)

    # Alinear las matrices de transición
    max_size = max(size1, size2)
    aligned_matrix1 = np.zeros((max_size, max_size))
    aligned_matrix2 = np.zeros((max_size, max_size))

    aligned_matrix1[:size1, :size1] = matrix1
    aligned_matrix2[:size2, :size2] = matrix2

    # Coseno de similaridad
    similarity = cosine_similarity(aligned_matrix1, aligned_matrix2)
    return 1 if similarity > 0.7 else 0

    # Originalmente: distancia de edición
    # return 1 if Levenshtein.distance(node1, node2) < 0.5 else 0

# Función para comparar aristas (dependencias entre funciones)
def edge_kernel(edge1_graph1, edge2_graph1, edge1_graph2, edge2_graph2):
    # Comparación entre aristas (binarias)
    return 1 if edge1_graph1 == edge1_graph2 and edge2_graph1 == edge2_graph2 else 0

# Función para calcular el kernel de grafo de caminata aleatoria modificada
def modified_random_walk_kernel(graph1, graph2, node_names1, node_names2):
    # Obtener las dimensiones de los grafos
    n1, m1 = graph1.shape

    # Inicializar el kernel de caminata aleatoria modificada
    modified_kernel = 0
    total_operations = 0

    # Recorrer todas las aristas del producto directo de los grafos
    for i in range(n1 - 1):  # Se detiene en n-1 para permitir la comparación con el siguiente nodo
        for j in range(m1 - 1):
            # Obtener los nombres de los nodos correspondientes en ambos grafos
            node_name1 = get_node_name(i, node_names1)
            node_name2 = get_node_name(i, node_names2)
            next_node_name1 = get_node_name(i + 1, node_names1)
            next_node_name2 = get_node_name(i + 1, node_names2)

            # Calcular la contribución de esta arista al kernel
            kernel_contribution = (node_kernel(node_name1, node_name2) *
                                   node_kernel(next_node_name1, next_node_name2) *
                                   edge_kernel(graph1[i, j], graph1[i+1, j+1], graph2[i, j], graph2[i+1, j+1]))

            # Sumar la contribución al kernel modificado
            modified_kernel += kernel_contribution
            total_operations += 1

    # Calcular el resultado normalizado
    normalized_kernel = modified_kernel / total_operations

    return normalized_kernel

# Parse tree -----------------------------------------------------------
def remove_comments_and_includes(input_file, output_file):
    # Leer el contenido del archivo de entrada
    with open(input_file, 'r') as f:
        content = f.read()

    # Eliminar los comentarios usando una expresión regular
    cleaned_content = re.sub(r'/\*.*?\*/|//.*?\n', '', content, flags=re.DOTALL)

    # Eliminar los includes usando una expresión regular
    cleaned_content = re.sub(r'#include\s*<.*>\n', '', cleaned_content)

    # Escribir el contenido limpio en el archivo de salida
    with open(output_file, 'w') as f:
        f.write(cleaned_content)

def string_similarity(str1, str2):
    # Comparación entre cadenas mediante cadenas de Markov
    matrix1, size1 = transition_matrix_from_string(str1)
    matrix2, size2 = transition_matrix_from_string(str2)

    # Alinear las matrices de transición
    max_size = max(size1, size2)
    aligned_matrix1 = np.zeros((max_size, max_size))
    aligned_matrix2 = np.zeros((max_size, max_size))

    aligned_matrix1[:size1, :size1] = matrix1
    aligned_matrix2[:size2, :size2] = matrix2

    # Coseno de similaridad
    similarity = cosine_similarity(aligned_matrix1, aligned_matrix2)
    return similarity

def compute_similarity(node1, node2, decay_factor):
    if isinstance(node1, c_ast.FuncDef) and isinstance(node2, c_ast.FuncDef):
        # Si ambos nodos son FuncDef, calcular la similitud entre sus cuerpos
        return compute_body_similarity(node1.body, node2.body)
    else:
        # Si los nodos son diferentes tipos, su similitud es cero
        return 0.0

def compute_body_similarity(body1, body2):
    if body1 is None or body2 is None:
        # Si alguno de los cuerpos es None, su similitud es cero
        return 0.0

    # Calcular la similitud entre las declaraciones de cada cuerpo
    body_similarity = 0
    block1 = body1.block_items
    block2 = body2.block_items

    for stmt1 in block1:
        for stmt2 in block2:
            stmt_similarity = compute_statement_similarity(stmt1, stmt2)
            body_similarity = max(body_similarity, stmt_similarity)

    # Retornar la similitud
    return body_similarity

def compute_statement_similarity(statement1, statement2):
    # Verificar si las declaraciones tienen el mismo tipo
    if type(statement1) != type(statement2):
        return 0.0

    # Inicializar la similitud
    similarity = 0.0

    if isinstance(statement1, c_ast.FuncDef):
        # Calcular la similitud entre los nombres de las funciones, número de parámetros, tipos de retorno
        name_similarity = string_similarity(statement1.decl.name, statement2.decl.name)
        param_count_similarity = min(1.0, abs(len(statement1.param_decls) - len(statement2.param_decls)) / 2)
        return_type_similarity = 1.0 if statement1.decl.type.type == statement2.decl.type.type else 0.0
        similarity += (name_similarity + param_count_similarity + return_type_similarity) / 3.0

    elif isinstance(statement1, c_ast.Decl):
        # Calcular la similitud entre los nombres de las variables, tipos de variables e inicializadores
        name_similarity = string_similarity(statement1.name, statement2.name)
        type_similarity = 1.0 if type(statement1.type) == type(statement2.type) else 0.0
        initializer_similarity = 1.0 if statement1.init == statement2.init else 0.0
        similarity += (name_similarity + type_similarity + type_similarity) / 3.0

    elif isinstance(statement1, c_ast.Assignment):
        # Calcular la similitud entre las operaciones de asignación y los lados izquierdo y derecho de la asignación
        op_similarity = 1.0 if statement1.op == statement2.op else 0.0
        lhs_similarity = compute_expression_similarity(statement1.lvalue, statement2.lvalue)
        rhs_similarity = compute_expression_similarity(statement1.rvalue, statement2.rvalue)
        similarity += (op_similarity + lhs_similarity + rhs_similarity) / 3.0

    elif isinstance(statement1, c_ast.For):
        ## Calcular la similitud entre los atributos del bucle for (inicialización, condición y actualización)
        init_similarity = compute_statement_similarity(statement1.init, statement2.init)
        cond_similarity = compute_expression_similarity(statement1.cond, statement2.cond)
        next_similarity = compute_expression_similarity(statement1.next, statement2.next)
        body_similarity = compute_body_similarity(statement1.stmts, statement2.stmts)
        similarity += (init_similarity + cond_similarity + next_similarity + body_similarity) / 4.0

    elif isinstance(statement1, c_ast.If):
        ## Calcular la similitud entre los atributos del condicional if (condición, cuerpo del if, cuerpo del else)
        if_similarity = compute_expression_similarity(statement1.cond, statement2.cond)
        then_similarity = compute_statement_similarity(statement1.iftrue, statement2.iftrue)
        else_similarity = compute_statement_similarity(statement1.iffalse, statement2.iffalse)
        similarity += (if_similarity + then_similarity + else_similarity) / 3.0

    elif isinstance(statement1, c_ast.DoWhile):
        ## Calcular la similitud entre los atributos del bucle do-while (cuerpo del bucle, condición)
        body_similarity = compute_statement_similarity(statement1.stmts, statement2.stmts)
        cond_similarity = compute_expression_similarity(statement1.cond, statement2.cond)
        body_similarity = compute_body_similarity(statement1.stmts, statement2.stmts)
        similarity += (body_similarity + cond_similarity) / 2.0

    elif isinstance(statement1, c_ast.While):
        ## Calcular la similitud entre los atributos del bucle while (condición, cuerpo del bucle)
        cond_similarity = compute_expression_similarity(statement1.cond, statement2.cond)
        body_similarity = compute_statement_similarity(statement1.stmts, statement2.stmts)
        similarity += (cond_similarity + body_similarity) / 2.0

    elif isinstance(statement1, c_ast.Switch):
        ## Calcular la similitud entre los atributos del switch (condición, cuerpo)
        expr_similarity = compute_expression_similarity(statement1.cond, statement2.cond)
        body_similarity = compute_body_similarity(statement1.stmts, statement2.stmts)
        similarity += (expr_similarity + body_similarity) / 2.0

    elif isinstance(statement1, c_ast.Case):
        ## Calcular la similitud entre los atributos del case (expresión, cuerpo)
        value_similarity = compute_expression_similarity(statement1.expr, statement2.expr)
        body_similarity = compute_body_similarity(statement1.stmts, statement2.stmts)
        similarity += (value_similarity + body_similarity) / 2.0

    elif isinstance(statement1, c_ast.Default):
      ## Calcular la similitud entre los atributos del default case (cuerpo)
      body_similarity = compute_body_similarity(statement1.stmts, statement2.stmts)
      similarity += body_similarity

    elif isinstance(statement1, c_ast.Struct):
        # Calcular la similitud entre los atributos de la estructura (nombre, campos)
        name_similarity = string_similarity(statement1.name, statement2.name)

        # Inicializar la similitud de los campos de la estructura
        fields_similarity = 0.0

        # Comparar los campos de las estructuras
        if len(statement1.decls) != len(statement2.decls):
            # Si las estructuras tienen diferente número de campos, la similitud es cero
            similarity += 0.0
        else:
            # Comparar cada par de campos de las estructuras
            for decl1, decl2 in zip(statement1.decls, statement2.decls):
                if decl1.name == decl2.name:
                    # Si los nombres de los campos son iguales, aumentar la similitud
                    fields_similarity += 1.0
            # Normalizar la similitud de los campos de la estructura
            fields_similarity /= len(statement1.decls)


        similarity += (name_similarity + fields_similarity) / 2.0

    elif isinstance(statement1, c_ast.Enum):
        ## Calcular la similitud entre las listas de constantes enumeradas
        if len(statement1.values) != len(statement2.values):
            # Si las listas tienen diferente longitud, la similitud es cero
            similarity += 0.0
        else:
            # Inicializar la similitud de las constantes enumeradas
            constants_similarity = 0.0
            # Comparar cada par de constantes enumeradas
            for val1, val2 in zip(statement1.values, statement2.values):
                if val1.name == val2.name:
                    # Si los nombres de las constantes son iguales, aumentar la similitud
                    constants_similarity += 1.0
            # Normalizar la similitud de las constantes enumeradas
            constants_similarity /= len(statement1.values)
            # Agregar la similitud de las constantes enumeradas a la similitud total
            similarity += constants_similarity

    return similarity

def compute_expression_similarity(expr1, expr2):
    # Comparación de expresiones en las asignaciones

    # Verificar si los tipos de expresión son iguales
    if type(expr1) != type(expr2):
        return 0.0

    # Comparar diferentes tipos de expresiones y calcular su similitud
    if isinstance(expr1, c_ast.Constant):
        # Si las expresiones son constantes, comparar sus valores
        return 1.0 if expr1.value == expr2.value else 0.0
    elif isinstance(expr1, c_ast.ID):
        # Si las expresiones son identificadores, comparar sus nombres
        return string_similarity(expr1.name, expr2.name)
    elif isinstance(expr1, c_ast.UnaryOp) and isinstance(expr2, c_ast.UnaryOp):
        # Si las expresiones son operaciones unarias, comparar las expresiones internas recursivamente
        return compute_expression_similarity(expr1.expr, expr2.expr)
    elif isinstance(expr1, c_ast.BinaryOp) and isinstance(expr2, c_ast.BinaryOp):
        # Si las expresiones son operaciones binarias, comparar las subexpresiones izquierda y derecha
        lhs_similarity = compute_expression_similarity(expr1.left, expr2.left)
        rhs_similarity = compute_expression_similarity(expr1.right, expr2.right)
        return (lhs_similarity + rhs_similarity) / 2.0
    else:
        # Si las expresiones no se pueden comparar, retornar 0.0
        return 0.0

def modified_parse_tree_kernel(tree1, tree2, decay_factor):
    # Inicializar el kernel modificado
    modified_kernel = 0

    # Calcular el kernel modificado para cada par de nodos de los árboles
    for decl1 in tree1.ext:
        max_similarity = 0.0
        for decl2 in tree2.ext:
            # Calcular la similitud entre las declaraciones y sus cuerpos
            child_similarity = 1 + compute_similarity(decl1, decl2, decay_factor)
            max_similarity = max(max_similarity, child_similarity)

            # Actualizar el kernel modificado con la similitud calculada
            modified_kernel += child_similarity

    # Aplicar la fórmula de similitud
    similarity = decay_factor * modified_kernel

    # Normalizar el kernel modificado para que esté en el rango [0, 1]
    max_possible_similarity = len(tree1.ext) * len(tree2.ext)
    normalized_kernel = similarity / (1 + max_possible_similarity)

    return normalized_kernel

# Detección de plagio -----------------------------------------------------------
def composite_kernel(K_mpt, K_mg, gamma):
    # Cálculo de Composite kernel usando la fórmula
    # K_co(s, s') = (1 - gamma) * K_mpt(Ts, Ts') + gamma * K_mg(Gs, Gs')
    composite_kernel = (1 - gamma) * K_mpt + gamma * K_mg
    return composite_kernel

def dfs(graph, node, visited):
    visited[node] = True
    for neighbor, is_connected in enumerate(graph[node]):
        if is_connected and not visited[neighbor]:
            dfs(graph, neighbor, visited)

def count_connected_components(graph):
    num_nodes = len(graph)
    visited = [False] * num_nodes
    num_components = 0
    for node in range(num_nodes):
        if not visited[node]:
            dfs(graph, node, visited)
            num_components += 1
    return num_components

def calculate_cyclomatic_complexity(graph):
    num_edges = sum(sum(row) for row in graph) // 2  # Dividir por 2 para evitar contar cada arista dos veces
    num_nodes = len(graph)
    num_components = count_connected_components(graph)
    complexity = num_edges - num_nodes + 2 * num_components
    return complexity

# Directorio donde se encuentran los archivos de código fuente
source_directory = "/mnt/c/Users/Administrador/Downloads/ITESM/Clases/8to Semestre/Desarrollo de aplicaciones avanzadas de ciencias computacionales/PlagiarismDetector/Dataset_C/A2016/Z1/Z1/"

# Nombres de los archivos de código fuente
source_file1 = "student2821.c"
source_file2 = "student8295.c"

# Nombres de los archivos DOT de salida
output_dot1 = source_file1[:-2]
output_dot2 = source_file2[:-2]
output_cleaned_file1 = "cleaned_" + source_file1
output_cleaned_file2 = "cleaned_" + source_file2

# Crear la carpeta outputs si no existe
output_folder = os.path.join(source_directory, "outputs")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"La carpeta 'outputs' ha sido creada en {source_directory}")

# Generar archivos DOT para los archivos de código fuente en C
generate_call_graph_c(os.path.join(source_directory, source_file1),os.path.join(output_folder, output_dot1))
generate_call_graph_c(os.path.join(source_directory, source_file2), os.path.join(output_folder, output_dot2))

# Convertir los archivos DOT a matrices de adyacencia
graph1, node_names1 = dot_to_adjacency_matrix(os.path.join(output_folder, output_dot1 + "0.dot"))
graph2, node_names2 = dot_to_adjacency_matrix(os.path.join(output_folder, output_dot2 + "0.dot"))

# Eliminar los includes del archivo de entrada y guardar el resultado en el archivo limpio
remove_comments_and_includes(os.path.join(source_directory, source_file1), os.path.join(output_folder, output_cleaned_file1))
remove_comments_and_includes(os.path.join(source_directory, source_file2), os.path.join(output_folder, output_cleaned_file2))

# Generar el AST del archivo limpio
ast1 = parse_file(os.path.join(output_folder, output_cleaned_file1))
ast2 = parse_file(os.path.join(output_folder, output_cleaned_file1))

# Calcular el kernel de caminata aleatoria modificada
modified_kernel = modified_random_walk_kernel(graph1, graph2, node_names1, node_names2)
print("Modified Graph Kernel:", modified_kernel)

# Calcular el kernel del árbol de análisis sintáctico modificado para los dos archivos
decay_factor = 0.9
modified_kernel_value = modified_parse_tree_kernel(ast1, ast2, decay_factor)
print("Modified Parse Tree Kernel:", modified_kernel_value)

# Calcular similitud combinada utilizando un kernel compuesto
c1 = calculate_cyclomatic_complexity(graph1)
c2 = calculate_cyclomatic_complexity(graph2)
gamma = 1 / (1 + math.exp(-(min(c1, c2) - 25)))
composite_similarity = composite_kernel(modified_kernel_value, modified_kernel, gamma)

# Impresión de similitud
print("Similaridad [", source_file1, ",", source_file2, "] de:", round(composite_similarity, 4))

# Eliminar el directorio outputs al finalizar
shutil.rmtree(output_folder)