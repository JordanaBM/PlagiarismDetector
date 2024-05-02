<h1 align="center">PlagiarismDetector</h1>

<p align="center">
  <img src="https://javier.rodriguez.org.mx/itesm/2014/tecnologico-de-monterrey-blue.png" width="50%">
</p>

<h2 align="center">Tecnológico de Monterrey Campus Querétaro</h2>
<h3 align="center">Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 201)</h3>
<p align="center">
  Jordana Betancourt Menchaca - A01707434<br>
  Ana Karen López Baltazar - A01707750<br>
  Arisbeth Aguirre Pontaza - A01274803
</p>

---

<h1 align="center">Análisis de similitud entre archivos de código en C</h1>

## Dataset

El Dataset utilizado fue el [siguiente](https://drive.google.com/drive/folders/1YruqwHxyq_UnMFYW_G828Nm9lqZur9M4?usp=sharing), sustraido de [PROGRAMMING HOMEWORK DATASET FOR PLAGIARISM DETECTION](https://ieee-dataport.org/open-access/programming-homework-dataset-plagiarism-detection)
 

## Pasos previos para ejecutar el código

### Para Windows

#### Instalar Python en Windows:

1. **Descargar Python:** Ve al [sitio web oficial de Python](https://www.python.org/) y descarga la última versión de Python para Windows. Asegúrate de seleccionar la opción para agregar Python al PATH durante la instalación.

2. **Instalar Python:** Ejecuta el instalador descargado y sigue las instrucciones en pantalla. Asegúrate de marcar la casilla que dice "Add Python to PATH" (Agregar Python al PATH) durante la instalación.

3. **Verificar la instalación:** Abre una ventana de comandos (`cmd`) y escribe `python --version` para verificar que Python se ha instalado correctamente. Deberías ver la versión de Python que has instalado.

### Instalar MinGW en Windows:

MinGW es un conjunto de herramientas que incluye un compilador de C/C++ para Windows.

1. **Descargar MinGW:** Ve al [sitio web oficial de MinGW](http://www.mingw.org/) y descarga el instalador de MinGW para Windows.

2. **Instalar MinGW:** Ejecuta el instalador descargado y sigue las instrucciones en pantalla. Durante la instalación, selecciona los componentes que deseas instalar, incluyendo el compilador de C/C++.

3. **Configurar MinGW en el PATH:** Después de instalar MinGW, debes añadir la ruta de la carpeta `bin` de MinGW al PATH de Windows. Para hacerlo, sigue estos pasos:
   - Haz clic con el botón derecho en "Este PC" o "Mi PC" en el escritorio o en el Explorador de archivos.
   - Selecciona "Propiedades" y luego "Configuración avanzada del sistema" en el panel izquierdo.
   - En la ventana de Propiedades del sistema, haz clic en "Variables de entorno".
   - En la sección "Variables del sistema", busca la variable PATH y haz clic en "Editar".
   - Haz clic en "Nuevo" y añade la ruta de la carpeta `bin` de MinGW (por ejemplo, `C:\MinGW\bin`).
   - Haz clic en "Aceptar" en todas las ventanas para guardar los cambios.

4. **Verificar la instalación:** Abre una nueva ventana de comandos (`cmd`) y escribe `gcc --version` para verificar que el compilador de C/C++ de MinGW se ha instalado correctamente. Deberías ver la versión de GCC que has instalado.


### Para macOS

#### Instalar Python en macOS:

1. **Descargar Python:** Abre un navegador web y ve al [sitio web oficial de Python](https://www.python.org/downloads/). Descarga la última versión de Python para macOS.

2. **Instalar Python:** Abre el archivo descargado y sigue las instrucciones en pantalla para instalar Python en tu sistema. Asegúrate de marcar la casilla que dice "Add Python to PATH" (Agregar Python al PATH) durante la instalación.

3. **Verificar la instalación:** Abre una terminal y escribe `python3 --version` para verificar que Python se ha instalado correctamente. Deberías ver la versión de Python que has instalado.

#### Instalar Xcode Command Line Tools (para compilación de C/C++):

1. **Abrir Terminal:** Abre la aplicación Terminal en tu Mac. Puedes encontrarla en la carpeta Utilidades dentro de la carpeta de Aplicaciones.

2. **Instalar Xcode Command Line Tools:** En la terminal, ejecuta el siguiente comando:
   ```sh
   xcode-select --install
    ```
Esto abrirá una ventana de instalación. Sigue las instrucciones en pantalla para instalar las Herramientas de línea de comandos de Xcode.

3. **Verificar la instalación:** Después de instalar las Herramientas de línea de comandos de Xcode, puedes verificar la instalación ejecutando el siguiente comando en la terminal:
 ```sh
   gcc --version
   ```

### Para Linux

#### Instalar Python en Linux:

1. **Usando el gestor de paquetes de tu distribución:** La mayoría de las distribuciones de Linux vienen con Python preinstalado. Si no es así, puedes instalarlo utilizando el gestor de paquetes de tu distribución. Por ejemplo, en Ubuntu puedes usar el siguiente comando:
   ```sh
   sudo apt-get update
   sudo apt-get install python3
    ```
2. **Verificar la instalación:** Abre una terminal y escribe python3 --version para verificar que Python se ha instalado correctamente. Deberías ver la versión de Python que has instalado.
 ```sh
   gcc --version
  g++ --version
  ```

---

## ¿Cómo correr el código?

1. **Instalar las librerías necesarias:** Antes de ejecutar el código, asegúrate de tener instaladas las siguientes librerías de Python:
   - `numpy`: Para cálculos numéricos eficientes.
   - `pycparser`: Para parsear archivos de código C en un AST.
   - `re`: Para trabajar con expresiones regulares.
   - `os`: Para operaciones con el sistema operativo.
   - `random`: Para generación de números aleatorios.

   Puedes instalar estas librerías usando `pip` si aún no las tienes instaladas:

   ```
   pip install numpy pycparser
   ```

2. **Ajustar las rutas de los archivos:** Antes de ejecutar el código, asegúrate de ajustar las rutas de los archivos de código C que quieres analizar. Puedes modificar las siguientes líneas para apuntar a tus archivos:
```python
archivo_seleccionado_1 = "ruta_del_archivo_1.c"
archivo_seleccionado_2 = "ruta_del_archivo_2.c"
```


3. **Ajustar la ruta del cpp.exe:** Antes de ejecutar el código, asegúrate de ajustar la ruta del archivo `cpp.exe` de MinGW. Puedes modificar la siguiente línea para que apunte a la ubicación correcta en tu sistema:
```python
ast_1 = parse_file(archivo_salida_1, use_cpp=True, cpp_path=r'RUTA_AL_CPP.EXE')
```

4. **Ejecutar el código:**  Una vez que hayas instalado las librerías y ajustado las rutas de los archivos, puedes ejecutar el código Python. Copia y pega todo el código en un archivo .py y ejecútalo desde la línea de comandos o desde tu entorno de desarrollo Python favorito.Por ejemplo, si no cambias el nombre de los archivos, puedes ejecutar el programa así desde la línea de comandos:
```python
python main.py
```

---

5. **Revisar los resultados:** Después de ejecutar el código, revisa la salida en la consola. El código imprimirá un mensaje indicando si se detectó plagio entre los archivos analizados y el nivel de similitud encontrado.Si se encuentra similitud y se supera un umbral predefinido, el código imprimirá un mensaje indicando que se detectó plagio y mostrará el nivel de similitud en porcentaje. Si no se supera el umbral, se imprimirá un mensaje indicando que no se detectó plagio.

<h1 align="center">Funcionamiento del código</h1>

1. **Importación de librerías:** El código importa varias librerías necesarias para su funcionamiento, como `numpy` para cálculos numéricos, `pycparser` para parsear código en C, `re` para expresiones regulares, `os` para operaciones con el sistema operativo y `random` para generación de números aleatorios.

2. **Selección de archivo aleatorio:** La función `seleccionar_archivo_aleatorio(carpeta)` toma una carpeta como entrada y selecciona aleatoriamente un archivo de esa carpeta.

3. **Eliminación de líneas de inclusión y comentarios:** La función `eliminar_includes(input_file, output_file)` elimina las líneas de inclusión (`#include`), comentarios (`/* comentario */` y `// comentario`) y líneas que contienen la palabra `namespace` de un archivo C.

4. **Parseo de archivos C en un AST:** Utiliza la función `parse_file` de `pycparser` para parsear los archivos de código C en un Abstract Syntax Tree (AST).

En el contexto de análisis de código en lenguaje C, el proceso de "parseo" se refiere a la acción de analizar y descomponer un archivo de código fuente C en sus componentes sintácticos fundamentales. Esto se hace para que el código pueda ser procesado y entendido de manera estructurada por programas informáticos, en lugar de simplemente leerlo como texto plano.

El parseo de archivos C en un Abstract Syntax Tree (AST) es una técnica comúnmente utilizada en el análisis de código. Un AST es una representación jerárquica de la estructura sintáctica del código fuente, donde cada nodo del árbol representa una construcción sintáctica del lenguaje, como una declaración, una expresión, un bucle, etc.

En el código se utiliza la función parse_file de la biblioteca pycparser para realizar este parseo. pycparser es una biblioteca de Python que se utiliza para analizar código C y construir un AST correspondiente. La función parse_file toma como entrada un archivo de código C y devuelve un objeto AST que representa la estructura sintáctica del código.

Aquí un ejemplo de como funcionaría suponiendo que tienes un archivo de código C llamado ejemplo.c con el siguiente contenido:

```C
#include <stdio.h>

int main() {
    printf("Hola, mundo!\n");
    return 0;
}
```

Así es como se vería el AST resultante al parsear el código de ejemplo que te proporcioné anteriormente:

```yaml
FileAST: 
  FuncDef: 
    Decl: main, [], [], []
      FuncDecl: []
        IdentifierType: ['int']
    Compound: 
      FuncCall: 
        ID: printf
        ExprList: 
          Constant: string, "Hola, mundo!\n"
      Return: 
        Constant: int, 0

```

5. **Generación de matrices de transición:** Utiliza las cadenas de texto generadas a partir de los AST para crear matrices de transición, que representan la probabilidad de transición entre tokens en las cadenas de texto.

En el contexto del análisis de similitud entre dos archivos de código fuente, la generación de matrices de transición se refiere al proceso de crear matrices que representan la probabilidad de transición entre tokens (palabras individuales) en las cadenas de texto generadas a partir de los Abstract Syntax Trees (AST) de los archivos de código fuente.

Para generar estas matrices, primero se convierten los AST de los archivos de código en cadenas de texto. Estas cadenas contienen representaciones estructuradas del código fuente en forma de tokens, donde cada token representa una palabra o símbolo en el código. Luego, se analizan estas cadenas para determinar la probabilidad de transición entre los tokens.

Por ejemplo, si tienes dos cadenas de texto generadas a partir de los AST de dos archivos de código C:

```
Cadena 1: "for (i = 0; i < 10; i++)"
Cadena 2: "for (int i = 0; i < 10; i++)"
```


Puedes crear una matriz de transición que muestre la probabilidad de transición entre los tokens en estas cadenas. La matriz podría verse así:

| Token   | for | (   | i   | =   | 0   | ;   | i   | <   | 10  | ;   | i++ | )   |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| for     | 0   | 1.0 | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   |
| (       | 0   | 0   | 1.0 | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   |
| i       | 0   | 0   | 0   | 1.0 | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   |
| =       | 0   | 0   | 0   | 0   | 1.0 | 0   | 0   | 0   | 0   | 0   | 0   | 0   |
| 0       | 0   | 0   | 0   | 0   | 0   | 1.0 | 0   | 0   | 0   | 0   | 0   | 0   |
| ;       | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 1.0 | 0   | 0   | 0   | 0   |
| i       | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 1.0 | 0   | 0   | 0   |
| <       | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 1.0 | 0   | 0   |
| 10      | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 1.0 | 0   |
| ;       | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 1.0 |
| i++     | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 1.0 |
| )       | 0   | 0   | ```

En esta matriz de transición, cada fila y columna representan un token (palabra o símbolo) en las cadenas de texto. Los valores en la matriz indican la probabilidad de transición entre dos tokens. Por ejemplo, en la fila correspondiente al token "for", la columna correspondiente al token "(" tiene un valor de 1.0, lo que significa que el token "(" siempre sigue al token "for" en la cadena de texto.

La generación de estas matrices de transición permite cuantificar la similitud entre dos cadenas de texto basándose en la similitud de sus estructuras sintácticas. Se pueden usar técnicas de comparación de matrices, como la similitud coseno, para determinar la similitud entre las cadenas de texto y, por lo tanto, entre los archivos de código fuente.

6. **Cálculo de similitud coseno:** Calcula la similitud coseno entre las matrices de transición de los dos archivos, lo que proporciona una medida de la similitud entre los archivos en términos de su estructura y contenido.

La similitud coseno es una medida utilizada para determinar la similitud entre dos vectores en un espacio euclidiano. En el contexto del análisis de código fuente, las matrices de transición se pueden ver como representaciones vectoriales de las cadenas de texto generadas a partir de los Abstract Syntax Trees (AST) de los archivos de código.

Para calcular la similitud coseno entre dos matrices de transición, primero se convierten las matrices en vectores unidimensionales. Luego, se calcula el producto punto entre los dos vectores y se divide por el producto de las magnitudes de los vectores para obtener un valor que varía entre -1 y 1. Un valor de 1 indica una similitud total, mientras que un valor de -1 indica una similitud inversa.

En el contexto del análisis de código fuente, el cálculo de la similitud coseno entre las matrices de transición de dos archivos proporciona una medida de la similitud entre los archivos en términos de su estructura y contenido. Una similitud cercana a 1 indica que los dos archivos son muy similares en términos de la estructura de su código, mientras que una similitud cercana a -1 indica que los archivos son muy diferentes.

7. **Comparación con umbral de similitud:** Compara la similitud calculada con un umbral predefinido para determinar si los archivos son similares o no. Si la similitud es mayor al umbral, se considera que hay plagio.
---









