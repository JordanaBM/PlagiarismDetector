#include <stdio.h>

// Declaración de la función suma
int suma(int a, int b) {
    return a + b;
}

// Declaración de la función resta
int resta(int a, int b) {
    return a - b;
}

int main() {
    int num1 = 10, num2 = 5;
    int resultado_suma, resultado_resta;

    // Llamada a la función suma
    resultado_suma = suma(num1, num2);
    printf("La suma de %d y %d es: %d\n", num1, num2, resultado_suma);

    // Llamada a la función resta
    resultado_resta = resta(num1, num2);
    printf("La resta de %d y %d es: %d\n", num1, num2, resultado_resta);

    return 0;
}
