#include <stdio.h>

int main() {
    int numero1 = 15, numero2 = 7;
    int resultado_suma, resultado_resta;

    // Llamada a la función sumar
    resultado_suma = numero1 + numero2;
    printf("La suma de %d y %d es: %d\n", numero1, numero2, resultado_suma);

    // Llamada a la función restar
    resultado_resta = numero1 - numero2;
    printf("La resta de %d y %d es: %d\n", numero1, numero2, resultado_resta);

    return 0;
}
