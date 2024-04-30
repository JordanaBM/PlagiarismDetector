#include <stdio.h>

void intercambiar(int *xp, int *yp) {
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                intercambiar(&arr[j], &arr[j+1]);
            }
        }
    }
}

void imprimirArray(int arr[], int size) {
    for (int i=0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int arreglo[] = {64, 34, 25, 12, 22, 11, 90};
    int tamano = sizeof(arreglo)/sizeof(arreglo[0]);
    printf("Arreglo original: \n");
    imprimirArray(arreglo, tamano);
    bubbleSort(arreglo, tamano);
    printf("Arreglo ordenado en forma ascendente: \n");
    imprimirArray(arreglo, tamano);
    return 0;
}
