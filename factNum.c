int factorial(int n) {
    if (n == 0 || n == 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

int main() {
    int num = 5;
    int resultado = factorial(num);

    printf("El factorial de %d es: %d\n", num, resultado);

    return 0;
}
