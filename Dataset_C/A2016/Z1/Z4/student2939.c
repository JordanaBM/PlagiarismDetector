#include <stdio.h>
int main() {
	int i,j,n;
	printf("Unesite broj n: ");
	scanf("%d", &n);
	while(n<=0 ||n>50){
		printf("Pogresan unos");
		printf("\nUnesite broj n: ");
		scanf("%d", &n);}
	if(n==1){
		printf("***");}
	else {
	for(i=0; i<n; i++){
	for(j=0; j<=4*n-4; j++){
	if(i==j || j==2*n-2-i || j==2*n-2+i || j==4*n-4-i){
		printf("*");}
	else{printf(" ");}} 
	   	printf("\n");}}
	return 0;
	}
