#include <stdio.h>
int main () {
	float a1, b1, a2, b2, x, y;
	printf("Unesite a1,b1,a2,b2: ");
	scanf("%f,%f,%f,%f", &a1, &b1, &a2, &b2);
	x=(b2-b1)/(a1-a2);
	y=a1*x+b1;
	if(b1==0 || b2==0 || a1==0 || a2==0 || (a1==a2 && b1!=b2)) {
		printf("Paralelne su");
		return 0;
	}
	else if(a1==a2 && b1==b2) {
		printf("Poklapaju se");
		return 0;
	}
		printf("Prave se sijeku u tacci (%.1f,%.1f)", x, y);
	return 0;
}