#include <stdio.h>
#include <math.h>

int main() {
	float a1,b1,a2,b2,x,y,e=0.000001;
	printf("Unesite a1,b1,a2,b2: ");
	scanf("%f,%f,%f,%f",&a1, &b1, &a2, &b2);
	x=(b2-b1)/(a1-a2);
	y=a1*x+b1;
	
	if (fabs(a1-a2)<=e && fabs(b1-b2)>e)
	printf("Paralelne su");
	else 
	if (fabs(a1-a2)<=e && fabs(b1-b2)<=e)
	printf ("Poklapaju se");
	else
	printf("Prave se sijeku u tacci (%.1f,%.1f)",x,y);
	
	return 0;
}
