#include <stdio.h>
#define epsilon 0.01
#include <math.h>

int main() {
	float a1,b1,a2,b2,y,x;
	printf ("Unesite a1,b1,a2,b2: ");
	scanf ("%f,%f,%f,%f", &a1,&b1,&a2,&b2);
	if (fabs(a1-a2)<epsilon && fabs(b1-b2)>epsilon)
	{
		printf ("Paralelne su");
		return 0;
	}
	else if (fabs(a1-a2)<epsilon && fabs(b1-b2)<epsilon)
	{
		printf ("Poklapaju se");
		return 0;
	}
	
	else 
	{
	x=(b2-b1)/(a1-a2);
	y=a1*x+b1;
	printf ("Prave se sijeku u tacci (%.1f,%.1f)", x,y);
	return 0;
	}
	return 0;
}
