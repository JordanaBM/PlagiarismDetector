#include <stdio.h>
int main ()
{
	float a1,b1,a2,b2,y1,x,y2;
	printf("Unesite a1,b1,a2,b2: ");
	scanf("%f, %f , %f , %f", &a1,&b1,&a2,&b2);
	y1=a1*x+b1;
	y2=a2*x+b2;
	if(a1==a2 && b1!=b2){
		printf("Paralelne su");
	}
	if(a1!=a2){
		x=(b1-b2)/(a2-a1);
		y1=a1*x+b1;
		printf("Prave se sijeku u tacci ""(%.1f,%.1f)",x,y1);
	}
	if(a1==a2 && b1==b2){
		printf("Poklapaju se");
	}
	return 0;
}
