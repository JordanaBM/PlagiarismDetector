#include <stdio.h>

int main() {
	
	int n,i,j;
	do{
	printf("Unesite broj n: ");
	scanf("%d",&n);
	if(n<=0||n>50) 
	 
	   printf("Pogresan unos\n");
	   
	  }while(n<=0||n>50);
	if(n==1) printf("***");
	else{
	for(i=0;i<n;i++)
	{
		for(j=0;j<=4*(n-1);j++)
		{
			
		    if(i==j&&j<=n-1) printf("*");
			else if(i==2*(n-1)-j &&j>n-1&&j<=2*(n-1)) printf("*");
			else if(i+(2*(n-1))==j &&j>2*(n-1)&&j<=3*(n-1)) printf("*");
			else if(i==4*(n-1)-j&&j>3*(n-1)&&j<=4*(n-1)) printf("*");
			
			else printf(" ");
		}
		printf("\n");
	}
	}
	return 0;
}
