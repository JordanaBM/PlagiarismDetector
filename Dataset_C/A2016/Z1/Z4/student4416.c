#include <stdio.h>

int main() {
	int n,i,j;
	for (;;) {
	printf("Unesite broj n: ");		
	scanf("%d",&n);
	if (n<=0 || n>50) printf ("Pogresan unos\n");
	else {
		 if (n==1) { printf ("***"); return 0;
		 }
		 else
		 {
			for (i=1;i<=n;i++)
			{
				for (j=1;j<=4*n-3;j++) {
					if (i==j || i+j==2*n || j-i==2*n-2 || j+i==4*n-2)
					printf ("*");
					else printf (" ");
			}
			printf ("\n");
	}
	}
	return 0;
	}
	}
	return 0;
}
