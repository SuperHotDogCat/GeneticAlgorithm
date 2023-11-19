/* breadth first search */
#include <stdio.h>

struct BOARD {
    char cell[9];
    struct BOARD *next;
    struct BOARD *back;
};

struct BOARD B0 = {
/*    2,3,5,7,1,6,4,8,0,NULL,NULL  */
    1,5,2,4,0,3,7,8,6,NULL,NULL
};

struct BOARD BG = {
    1,2,3,4,5,6,7,8,0
};

struct BOARD *q0 = NULL, *q1 = &B0, *q2 = &B0;

int nc = 0;
int ns = 0;

#define FN1 "Init.dat"
#define FN2 "Goal.dat"

main(){
    struct BOARD *b;

    readdata(&B0,FN1);
    readdata(&BG,FN2);

    printf("Initial state is \n",ns);
    printboard(&B0);
    printf("Goal state is \n",ns);
    printboard(&BG);
    printf("Execution begins..\n");
    

    while (q1 != NULL) {
	for (b=q1; b!=NULL; b=b->next) expand(b);
	q0=q1; q1=q2; q2=NULL;
    }
}

readdata(b,buf)
struct BOARD *b;
char *buf;
{
    FILE *fn;
    int num[9],k;
  
    if ((fn = fopen(buf,"r")) == (FILE *)NULL) return;
    fscanf(fn,"%1d %1d %1d %1d %1d %1d %1d %1d %1d",
	   num,num+1,num+2,num+3,num+4,num+5,
	   num+6,num+7,num+8);

    for (k=0;k<9;k++) b->cell[k] = *(num+k);

    fclose(fn);
}


expand(b)
struct BOARD *b; {
    register int i;
    for (i=0; i<9; i++)
	if (b->cell[i]==0) break;
    if (i%3>0) exchange(b,i,i-1);
    if (i>2) exchange(b,i,i-3);
    if (i%3<2) exchange(b,i,i+1);
    if (i<6) exchange(b,i,i+3);
}

exchange(b,i,j)
struct BOARD *b; int i,j; {
    struct BOARD *m;
    int k;
    m=(struct BOARD*)malloc (sizeof(struct BOARD));
    if (m==NULL) {
	printf("memory overflow\n");
	exit(2);
    }
    for (k=0;k<9;k++) m-> cell[k] = b->cell[k];
    m-> cell[i] = b->cell[j];
    m-> cell[j] = 0;
    m-> back = b;
    m-> next = NULL;
    if (equal(m,&BG)) { 
	bornprint(m);
	printf("**** Answer found... ****\n");
	traceback(m); 
	printf("No. of children is %d\n",nc);
	printf("Length of solution is %d\n",ns);
	exit(0); 
    }
    putq(m);
}

equal(b1,b2)
struct BOARD *b1,*b2; {
    register int i;
    for (i=0; i<9; i++) 
	if (b1->cell[i]!=b2->cell[i]) return 0;
    return 1;
}

traceback(b)
struct BOARD *b; {
    for (; b!=NULL; b=b->back) {
	ns++;
	printboard(b);
    }
}

printboard(b)
struct BOARD *b; {
    int i;
    for (i=0;i<9;i++) {
	if (b->cell[i]==0) putchar('X');
	else printf("%1d",b->cell[i]);
	if (i%3==2) {
	    putchar('\n');
	    if (i!=8) {
		putchar('-');
		putchar('+');
		putchar('-');
		putchar('+');
		putchar('-');
		putchar('\n');
	    }
	}
	else putchar('|');
    }
    putchar('\n');
}


bornprint(b)
struct BOARD *b; {
    nc++;
    printboard(b->back);
    printf(" ----> \n");
    printboard(b);
   printf("\n");
}

putq(b)
struct BOARD *b; {
    struct BOARD *n; 
    for (n=q0; n!=NULL; n=n->next)
	if (equal(b,n)) {free(b); return; }
    if (q2==NULL) {
	q2=b;
	bornprint(q2);
    }
    else {
	for (n=q2; n->next!=NULL; n=n->next)
	    if (equal(b,n)) {free(b); return; }
	n -> next = b;
	bornprint(b);
    }
}


    
