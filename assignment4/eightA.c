/* A star search */
#include <stdio.h>

struct BOARD {
    char cell[9];
    struct BOARD *next;
    struct BOARD *back;
    int depth,f;
};

struct BOARD B0 = {
/*    2,3,5,7,1,6,4,8,0,NULL,NULL,0,0  */
/* problem 1 */
    1,5,2,4,0,3,7,8,6,NULL,NULL,0,0
/* problem 2 */
/*    1,2,0,4,5,3,7,8,6,NULL,NULL,0,0  */
};

struct BOARD BG = {
    1,2,3,4,5,6,7,8,0
};

struct BOARD *q0 = NULL, *q2 = &B0;

struct BOARD *getq() {
    struct BOARD *q; 
    if (q2==NULL) return NULL;
    q = q2;
    q2 = q2->next;
    return (q);
}

int nc = 0;
int ns = 0;

#define FN1 "Init.dat"
#define FN2 "Goal.dat"

main(){
    struct BOARD *b;
    struct BOARD *q1;

    readdata(&B0,FN1);
    readdata(&BG,FN2);

    B0.depth = 1;

    printf("Initial state is \n",ns);
    printboard(&B0);
    printf("Goal state is \n",ns);
    printboard(&BG);
    printf("Execution begins..\n");

    
    f_value(&B0);

    while ((q1 = getq()) != NULL) {
	expand(q1);
	if (q0==NULL) q0 = q1;
	else {
	    q1 -> next = q0;
	    q0 = q1;
	}
/*	printf("Q0 is ...\n");
	printQ(q0);
	printf("Q2 is ...\n");
	printQ(q2);
 */
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


int diff(b1,b2)
struct BOARD *b1,*b2; {
    int ans=0,k;
    for (k=0;k<9;k++) if(b1->cell[k] != b2->cell[k]) ans++;
    return(ans);
}


f_value(b)
struct BOARD *b; {
    b->f = b->depth + diff(b,&BG);
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
    m-> depth = b -> depth + 1;
    if (equal(m,&BG)) { 
	f_value(b);
	bornprint(m);
	printf("**** Answer found... ****\n");
	traceback(m); 
	printf("No. of children is %d\n",nc);
	printf("Length of solution is %d\n",ns);
	exit(0); 
    }
    putq(m);
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


printQ(b)
struct BOARD *b; {
    if (b==NULL) return;
    printboard(b);
    printf(" %d ==> \n",b->f);
    printQ(b->next);
}


bornprint(b)
struct BOARD *b; {
    nc++;
    printboard(b->back);
    printf(" %d ----> %d \n",b->back->f,b->f);
    printboard(b);
   printf("\n");
}

int putq(b)
struct BOARD *b; {
    struct BOARD *n,*oldn=NULL; 
    for (n=q0; n!=NULL; n=n->next)
	if (equal(b,n)) {free(b); return; }
    if (q2==NULL) {
	f_value(b);
	q2=b;
	bornprint(b);
    }
    else {
	f_value(b);
	for (n=q2; n->next!=NULL; n=n->next)
	    if (equal(b,n)) {free(b); return; }
	for (n=q2; n!=NULL; oldn=n, n=n->next)
	    if (n->f > b->f) break;
	b -> next = n;
	if (oldn != NULL) oldn->next = b;
	else q2 = b;
	bornprint(b);
    }
}


    
