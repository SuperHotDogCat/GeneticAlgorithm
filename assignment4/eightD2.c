/* depth first search */
#include <stdio.h>

struct BOARD {
    char cell[9];
    struct BOARD *next;
    struct BOARD *back;
    int depth;
};

struct BOARD B0 = {
/*    2,3,5,7,1,6,4,8,0,NULL,NULL  */
/* problem 1 */
/*     1,5,2,4,0,3,7,8,6,NULL,NULL */ 
/* problem 2 */
    1,2,0,4,5,3,7,8,6,NULL,NULL
};

struct BOARD BG = {
    1,2,3,4,5,6,7,8,0
};

struct BOARD *q0 = NULL, *q2 = &B0;

int nc = 0;
int ns = 0;

#define FN1 "Init.dat"
#define FN2 "Goal.dat"

int depth_limit = 4;
int depth_increase = 6;

struct BOARD *getq() {
    struct BOARD *q,*oldn,*n; 
    if (q2==NULL) return NULL;

    if (depth_limit < 0) {
	q = q2;
	q2 = q2->next;
	return (q);
    }

    while (1){
	oldn = NULL;
	for (n=q2; n != NULL; oldn=n, n=n->next)
	    if (n->depth <= depth_limit) break;
	if (n==NULL) {
	    depth_limit = depth_limit + depth_increase;
	    continue;
	}
	if (oldn==NULL) q2 = n->next;
	else oldn->next = n->next;
	return(n);
    }
}


main(){
    struct BOARD *b;
    struct BOARD *q1;


    readdata(&B0,FN1);
    readdata(&BG,FN2);

    B0.depth = 0;

    printf("Initial state is \n",ns);
    printboard(&B0);
    printf("Goal state is \n",ns);
    printboard(&BG);
    printf("Execution begins..\n");
    

    while ((q1 = getq()) != NULL) {
	expand(q1);
	if (q0==NULL) q0 = q1;
	else {
	    q1 -> next = q0;
	    q0 = q1;
	}
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


struct BOARD *exchange(b,i,j)
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
	bornprint(m);
	printf("**** Answer found... ****\n");
	traceback(m); 
	printf("No. of children is %d\n",nc);
	printf("Length of solution is %d\n",ns);
	exit(0); 
    }
    return(m);
}

expand(b)
struct BOARD *b; {
struct BOARD *m1=NULL,*m2=NULL,*m3=NULL,*m4=NULL;
    register int i, k1,k2,k3,k4;
    for (i=0; i<9; i++)
	if (b->cell[i]==0) break;

    if (i%3>0) m1=exchange(b,i,i-1);
    if (i>2) m2=exchange(b,i,i-3);
    if (i%3<2) m3=exchange(b,i,i+1);
    if (i<6) m4=exchange(b,i,i+3);

    if (m4!=NULL) k4=putq(m4);
    if (m3!=NULL) k3=putq(m3);
    if (m2!=NULL) k2=putq(m2);
    if (m1!=NULL) k1=putq(m1);

    if ((m1!=NULL)&&(k1==1)) bornprint(m1);
    if ((m2!=NULL)&&(k2==1)) bornprint(m2);
    if ((m3!=NULL)&&(k3==1)) bornprint(m3);
    if ((m4!=NULL)&&(k4==1)) bornprint(m4);

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

int putq(b)
struct BOARD *b; {
    struct BOARD *n; 
    for (n=q0; n!=NULL; n=n->next)
	if (equal(b,n)) {free(b); return 0; }
    if (q2==NULL) {
	q2=b;
	return 1;
    }
    else {
	for (n=q2; n->next!=NULL; n=n->next)
	    if (equal(b,n)) {free(b); return 0; }
	b -> next = q2;
	q2 = b;
	return 1;
    }
}


    
