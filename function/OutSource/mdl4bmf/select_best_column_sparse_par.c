#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include "mex.h"
#include "matrix.h"

#define element(rows, row, col) (rows*col + row)
#define MAX(a,b) ((a>b)?a:b)
#define MIN(a,b) ((a<b)?a:b)


/* The struct that contains all necessary pointers for thread-worker */
typedef struct
{
  /* Input variables; read-only outside mexFunction */
  mwSize n; 
  mwSize m;
  mwSize c;
  const mwIndex *AIr;
  const mwIndex *AJc;
  const mwIndex *CIr;
  const mwIndex *CJc;
  const uint8_t * restrict mask;
  const mwIndex * restrict available_cols;
  mwSize *slices;
  /* Output variables; need mutex */
  long coverval;
  double * restrict id;
  double * restrict r;
} mydata_t;

/* Global data pointers */
mydata_t data;

/* Mutex */
pthread_mutex_t mutexoutput;


/* The worker function */
/*
static long
cover(const uint8_t * restrict A, 
      const mwSize n, 
      const mwSize m, 
      const uint8_t * restrict mask, 
      const uint8_t * restrict C, 
      const mwSize j, 
      long long * restrict tmpR);
*/
static long
cover(const mwSize j,
      long long * restrict tmpR);
 
/* Thread-worker */
void *
threadWorker(void *id);

 
/*
 * [id, r] = select_best_column(A, C, mask, parnum)
 *
 * Selects the best column of C to add to an BMF of A of which mask elements 
 * are already covered. Returns the index of the column in C, and r, the row 
 * to define the use of the column. Lauches parnum threads (default = 1) to 
 * do the computation.
 *
 */ 

void 
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  mwSize n, m, c, foo;
  double * restrict id, * restrict r, * restrict tmp;
  long long * restrict tmpR;
  uint8_t * restrict mask;
  mwIndex * restrict available_cols;
  mwSize i, j;
  long tmpCoverval, coverval;
  int parnum; 
  unsigned tnum;
  pthread_attr_t attr;
  pthread_t *callThd;

  /* Check for input and output argument numbers */
  if (nrhs < 4) 
    mexErrMsgTxt("Four input arguments, A, C, available_cols, and mask, required.");
  if (nrhs > 5)
    mexErrMsgTxt("At most five input arguments allowed.");
  if (nlhs != 2)
    mexErrMsgTxt("Two output arguments required.");
  
  /* Check and load input matrices. */
  if (!mxIsSparse(prhs[0]))
    mexErrMsgTxt("Data matrix (first input) must be sparse.");
  /*A = (uint8_t *)mxGetData(prhs[0]);*/
  n = mxGetM(prhs[0]);
  m = mxGetN(prhs[0]);
  
  if (!mxIsSparse(prhs[1]))
    mexErrMsgTxt("Candidate matrix (second input) must be sparse.");
  /*C = (uint8_t *)mxGetData(prhs[1]);*/
  foo = mxGetM(prhs[1]);
  if (foo != n) 
    mexErrMsgTxt("Matrix C must have same number of rows as A.");
  /*c = mxGetN(prhs[1]);*/
  

  foo = mxGetM(prhs[2]);
  c = mxGetN(prhs[2]);
  if (MIN(c, foo) != 1) 
    mexErrMsgTxt("available_cols must be a vector");
  c = MAX(c, foo);

  /* DEBUG */
  //fprintf(stderr, "DEBUG: c = %i\n", c);

  /* Allocate space for indices */
  available_cols = (mwIndex *)mxMalloc(c*sizeof(mwIndex));
  tmp = mxGetPr(prhs[2]);
  for (i=0; i < c; i++)
    available_cols[i] = (mwIndex)tmp[i];

  if (!mxIsUint8(prhs[3]))
    mexErrMsgTxt("Mask matrix (third input) must be of uint8_t type.");
  mask = (uint8_t *)mxGetData(prhs[3]);
  foo = mxGetM(prhs[3]);
  if (foo != n)
    mexErrMsgTxt("Mask must have same number of rows as A.");
  foo = mxGetN(prhs[3]);
  if (foo != m)
    mexErrMsgTxt("Mask must have same number of columns as A.");
  
  if (nrhs == 5) {
    parnum = (int)mxGetScalar(prhs[4]);
    if (parnum < 1)
      mexErrMsgTxt("Number of threads must be positive.");
  } else {
    parnum = 1;
  }

  //fprintf(stderr, "DEBUG: done input data initialization\n");

  /* Output argument id */
  plhs[0] = mxCreateDoubleScalar(mxGetNaN());
  id = mxGetPr(plhs[0]);
  
  /* Output argument r */
  plhs[1] = mxCreateDoubleMatrix(1, m, mxREAL);
  r = mxGetPr(plhs[1]);
  
  /* Copy all this to data */
  data.n = n;
  data.m = m;
  data.c = c;
  data.AIr = mxGetIr(prhs[0]);
  data.AJc = mxGetJc(prhs[0]);
  data.CIr = mxGetIr(prhs[1]);
  data.CJc = mxGetJc(prhs[1]);
  data.available_cols = available_cols;
  data.mask = mask;
  data.coverval = -1;
  data.id = id;
  data.r = r;

  /* Initialize slices */
  data.slices = (mwSize *)mxMalloc((parnum+1)*sizeof(mwSize));
  data.slices[0] = 0;
  for (tnum = 1; tnum < parnum; tnum++)
    data.slices[tnum] = (mwSize)tnum*c/parnum;
  data.slices[parnum] = c;

  //fprintf(stderr, "DEBUG: done output data initialization\n");

  /* Allocate space for pointers to threads */
  callThd = (pthread_t *)mxMalloc(parnum*sizeof(pthread_t));

  /* Init mutex */
  pthread_mutex_init(&mutexoutput, NULL);

  /* Assign attributes */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  //fprintf(stderr, "DEBUG: done thread data initialization\n");

  /* Create threads */
  for (tnum=0; tnum < parnum; tnum++)
    pthread_create(&callThd[tnum], &attr, threadWorker, (void *)tnum);

  pthread_attr_destroy(&attr);

  /* Wait for threads */
  for (tnum=0; tnum < parnum; tnum++)
    pthread_join(callThd[tnum], NULL);


  //fprintf(stderr, "DEBUG: cleaning up\n");

  /* Clean-up */
  mxFree(callThd);
  pthread_mutex_destroy(&mutexoutput);
  mxFree(data.slices);
  mxFree(available_cols);
  //fprintf(stderr, "DEBUG: that's all folks\n");



}


/*
 * Thread worker; runs cover() for the slice of columns and updates coverval 
 * and others if got better results; honors mutex
 */

void *
threadWorker(void *tmpId)
{
  mwSize i, j;
  long coverVal;
  long long *R;
  unsigned id;

  /* convert tmpId */
  id = (unsigned)tmpId;


  //fprintf(stderr, "DEBUG: in thread %u\n", id);

  /* allocate R */
  R = (long long *)malloc(data.m*sizeof(long long));
  if (R == NULL){
    fprintf(stderr, "Memory allocation failed. Aborting.\n");
    pthread_exit((void *)1);
  }

  //fprintf(stderr, "DEBUG[%u]: memory allocation done\n", id);

  /* run cover */
  for (i=data.slices[id]; i < data.slices[id+1]; i++) {
    /*coverVal = cover(data.A, data.n, data.m, data.mask, data.C, i, R);*/
    //fprintf(stderr, "DEBUG[%u]: calling cover for column %i\n", id, (int)i);
    coverVal = cover(i, R);
    /* honor mutex */
    //fprintf(stderr, "DEBUG[%u]: calling mutex\n", id);
    pthread_mutex_lock(&mutexoutput);
    //fprintf(stderr, "DEBUG[%u]: within mutex\n", id);
    if (coverVal > data.coverval) { /* better than best-yet */
      data.coverval = coverVal;
      data.id[0] = (double)(i+1); /* correct for matlab-style indices */
      for (j=0; j < data.m; j++) /* copy row R to data.r */
	data.r[j] = (double)R[j];
    }
    /* open mutex */
    //fprintf(stderr, "DEBUG[%u]: releasing mutex\n", id);
    pthread_mutex_unlock(&mutexoutput);
  }

  /* free R */
  //fprintf(stderr, "DEBUG[%u]: freeing R\n", id);
  free(R);

  pthread_exit((void *)0);
}


/*
 * Return the overall cover value for column c of C and the corresponding 
 * usage row in tmpR.
 */

static long
cover(const mwSize idx,
      long long * restrict tmpR)
{
  mwIndex Acol, Ccol, j;
  
  const mwIndex *CJc; 
  const mwIndex *CIr; 
  const mwIndex *AJc;
  const mwIndex *AIr;
  long coverval;

  /* DEBUG */
  //fprintf(stderr, "DEBUG: In cover for idx = %i\n", idx); 

  /* Get the column of C we're working with */
  Ccol = data.available_cols[idx] - 1; /* Correct from Matlab indices */
  CJc = data.CJc; /*mxGetJc(data.C); */
  CIr = data.CIr; /*mxGetIr(data.C); */

  /* Get the vectors for A */
  AJc = data.AJc; /*mxGetJc(data.A); */
  AIr = data.AIr; /*mxGetIr(data.A); */

  /* DEBUG */
  //fprintf(stderr, "DEBUG: at cover for column %i (%u)\n", idx, Ccol);

  for (Acol=0; Acol<data.m; Acol++) {
    /* For all columns of A */
    long long v = 0;
    mwIndex Arow, Crow;
    
    Arow = AJc[Acol];
    Crow = CJc[Ccol];
        
    while (Arow < AJc[Acol+1] || Crow < CJc[Ccol+1]) {
      if (AIr[Arow] == CIr[Crow] 
	  && data.mask[element(data.n, AIr[Arow], Acol)] == 0) {
	/* A(i,j) = 1 and C(i, Ccol) = 1 and A(i,j) is not covered */
	v++;
	Arow++;
	Crow++;
      } else if (AIr[Arow] < CIr[Crow]) {
	/* There is an uncovered 1 in A */
	/* Advance A if possible; o/w pay for C */
	if (Arow < AJc[Acol+1]) Arow++;
	else {
	  /* C must cover some 0s, check if they're masked */
	  if (data.mask[element(data.n, CIr[Crow], Acol)] == 0)
	    v--; /* Element is not masked */
	  Crow++; /* Advance Crow */
	}
      } else { /*if (CIr[Crow] < AIr[Arow]) { */
	/* Either we are at the bottom of C or have covered something */
	if (Crow < CJc[Ccol+1]) {
	  /* We have covered something, is it masked */
	  if (data.mask[element(data.n, CIr[Crow], Acol)] == 0)
	    v--; /* Element is not masked */
	  Crow++; /* Advance Crow */
	}
	else Arow++;
      }
    }
    
    tmpR[Acol] = v;
  }

  /* Now we know, for each col of A, should we use idx to cover it or not */
  /* Let's update tempR and sum up the coverval */

  coverval = 0;

  for (j=0; j<data.m; j++) {
    if (tmpR[j] > 0) {
      /* Use idx to cover j */
      coverval += tmpR[j]; /* add to coverval */
      tmpR[j] = 1; /* Booleanize tmpR */
    } else {
      tmpR[j] = 0; /* Booleanize tmpR */
    }
  }
  return coverval;
}
