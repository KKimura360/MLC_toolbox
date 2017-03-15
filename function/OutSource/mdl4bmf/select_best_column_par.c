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
  uint8_t * restrict A;
  uint8_t * restrict mask;
  uint8_t * restrict C;
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
  double * restrict id, * restrict r; 
  long long * restrict tmpR;
  uint8_t * restrict A, * restrict C, * restrict mask;
  mwSize i, j;
  long tmpCoverval, coverval;
  int parnum; 
  unsigned tnum;
  pthread_attr_t attr;
  pthread_t *callThd;

  /* Check for input and output argument numbers */
  if (nrhs < 3) 
    mexErrMsgTxt("Three input arguments, A, C, and mask, required.");
  if (nrhs > 4)
    mexErrMsgTxt("At most four input arguments allowed.");
  if (nlhs != 2)
    mexErrMsgTxt("Two output arguments required.");
  
  /* Check and load input matrices. */
  if (!mxIsUint8(prhs[0]))
    mexErrMsgTxt("Input matrices must be of uint8_t; first is not.");
  A = (uint8_t *)mxGetData(prhs[0]);
  n = mxGetM(prhs[0]);
  m = mxGetN(prhs[0]);
  
  if (!mxIsUint8(prhs[1]))
    mexErrMsgTxt("Input matrices must be of uint8_t; second is not.");
  C = (uint8_t *)mxGetData(prhs[1]);
  foo = mxGetM(prhs[1]);
  if (foo != n) 
    mexErrMsgTxt("Matrix C must have same number of rows as A.");
  c = mxGetN(prhs[1]);
  
  if (!mxIsUint8(prhs[2]))
    mexErrMsgTxt("Input matrices must be of uint8_t; third is not.");
  mask = (uint8_t *)mxGetData(prhs[2]);
  foo = mxGetM(prhs[2]);
  if (foo != n)
    mexErrMsgTxt("Mask must have same number of rows as A.");
  foo = mxGetN(prhs[2]);
  if (foo != m)
    mexErrMsgTxt("Mask must have same number of columns as A.");
  
  if (nrhs == 4) {
    parnum = (int)mxGetScalar(prhs[3]);
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
  data.A = A;
  data.C = C;
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
  //fprintf(stderr, "DEBUG: that's all folks\n");



  /* historical code, comment out */
  #if 0
  /* Initialize temporary row and "best" covervalue */
  coverval = -1; /* by definition, cover() returns values >= 0. */
  tmpR = (long long *)mxMalloc(m*sizeof(long long));
  /* Try each column */
  
  for (i=0; i<c; i++) {
    tmpCoverval = cover(A, n, m, mask, C, i, tmpR);
    if (tmpCoverval > coverval) {
      coverval = tmpCoverval;
      id[0] = (double)(i+1); /* correct for Matlab-style indices */
      for (j=0; j < m; j++) /* copy the row tmpR to r*/
	r[j] = (double)tmpR[j];
    }
  }
  #endif
  /* end historical data commenting */
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
/*
static long
cover(const uint8_t * restrict A, 
      const mwSize n, 
      const mwSize m, 
      const uint8_t * restrict mask, 
      const uint8_t * restrict C, 
      const mwSize idx, 
      long long * restrict tmpR)
*/
static long
cover(const mwSize idx,
      long long * restrict tmpR)
{
  mwSize i, j;
  long coverval;
 
  /* Empty tmpR */
  /* Obsolete, will use v instead */
  /* memset(tmpR, 0, m*sizeof(long long)); */

  for (j=0; j<data.m; j++) {
    /* For all columns of A */
    long long v = 0;
    for (i=0; i<data.n; i++) {
      /* How good column idx is in covering column j */
      if (data.mask[element(data.n, i, j)] == 0 
	  && data.A[element(data.n, i, j)] == 1 
	  && data.C[element(data.n, i, idx)] == 1) {
	/* A(i,j) = 1 and C(i,idx) = 1, and A(i,j) is not covered */
	//tmpR[j]++;
	v++;
      } else if (data.mask[element(data.n, i, j)] == 0 
		 && data.A[element(data.n, i, j)] == 0
		 && data.C[element(data.n, i, idx)] == 1) {
	/* This would cover an uncovered zero element */
	//tmpR[j]--;
	v--;
      }
    }
    tmpR[j] = v;
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
