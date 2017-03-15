/*
 * asso.c
 *
 * Tries to solve DBP via the association method.
 *
 * Pauli Miettinen
 * 20.6.2005
 *
 * Last modified
 * 19.8.2008
 */

#include "approx.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef MATLAB
#include "matrix.h"
#endif

#ifdef MATLAB
#define MALLOC mxMalloc
#else
#define MALLOC malloc
#endif

/* A global variable for progress printing */
char progress[] = {'|', '/', '-', '\\'};

smatrix
calculate_association(matrix S, int size, int dim, options *opti);

int
solve_basis(matrix S, int size, int dim, matrix B, int k, smatrix D, 
	    matrix O, options *opti);

int 
vect_max(int *vect, int size);

int
approximate(matrix S, int size, int dim, matrix B, int k, matrix O, 
	    options *opti)
{
  smatrix D; /* let D be a sparse matrix */

  if (opti->verbose > 0) {
    fprintf(stderr, "Calculating associations...\n");
  }
  D = calculate_association(S, size, dim, opti);

  if (D == NULL)
    return 0;

  if (opti->verbose > 0) {
    fprintf(stderr, "Solving basis...\n");
  }
  return solve_basis(S, size, dim, B, k, D, O, opti);

}

smatrix
calculate_association(matrix S, int size, int dim, options *opti)
{
  int i, j, bit;
  double *A; /* We only need a row at a time */
  double sum;
  smatrix D;
  selement *ptr, *tmp;

  A = (double *)MALLOC(dim * sizeof(double));
  if (A == NULL) {
    perror("Error while allocating space for association matrix");
    return NULL;
  }

  /* Allocate space for sparse matrix D */
  D = (smatrix)MALLOC(dim * sizeof(selement *));
  if (D == NULL) {
    perror("Error while allocating space for sparse association matrix");
    return NULL;
  }
  /* Set D to be empty matrix, i.e., full of NULLs */
  memset(D, 0, dim * sizeof(selement *));

  for (bit = 0; bit < dim; bit++) {
    if (opti->verbose > 0) {
      fprintf(stderr, "\r  column %i   ", bit+1);
    }
    /* set A as zero vector */
    memset(A, 0, dim * sizeof(double));

    /* summarize over set */
    for (i = 0; i < size; i++) {
      if (S[i][bit]) {
	for (j = 0; j < dim; j++) 
	  A[j] += S[i][j];
      }
    }

    /* 'normalize' sums, A[bit] is allways largest; calculate D */
    sum = A[bit];
    ptr = tmp = NULL;
    for (i = 0; i < dim; i++) {
      A[i] /= sum;
      /* Build sparse matrix D */
      if (A[i] > opti->threshold) {
	tmp = (selement *)MALLOC(sizeof(selement));
	if (tmp == NULL) {
	  perror("Error while allocating space for sparse matrix element");
	  return NULL;
	}
	tmp->c = i;
	tmp->n = NULL;
	if (D[bit] == NULL) {
	  /* This is the first 1 in this row */
	  D[bit] = ptr = tmp;
	} else {
	  ptr->n=tmp;
	  ptr = tmp;
	}
      }
    }
  }
  if (opti->verbose > 0) {
    fprintf(stderr, "\n");
  }

  return D;
}

int
solve_basis(matrix S, int size, int dim, matrix B, int k, smatrix D, 
	    matrix O, options *opti)
{
  int i, basis, row;
  matrix covered;
  int best;
  int *best_rowcount, *rowcount;
  int best_covers, covers;
  selement *element;
  FILE *fp, *decompfp;

  /* Open basis file */
  if (opti->original_basis != NULL) {
    fp = fopen(opti->original_basis, "w");
    if (fp == NULL) {
      perror("Error when opening file for printing basis");
    }
  } else fp = NULL;

  /* Open decomposition file */
  if (opti->decomp_matrix != NULL) {
    decompfp = fopen(opti->decomp_matrix, "w");
    if (decompfp == NULL) {
      perror("Error when opening file for printing decomp");
    }
    /* Print #rows and #cols; this is a transpose */
    fprintf(decompfp, "%i\n%i\n", k, size);
  } else decompfp = NULL;

  covered = (matrix)MALLOC(size * sizeof(vector));
  if (covered == NULL) {
    perror("Error while allocating space for covered matrix");
    return 0;
  }
  for (i = 0; i < size; i++) {
    covered[i] = (vector)MALLOC(dim * sizeof(char));
    if (covered[i] == NULL) {
      perror("Error while allocating space for covered matrix");
      return 0;
    }
    memset(covered[i], 0, dim);
  }

  /* best is thus far an integer giving the correct row */

  best_rowcount = (int *)MALLOC(size * sizeof(int));
  if (best_rowcount == NULL) {
    perror("Error while allocating space for 'best_rowcount' vector");
    return 0;
  }

  rowcount = (int *)MALLOC(size * sizeof(int));
  if (rowcount == NULL) {
    perror("Error while allocating space for 'rowcount' vector");
    return 0;
  }

  /* iterate thru all asked basis vectors */
  for (basis = 0; basis < k; basis++) {
    if (opti->verbose > 0) {
      fprintf(stderr, "\r  basis vector %i   ", basis+1);
    }

    best = -1;
    best_covers = 0;
    memset(best_rowcount, 0, size * sizeof(int));
  
    /* iterate thru all rows in D */
    for (row = 0; row < dim; row++) {
      /* find best row */

      if (opti->verbose > 0) fprintf(stderr, "\b%c", progress[row%4]);

      covers = 0;
      memset(rowcount, 0, size * sizeof(int));

      for (i = 0; i < size; i++) {
	element = D[row];
	while (element != NULL) {
	  /* UPDATE: penalize covered 0s only when somebody else doesn't
	   * cover them already! */
	  if (S[i][element->c] == 0) rowcount[i] -= opti->penalty_overcovered
	    * (1 - covered[i][element->c]);
	  else rowcount[i] += opti->bonus_covered * (S[i][element->c] 
						     - covered[i][element->c]);
	  element = element->n;
	}
	if (rowcount[i] > 0) covers += rowcount[i];
      }

      if (covers > best_covers) {
	/* we found the best */
	best_covers = covers;
	/* 'best' is this row */
	best = row;
	memcpy(best_rowcount, rowcount, size*sizeof(int));
      }
    }


    if (best > -1) {
      /* We have found the best - if best == -1, this basis vector will be
       * empty.
       */
      for (i = 0; i < size; i++) {
	if (best_rowcount[i] > 0) {
	  element = D[best];
	  while (element != NULL) {
	    covered[i][element->c] = 1;
	    element = element->n;
	  }
	}
      }
      element = D[best];
      while (element != NULL) {
	B[basis][element->c] = 1;
	element = element->n;
      }
    }

    /* And finally, let's print each basis vector as soon as they are ready */
    if (fp != NULL) {
      for (i=0; i < dim; i++) fprintf(fp, "%c ", '0'+B[basis][i]);
      fprintf(fp, "\n");
      fflush(fp);
    }
    for (i=0; i < size; i++) {
      /* set cols of O correctly */
      if (best_rowcount[i] > 0)
	O[i][basis] = 1;
      /* And print each decomp vector, too */
      if (decompfp != NULL) {
	fprintf(decompfp, "%c ", (best_rowcount[i] > 0) ? '1' : '0');
      }
    }
    if (decompfp != NULL) {
      fprintf(decompfp, "\n");
      fflush(decompfp);
    }
  }
  if (opti->verbose > 0) {
    fprintf(stderr, "\n");
  }

  if (fp != NULL) fclose(fp);
  if (decompfp != NULL) fclose(decompfp);
  
  return 1; 
}


int
vect_max(int *vect, int size)
{
  int i, best;
  best = vect[0];
  for (i = 1; i < size; i++) {
    if (vect[i] > best)
      best = vect[i];
  }
  return best;
}

void
approx_help()
{
  printf("-t, --threshold=f\n"
	 "\t Threshold giving the floating-point number where to discretize.\n"
	 "\t Defaults to 1.0.\n"
	 "-B, --temporary-basis=FILE\n"
	 "\t A file where the basis will be printed during computation.\n"
	 "-p, --bonus-covered=p\n"
	 "\t Bias the object function to give `p' times more points for\n"
	 "\t covering 1s.\n");
  printf("-P, --penalty-overcovered=P\n"
	 "\t Bias the object function to penalize each covered 0 by 'P'.\n"
	 "-D, --decomp-matrix=FILE\n"
	 "\t The file where the decomposition matrix will be printed.\n"
	 "\t The decomposition matrix will be printed as a transpose.\n\n");
}
