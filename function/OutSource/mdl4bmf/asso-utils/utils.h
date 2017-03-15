#ifndef _DBP_UTILS
#define _DBP_UTILS

#include <stdio.h>

#ifndef DBP_TYPES /* to make sure that we typedef these just once */
#define DBP_TYPES
typedef char *vector;
typedef char **matrix;
typedef unsigned long int *ivector; /* to save integer vectors */
typedef unsigned long int **imatrix; /*       "        matrices */
#endif

#define MAX_LINELENGTH 4096 /* Longest line in sparse input matrix */

/* Some new types here... */

struct selestr {
  unsigned int c;
  struct selestr *n;
};

typedef struct selestr selement;
typedef selement **smatrix;

matrix 
read_matrix(const char *file, int *s, int *d);

matrix
read_sparse_matrix(const char *file, int *s, int *d);

int 
print_matrix(const char *file, matrix S, int s, int d);

int
print_sparse_matrix(const char *file, matrix S, int s, int d);

void 
free_matrix(matrix M, int n);

/* Initials a seed for random number generator and return NULL if 'seed' != 0.
 * Otherwise returns a pointer to random number character device to be used
 * with 'give_rand()'.
 */
FILE *
init_seed(unsigned int seed);

/* Returns a random number as an unsigned integer. If 'randdev' is is NULL,
 * uses standard library 'rand()', otherwise uses given character device. */
unsigned int
give_rand(FILE *randdev);


#endif
