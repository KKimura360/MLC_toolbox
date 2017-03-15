#ifndef APPROX
#define APPROX
/* Structures */

struct options_s {
  unsigned int error_max;
  unsigned int cut_size;
  unsigned int noisy_vectors;
  unsigned int iterations;
  unsigned int remove_covered;
  unsigned int seed;
  unsigned int verbose;
  char *original_basis;
  double threshold;
  char majority;
  unsigned int bonus_covered;
  unsigned int penalty_overcovered;
  char *decomp_matrix;
};

/* type definitions */

#ifndef DBP_TYPES /* to make sure that we include these just once*/
#define DBP_TYPES
typedef char *vector;
typedef char **matrix;
typedef unsigned long int *ivector; /* to save integer vectors */
typedef unsigned long int **imatrix; /*       "        matrices */
#endif
typedef struct options_s options;

/* procedures */

int approximate(matrix Set, 
		int size, 
		int dim, 
		matrix B, 
		int k, 
		matrix O,
		options *opti);

void approx_help();

#endif
