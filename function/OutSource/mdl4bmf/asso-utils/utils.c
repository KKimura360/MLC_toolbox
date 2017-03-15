/*
 * utils.c
 *
 * Some useful utils to use with programs.
 */
#include "utils.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


/* Portability code */
#ifndef _GNU_SOURCE
char *program_invocation_short_name = "";
#endif

matrix 
read_matrix(const char *file, int *s, int *d) 
{
  int i, j, value;
  FILE *fp;
  matrix B;

  if (file[0] == '-' && file[1] == '\0') fp = stdin;
  else {
    if ((fp=fopen(file, "r")) == NULL) {
      fprintf(stderr, "%s: Couldn't open file %s; %s\n",
	      program_invocation_short_name, file, strerror(errno));
      return NULL;
    }
    if (setvbuf(fp, NULL, _IOFBF, BUFSIZ)) {
      fprintf(stderr, "%s: Couldn't set buffer for file %s; %s\n",
	      program_invocation_short_name, file, strerror(errno));
      return NULL;
    }
  }

  if (fscanf(fp, "%i %i", s, d) == EOF) {
    fprintf(stderr, "%s: Error when reading file %s; %s\n",
	    program_invocation_short_name, file, strerror(errno));
    if (fp!=stdin) fclose(fp);
    return NULL;
  }

  if ((B = malloc((*s) * sizeof(vector))) == NULL) {
    fprintf(stderr, "%s: Error when allocating space; %s\n",
	    program_invocation_short_name, strerror(errno));
    if (fp!=stdin) fclose(fp);
    return NULL;
  }
  for (i=0; i<(*s); i++) {
    if ((B[i] = malloc((*d) * sizeof(char))) == NULL) {
      fprintf(stderr, "%s: Error when allocating space; %s\n",
	      program_invocation_short_name, strerror(errno));
      if (fp!=stdin) fclose(fp);
      free_matrix(B, i);
      return NULL;
    }
  }

  for (i=0; i<(*s); i++) {
    for (j=0; j<(*d); j++) {
      if (fscanf(fp, "%i", &value) == EOF) {
	fprintf(stderr, "%s: Error when reading file %s; %s\n",
		program_invocation_short_name, file, strerror(errno));
	if (fp != stdin) fclose(fp);
	free_matrix(B, *s);
	return NULL;
      }
      B[i][j] = (char)value;
    }
  }

  return B;
}


matrix
read_sparse_matrix(const char *file, int *s, int *d)
{
  int i, value, ones, j;
  FILE *fp;
  matrix B;


  if (file[0] == '-' && file[1] == '\0') fp = stdin;
  else {
    if ((fp=fopen(file, "r")) == NULL) {
      fprintf(stderr, "%s: Couldn't open file %s; %s\n",
	      program_invocation_short_name, file, strerror(errno));
      return NULL;
    }
    if (setvbuf(fp, NULL, _IOFBF, BUFSIZ)) {
      fprintf(stderr, "%s: Couldn't set buffer for file %s; %s\n",
	      program_invocation_short_name, file, strerror(errno));
      return NULL;
    }
  }

  if (fscanf(fp, "%i %i", s, d) == EOF) {
    fprintf(stderr, "%s: Error when reading file %s; %s\n",
	    program_invocation_short_name, file, strerror(errno));
    if (fp!=stdin) fclose(fp);
    return NULL;
  }

  if ((B = malloc((*s) * sizeof(vector))) == NULL) {
    fprintf(stderr, "%s: Error when allocating space; %s\n",
	    program_invocation_short_name, strerror(errno));
    if (fp!=stdin) fclose(fp);
    return NULL;
  }
  for (i=0; i<(*s); i++) {
    if ((B[i] = malloc((*d) * sizeof(char))) == NULL) {
      fprintf(stderr, "%s: Error when allocating space; %s\n",
	      program_invocation_short_name, strerror(errno));
      if (fp!=stdin) fclose(fp);
      free_matrix(B, i);
      return NULL;
    }
    memset(B[i], 0, *d);
  }


  /* Read the matrix; at first there should be an integer representing
   * how many 1s there are in this row.
   */
  for (i = 0; i < *s; i++) {
    if (fscanf(fp, "%i", &ones) == EOF) {
      fprintf(stderr, "%s: Error when reading file %s; %s\n",
	      program_invocation_short_name, file, strerror(errno));
      return NULL;
    }
    for (j = 0; j < ones; j++) {
      if (fscanf(fp, "%i", &value) == EOF) {
	fprintf(stderr, "%s: Error when reading file %s; %s\n",
		program_invocation_short_name, file, strerror(errno));
	return NULL;
      }
      if (value < 1) {
	fprintf(stderr, "%s: Error when reading file %s: value %i (row %i, "
		"%i:th value) lesser than 1.\n", 
		program_invocation_short_name, file, value, i, j);
	return NULL;
      } else if (value > *d) {
	fprintf(stderr, "%s: Error when reading file %s: value %i (row %i, "
		"%i:th value) greater than number of columns (%i)\n", 
		program_invocation_short_name, file, value, i, j, *d);
	return NULL;
      }
      B[i][value-1] = 1;
    }
  }

  return B;
}


int 
print_matrix(const char *file, matrix S, int s, int d) 
{
  int i, j, c;
  FILE *fp;

  if (file[0] == '-' && file[1] == '\0') fp = stdout;
  else {
    fp = fopen(file, "w");
    if (fp == NULL) {
      fprintf(stderr, "%s: Couldn't open file %s for writing; %s\n",
	      program_invocation_short_name, file, strerror(errno));
      return 0;
    }
    if (setvbuf(fp, NULL, _IOFBF, BUFSIZ)) {
      fprintf(stderr, "%s: Couldn't set buffer for file %s; %s\n",
	      program_invocation_short_name, file, strerror(errno));
      fclose(fp);
      return 0;
    }
  }

  c = fprintf(fp, "%i\n%i\n", s, d);
  if (c < 0) {
    fprintf(stderr, "%s: Couldn't write to file %s; %s\n",
	    program_invocation_short_name, file, strerror(errno));
    if (fp != stdout) fclose(fp);
    return 0;
  }
  for (i=0; i<s; i++) {
    for (j=0; j<d; j++) {
      c = fprintf(fp, "%i ", (int)S[i][j]);
      if (c < 0) {
	fprintf(stderr, "%s: Couldn't write to file %s; %s\n",
		program_invocation_short_name, file, strerror(errno));
	if (fp != stdout) fclose(fp);
	return 0;
      }
    }
    if (putc('\n', fp) == EOF) {
      fprintf(stderr, "%s: Couldn't write to file %s; %s\n",
	      program_invocation_short_name, file, strerror(errno));
      if (fp != stdout) fclose(fp);
      return 0;
    }
  }

  if (fp != stdout)
    fclose(fp);
  return 1;
}

int
print_sparse_matrix(const char *file, matrix S, int s, int d)
{
  int i, j, ones, c;
  FILE *fp;
  if (file[0] == '-' && file[1] == '\0') fp = stdout;
  else {
    fp = fopen(file, "w");
    if (fp == NULL) {
      fprintf(stderr, "%s: Couldn't open file %s for writing; %s\n",
	      program_invocation_short_name, file, strerror(errno));
      return 0;
    }
    if (setvbuf(fp, NULL, _IOFBF, BUFSIZ)) {
      fprintf(stderr, "%s: Couldn't set buffer for file %s; %s\n",
	      program_invocation_short_name, file, strerror(errno));
      fclose(fp);
      return 0;
    }
  }

  c = fprintf(fp, "%i\n%i\n", s, d);
  if (c < 0) {
    fprintf(stderr, "%s: Couldn't write to file %s; %s\n",
	    program_invocation_short_name, file, strerror(errno));
    if (fp != stdout) fclose(fp);
    return 0;
  }
  for (i=0; i<s; i++) {
    /* compute ones */
    for (j=0, ones=0; j<d; j++) ones+=S[i][j];
    /* print ones */
    c = fprintf(fp, "%i ", ones);
    if (c < 0) {
      fprintf(stderr, "%s: Couldn't write to file %s; %s\n",
	      program_invocation_short_name, file, strerror(errno));
      if (fp != stdout) fclose(fp);
      return 0;
    }
    for (j=0; j<d; j++) {
      if (S[i][j]) {
	c = fprintf(fp, "%i ", j+1);
	if (c < 0) {
	  fprintf(stderr, "%s: Couldn't write to file %s; %s\n",
		  program_invocation_short_name, file, strerror(errno));
	  if (fp != stdout) fclose(fp);
	  return 0;
	}
      }
    }
    if (putc('\n', fp) == EOF) {
      fprintf(stderr, "%s: Couldn't write to file %s; %s\n",
	      program_invocation_short_name, file, strerror(errno));
      if (fp != stdout) fclose(fp);
      return 0;
    }
  }

  if (fp != stdout)
    fclose(fp);
  return 1;
}


void 
free_matrix(matrix M, int n) 
{
  int i;

  for (i=0; i<n; i++) free(M[i]);
  free(M);
}


FILE *
init_seed(unsigned int seed)
{
  FILE *fp = NULL;

  if (seed != 0) {
    srand(seed);
  } else {
    fp = fopen("/dev/urandom", "r");
    if (fp == NULL) {
      fprintf(stderr, "%s: Couldn't open file /dev/urandom; %s\n",
	      program_invocation_short_name, strerror(errno));
      exit(1);
    }
  }
  return fp;
}


unsigned int
give_rand(FILE *randdev)
{
  unsigned int random;
  int i;

  if (randdev == NULL) {
    return (unsigned)rand();
  } else {
    i = fread(&random, sizeof(random), 1, randdev);
    if (i == 0) {
      if (feof(randdev)) {
	fprintf(stderr, "%s: EOF occured when reading RNG\n",
		program_invocation_short_name);
	fclose(randdev);
	exit(1);
      } else if (ferror(randdev)) {
	fprintf(stderr, "%s: Error while reading RNG; %s\n",
		program_invocation_short_name, strerror(errno));
	fclose(randdev);
	exit(1);
      }
    }
    return random;
  }
}

  
