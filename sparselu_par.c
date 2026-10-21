// Extraido de https://gitlab.inria.fr/openmp/kastors/-/tree/affinity/sparselu

/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <float.h>
#include <math.h>
#include <libgen.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* Antes vindos de -DMSIZE -DSMSIZE no Makefile.am */
#define MSIZE
#define SMSIZE

#define min(a, b) ((a<b)?a:b)
#define max(a, b) ((a>b)?a:b)

#define EPSILON 1.0E-6

/* ---- main.h ---- */
struct user_parameters {
    int check;
    int succeed;
    char* string2display;
    int niter;
#ifdef TITER
    int titer;
#endif
#ifdef MSIZE
    int matrix_size;
#endif
#ifdef SMSIZE
    int submatrix_size;
#endif
#ifdef BSIZE
    int blocksize;
#endif
#ifdef IBSIZE
    int iblocksize;
#endif
#ifdef CUTOFF_DEPTH
    int cutoff_depth;
#endif
#ifdef CUTOFF_SIZE
    int cutoff_size;
#endif
};

extern double run(struct user_parameters* params, uint64_t *startTime, uint64_t *endTime);

/* ---- timer.h ---- */
#if defined(__linux__) || defined(__MIC__)
#include <time.h>
#define START_TIMER\
    struct timespec rss;\
    clock_gettime(CLOCK_MONOTONIC, &rss);

#define END_TIMER\
    struct timespec rse;\
    clock_gettime(CLOCK_MONOTONIC, &rse);\
    double wtime = rse.tv_sec - rss.tv_sec + (rse.tv_nsec - rss.tv_nsec) / 1000000000.;

#elif defined(__APPLE__)
#include <sys/time.h>

#define START_TIMER\
    struct timeval rss;\
    gettimeofday(&rss, 0);

#define END_TIMER\
    struct timeval rse;\
    gettimeofday(&rse, 0);\
    double wtime = rse.tv_sec - rss.tv_sec + (rse.tv_usec - rss.tv_usec) / 1000000.;

#endif

#define TIMER wtime

/* ---- sparselu.h ---- */
float * allocate_clean_block(int submatrix_size);
void lu0(float *diag, int submatrix_size);
void bdiv(float *diag, float *row, int submatrix_size);
void bmod(float *row, float *col, float *inner, int submatrix_size);
void fwd(float *diag, float *col, int submatrix_size);

void sparselu_seq_call(float **BENCH, int matrix_size, int submatrix_size);
void sparselu_par_call(float **BENCH, int matrix_size, int submatrix_size);

/***********************************************************************
 * genmat:
 **********************************************************************/
static void genmat (float *M[], int matrix_size, int submatrix_size)
{
    int null_entry, init_val, i, j, ii, jj;

    init_val = 1325;

    /* generating the structure */
    for (ii=0; ii < matrix_size; ii++)
    {
        for (jj=0; jj < matrix_size; jj++)
        {
#pragma omp task shared(M)
{
            float *p;
            /* computing null entries */
            null_entry=0;
            if ((ii<jj) && (ii%3 !=0)) null_entry = 1;
            if ((ii>jj) && (jj%3 !=0)) null_entry = 1;
            if (ii%2==1) null_entry = 1;
            if (jj%2==1) null_entry = 1;
            if (ii==jj) null_entry = 0;
            if (ii==jj-1) null_entry = 0;
            if (ii-1 == jj) null_entry = 0;
            /* allocating matrix */
            if (null_entry == 0){
                M[ii*matrix_size+jj] = (float *) malloc(submatrix_size*submatrix_size*sizeof(float));
                if (M[ii*matrix_size+jj] == NULL)
                    exit(101);
                /* initializing matrix */
                p = M[ii*matrix_size+jj];
                for (i = 0; i < submatrix_size; i++)
                {
                    for (j = 0; j < submatrix_size; j++)
                    {
                        init_val = (3125 * init_val) % 65536;
                        (*p) = (float)((init_val - 32768.0) / 16384.0);
                        p++;
                    }
                }
            }
            else
            {
                M[ii*matrix_size+jj] = NULL;
            }
}
        }
    }
#pragma omp taskwait
}
/***********************************************************************
 * allocate_clean_block:
 **********************************************************************/
float * allocate_clean_block(int submatrix_size)
{
    int i,j;
    float *p, *q;

    p = (float *) malloc(submatrix_size*submatrix_size*sizeof(float));
    q=p;
    if (p!=NULL){
        for (i = 0; i < submatrix_size; i++)
            for (j = 0; j < submatrix_size; j++){(*p)=0.0; p++;}

    }
    else
        exit (101);
    return (q);
}

/***********************************************************************
 * lu0:
 **********************************************************************/
void lu0(float *diag, int submatrix_size)
{
    int i, j, k;

    for (k=0; k<submatrix_size; k++)
        for (i=k+1; i<submatrix_size; i++)
        {
            diag[i*submatrix_size+k] = diag[i*submatrix_size+k] / diag[k*submatrix_size+k];
            for (j=k+1; j<submatrix_size; j++)
                diag[i*submatrix_size+j] = diag[i*submatrix_size+j] - diag[i*submatrix_size+k] * diag[k*submatrix_size+j];
        }
}

/***********************************************************************
 * bdiv:
 **********************************************************************/
void bdiv(float *diag, float *row, int submatrix_size)
{
    int i, j, k;
    for (i=0; i<submatrix_size; i++)
        for (k=0; k<submatrix_size; k++)
        {
            row[i*submatrix_size+k] = row[i*submatrix_size+k] / diag[k*submatrix_size+k];
            for (j=k+1; j<submatrix_size; j++)
                row[i*submatrix_size+j] = row[i*submatrix_size+j] - row[i*submatrix_size+k]*diag[k*submatrix_size+j];
        }
}
/***********************************************************************
 * bmod:
 **********************************************************************/
void bmod(float *row, float *col, float *inner, int submatrix_size)
{
    int i, j, k;
    for (i=0; i<submatrix_size; i++)
        for (j=0; j<submatrix_size; j++)
            for (k=0; k<submatrix_size; k++)
                inner[i*submatrix_size+j] = inner[i*submatrix_size+j] - row[i*submatrix_size+k]*col[k*submatrix_size+j];
}
/***********************************************************************
 * fwd:
 **********************************************************************/
void fwd(float *diag, float *col, int submatrix_size)
{
    int i, j, k;
    for (j=0; j<submatrix_size; j++)
        for (k=0; k<submatrix_size; k++)
            for (i=k+1; i<submatrix_size; i++)
                col[i*submatrix_size+j] = col[i*submatrix_size+j] - diag[i*submatrix_size+k]*col[k*submatrix_size+j];
}


static void sparselu_init (float ***pBENCH, int matrix_size, int submatrix_size)
{
    *pBENCH = (float **) malloc(matrix_size*matrix_size*sizeof(float *));
    genmat(*pBENCH, matrix_size, submatrix_size);
}

#ifdef _OPENMP

/***********************************************************************
 * checkmat:
 **********************************************************************/
static int checkmat (float *M, float *N, int submatrix_size)
{
    int i, j;
    float r_err;

    for (i = 0; i < submatrix_size; i++)
    {
        for (j = 0; j < submatrix_size; j++)
        {
            r_err = M[i*submatrix_size+j] - N[i*submatrix_size+j];
            if ( r_err == 0.0 ) continue;

            if (r_err < 0.0 ) r_err = -r_err;

            if ( M[i*submatrix_size+j] == 0 )
                return 0;
            r_err = r_err / M[i*submatrix_size+j];
            if(r_err > EPSILON)
                return 0;
        }
    }
    return 1;
}

static int sparselu_check(float **BENCH_SEQ, float **BENCH, int matrix_size, int submatrix_size)
{
    int ii,jj,ok=1;

    for (ii=0; ((ii<matrix_size) && ok); ii++)
    {
        for (jj=0; ((jj<matrix_size) && ok); jj++)
        {
            if ((BENCH_SEQ[ii*matrix_size+jj] == NULL) && (BENCH[ii*matrix_size+jj] != NULL)) ok = 0;
            if ((BENCH_SEQ[ii*matrix_size+jj] != NULL) && (BENCH[ii*matrix_size+jj] == NULL)) ok = 0;
            if ((BENCH_SEQ[ii*matrix_size+jj] != NULL) && (BENCH[ii*matrix_size+jj] != NULL))
                ok = checkmat(BENCH_SEQ[ii*matrix_size+jj], BENCH[ii*matrix_size+jj], submatrix_size);
        }
    }
    return ok;
}
#endif

/***********************************************************************
 * sparselu_seq_call:
 **********************************************************************/
void sparselu_seq_call(float **BENCH, int matrix_size, int submatrix_size)
{
   int ii, jj, kk;

   for (kk=0; kk<matrix_size; kk++)
   {
      lu0(BENCH[kk*matrix_size+kk], submatrix_size);
      for (jj=kk+1; jj<matrix_size; jj++)
         if (BENCH[kk*matrix_size+jj] != NULL)
         {
            fwd(BENCH[kk*matrix_size+kk], BENCH[kk*matrix_size+jj], submatrix_size);
         }
      for (ii=kk+1; ii<matrix_size; ii++)
         if (BENCH[ii*matrix_size+kk] != NULL)
         {
            bdiv (BENCH[kk*matrix_size+kk], BENCH[ii*matrix_size+kk], submatrix_size);
         }
      for (ii=kk+1; ii<matrix_size; ii++)
         if (BENCH[ii*matrix_size+kk] != NULL)
            for (jj=kk+1; jj<matrix_size; jj++)
               if (BENCH[kk*matrix_size+jj] != NULL)
               {
                     if (BENCH[ii*matrix_size+jj]==NULL) BENCH[ii*matrix_size+jj] = allocate_clean_block(submatrix_size);
                     bmod(BENCH[ii*matrix_size+kk], BENCH[kk*matrix_size+jj], BENCH[ii*matrix_size+jj], submatrix_size);
               }

   }
}

/***********************************************************************
 * sparselu_par_call (task + depend):
 **********************************************************************/
void sparselu_par_call(float **BENCH, int matrix_size, int submatrix_size)
{
    int ii, jj, kk;

#pragma omp parallel private(kk,ii,jj) shared(BENCH)
#pragma omp single /* nowait */
    {
        /*#pragma omp task untied*/
        for (kk=0; kk<matrix_size; kk++)
        {
#pragma omp task firstprivate(kk) shared(BENCH) depend(inout: BENCH[kk*matrix_size+kk:submatrix_size*submatrix_size])
            lu0(BENCH[kk*matrix_size+kk], submatrix_size);
            for (jj=kk+1; jj<matrix_size; jj++)
                if (BENCH[kk*matrix_size+jj] != NULL)
                {
#pragma omp task firstprivate(kk, jj) shared(BENCH) depend(in: BENCH[kk*matrix_size+kk:submatrix_size*submatrix_size]) depend(inout: BENCH[kk*matrix_size+jj:submatrix_size*submatrix_size])
                    fwd(BENCH[kk*matrix_size+kk], BENCH[kk*matrix_size+jj], submatrix_size);
                }
            for (ii=kk+1; ii<matrix_size; ii++)
                if (BENCH[ii*matrix_size+kk] != NULL)
                {
#pragma omp task firstprivate(kk, ii) shared(BENCH) depend(in: BENCH[kk*matrix_size+kk:submatrix_size*submatrix_size]) depend(inout: BENCH[ii*matrix_size+kk:submatrix_size*submatrix_size])
                    bdiv (BENCH[kk*matrix_size+kk], BENCH[ii*matrix_size+kk], submatrix_size);
                }

            for (ii=kk+1; ii<matrix_size; ii++)
                if (BENCH[ii*matrix_size+kk] != NULL)
                    for (jj=kk+1; jj<matrix_size; jj++)
                        if (BENCH[kk*matrix_size+jj] != NULL)
                        {
                            if (BENCH[ii*matrix_size+jj]==NULL) BENCH[ii*matrix_size+jj] = allocate_clean_block(submatrix_size);
#pragma omp task firstprivate(kk, jj, ii) shared(BENCH) \
                            depend(in: BENCH[ii*matrix_size+kk:submatrix_size*submatrix_size], BENCH[kk*matrix_size+jj:submatrix_size*submatrix_size]) \
                            depend(inout: BENCH[ii*matrix_size+jj:submatrix_size*submatrix_size])
                            bmod(BENCH[ii*matrix_size+kk], BENCH[kk*matrix_size+jj], BENCH[ii*matrix_size+jj], submatrix_size);
                        }

        }
#pragma omp taskwait
    }
}

/***********************************************************************
 * run:
 **********************************************************************/
double run(struct user_parameters* params, uint64_t *startTime, uint64_t *endTime)
{
    startTime = 0;
    endTime = 0;
    float **BENCH;
    int matrix_size = params->matrix_size;
    if (matrix_size <= 0) {
        matrix_size = 64;
        params->matrix_size = matrix_size;
    }
    int submatrix_size = params->submatrix_size;
    if (submatrix_size <= 0) {
        submatrix_size = 64;
        params->submatrix_size = submatrix_size;
    }

#pragma omp parallel
#pragma omp master
    sparselu_init(&BENCH, matrix_size, submatrix_size);

    /// KERNEL INTENSIVE COMPUTATION
    START_TIMER;
#ifndef _OPENMP
    sparselu_seq_call(BENCH, matrix_size, submatrix_size);
#else
    sparselu_par_call(BENCH, matrix_size, submatrix_size);
#endif
    END_TIMER;

#ifdef _OPENMP
    if(params->check) {
        float **BENCH_SEQ;
        sparselu_init(&BENCH_SEQ, matrix_size, submatrix_size);
        sparselu_seq_call(BENCH_SEQ, matrix_size, submatrix_size);
        params->succeed = sparselu_check(BENCH_SEQ, BENCH, matrix_size, submatrix_size);
    }
#else
    params->succeed = 1;
#endif
    return TIMER;
}

/***********************************************************************
 * main.c (KaStORS):
 **********************************************************************/
void parse(int argc, char* argv[], struct user_parameters* params)
{
    int i;
    for(i=1; i<argc; i++) {
        if(!strcmp(argv[i], "-c"))
            params->check = 1;
        else if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("----------------------------------------------\n");
            printf("-                KaStORS                     -\n");
            printf("-   Kaapi Starpu OpenMP Runtime task Suite   -\n");
            printf("----------------------------------------------\n");
            printf("-h, --help : Show help information\n");
            printf("-c : Ask to check result\n");
            printf("-i : Number of iterations\n");
#ifdef TITER
            printf("-r : Number ot timestep iteration\n");
#endif
#ifdef MSIZE
            printf("-n : Matrix size\n");
#endif
#ifdef SMSIZE
            printf("-m : SubMatrix size\n");
#endif
#ifdef BSIZE
            printf("-b : Block size\n");
#endif
#ifdef IBSIZE
            printf("-ib : Internal Block size\n");
#endif
#ifdef CUTOFF_SIZE
            printf("-s : Cutoff (Size of the matrix)\n");
#endif
#ifdef CUTOFF_DEPTH
            printf("-d : Cutoff (depth)\n");
#endif
            exit(EXIT_SUCCESS);
        } else if(!strcmp(argv[i], "-i")) {
            if (++i < argc)
                params->niter = atoi(argv[i]);
            else {
                fprintf(stderr, "-i requires a number\n");
                exit(EXIT_FAILURE);
            }
#ifdef TITER
        } else if(!strcmp(argv[i], "-r")) {
            if (++i < argc)
                params->titer = atoi(argv[i]);
            else {
                fprintf(stderr, "-r requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
#ifdef MSIZE
        } else if(!strcmp(argv[i], "-n")) {
            if (++i < argc)
                params->matrix_size = atoi(argv[i]);
            else {
                fprintf(stderr, "-n requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
#ifdef SMSIZE
        } else if(!strcmp(argv[i], "-m")) {
            if (++i < argc)
                params->submatrix_size = atoi(argv[i]);
            else {
                fprintf(stderr, "-m requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
#ifdef BSIZE
        } else if(!strcmp(argv[i], "-b")) {
            if (++i < argc)
                params->blocksize = atoi(argv[i]);
            else {
                fprintf(stderr, "-b requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
#ifdef IBSIZE
        } else if(!strcmp(argv[i], "-ib")) {
            if (++i < argc)
                params->iblocksize = atoi(argv[i]);
            else {
                fprintf(stderr, "-ib requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
#ifdef CUTOFF_SIZE
        } else if(!strcmp(argv[i], "-s")) {
            if (++i < argc)
                params->cutoff_size = atoi(argv[i]);
            else {
                fprintf(stderr, "-s requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
#ifdef CUTOFF_DEPTH
        } else if(!strcmp(argv[i], "-d")) {
            if (++i < argc)
                params->cutoff_depth = atoi(argv[i]);
            else {
                fprintf(stderr, "-d requires a number\n");
                exit(EXIT_FAILURE);
            }
#endif
        } else
            fprintf(stderr, "Unknown parameter : %s\n", argv[i]);
    }
}

int comp (const void * elem1, const void * elem2) 
{
    double f = *((double*)elem1);
    double s = *((double*)elem2);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}

int main(int argc, char* argv[])
{
    int num_threads = 1;
    struct user_parameters params;
    memset(&params, 0, sizeof(params));

    /* default value */
    params.niter = 1;

    parse(argc, argv, &params);

// get Number of thread if OpenMP is activated
#ifdef _OPENMP
    #pragma omp parallel
    #pragma omp master
    num_threads = omp_get_num_threads();
#endif

    // warmup
    uint64_t startTime;
    uint64_t endTime;

    run(&params, &startTime, &endTime);

    double mean = 0.0;
    double meansqr = 0.0;
    double min_ = DBL_MAX;
    double max_ = -1;
    double* all_times = (double*)malloc(sizeof(double) * params.niter); 

    for (int i=0; i<params.niter; ++i)
    {
      double cur_time = run(&params, &startTime, &endTime);
      all_times[i] = cur_time;
      mean += cur_time;
      min_ = min(min_, cur_time);
      max_ = max(max_, cur_time);
      meansqr += cur_time * cur_time;
      sleep(5);
      }
    mean /= params.niter;
    meansqr /= params.niter;
    double stddev = sqrt(meansqr - mean * mean);

    qsort(all_times, params.niter, sizeof(double), comp);
    double median = all_times[params.niter / 2];

    free(all_times);

    char *progname = argv[0], *slash;
    while ((slash = strchr(progname, '/'))) {
        progname = slash + 1;
    }
    printf("#Output follows this header :\n");
    printf("##Progname ");
#ifdef MSIZE
    printf("Size ");
#endif
#ifdef SMSIZE
    printf("Submatrix_size ");
#endif
#ifdef BSIZE
    printf("Blocksize ");
#endif
#ifdef IBSIZE
    printf("Internal_Blocksize ");
#endif
#ifdef TITER
    printf("Iteration_time ");
#endif
    printf("Iterations ");
#ifdef CUTOFF_SIZE
    printf("Cutoff_Size ");
#endif
#ifdef CUTOFF_DEPTH
    printf("Cutoff_Depth ");
#endif
    printf("Threads ");
#ifdef GFLOPS
    printf("Gflops(Mean) ");
#else
    printf("Time(Mean,sec) ");
#endif
    printf("startTime endTime Stddev");
    printf("\n");

    //Actual values
    printf("%s ", progname);
#ifdef MSIZE
    printf("%d ", params.matrix_size);
#endif
#ifdef SMSIZE
    printf("%d ", params.submatrix_size);
#endif
#ifdef BSIZE
    printf("%d ", params.blocksize);
#endif
#ifdef IBSIZE
    printf("%d ", params.iblocksize);
#endif
#ifdef TITER
    printf("%d ", params.titer);
#endif
    printf("%d ", params.niter);
#ifdef CUTOFF_SIZE
    printf("%d ", params.cutoff_size);
#endif
#ifdef CUTOFF_DEPTH
    printf("%d ", params.cutoff_depth);
#endif
    printf("%d ", num_threads);
    printf("%lf ", mean);
    printf("%lu ", startTime);
    printf("%lu ", endTime);
    printf("%lf\n", stddev);
    printf("#Experience summarry : avg : %lf :: std : %lf :: min : %lf :: max : %lf :: median : %lf\n",
           mean, stddev, min_, max_, median);
    if(params.check)
        printf("Check : %s\n", (params.succeed)?
                ((params.succeed > 1)?"not implemented":"success")
                :"fail");
    if (params.string2display !=0)
      printf("%s", params.string2display);
    printf("\n");

    return 0;
}
