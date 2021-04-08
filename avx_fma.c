/*
gcc -Wall -Wextra -O3 -march=native -mtune=native -o avx_fma avx_fma.c -fsanitize=undefined
objdump -d avx_fma | grep -10 vfma

https://www.uio.no/studier/emner/matnat/ifi/IN3200/v19/teaching-material/avx256.pdf
*/

#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <x86intrin.h>

volatile sig_atomic_t stop;

void inthand(int signum) {
    stop = 1;
}

static __inline__ u_int64_t start_clock() {
    // See: Intel Doc #324264, "How to Benchmark Code Execution Times on Intel...",
    u_int32_t hi, lo;
    __asm__ __volatile__ (
        "CPUID\n\t"
        "RDTSC\n\t"
        "mov %%edx, %0\n\t"
        "mov %%eax, %1\n\t": "=r" (hi), "=r" (lo)::
        "%rax", "%rbx", "%rcx", "%rdx");
    return ( (u_int64_t)lo) | ( ((u_int64_t)hi) << 32);
}

static __inline__ u_int64_t stop_clock() {
    // See: Intel Doc #324264, "How to Benchmark Code Execution Times on Intel...",
    u_int32_t hi, lo;
    __asm__ __volatile__(
        "RDTSCP\n\t"
        "mov %%edx, %0\n\t"
        "mov %%eax, %1\n\t"
        "CPUID\n\t": "=r" (hi), "=r" (lo)::
        "%rax", "%rbx", "%rcx", "%rdx");
    return ( (u_int64_t)lo) | ( ((u_int64_t)hi) << 32);
}

int main () {
  signal(SIGINT, inthand);
  signal(SIGTERM, inthand);

  double a[4];
  double b[4];
  double c[4];

  for (int i = 0; i < 4; i++) {
    a[i] = drand48();
    b[i] = drand48();
    c[i] = 0.0;
  }
  a[0] = 1.0;
  b[0] = 1.0;

  __m256d av = _mm256_loadu_pd(a);      // load 4 double from a
  __m256d bv = _mm256_loadu_pd(b);      // load 4 double from b
  __m256d cv = _mm256_loadu_pd(c);      // load 4 double from c

  printf("Starting computation. Use Ctrl-C to stop it.\n");
  u_int64_t start_rdtsc = start_clock();
  while (!stop) {
    cv =  _mm256_fmadd_pd(av, bv, cv); // cv = av*bv + cv
  }
  u_int64_t stop_rdtsc = stop_clock();
  u_int64_t diff = (stop_rdtsc-start_rdtsc);
  _mm256_storeu_pd(c, cv);             // write cv to c array


  printf("\n\nPerformed %g AVX256 FMA3 operations. Avg: %g CPU cycles/call\n", c[0], diff/c[0]);
  for (int i = 0; i < 2; i++) {
    printf("%g\t%g\n", c[2*i], c[2*i+1]);
  }

  return(0);
}
