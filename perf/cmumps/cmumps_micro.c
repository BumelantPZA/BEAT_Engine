// Single-precision (cmumps) Schur factorization of an exported system; times 6 factorizations.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cmumps_c.h"
extern int lbt_forward(const char *path, int clear, int verbose, const char *suffix_hint);
static void *rd(const char *dir, const char *f, size_t *n) {
  char p[1024]; snprintf(p, sizeof p, "%s/%s", dir, f); FILE *fp = fopen(p, "rb");
  fseek(fp, 0, SEEK_END); *n = ftell(fp); rewind(fp); void *b = malloc(*n); fread(b, 1, *n, fp); fclose(fp); return b;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
  const char *dir = argv[1]; size_t sz;
  lbt_forward("/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate", 1, 0, "\x1a$NEWLAPACK");
  long long *sizes = rd(dir, "sizes.bin", &sz); int n = sizes[0]; long long nz = sizes[1]; int ns = sizes[2];
  int *irn = rd(dir, "irn.bin", &sz), *jcn = rd(dir, "jcn.bin", &sz), *vars = rd(dir, "schur_vars.bin", &sz);
  double *a64 = rd(dir, "a.bin", &sz); mumps_complex *a = malloc(nz * sizeof *a);
  for (long long i = 0; i < nz; i++) { a[i].r = a64[2*i]; a[i].i = a64[2*i+1]; }
  mumps_complex *schur = calloc((size_t)ns * ns, sizeof *schur);
  CMUMPS_STRUC_C id; memset(&id, 0, sizeof id);
  id.job = -1; id.par = 1; id.sym = 2; id.comm_fortran = -987654; cmumps_c(&id);
  id.icntl[0] = id.icntl[1] = id.icntl[2] = -1; id.icntl[3] = 0; id.icntl[5] = 0; id.icntl[6] = 7; id.icntl[7] = 0;
  id.icntl[11] = 1; id.icntl[17] = 0; id.icntl[18] = 3; id.cntl[0] = 0.0f;
  id.n = n; id.nnz = nz; id.irn = irn; id.jcn = jcn; id.a = a;
  id.size_schur = ns; id.listvar_schur = vars; id.schur = schur; id.schur_lld = ns;
  id.job = 1; cmumps_c(&id); if (id.infog[0] < 0) { printf("analysis INFOG(1)=%d\n", id.infog[0]); return 1; }
  double best = 1e9, sum = 0;
  for (int k = 0; k < 6; k++) {
    double t = now(); id.job = 2; cmumps_c(&id); t = now() - t;
    if (id.infog[0] < 0) { printf("factor INFOG(1)=%d INFOG(2)=%d\n", id.infog[0], id.infog[1]); return 1; }
    if (k) { sum += t; if (t < best) best = t; }
  }
  printf("cmumps fac median-ish mean %.1f ms (min %.1f) flops %.2f G\n", sum / 5 * 1e3, best * 1e3, id.rinfog[0] / 1e9);
  char p[1024]; snprintf(p, sizeof p, "%s/schur32.bin", dir); FILE *fp = fopen(p, "wb"); fwrite(schur, sizeof *schur, (size_t)ns * ns, fp); fclose(fp);
  id.job = -2; cmumps_c(&id); return 0;
}
