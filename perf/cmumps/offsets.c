#include <stdio.h>
#include <stddef.h>
#include "cmumps_c.h"
#define O(f) printf("%s=%zu, ", #f, offsetof(CMUMPS_STRUC_C, f))
int main(void) {
  printf("sizeof=%zu,\n", sizeof(CMUMPS_STRUC_C));
  O(icntl); O(keep); O(cntl); O(dkeep); O(keep8); O(n); O(nnz); O(irn); O(a); printf("\n");
  O(colsca_from_mumps); O(rowind); O(rhs); O(nrhs); O(ld_rhsintr); O(info); printf("\n");
  O(infog); O(rinfog); O(nb_singular_values); O(size_schur); O(listvar_schur); printf("\n");
  O(schur); O(wk_user); O(version_number); O(lwk_user); O(metis_options); printf("\n");
  O(instance_number); printf("\n"); return 0;
}
