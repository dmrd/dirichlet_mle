#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_psi.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include "dirichlet.h"

double TOL = 1.48e-8;

void test_ipsi() {
    /* Test the inverse psi method */

    /* Sanity check */
    double res = _ipsi(gsl_sf_psi(0.01));
    assert(res - 0.01 < TOL);

    /* Check steps along logspace from 10^-5 to 10^5 */
    double start = -5;
    double end = 5;
    double steps = 100;
    for (size_t i = 0; i < steps; i++) {
        double y = pow(10, start + i * (end - start) / steps);
        double res = _ipsi(gsl_sf_psi(y));
        // printf("%f\t%f\n", y, res);
        assert(fabs(res - y) < TOL);
    }
}

double norm_diff(gsl_vector *orig, gsl_vector *fit) {
    //norm(self.a0 - a0_fit)/norm(self.a0)
    gsl_vector *tmp = gsl_vector_alloc(orig->size);
    gsl_vector_memcpy(tmp, orig);
    gsl_vector_sub(tmp, fit);
    return gsl_blas_dnrm2(tmp) / gsl_blas_dnrm2(orig);
}

void test_dirichlet_mle() {
    /* Test MLE values */
    gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);

    size_t tests = 5;
    size_t K = 5;    // number parameters
    size_t N = 1000; // number observations
    gsl_vector *a = gsl_vector_alloc(K);
    gsl_matrix *D = gsl_matrix_alloc(N, K);
    for (size_t test = 0; test < tests; test++) {
        for (size_t i = 0; i < K; i++) {
            a->data[i] = gsl_rng_uniform_int(r, 1000) + 1;
        }

        for (size_t i = 0; i < N; i++) {
            gsl_vector_view vv = gsl_matrix_row(D, i);
            gsl_ran_dirichlet(r, K, a->data, vv.vector.data);
        }

        gsl_vector *rec = dirichlet_mle(D);
        /*double diff = norm_diff(a, rec);*/
        /*printf("%f\n", diff);*/
        /*assert(norm_diff(a, rec) < 0.1);*/
        printf("Parameter\tMLE Estimate\n");
        for (size_t i = 0; i < K; i++) {
            printf("%f\t%f\n", a->data[i], rec->data[i]);
        }
        printf("\n");
        gsl_vector_free(rec);
    }
    gsl_matrix_free(D);
    gsl_vector_free(a);
}

int main() {
    test_ipsi();
    test_dirichlet_mle();
}
