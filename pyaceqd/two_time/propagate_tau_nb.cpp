// C++ reimplementation of calc_onetime_simple (the "no phonon" G1 case)
// exposed to Python via nanobind.
//
// Memory layout convention: all arrays are C-order (row-major).
//   dm_block : shape (n_map, dim2, dim2)  — matrix j at ptr + j*dim2*dim2
//   dm_s     : shape (dim2, dim2)
//   opA/B/C  : shape (dim, dim)
//   rho_init : shape (dim2,)
//   result   : shape (n_t, n_tb+1)   [allocated here, returned to Python]
//
// Using CblasRowMajor throughout matches C-order numpy arrays directly,
// eliminating the asfortranarray/transpose needed by the Fortran binding.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cblas.h>
#include <complex>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nb = nanobind;
using cplx = std::complex<double>;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// y = M * x,  M is (n x n) row-major
static inline void matvec(const cplx* M, const cplx* x, cplx* y, int n) {
    const cplx one{1.0, 0.0};
    const cplx zero{0.0, 0.0};
    cblas_zgemv(CblasRowMajor, CblasNoTrans,
                n, n, &one, M, n, x, 1, &zero, y, 1);
}

// C = A * B,  all (dim x dim) row-major
static inline void matmul_sq(const cplx* A, const cplx* B, cplx* C, int dim) {
    const cplx one{1.0, 0.0};
    const cplx zero{0.0, 0.0};
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                dim, dim, dim, &one, A, dim, B, dim, &zero, C, dim);
}

// Tr(M * rho_vec) where rho_vec is a dim2-vector representing rho[i,j]=rho_vec[i*dim+j]
// Tr(M * rho) = sum_i sum_k M[i,k] * rho[k,i]
static inline cplx trace_Mrho(const cplx* M, const cplx* rho_vec, int dim) {
    cplx tr{0.0, 0.0};
    for (int i = 0; i < dim; ++i)
        for (int k = 0; k < dim; ++k)
            tr += M[i * dim + k] * rho_vec[k * dim + i];
    return tr;
}

// static inline cplx trace_Mrho(const cplx* M, const cplx* rho_vec, int dim) {
//     cplx tr{0.0, 0.0};
//     for (int i = 0; i < dim; ++i)
//         for (int k = 0; k < dim; ++k)
//             tr += M[i * dim + k] * rho_vec[k * dim + i];
//     return tr;
// }

// ---------------------------------------------------------------------------
// TimeLocalPhononfree class
// ---------------------------------------------------------------------------

class TimeLocalPhononfree {
public:
    // dm_block: C-order (n_map, dim2, dim2) complex array — no copy, pointer stored.
    // dm_s    : C-order (dim2, dim2)        complex array — no copy.
    // The Python caller must keep both arrays alive while this object exists.
    TimeLocalPhononfree(
        nb::ndarray<cplx, nb::ndim<3>, nb::device::cpu> dm_block,
        nb::ndarray<cplx, nb::ndim<2>, nb::device::cpu> dm_s,
        int dim)
        : dim_(dim), dim2_(dim * dim)
    {
        if ((int)dm_block.shape(1) != dim2_ || (int)dm_block.shape(2) != dim2_)
            throw std::invalid_argument("dm_block inner dims must equal dim*dim");
        if ((int)dm_s.shape(0) != dim2_ || (int)dm_s.shape(1) != dim2_)
            throw std::invalid_argument("dm_s shape must be (dim2, dim2)");

        n_map_        = (int)dm_block.shape(0);
        dm_block_ptr_ = dm_block.data();
        dm_s_ptr_     = dm_s.data();
    }

    // Compute G1[n_t, n_tb+1]:
    //   G1[i, 0]   = Tr(opA * opB * opC * rho(t_i))
    //   G1[i, k>0] = Tr(opB * propagate(opC * rho(t_i) * opA, k steps from t_i))
    //
    // Mirrors calc_onetime_simple in propagate_tau.f90 step-for-step.
    void
    calc_G1(
        nb::ndarray<cplx,   nb::ndim<1>, nb::device::cpu> rho_init,
        nb::ndarray<cplx,   nb::ndim<2>, nb::device::cpu> opA,
        nb::ndarray<cplx,   nb::ndim<2>, nb::device::cpu> opB,
        nb::ndarray<cplx,   nb::ndim<2>, nb::device::cpu> opC,
        nb::ndarray<double, nb::ndim<1>, nb::device::cpu> time,
        nb::ndarray<double, nb::ndim<1>, nb::device::cpu> time_sparse,
        int n_tb,
        nb::ndarray<cplx,   nb::ndim<2>, nb::device::cpu> out)
    {
        const int dim     = dim_;
        const int dim2    = dim2_;
        const int n_t     = (int)time_sparse.shape(0);
        const int n_tfull = (int)time.shape(0);
        const int n_tau   = n_tb + 1;

        if ((int)rho_init.shape(0) != dim2)
            throw std::invalid_argument("rho_init length must be dim*dim");
        if ((int)out.shape(0) != n_t || (int)out.shape(1) != n_tau)
            throw std::invalid_argument("out shape must be (n_t, n_tb+1)");
        const cplx*   dm_block = dm_block_ptr_;
        const cplx*   dm_s     = dm_s_ptr_;
        const cplx*   pA       = opA.data();
        const cplx*   pB       = opB.data();
        const cplx*   pC       = opC.data();
        const double* t_full   = time.data();
        const double* t_sparse = time_sparse.data();

        // Output written directly into caller-supplied buffer — no allocation here.
        cplx* result = out.data();

        // Release GIL for pure C++ computation; re-acquired when `release` is destroyed.
        {
            nb::gil_scoped_release release;

            // Working buffers for sequential phase
            std::vector<cplx> rho_vec(dim2), rho_res(dim2);
            std::vector<cplx> ABC(dim * dim), tmp(dim * dim);
            std::vector<cplx> rho_buffer((size_t)dim2 * n_t);
            std::vector<int>  j_array(n_t);

            std::memcpy(rho_vec.data(), rho_init.data(), dim2 * sizeof(cplx));

            // Precompute opA*opB*opC once for tau=0 traces
            matmul_sq(pB, pC, tmp.data(), dim);
            matmul_sq(pA, tmp.data(), ABC.data(), dim);
            
            int j = 0;  // 0-based index into t_full[]
            
            auto t0 = std::chrono::steady_clock::now();

            for (int i = 0; i < n_t; ++i) {
                // Step 2: advance rho_vec along t until t_full[j] >= t_sparse[i]
                while (j < n_tfull - 1 && t_full[j] < t_sparse[i]) {
                    const cplx* M = (j < n_map_)
                        ? dm_block + (size_t)j * dim2 * dim2
                        : dm_s;
                    matvec(M, rho_vec.data(), rho_res.data(), dim2);
                    std::swap(rho_vec, rho_res);
                    ++j;
                }

                // Step 3: result[i, 0] = Tr(opA*opB*opC * rho(t_i))
                result[(size_t)i * n_tau + 0] = trace_Mrho(ABC.data(), rho_vec.data(), dim);

                // Step 4: rho_mod = opC * rho_mat * opA
                matmul_sq(pC, rho_vec.data(), tmp.data(), dim);
                matmul_sq(tmp.data(), pA, rho_res.data(), dim);

                std::memcpy(&rho_buffer[(size_t)i * dim2], rho_res.data(), dim2 * sizeof(cplx));
                j_array[i] = j;
            }

            auto t1 = std::chrono::steady_clock::now();
            // Steps 5+6: parallel tau propagation — each i is independent
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_t; ++i) {
                std::vector<cplx> lrho(dim2), ltmp(dim2);
                std::memcpy(lrho.data(), &rho_buffer[(size_t)i * dim2], dim2 * sizeof(cplx));

                int jj = j_array[i];

                for (int k = 1; k <= n_tb; ++k) {
                    const cplx* M = (jj < n_map_)
                        ? dm_block + (size_t)jj * dim2 * dim2
                        : dm_s;
                    matvec(M, lrho.data(), ltmp.data(), dim2);
                    // matvec(dm_s, lrho.data(), ltmp.data(), dim2);
                    std::swap(lrho, ltmp);
                    result[(size_t)i * n_tau + k] = trace_Mrho(pB, lrho.data(), dim);
                    ++jj;
                }
                // for (int k = 1; k <= n_tb; ++k) {
                //     jj < n_map_ 
                //         ? matvec(dm_block + (size_t)jj * dim2 * dim2, lrho.data(), ltmp.data(), dim2) 
                //         : matvec(dm_s, lrho.data(), ltmp.data(), dim2);
                //     std::swap(lrho, ltmp);
                //     result[(size_t)i * n_tau + k] = trace_Mrho(pB, lrho.data(), dim);
                //     ++jj;
                // }
            }

            auto t2 = std::chrono::steady_clock::now();
            printf("C++ G1: seq=%.3fs  tau=%.3fs\n",
                std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count());
            fflush(stdout);
        } // GIL re-acquired here
    }

private:
    int         dim_;
    int         dim2_;
    int         n_map_;
    const cplx* dm_block_ptr_;
    const cplx* dm_s_ptr_;
};

// ---------------------------------------------------------------------------
// nanobind module
// ---------------------------------------------------------------------------

NB_MODULE(propagate_tau_nb_module, m) {
    m.doc() = "C++ time-local propagator module (phonon-free G1)";

    nb::class_<TimeLocalPhononfree>(m, "TimeLocalPhononfree")
        .def(nb::init<
                nb::ndarray<cplx, nb::ndim<3>, nb::device::cpu>,
                nb::ndarray<cplx, nb::ndim<2>, nb::device::cpu>,
                int>(),
             nb::arg("dm_block"), nb::arg("dm_s"), nb::arg("dim"),
             "Construct from dm_block (n_map, dim2, dim2), dm_s (dim2, dim2), dim.\n"
             "Arrays must be C-contiguous and remain alive for the object's lifetime.")
        .def("calc_G1", &TimeLocalPhononfree::calc_G1,
             nb::arg("rho_init"), nb::arg("opA"), nb::arg("opB"), nb::arg("opC"),
             nb::arg("time"), nb::arg("time_sparse"), nb::arg("n_tb"), nb::arg("out"),
             "Fills pre-allocated out array (n_t, n_tb+1) in-place. Prints timing to stdout.\n"
             "out[i,0]   = Tr(opA*opB*opC * rho(t_i))\n"
             "out[i,k>0] = Tr(opB * propagate(opC*rho(t_i)*opA, k steps))");
}
