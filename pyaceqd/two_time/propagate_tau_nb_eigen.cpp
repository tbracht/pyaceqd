// Eigen-based reimplementation of the TimeLocalPhononfree / calc_G1 module.
// Identical algorithm to propagate_tau_nb.cpp but uses Eigen for matrix ops
// instead of raw CBLAS calls.
//
// Memory layout convention: all arrays are C-order (row-major).
//   dm_block : shape (n_map, dim2, dim2)
//   dm_s     : shape (dim2, dim2)
//   opA/B/C  : shape (dim, dim)
//   rho_init : shape (dim2,)
//   out      : shape (n_t, n_tb+1)   [pre-allocated, filled in-place]

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Eigen with row-major storage to match C-order numpy arrays
#include <Eigen/Dense>

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

// Row-major Eigen typedefs
using MatRM  = Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VecE   = Eigen::Matrix<cplx, Eigen::Dynamic, 1>;

// Map a raw pointer to a row-major Eigen matrix (no copy)
static inline Eigen::Map<const MatRM> map_mat(const cplx* p, int r, int c) {
    return Eigen::Map<const MatRM>(p, r, c);
}
static inline Eigen::Map<MatRM> map_mat_mut(cplx* p, int r, int c) {
    return Eigen::Map<MatRM>(p, r, c);
}
static inline Eigen::Map<const VecE> map_vec(const cplx* p, int n) {
    return Eigen::Map<const VecE>(p, n);
}
static inline Eigen::Map<VecE> map_vec_mut(cplx* p, int n) {
    return Eigen::Map<VecE>(p, n);
}

// ---------------------------------------------------------------------------
// TimeLocalPhononfreeEigen class
// ---------------------------------------------------------------------------

class TimeLocalPhononfreeEigen {
public:
    TimeLocalPhononfreeEigen(
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

    // Fills out[n_t, n_tb+1] in-place.
    //   out[i, 0]   = Tr(opA * opB * opC * rho(t_i))
    //   out[i, k>0] = Tr(opB * propagate(opC * rho(t_i) * opA, k steps))
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
        const double* t_full   = time.data();
        const double* t_sparse = time_sparse.data();
        cplx*         result   = out.data();

        // Build Eigen views for the small (dim x dim) operators — no copy
        Eigen::Map<const MatRM> A = map_mat(opA.data(), dim, dim);
        Eigen::Map<const MatRM> B = map_mat(opB.data(), dim, dim);
        Eigen::Map<const MatRM> C = map_mat(opC.data(), dim, dim);

        // Precompute opA*opB*opC for tau=0 traces
        MatRM ABC = A * B * C;

        // Precompute non-zero entries of B once — used in the hot tau trace.
        // For typical operators (e.g. a single lowering op) this collapses the
        // double loop to O(1) multiplications.
        struct BEntry { int ii, kk; cplx val; };
        std::vector<BEntry> B_nz;
        B_nz.reserve(dim * dim);
        for (int ii = 0; ii < dim; ++ii)
            for (int kk = 0; kk < dim; ++kk)
                if (B(ii, kk) != cplx{0, 0})
                    B_nz.push_back({ii, kk, B(ii, kk)});
        printf("B has %zu non-zero entries\n", B_nz.size());
        
        {
            nb::gil_scoped_release release;

            // rho_vec as an Eigen vector (dim2)
            VecE rho_vec = map_vec(rho_init.data(), dim2);

            std::vector<cplx> rho_buffer((size_t)dim2 * n_t);
            std::vector<int>  j_array(n_t);

            int j = 0;

            auto t0 = std::chrono::steady_clock::now();

            for (int i = 0; i < n_t; ++i) {
                // Advance rho_vec along t until t_full[j] >= t_sparse[i]
                while (j < n_tfull - 1 && t_full[j] < t_sparse[i]) {
                    Eigen::Map<const MatRM> M = (j < n_map_)
                        ? map_mat(dm_block + (size_t)j * dim2 * dim2, dim2, dim2)
                        : map_mat(dm_s, dim2, dim2);
                    VecE tmp = M * rho_vec;
                    rho_vec  = std::move(tmp);
                    ++j;
                }

                // result[i, 0] = Tr(ABC * rho(t_i))
                // ABC is (dim x dim), rho_vec is (dim2,) = vectorised rho[row,col]
                // Tr(M * rho) = sum_i sum_k M[i,k] * rho_vec[k*dim + i]
                {
                    cplx tr{0, 0};
                    for (int ii = 0; ii < dim; ++ii)
                        for (int kk = 0; kk < dim; ++kk)
                            tr += ABC(ii, kk) * rho_vec[kk * dim + ii];
                    result[(size_t)i * n_tau + 0] = tr;
                }

                // rho_mod = opC * rho_mat * opA  (vectorised: C * rho_vec + rho_vec * A)
                // We work with the vectorised form: treat rho_vec as a (dim x dim) matrix
                {
                    Eigen::Map<const MatRM> rho_mat = map_mat(rho_vec.data(), dim, dim);
                    MatRM rho_mod = C * rho_mat * A;
                    std::memcpy(&rho_buffer[(size_t)i * dim2],
                                rho_mod.data(), dim2 * sizeof(cplx));
                }
                j_array[i] = j;
            }

            auto t1 = std::chrono::steady_clock::now();

            // Parallel tau propagation — each i is independent
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_t; ++i) {
                VecE lrho = map_vec(&rho_buffer[(size_t)i * dim2], dim2);
                int jj = j_array[i];

                for (int k = 1; k <= n_tb; ++k) {
                    Eigen::Map<const MatRM> M = (jj < n_map_)
                        ? map_mat(dm_block + (size_t)jj * dim2 * dim2, dim2, dim2)
                        : map_mat(dm_s, dim2, dim2);
                    VecE ltmp = M * lrho;
                    lrho = std::move(ltmp);

                    // Tr(opB * lrho_mat) — sparse: only non-zero entries of B
                    cplx tr{0, 0};
                    for (const BEntry& e : B_nz)
                        tr += e.val * lrho[e.kk * dim + e.ii];
                    result[(size_t)i * n_tau + k] = tr;
                    ++jj;
                }
            }

            auto t2 = std::chrono::steady_clock::now();
            printf("Eigen G1: seq=%.3fs  tau=%.3fs\n",
                std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count());
            fflush(stdout);
        } // GIL re-acquired
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

NB_MODULE(propagate_tau_nb_eigen_module, m) {
    m.doc() = "Eigen-based time-local propagator module (phonon-free G1)";

    nb::class_<TimeLocalPhononfreeEigen>(m, "TimeLocalPhononfreeEigen")
        .def(nb::init<
                nb::ndarray<cplx, nb::ndim<3>, nb::device::cpu>,
                nb::ndarray<cplx, nb::ndim<2>, nb::device::cpu>,
                int>(),
             nb::arg("dm_block"), nb::arg("dm_s"), nb::arg("dim"),
             "Construct from dm_block (n_map, dim2, dim2), dm_s (dim2, dim2), dim.\n"
             "Arrays must be C-contiguous and remain alive for the object's lifetime.")
        .def("calc_G1", &TimeLocalPhononfreeEigen::calc_G1,
             nb::arg("rho_init"), nb::arg("opA"), nb::arg("opB"), nb::arg("opC"),
             nb::arg("time"), nb::arg("time_sparse"), nb::arg("n_tb"), nb::arg("out"),
             "Fills pre-allocated out array (n_t, n_tb+1) in-place. Prints timing to stdout.\n"
             "out[i,0]   = Tr(opA*opB*opC * rho(t_i))\n"
             "out[i,k>0] = Tr(opB * propagate(opC*rho(t_i)*opA, k steps))");
}
