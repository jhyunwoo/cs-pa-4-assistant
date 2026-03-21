#include "gpt_mini.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <immintrin.h>

#pragma GCC optimize("O3,unroll-loops,no-trapping-math,tree-vectorize")

static inline void* alloc_aligned(size_t size) {
    void* ptr;
    if (posix_memalign(&ptr, 64, size)) return nullptr;
    return ptr;
}

static inline void free_aligned(void* ptr) {
    free(ptr);
}

using std::max;
using std::min;
using std::unique_ptr;
using std::vector;

namespace {

int sample_from(const float* p, int size) {
    int best_idx = 0;
    float best_val = p[0];
    for (int i = 1; i < size; i++) {
        if (p[i] > best_val) {
            best_val = p[i];
            best_idx = i;
        }
    }
    return best_idx;
}

float* softmax(const float* x, int size) {
    // Fast max with SIMD
    float maxv = x[0];
    int i = 1;
    if (size >= 16) {
        __m512 vmax = _mm512_loadu_ps(x);
        for (i = 16; i <= size - 16; i += 16) {
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(x + i));
        }
        maxv = _mm512_reduce_max_ps(vmax);
    }
    for (; i < size; i++) if (x[i] > maxv) maxv = x[i];
    
    float* y = new float[size];
    float sum = 0;
    for (i = 0; i < size; i++) {
        y[i] = std::exp(x[i] - maxv);
        sum += y[i];
    }
    
    // SIMD normalization
    float inv = 1.0f / sum;
    __m512 vinv = _mm512_set1_ps(inv);
    i = 0;
    for (; i <= size - 16; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(y + i), vinv));
    }
    for (; i < size; i++) y[i] *= inv;
    return y;
}

void transpose(const float* __restrict__ M, int rows, int cols, float* __restrict__ T) {
    const int BLOCK = 32;
    for (int i = 0; i < rows; i += BLOCK) {
        for (int j = 0; j < cols; j += BLOCK) {
            int i_lim = min(rows, i + BLOCK);
            int j_lim = min(cols, j + BLOCK);
            for (int ii = i; ii < i_lim; ++ii) {
                for (int jj = j; jj < j_lim; ++jj) {
                    T[jj * rows + ii] = M[ii * cols + jj];
                }
            }
        }
    }
}

float* transpose(const float* M, int rows, int cols) {
    float* T = new float[cols * rows];
    transpose(M, rows, cols, T);
    return T;
}

// ==================== Optimized Matrix Multiplication ====================

__attribute__((always_inline, hot))
inline void kernel_16x16(int k, const float* __restrict__ packedA, 
                         const float* __restrict__ packedB, 
                         float* __restrict__ C, int ldc) {
    __m512 c0  = _mm512_loadu_ps(C + 0*ldc);
    __m512 c1  = _mm512_loadu_ps(C + 1*ldc);
    __m512 c2  = _mm512_loadu_ps(C + 2*ldc);
    __m512 c3  = _mm512_loadu_ps(C + 3*ldc);
    __m512 c4  = _mm512_loadu_ps(C + 4*ldc);
    __m512 c5  = _mm512_loadu_ps(C + 5*ldc);
    __m512 c6  = _mm512_loadu_ps(C + 6*ldc);
    __m512 c7  = _mm512_loadu_ps(C + 7*ldc);
    __m512 c8  = _mm512_loadu_ps(C + 8*ldc);
    __m512 c9  = _mm512_loadu_ps(C + 9*ldc);
    __m512 c10 = _mm512_loadu_ps(C + 10*ldc);
    __m512 c11 = _mm512_loadu_ps(C + 11*ldc);
    __m512 c12 = _mm512_loadu_ps(C + 12*ldc);
    __m512 c13 = _mm512_loadu_ps(C + 13*ldc);
    __m512 c14 = _mm512_loadu_ps(C + 14*ldc);
    __m512 c15 = _mm512_loadu_ps(C + 15*ldc);

    const float* b_ptr = packedB;
    const float* a_ptr = packedA;

    int p = 0;
    for (; p <= k - 8; p += 8) {
        __m512 b0 = _mm512_load_ps(b_ptr);
        __m512 b1 = _mm512_load_ps(b_ptr + 16);
        __m512 b2 = _mm512_load_ps(b_ptr + 32);
        __m512 b3 = _mm512_load_ps(b_ptr + 48);
        __m512 b4 = _mm512_load_ps(b_ptr + 64);
        __m512 b5 = _mm512_load_ps(b_ptr + 80);
        __m512 b6 = _mm512_load_ps(b_ptr + 96);
        __m512 b7 = _mm512_load_ps(b_ptr + 112);

        #define PROCESS_ROW(row) \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row]), b0, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 16]), b1, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 32]), b2, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 48]), b3, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 64]), b4, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 80]), b5, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 96]), b6, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 112]), b7, c##row);

        PROCESS_ROW(0) PROCESS_ROW(1) PROCESS_ROW(2) PROCESS_ROW(3)
        PROCESS_ROW(4) PROCESS_ROW(5) PROCESS_ROW(6) PROCESS_ROW(7)
        PROCESS_ROW(8) PROCESS_ROW(9) PROCESS_ROW(10) PROCESS_ROW(11)
        PROCESS_ROW(12) PROCESS_ROW(13) PROCESS_ROW(14) PROCESS_ROW(15)
        
        #undef PROCESS_ROW

        b_ptr += 128;
        a_ptr += 128;
    }
    
    for (; p < k; ++p) {
        __m512 b = _mm512_load_ps(b_ptr);
        b_ptr += 16;
        
        c0  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[0]), b, c0);
        c1  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[1]), b, c1);
        c2  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[2]), b, c2);
        c3  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[3]), b, c3);
        c4  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[4]), b, c4);
        c5  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[5]), b, c5);
        c6  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[6]), b, c6);
        c7  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[7]), b, c7);
        c8  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[8]), b, c8);
        c9  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[9]), b, c9);
        c10 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[10]), b, c10);
        c11 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[11]), b, c11);
        c12 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[12]), b, c12);
        c13 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[13]), b, c13);
        c14 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[14]), b, c14);
        c15 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[15]), b, c15);
        a_ptr += 16;
    }

    _mm512_storeu_ps(C + 0*ldc, c0);
    _mm512_storeu_ps(C + 1*ldc, c1);
    _mm512_storeu_ps(C + 2*ldc, c2);
    _mm512_storeu_ps(C + 3*ldc, c3);
    _mm512_storeu_ps(C + 4*ldc, c4);
    _mm512_storeu_ps(C + 5*ldc, c5);
    _mm512_storeu_ps(C + 6*ldc, c6);
    _mm512_storeu_ps(C + 7*ldc, c7);
    _mm512_storeu_ps(C + 8*ldc, c8);
    _mm512_storeu_ps(C + 9*ldc, c9);
    _mm512_storeu_ps(C + 10*ldc, c10);
    _mm512_storeu_ps(C + 11*ldc, c11);
    _mm512_storeu_ps(C + 12*ldc, c12);
    _mm512_storeu_ps(C + 13*ldc, c13);
    _mm512_storeu_ps(C + 14*ldc, c14);
    _mm512_storeu_ps(C + 15*ldc, c15);
}

__attribute__((always_inline, hot))
inline void kernel_16x16_masked(int k, const float* __restrict__ packedA, 
                                const float* __restrict__ packedB, 
                                float* __restrict__ C, int ldc, __mmask16 mask) {
    __m512 c0  = _mm512_maskz_loadu_ps(mask, C + 0*ldc);
    __m512 c1  = _mm512_maskz_loadu_ps(mask, C + 1*ldc);
    __m512 c2  = _mm512_maskz_loadu_ps(mask, C + 2*ldc);
    __m512 c3  = _mm512_maskz_loadu_ps(mask, C + 3*ldc);
    __m512 c4  = _mm512_maskz_loadu_ps(mask, C + 4*ldc);
    __m512 c5  = _mm512_maskz_loadu_ps(mask, C + 5*ldc);
    __m512 c6  = _mm512_maskz_loadu_ps(mask, C + 6*ldc);
    __m512 c7  = _mm512_maskz_loadu_ps(mask, C + 7*ldc);
    __m512 c8  = _mm512_maskz_loadu_ps(mask, C + 8*ldc);
    __m512 c9  = _mm512_maskz_loadu_ps(mask, C + 9*ldc);
    __m512 c10 = _mm512_maskz_loadu_ps(mask, C + 10*ldc);
    __m512 c11 = _mm512_maskz_loadu_ps(mask, C + 11*ldc);
    __m512 c12 = _mm512_maskz_loadu_ps(mask, C + 12*ldc);
    __m512 c13 = _mm512_maskz_loadu_ps(mask, C + 13*ldc);
    __m512 c14 = _mm512_maskz_loadu_ps(mask, C + 14*ldc);
    __m512 c15 = _mm512_maskz_loadu_ps(mask, C + 15*ldc);

    const float* b_ptr = packedB;
    const float* a_ptr = packedA;

    int p = 0;
    for (; p <= k - 8; p += 8) {
        __m512 b0 = _mm512_load_ps(b_ptr);
        __m512 b1 = _mm512_load_ps(b_ptr + 16);
        __m512 b2 = _mm512_load_ps(b_ptr + 32);
        __m512 b3 = _mm512_load_ps(b_ptr + 48);
        __m512 b4 = _mm512_load_ps(b_ptr + 64);
        __m512 b5 = _mm512_load_ps(b_ptr + 80);
        __m512 b6 = _mm512_load_ps(b_ptr + 96);
        __m512 b7 = _mm512_load_ps(b_ptr + 112);

        #define PROCESS_ROW(row) \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row]), b0, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 16]), b1, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 32]), b2, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 48]), b3, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 64]), b4, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 80]), b5, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 96]), b6, c##row); \
            c##row = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[row + 112]), b7, c##row);

        PROCESS_ROW(0) PROCESS_ROW(1) PROCESS_ROW(2) PROCESS_ROW(3)
        PROCESS_ROW(4) PROCESS_ROW(5) PROCESS_ROW(6) PROCESS_ROW(7)
        PROCESS_ROW(8) PROCESS_ROW(9) PROCESS_ROW(10) PROCESS_ROW(11)
        PROCESS_ROW(12) PROCESS_ROW(13) PROCESS_ROW(14) PROCESS_ROW(15)
        
        #undef PROCESS_ROW

        b_ptr += 128;
        a_ptr += 128;
    }
    
    for (; p < k; ++p) {
        __m512 b = _mm512_load_ps(b_ptr);
        b_ptr += 16;
        
        c0  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[0]), b, c0);
        c1  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[1]), b, c1);
        c2  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[2]), b, c2);
        c3  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[3]), b, c3);
        c4  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[4]), b, c4);
        c5  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[5]), b, c5);
        c6  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[6]), b, c6);
        c7  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[7]), b, c7);
        c8  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[8]), b, c8);
        c9  = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[9]), b, c9);
        c10 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[10]), b, c10);
        c11 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[11]), b, c11);
        c12 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[12]), b, c12);
        c13 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[13]), b, c13);
        c14 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[14]), b, c14);
        c15 = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[15]), b, c15);
        a_ptr += 16;
    }

    _mm512_mask_storeu_ps(C + 0*ldc, mask, c0);
    _mm512_mask_storeu_ps(C + 1*ldc, mask, c1);
    _mm512_mask_storeu_ps(C + 2*ldc, mask, c2);
    _mm512_mask_storeu_ps(C + 3*ldc, mask, c3);
    _mm512_mask_storeu_ps(C + 4*ldc, mask, c4);
    _mm512_mask_storeu_ps(C + 5*ldc, mask, c5);
    _mm512_mask_storeu_ps(C + 6*ldc, mask, c6);
    _mm512_mask_storeu_ps(C + 7*ldc, mask, c7);
    _mm512_mask_storeu_ps(C + 8*ldc, mask, c8);
    _mm512_mask_storeu_ps(C + 9*ldc, mask, c9);
    _mm512_mask_storeu_ps(C + 10*ldc, mask, c10);
    _mm512_mask_storeu_ps(C + 11*ldc, mask, c11);
    _mm512_mask_storeu_ps(C + 12*ldc, mask, c12);
    _mm512_mask_storeu_ps(C + 13*ldc, mask, c13);
    _mm512_mask_storeu_ps(C + 14*ldc, mask, c14);
    _mm512_mask_storeu_ps(C + 15*ldc, mask, c15);
}

inline void pack_A(int k, const float* A, int lda, int i0, int i_max, int p0, int p_max, float* packed) {
    (void)k;
    int indices[16];
    for(int i=0; i<16; ++i) indices[i] = i * lda;
    __m512i vindex = _mm512_loadu_si512(indices);

    if (i0 + 16 <= i_max) {
        for (int p = p0; p < p_max; ++p) {
            __m512 a = _mm512_i32gather_ps(vindex, &A[i0 * lda + p], 4);
            _mm512_store_ps(packed, a);
            packed += 16;
        }
    } else {
        __mmask16 mask = (1 << (i_max - i0)) - 1;
        for (int p = p0; p < p_max; ++p) {
            __m512 a = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, vindex, &A[i0 * lda + p], 4);
            _mm512_store_ps(packed, a);
            packed += 16;
        }
    }
}

inline void pack_B(int k, const float* B, int ldb, int p0, int p_max, int j0, int j_max, int nr, float* packed) {
    (void)k;
    if (nr == 16 && j0 + 16 <= j_max) {
        for (int p = p0; p < p_max; ++p) {
            __m512 b = _mm512_loadu_ps(&B[p * ldb + j0]);
            _mm512_store_ps(packed, b);
            packed += 16;
        }
    } else {
        for (int p = p0; p < p_max; ++p) {
            for (int j = 0; j < nr; ++j) {
                if (j0 + j < j_max) {
                    *packed++ = B[p * ldb + (j0 + j)];
                } else {
                    *packed++ = 0.0f;
                }
            }
        }
    }
}

inline float dot_product(const float* A, const float* B, int k) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    
    int p = 0;
    for (; p <= k - 64; p += 64) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(A + p), _mm512_loadu_ps(B + p), sum0);
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(A + p + 16), _mm512_loadu_ps(B + p + 16), sum1);
        sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(A + p + 32), _mm512_loadu_ps(B + p + 32), sum2);
        sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(A + p + 48), _mm512_loadu_ps(B + p + 48), sum3);
    }
    
    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    
    for (; p <= k - 16; p += 16) {
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(A + p), _mm512_loadu_ps(B + p), sum);
    }
    
    float res = _mm512_reduce_add_ps(sum);
    for (; p < k; ++p) {
        res += A[p] * B[p];
    }
    return res;
}

inline void scale_vector(const float* src, float scale, int n, float* dst) {
    __m512 s = _mm512_set1_ps(scale);
    int j = 0;
    for (; j <= n - 64; j += 64) {
        _mm512_storeu_ps(dst + j, _mm512_mul_ps(_mm512_loadu_ps(src + j), s));
        _mm512_storeu_ps(dst + j + 16, _mm512_mul_ps(_mm512_loadu_ps(src + j + 16), s));
        _mm512_storeu_ps(dst + j + 32, _mm512_mul_ps(_mm512_loadu_ps(src + j + 32), s));
        _mm512_storeu_ps(dst + j + 48, _mm512_mul_ps(_mm512_loadu_ps(src + j + 48), s));
    }
    for (; j <= n - 16; j += 16) {
        _mm512_storeu_ps(dst + j, _mm512_mul_ps(_mm512_loadu_ps(src + j), s));
    }
    for (; j < n; ++j) {
        dst[j] = src[j] * scale;
    }
}

static constexpr int MR = 16;
static constexpr int NR = 16;
static constexpr int MC = 256;
static constexpr int KC = 256;
static constexpr int NC = 256;

float* matrix_matrix_multiply(const float* A, int m, int k, const float* B, int n) {
    if (m == 1 && n == 1) {
        float* C = new float[1];
        C[0] = dot_product(A, B, k);
        return C;
    }

    if (m == 1 && k == 1) {
        float* C = new float[n];
        scale_vector(B, A[0], n, C);
        return C;
    }
    
    int m_padded = (m + MR - 1) & ~(MR - 1);
    float* C = new float[m_padded * n];
    std::fill(C, C + m_padded * n, 0.0f);

    static float* packedA = nullptr;
    static float* packedB = nullptr;
    static size_t packedA_size = 0;
    static size_t packedB_size = 0;
    
    size_t needed_A = (size_t)MC * KC;
    size_t needed_B = (size_t)KC * NC;
    
    if (packedA_size < needed_A) {
        if (packedA) free_aligned(packedA);
        packedA = (float*)alloc_aligned(needed_A * sizeof(float));
        packedA_size = needed_A;
    }
    if (packedB_size < needed_B) {
        if (packedB) free_aligned(packedB);
        packedB = (float*)alloc_aligned(needed_B * sizeof(float));
        packedB_size = needed_B;
    }

    for (int p0 = 0; p0 < k; p0 += KC) {
        int p_lim = min(k, p0 + KC);
        
        for (int i0 = 0; i0 < m_padded; i0 += MC) {
            int i_lim = min(m_padded, i0 + MC);
            for (int i = i0; i < i_lim; i += MR) {
                pack_A(k, A, k, i, m, p0, p_lim, &packedA[(i - i0) * (p_lim - p0)]);
            }
        }
        
        for (int j0 = 0; j0 < n; j0 += NC) {
            int j_lim = min(n, j0 + NC);

            float* b_pack_ptr = packedB;
            for (int j = j0; j < j_lim; j += NR) {
                int current_nr = min(NR, j_lim - j);
                    pack_B(k, B, n, p0, p_lim, j, j + current_nr, 16, b_pack_ptr);
                b_pack_ptr += (p_lim - p0) * 16;
            }

            const float* p0_a_ptr = packedA;
            for (int i0 = 0; i0 < m_padded; i0 += MC) {
                int i_lim = min(m_padded, i0 + MC);
                
                const float* current_a_block = p0_a_ptr;
                size_t block_size = (size_t)(i_lim - i0) * (p_lim - p0);
                p0_a_ptr += block_size;

                float* current_b_ptr = packedB;
                for (int j = j0; j < j_lim; j += NR) {
                    int current_nr = min(NR, j_lim - j);
                    
                    if (current_nr == 16) {
                        for (int i = i0; i < i_lim; i += MR) {
                            kernel_16x16(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], current_b_ptr, &C[i * n + j], n);
                        }
                    } else {
                        __mmask16 mask = (1 << current_nr) - 1;
                        for (int i = i0; i < i_lim; i += MR) {
                            kernel_16x16_masked(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], current_b_ptr, &C[i * n + j], n, mask);
                        }
                    }
                    current_b_ptr += (p_lim - p0) * 16;
                }
            }
        }
    }
    
    return C;
}

inline void kernel_16x1(int k, const float* packedA, const float* B, float* C) {
    __m512 c0 = _mm512_loadu_ps(C);
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    __m512 c3 = _mm512_setzero_ps();
    
    const float* a_ptr = packedA;
    const float* b_ptr = B;

    int p = 0;
    for (; p <= k - 4; p += 4) {
        c0 = _mm512_fmadd_ps(_mm512_load_ps(a_ptr), _mm512_set1_ps(b_ptr[0]), c0);
        c1 = _mm512_fmadd_ps(_mm512_load_ps(a_ptr + 16), _mm512_set1_ps(b_ptr[1]), c1);
        c2 = _mm512_fmadd_ps(_mm512_load_ps(a_ptr + 32), _mm512_set1_ps(b_ptr[2]), c2);
        c3 = _mm512_fmadd_ps(_mm512_load_ps(a_ptr + 48), _mm512_set1_ps(b_ptr[3]), c3);
        a_ptr += 64;
        b_ptr += 4;
    }
    
    __m512 c = _mm512_add_ps(_mm512_add_ps(c0, c1), _mm512_add_ps(c2, c3));

    for (; p < k; ++p) {
        c = _mm512_fmadd_ps(_mm512_load_ps(a_ptr), _mm512_set1_ps(*b_ptr++), c);
        a_ptr += 16;
    }
    
    _mm512_storeu_ps(C, c);
}

void matrix_matrix_multiply_prepacked(const float* packedA, int m, int k, const float* B, int n, float* C) {
    int m_padded = (m + MR - 1) & ~(MR - 1);
    std::fill(C, C + m_padded * n, 0.0f);

    static float* packedB = nullptr;
    static size_t packedB_size = 0;
    
    size_t needed_B = (size_t)KC * NC;
    if (packedB_size < needed_B) {
        if (packedB) free_aligned(packedB);
        packedB = (float*)alloc_aligned(needed_B * sizeof(float));
        packedB_size = needed_B;
    }

    if (n == 1) {
        const float* a_ptr = packedA;
        for (int p0 = 0; p0 < k; p0 += KC) {
            int p_lim = min(k, p0 + KC);
            for (int i0 = 0; i0 < m_padded; i0 += MC) {
                int i_lim = min(m_padded, i0 + MC);
                
                const float* current_a_block = a_ptr;
                size_t block_size = (size_t)(i_lim - i0) * (p_lim - p0);
                a_ptr += block_size;

                for (int i = i0; i < i_lim; i += MR) {
                    kernel_16x1(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], &B[p0], &C[i]);
                }
            }
        }
        return;
    }

    const float* a_ptr = packedA;

    for (int p0 = 0; p0 < k; p0 += KC) {
        int p_lim = min(k, p0 + KC);
        
        for (int j0 = 0; j0 < n; j0 += NC) {
            int j_lim = min(n, j0 + NC);

            float* b_pack_ptr = packedB;
            for (int j = j0; j < j_lim; j += NR) {
                int current_nr = min(NR, j_lim - j);
                    pack_B(k, B, n, p0, p_lim, j, j + current_nr, 16, b_pack_ptr);
                b_pack_ptr += (p_lim - p0) * 16;
            }

            const float* p0_a_ptr = a_ptr;
            for (int i0 = 0; i0 < m_padded; i0 += MC) {
                int i_lim = min(m_padded, i0 + MC);
                
                const float* current_a_block = p0_a_ptr;
                size_t block_size = (size_t)(i_lim - i0) * (p_lim - p0);
                p0_a_ptr += block_size;

                float* current_b_ptr = packedB;
                for (int j = j0; j < j_lim; j += NR) {
                    int current_nr = min(NR, j_lim - j);
                    
                    if (current_nr == 16) {
                        for (int i = i0; i < i_lim; i += MR) {
                            kernel_16x16(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], current_b_ptr, &C[i * n + j], n);
                        }
                    } else {
                        __mmask16 mask = (1 << current_nr) - 1;
                        for (int i = i0; i < i_lim; i += MR) {
                            kernel_16x16_masked(p_lim - p0, &current_a_block[(i - i0) * (p_lim - p0)], current_b_ptr, &C[i * n + j], n, mask);
                        }
                    }
                    current_b_ptr += (p_lim - p0) * 16;
                }
            }
        }
        
        for (int i0 = 0; i0 < m_padded; i0 += MC) {
             int i_lim = min(m_padded, i0 + MC);
             a_ptr += (size_t)(i_lim - i0) * (p_lim - p0);
        }
    }
}

float* pack_matrix_A(int m, int k, const float* A) {
    int m_padded = (m + MR - 1) & ~(MR - 1);

    size_t total_size = 0;
    for (int p0 = 0; p0 < k; p0 += KC) {
        int p_lim = min(k, p0 + KC);
        for (int i0 = 0; i0 < m_padded; i0 += MC) {
            int i_lim = min(m_padded, i0 + MC);
            total_size += (size_t)(i_lim - i0) * (p_lim - p0);
        }
    }
    
    float* packed = (float*)alloc_aligned(total_size * sizeof(float));
    float* ptr = packed;
    
    for (int p0 = 0; p0 < k; p0 += KC) {
        int p_lim = min(k, p0 + KC);
        for (int i0 = 0; i0 < m_padded; i0 += MC) {
            int i_lim = min(m_padded, i0 + MC);
            
            for (int i = i0; i < i_lim; i += MR) {
                pack_A(k, A, k, i, m, p0, p_lim, ptr);
                ptr += MR * (p_lim - p0);
            }
        }
    }
    return packed;
}

// ==================== Neural Network Layers ====================


struct Linear {
    int in_dim;
    int out_dim;
    float* W;
    float* WT;  // Transposed weights for GEMV
    float* packedW;
    mutable std::vector<float> workspace;
    mutable float* gemv_output;

    Linear(const Linear&) = delete;
    Linear& operator=(const Linear&) = delete;

    Linear(int in_dim, int out_dim, float* weights)
        : in_dim(in_dim), out_dim(out_dim) {
        // Create copy of weights
        W = new float[in_dim * out_dim];
        size_t size = (size_t)in_dim * out_dim;
        for (size_t i = 0; i < size; ++i) W[i] = weights[i];
        
        // Create transposed weights for fast GEMV
        WT = new float[in_dim * out_dim];
        transpose(W, out_dim, in_dim, WT);
        
        packedW = pack_matrix_A(out_dim, in_dim, W);
        gemv_output = new float[out_dim];
        // fprintf(stderr, "Linear %p constructed, gemv_output %p\n", (void*)this, (void*)gemv_output);
        
        // Delete original weights as we own them now
        delete[] weights;
    }

    ~Linear() { 
        delete[] W; 
        delete[] WT;
        free_aligned(packedW);
    }

    // Optimized GEMV: y = W @ x where W is [out_dim, in_dim], x is [in_dim]
    float* forward_gemv(const float* x) const {
        float* y = new float[out_dim];
        // fprintf(stderr, "Linear %p forward_gemv, gemv_output %p\n", (void*)this, (void*)y);
        
        // Process 16 output elements at a time for better ILP
        int i = 0;
        for (; i <= out_dim - 16; i += 16) {
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();
            __m512 sum4 = _mm512_setzero_ps();
            __m512 sum5 = _mm512_setzero_ps();
            __m512 sum6 = _mm512_setzero_ps();
            __m512 sum7 = _mm512_setzero_ps();
            __m512 sum8 = _mm512_setzero_ps();
            __m512 sum9 = _mm512_setzero_ps();
            __m512 sum10 = _mm512_setzero_ps();
            __m512 sum11 = _mm512_setzero_ps();
            __m512 sum12 = _mm512_setzero_ps();
            __m512 sum13 = _mm512_setzero_ps();
            __m512 sum14 = _mm512_setzero_ps();
            __m512 sum15 = _mm512_setzero_ps();
            
            const float* w0 = W + i * in_dim;
            const float* w1 = W + (i+1) * in_dim;
            const float* w2 = W + (i+2) * in_dim;
            const float* w3 = W + (i+3) * in_dim;
            const float* w4 = W + (i+4) * in_dim;
            const float* w5 = W + (i+5) * in_dim;
            const float* w6 = W + (i+6) * in_dim;
            const float* w7 = W + (i+7) * in_dim;
            const float* w8 = W + (i+8) * in_dim;
            const float* w9 = W + (i+9) * in_dim;
            const float* w10 = W + (i+10) * in_dim;
            const float* w11 = W + (i+11) * in_dim;
            const float* w12 = W + (i+12) * in_dim;
            const float* w13 = W + (i+13) * in_dim;
            const float* w14 = W + (i+14) * in_dim;
            const float* w15 = W + (i+15) * in_dim;
            
            int j = 0;
            for (; j <= in_dim - 16; j += 16) {
                __m512 vx = _mm512_loadu_ps(x + j);
                
                sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + j), vx, sum0);
                sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + j), vx, sum1);
                sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + j), vx, sum2);
                sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + j), vx, sum3);
                sum4 = _mm512_fmadd_ps(_mm512_loadu_ps(w4 + j), vx, sum4);
                sum5 = _mm512_fmadd_ps(_mm512_loadu_ps(w5 + j), vx, sum5);
                sum6 = _mm512_fmadd_ps(_mm512_loadu_ps(w6 + j), vx, sum6);
                sum7 = _mm512_fmadd_ps(_mm512_loadu_ps(w7 + j), vx, sum7);
                sum8 = _mm512_fmadd_ps(_mm512_loadu_ps(w8 + j), vx, sum8);
                sum9 = _mm512_fmadd_ps(_mm512_loadu_ps(w9 + j), vx, sum9);
                sum10 = _mm512_fmadd_ps(_mm512_loadu_ps(w10 + j), vx, sum10);
                sum11 = _mm512_fmadd_ps(_mm512_loadu_ps(w11 + j), vx, sum11);
                sum12 = _mm512_fmadd_ps(_mm512_loadu_ps(w12 + j), vx, sum12);
                sum13 = _mm512_fmadd_ps(_mm512_loadu_ps(w13 + j), vx, sum13);
                sum14 = _mm512_fmadd_ps(_mm512_loadu_ps(w14 + j), vx, sum14);
                sum15 = _mm512_fmadd_ps(_mm512_loadu_ps(w15 + j), vx, sum15);
            }
            
            y[i] = _mm512_reduce_add_ps(sum0);
            y[i+1] = _mm512_reduce_add_ps(sum1);
            y[i+2] = _mm512_reduce_add_ps(sum2);
            y[i+3] = _mm512_reduce_add_ps(sum3);
            y[i+4] = _mm512_reduce_add_ps(sum4);
            y[i+5] = _mm512_reduce_add_ps(sum5);
            y[i+6] = _mm512_reduce_add_ps(sum6);
            y[i+7] = _mm512_reduce_add_ps(sum7);
            y[i+8] = _mm512_reduce_add_ps(sum8);
            y[i+9] = _mm512_reduce_add_ps(sum9);
            y[i+10] = _mm512_reduce_add_ps(sum10);
            y[i+11] = _mm512_reduce_add_ps(sum11);
            y[i+12] = _mm512_reduce_add_ps(sum12);
            y[i+13] = _mm512_reduce_add_ps(sum13);
            y[i+14] = _mm512_reduce_add_ps(sum14);
            y[i+15] = _mm512_reduce_add_ps(sum15);
            
            for (; j < in_dim; j++) {
                float xj = x[j];
                y[i] += w0[j] * xj;
                y[i+1] += w1[j] * xj;
                y[i+2] += w2[j] * xj;
                y[i+3] += w3[j] * xj;
                y[i+4] += w4[j] * xj;
                y[i+5] += w5[j] * xj;
                y[i+6] += w6[j] * xj;
                y[i+7] += w7[j] * xj;
                y[i+8] += w8[j] * xj;
                y[i+9] += w9[j] * xj;
                y[i+10] += w10[j] * xj;
                y[i+11] += w11[j] * xj;
                y[i+12] += w12[j] * xj;
                y[i+13] += w13[j] * xj;
                y[i+14] += w14[j] * xj;
                y[i+15] += w15[j] * xj;
            }
        }
        
        for (; i < out_dim; i++) {
            __m512 sum = _mm512_setzero_ps();
            const float* wi = W + i * in_dim;
            int j = 0;
            for (; j <= in_dim - 16; j += 16) {
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(wi + j), _mm512_loadu_ps(x + j), sum);
            }
            float r = _mm512_reduce_add_ps(sum);
            for (; j < in_dim; j++) r += wi[j] * x[j];
            y[i] = r;
        }
        
        return y;
    }

    float* forward(const float* x, int batch_size) const {
        if (batch_size == 1) {
            return forward_gemv(x);
        }

        int m_padded = (out_dim + 15) & ~15;
        size_t required = (size_t)in_dim * batch_size + (size_t)m_padded * batch_size;
        if (workspace.size() < required) workspace.resize(required);
        
        float* x_transposed = workspace.data();
        float* y = workspace.data() + in_dim * batch_size;

        transpose(x, batch_size, in_dim, x_transposed);
        matrix_matrix_multiply_prepacked(packedW, out_dim, in_dim, x_transposed, batch_size, y);
        
        return transpose(y, out_dim, batch_size);
    }
};

struct LayerNorm {
    int dim;
    mutable float* buf;

    explicit LayerNorm(int dim) : dim(dim), buf(new float[dim]) {}

    ~LayerNorm() { delete[] buf; }

    float* forward(const float* x) const {
        // Single pass: compute sum and sum of squares
        __m512 vsum = _mm512_setzero_ps();
        __m512 vsum2 = _mm512_setzero_ps();
        int i = 0;
        for (; i <= dim - 16; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            vsum = _mm512_add_ps(vsum, vx);
            vsum2 = _mm512_fmadd_ps(vx, vx, vsum2);
        }
        float sum = _mm512_reduce_add_ps(vsum);
        float sum2 = _mm512_reduce_add_ps(vsum2);
        for (; i < dim; i++) {
            sum += x[i];
            sum2 += x[i] * x[i];
        }
        
        float mean = sum / dim;
        float var = sum2 / dim - mean * mean;
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        
        __m512 vmean = _mm512_set1_ps(mean);
        __m512 vinv_std = _mm512_set1_ps(inv_std);

        float* y = buf;
        i = 0;
        for (; i <= dim - 16; i += 16) {
            __m512 normalized = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(x + i), vmean), vinv_std);
            _mm512_storeu_ps(y + i, normalized);
        }
        for (; i < dim; i++) {
            y[i] = (x[i] - mean) * inv_std;
        }
        return y;
    }
};

struct FeedForward {
    Linear fc1;
    Linear fc2;
    mutable float* buffer;
    mutable int buffer_size;

    FeedForward(int d_model, int d_ff, float* fc1_weights, float* fc2_weights)
        : fc1(d_model, d_ff, fc1_weights), fc2(d_ff, d_model, fc2_weights),
          buffer(nullptr), buffer_size(0) {}
    
    ~FeedForward() {
        if (buffer) delete[] buffer;
    }

    float* forward(const float* x, int batch_size) {
        float* h = fc1.forward(x, batch_size);
        
        // Fused ReLU - modify in place
        int size = batch_size * fc1.out_dim;
        __m512 zero = _mm512_setzero_ps();
        int i = 0;
        for (; i <= size - 16; i += 16) {
            __m512 v = _mm512_loadu_ps(h + i);
            _mm512_storeu_ps(h + i, _mm512_max_ps(zero, v));
        }
        for (; i < size; i++) {
            if (h[i] < 0.0f) h[i] = 0.0f;
        }
        
        float* o = fc2.forward(h, batch_size);
        delete[] h;
        return o;
    }
};

// SelfAttention with KV caching for incremental generation
struct SelfAttention {
    int d_model;
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear o_proj;
    
    // KV cache
    mutable float* k_cache;
    mutable float* v_cache;
    mutable int cache_len;
    mutable int cache_capacity;
    
    // Preallocated buffers for incremental forward
    mutable float* scores_buf;
    mutable float* out_buf;
    mutable int scores_capacity;
    float scale;

    SelfAttention(int d_model, [[maybe_unused]] int n_head_unused, float* Wq_weights, float* Wk_weights,
                  float* Wv_weights, float* Wo_weights)
        : d_model(d_model),
          q_proj(d_model, d_model, transpose(Wq_weights, d_model, d_model)),
          k_proj(d_model, d_model, transpose(Wk_weights, d_model, d_model)),
          v_proj(d_model, d_model, transpose(Wv_weights, d_model, d_model)),
          o_proj(d_model, d_model, transpose(Wo_weights, d_model, d_model)),
          k_cache(nullptr), v_cache(nullptr), cache_len(0), cache_capacity(0),
          scores_buf(nullptr), out_buf(nullptr), scores_capacity(0),
          scale(1.0f / std::sqrt(static_cast<float>(d_model))) {
        delete[] Wq_weights;
        delete[] Wk_weights;
        delete[] Wv_weights;
        delete[] Wo_weights;
        out_buf = new float[d_model];
    }
    
    ~SelfAttention() {
        if (k_cache) delete[] k_cache;
        if (v_cache) delete[] v_cache;
        if (scores_buf) delete[] scores_buf;
        if (out_buf) delete[] out_buf;
    }
    
    void reset_cache() {
        cache_len = 0;
    }
    
    void ensure_cache_capacity(int new_len) const {
        if (new_len > cache_capacity) {
            int new_cap = max(64, new_len * 2);
            float* new_k = new float[new_cap * d_model];
            float* new_v = new float[new_cap * d_model];
            if (k_cache && cache_len > 0) {
                for (int i = 0; i < cache_len * d_model; i++) {
                    new_k[i] = k_cache[i];
                    new_v[i] = v_cache[i];
                }
            }
            if (k_cache) delete[] k_cache;
            if (v_cache) delete[] v_cache;
            k_cache = new_k;
            v_cache = new_v;
            cache_capacity = new_cap;
        }
        if (new_len > scores_capacity) {
            if (scores_buf) delete[] scores_buf;
            scores_buf = new float[new_len * 2];
            scores_capacity = new_len * 2;
        }
    }

    // Full forward for prefill (processes all tokens)
    float* forward_prefill(const float* x, int T) {
        float* Q = q_proj.forward(x, T);
        float* K = k_proj.forward(x, T);
        float* V = v_proj.forward(x, T);
        
        // Store K, V in cache
        ensure_cache_capacity(T);
        for (int i = 0; i < T * d_model; i++) {
            k_cache[i] = K[i];
            v_cache[i] = V[i];
        }
        cache_len = T;

        float* KT = transpose(K, T, d_model);
        float* scores = matrix_matrix_multiply(Q, T, d_model, KT, T);
        float scale = 1.0f / std::sqrt(static_cast<float>(d_model));
        for (int i = 0; i < T * T; i++) scores[i] *= scale;
        delete[] KT;

        float* attn = new float[T * T];
        for (int i = 0; i < T; i++) {
            float* row_softmax = softmax(scores + i * T, i + 1);
            for (int j = 0; j <= i; ++j) attn[i * T + j] = row_softmax[j];
            delete[] row_softmax;
            for (int j = i + 1; j < T; ++j) attn[i * T + j] = 0.0f;
        }
        delete[] scores;

        float* out = matrix_matrix_multiply(attn, T, T, V, d_model);
        delete[] attn;
        delete[] Q;
        delete[] K;
        delete[] V;

        float* out_proj = o_proj.forward(out, T);
        delete[] out;
        return out_proj;
    }
    
    // Incremental forward (processes only the last token) - highly optimized
    float* forward_incremental(const float* x) {
        float* q = q_proj.forward(x, 1);
        float* k = k_proj.forward(x, 1);
        float* v = v_proj.forward(x, 1);
        
        int new_pos = cache_len;
        ensure_cache_capacity(cache_len + 1);
        
        // SIMD copy to cache
        int i = 0;
        for (; i <= d_model - 16; i += 16) {
            _mm512_storeu_ps(k_cache + new_pos * d_model + i, _mm512_loadu_ps(k + i));
            _mm512_storeu_ps(v_cache + new_pos * d_model + i, _mm512_loadu_ps(v + i));
        }
        for (; i < d_model; i++) {
            k_cache[new_pos * d_model + i] = k[i];
            v_cache[new_pos * d_model + i] = v[i];
        }
        cache_len++;
        
        delete[] k;
        delete[] v;
        
        // Use preallocated scores buffer
        float* scores = scores_buf;
        
        for (int j = 0; j < cache_len; j++) {
            __m512 sum = _mm512_setzero_ps();
            const float* kj = k_cache + j * d_model;
            int p = 0;
            for (; p <= d_model - 64; p += 64) {
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(q + p), _mm512_loadu_ps(kj + p), sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(q + p + 16), _mm512_loadu_ps(kj + p + 16), sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(q + p + 32), _mm512_loadu_ps(kj + p + 32), sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(q + p + 48), _mm512_loadu_ps(kj + p + 48), sum);
            }
            for (; p <= d_model - 16; p += 16) {
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(q + p), _mm512_loadu_ps(kj + p), sum);
            }
            float dot = _mm512_reduce_add_ps(sum);
            for (; p < d_model; p++) dot += q[p] * kj[p];
            scores[j] = dot * scale;
        }
        
        // Fast softmax
        float maxv = scores[0];
        for (int j = 1; j < cache_len; j++) if (scores[j] > maxv) maxv = scores[j];
        
        float sum = 0.0f;
        for (int j = 0; j < cache_len; j++) {
            scores[j] = std::exp(scores[j] - maxv);
            sum += scores[j];
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < cache_len; j++) scores[j] *= inv_sum;
        
        // Use preallocated output buffer
        float* out = out_buf;
        
        // Initialize output with first weighted V
        {
            __m512 va = _mm512_set1_ps(scores[0]);
            const float* v0 = v_cache;
            int p = 0;
            for (; p <= d_model - 64; p += 64) {
                _mm512_storeu_ps(out + p, _mm512_mul_ps(va, _mm512_loadu_ps(v0 + p)));
                _mm512_storeu_ps(out + p + 16, _mm512_mul_ps(va, _mm512_loadu_ps(v0 + p + 16)));
                _mm512_storeu_ps(out + p + 32, _mm512_mul_ps(va, _mm512_loadu_ps(v0 + p + 32)));
                _mm512_storeu_ps(out + p + 48, _mm512_mul_ps(va, _mm512_loadu_ps(v0 + p + 48)));
            }
            for (; p <= d_model - 16; p += 16) {
                _mm512_storeu_ps(out + p, _mm512_mul_ps(va, _mm512_loadu_ps(v0 + p)));
            }
            for (; p < d_model; p++) out[p] = scores[0] * v0[p];
        }
        
        // Add remaining weighted Vs
        for (int j = 1; j < cache_len; j++) {
            __m512 va = _mm512_set1_ps(scores[j]);
            const float* vj = v_cache + j * d_model;
            int p = 0;
            for (; p <= d_model - 64; p += 64) {
                __m512 vo0 = _mm512_loadu_ps(out + p);
                __m512 vo1 = _mm512_loadu_ps(out + p + 16);
                __m512 vo2 = _mm512_loadu_ps(out + p + 32);
                __m512 vo3 = _mm512_loadu_ps(out + p + 48);
                _mm512_storeu_ps(out + p, _mm512_fmadd_ps(va, _mm512_loadu_ps(vj + p), vo0));
                _mm512_storeu_ps(out + p + 16, _mm512_fmadd_ps(va, _mm512_loadu_ps(vj + p + 16), vo1));
                _mm512_storeu_ps(out + p + 32, _mm512_fmadd_ps(va, _mm512_loadu_ps(vj + p + 32), vo2));
                _mm512_storeu_ps(out + p + 48, _mm512_fmadd_ps(va, _mm512_loadu_ps(vj + p + 48), vo3));
            }
            for (; p <= d_model - 16; p += 16) {
                __m512 vo = _mm512_loadu_ps(out + p);
                _mm512_storeu_ps(out + p, _mm512_fmadd_ps(va, _mm512_loadu_ps(vj + p), vo));
            }
            for (; p < d_model; p++) out[p] += scores[j] * vj[p];
        }
        
        delete[] q;
        
        float* out_proj = o_proj.forward(out, 1);
        return out_proj;
    }
};

struct TransformerBlock {
    SelfAttention attn;
    FeedForward ffn;
    LayerNorm ln1;
    LayerNorm ln2;
    int d_model;
    
    // Preallocated buffers for incremental forward
    mutable float* out_buf;
    mutable float* final_buf;

    TransformerBlock(int d_model, int n_head, int d_ff, float* Wq_weights, float* Wk_weights,
                     float* Wv_weights, float* Wo_weights, float* fc1_weights, float* fc2_weights)
        : attn(d_model, n_head, Wq_weights, Wk_weights, Wv_weights, Wo_weights),
          ffn(d_model, d_ff, fc1_weights, fc2_weights),
          ln1(d_model),
          ln2(d_model),
          d_model(d_model) {
        out_buf = new float[d_model];
        final_buf = new float[d_model];
    }
    
    ~TransformerBlock() {
        delete[] out_buf;
        delete[] final_buf;
    }
    
    void reset_cache() {
        attn.reset_cache();
    }

    // Prefill: process all tokens
    float* forward_prefill(const float* x, int T, int d_model) {
        float* x_norm = new float[T * d_model];
        for (int t = 0; t < T; t++) {
            float* norm = ln1.forward(x + t * d_model);
            for (int i = 0; i < d_model; i++) x_norm[t * d_model + i] = norm[i];
        }

        float* attn_out = attn.forward_prefill(x_norm, T);
        delete[] x_norm;

        float* out = new float[T * d_model];
        int size = T * d_model;
        int i = 0;
        for (; i <= size - 16; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 va = _mm512_loadu_ps(attn_out + i);
            _mm512_storeu_ps(out + i, _mm512_add_ps(vx, va));
        }
        for (; i < size; i++) out[i] = x[i] + attn_out[i];
        delete[] attn_out;

        float* y_norm = new float[T * d_model];
        for (int t = 0; t < T; t++) {
            float* norm = ln2.forward(out + t * d_model);
            for (int i = 0; i < d_model; i++) y_norm[t * d_model + i] = norm[i];
        }
        
        float* f = ffn.forward(y_norm, T);
        delete[] y_norm;
        
        float* final_out = new float[T * d_model];
        i = 0;
        for (; i <= size - 16; i += 16) {
            __m512 vo = _mm512_loadu_ps(out + i);
            __m512 vf = _mm512_loadu_ps(f + i);
            _mm512_storeu_ps(final_out + i, _mm512_add_ps(vo, vf));
        }
        for (; i < size; i++) final_out[i] = out[i] + f[i];
        
        delete[] f;
        delete[] out;
        return final_out;
    }
    
    // Incremental: process only the last token
    float* forward_incremental(const float* x) {
        float* x_norm = ln1.forward(x);
        float* attn_out = attn.forward_incremental(x_norm);

        // Use preallocated buffer for intermediate result
        float* out = out_buf;
        int i = 0;
        for (; i <= d_model - 16; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 va = _mm512_loadu_ps(attn_out + i);
            _mm512_storeu_ps(out + i, _mm512_add_ps(vx, va));
        }
        for (; i < d_model; i++) out[i] = x[i] + attn_out[i];
        delete[] attn_out;

        float* y_norm = ln2.forward(out);
        float* f = ffn.forward(y_norm, 1);
        
        // Use preallocated final buffer
        float* final_out = final_buf;
        i = 0;
        for (; i <= d_model - 16; i += 16) {
            __m512 vo = _mm512_loadu_ps(out + i);
            __m512 vf = _mm512_loadu_ps(f + i);
            _mm512_storeu_ps(final_out + i, _mm512_add_ps(vo, vf));
        }
        for (; i < d_model; i++) final_out[i] = out[i] + f[i];
        
        delete[] f;
        return final_out;
    }
};

}  // namespace

struct GPTMini::Impl {
    int vocab_size;
    int d_model;
    int n_head;
    int d_ff;
    int n_layer;
    vector<unique_ptr<TransformerBlock>> blocks;
    float* embed_W;
    Linear lm_head;
    bool dump_enabled = false;
    std::string dump_dir;
    
    // For KV caching
    mutable bool is_first_call;
    mutable float* last_hidden_states;
    mutable int last_T;
    mutable float* embed_buf;

    Impl(int vocab, int d_model, int n_head, int d_ff, int n_layer, float* embed_weights,
         float* lm_head_weights, const vector<GPTMini::BlockWeights>& block_weights)
        : vocab_size(vocab),
          d_model(d_model),
          n_head(n_head),
          d_ff(d_ff),
          n_layer(n_layer),
          blocks(),
          embed_W(embed_weights),
          lm_head(d_model, vocab, lm_head_weights),
          is_first_call(true),
          last_hidden_states(nullptr),
          last_T(0),
          embed_buf(new float[d_model]) {
        blocks.reserve(n_layer);
        for (int i = 0; i < n_layer; i++) {
            const auto& bw = block_weights[i];
            blocks.emplace_back(std::make_unique<TransformerBlock>(d_model, n_head, d_ff, bw.Wq,
                                                                   bw.Wk, bw.Wv, bw.Wo, bw.fc1,
                                                                   bw.fc2));
        }
    }

    ~Impl() {
        delete[] embed_W;
        if (last_hidden_states) delete[] last_hidden_states;
        delete[] embed_buf;
    }

    void enable_layer_dumping(const std::string& directory) {
        utils::enable_layer_dumping(dump_enabled, dump_dir, directory);
    }

    void dump_layer_output(const float* data, int rows, int cols) {
        utils::dump_layer_output(dump_enabled, dump_dir, data, rows, cols);
    }

    int generate_next(const vector<int>& context) {
        int T = context.size();
        
        if (is_first_call || T <= 3) {
            // First call or context reset: do full prefill
            is_first_call = false;
            
            // Reset caches
            for (int layer = 0; layer < n_layer; ++layer) {
                blocks[layer]->reset_cache();
            }
            
            // Embed all tokens
        float* x = new float[T * d_model];
        for (int t = 0; t < T; t++) {
                int id = context[t];
            for (int i = 0; i < d_model; i++) x[t * d_model + i] = embed_W[id * d_model + i];
    }

            // Forward through all layers (prefill)
        // Forward through all layers (prefill)
        for (int layer = 0; layer < n_layer; ++layer) {
            float* new_x = blocks[layer]->forward_prefill(x, T, d_model);
            delete[] x;
            x = new_x;
        }
            
        dump_layer_output(x, T, d_model);
            
            // Store for incremental generation
            if (last_hidden_states) delete[] last_hidden_states;
            last_hidden_states = new float[T * d_model];
            for (int i = 0; i < T * d_model; i++) last_hidden_states[i] = x[i];
            last_T = T;
            
            float* logits = lm_head.forward(x + (T - 1) * d_model, 1);
            float* probs = softmax(logits, vocab_size);
            int next = sample_from(probs, vocab_size);
            
            delete[] x;
            delete[] logits;
            delete[] probs;
            return next;
        } else {
            // Incremental: only process the new token
            int new_token = context[T - 1];
            
            // Embed new token into embed_buf
            for (int i = 0; i < d_model; i++) embed_buf[i] = embed_W[new_token * d_model + i];
            
            // Forward through all layers (incremental)
            // Each layer returns its internal buffer, so we don't need to delete
            float* x = embed_buf;
            for (int layer = 0; layer < n_layer; ++layer) {
                x = blocks[layer]->forward_incremental(x);
            }
            
            float* new_hidden = new float[T * d_model];
            for (int i = 0; i < last_T * d_model; i++) new_hidden[i] = last_hidden_states[i];
            for (int i = 0; i < d_model; i++) new_hidden[last_T * d_model + i] = x[i];
            
            delete[] last_hidden_states;
            last_hidden_states = new_hidden;
            last_T = T;
            
            dump_layer_output(last_hidden_states, T, d_model);
            
            float* logits = lm_head.forward(x, 1);
            float* probs = softmax(logits, vocab_size);
            int next = sample_from(probs, vocab_size);
            
            // x, logits are internal buffers - don't delete
            delete[] probs;
            return next;
        }
    }
};

GPTMini::GPTMini(int vocab, int d_model, int n_head, int d_ff, int n_layer,
                 float* embed_weights, float* lm_head_weights,
                 const vector<BlockWeights>& block_weights)
    : impl(new Impl(vocab, d_model, n_head, d_ff, n_layer, embed_weights, lm_head_weights,
                    block_weights)) {}

GPTMini::~GPTMini() { delete impl; }

int GPTMini::generate_next(const vector<int>& context) { return impl->generate_next(context); }

void GPTMini::enable_layer_dumping(const std::string& directory) {
    impl->enable_layer_dumping(directory);
}
