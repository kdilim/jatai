// Experimento: multiplicacao de matriz INT8 feita so com operacoes de bit (planos de bits + AND + popcount)
// contra a mesma multiplicacao com dp4a, na mesma estrutura de kernel. Resultado tem que ser identico (int32).
//
// Plano de bits: X int8 [R x K] vira Xp uint32 [R][K/32][8]; o bit k da palavra (r, w, p) e o bit p de
// X[r][32w + k]. Em complemento de dois, x = -128 b7 + sum_{p<7} 2^p b_p, entao
//   sum_k a_k b_k = sum_{p,q} peso_p peso_q popc(A_p & B_q),  peso_p = 2^p (p<7), peso_7 = -128.
// A conversao para planos usa __ballot_sync: cada lane segura um elemento, e o ballot do bit p ja e a palavra.
// Variante "nbits": usa so os n bits mais altos de cada lado (mostra como o custo cai com menos planos;
// nao e exata).
//
// Compilar: nvcc -O3 -arch=sm_75 bitplano.cu -o bitplano
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s linha %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int TM = 64, TN = 64, KW = 8;   // tile 64 x 64 saidas; 8 palavras de 32 (= 256 de K) por etapa

// int8 [R x K] -> planos [R][K/32][8]. 1 warp = 32 elementos consecutivos de uma linha.
__global__ void k_planos(const int8_t* X, uint32_t* Xp, int R, int K) {
    long long w = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= (long long)R * K / 32) return;
    int x = X[w * 32 + (threadIdx.x & 31)];
#pragma unroll
    for (int p = 0; p < 8; p++) {
        uint32_t b = __ballot_sync(0xffffffffu, (x >> p) & 1);
        if ((threadIdx.x & 31) == p) Xp[w * 8 + p] = b;
    }
}

// C = A . B^T por planos de bits; cada thread 4 x 4 saidas; NB = numero de planos usados (de cima para baixo)
template <int NB>
__global__ void __launch_bounds__(256) k_bits(const uint32_t* __restrict__ Ap, const uint32_t* __restrict__ Bp, int* C, int M, int N, int K) {
    __shared__ uint32_t As[KW][8][TM + 1], Bs[KW][8][TN + 1];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4, r0 = blockIdx.y * TM, c0 = blockIdx.x * TN, nw = K / 32;
    int acc[4][4] = {};
    for (int w0 = 0; w0 < nw; w0 += KW) {
        for (int i = tid; i < TM * KW * 8; i += 256) {
            int r = i / (KW * 8), w = (i / 8) % KW, p = i % 8;
            As[w][p][r] = Ap[((size_t)(r0 + r) * nw + w0 + w) * 8 + p];
            Bs[w][p][r] = Bp[((size_t)(c0 + r) * nw + w0 + w) * 8 + p];
        }
        __syncthreads();
#pragma unroll 1
        for (int w = 0; w < KW; w++) {
            uint32_t a[4][NB], b[4][NB];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int p = 0; p < NB; p++) { a[i][p] = As[w][7 - p][ty + 16 * i]; b[i][p] = Bs[w][7 - p][tx + 16 * i]; }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    int t = 0;
#pragma unroll
                    for (int p = 0; p < NB; p++) {
                        int s = 0;
#pragma unroll
                        for (int q = 0; q < NB; q++) s += (q == 0 ? -128 : (128 >> q)) * __popc(a[i][p] & b[j][q]);
                        t += (p == 0 ? -128 : (128 >> p)) * s;
                    }
                    acc[i][j] += t;
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) C[(size_t)(r0 + ty + 16 * i) * N + c0 + tx + 16 * j] = acc[i][j];
}

// mesma estrutura com dp4a: int8 [R x K], 4 valores por palavra
__global__ void __launch_bounds__(256) k_dp4a(const int* __restrict__ A, const int* __restrict__ B, int* C, int M, int N, int K) {
    constexpr int KW4 = KW * 8;               // mesmas 256 posicoes de K por etapa = 64 palavras
    __shared__ int As[KW4][TM + 1], Bs[KW4][TN + 1];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4, r0 = blockIdx.y * TM, c0 = blockIdx.x * TN, nw = K / 4;
    int acc[4][4] = {};
    for (int w0 = 0; w0 < nw; w0 += KW4) {
        for (int i = tid; i < TM * KW4; i += 256) {
            int r = i / KW4, w = i % KW4;
            As[w][r] = A[(size_t)(r0 + r) * nw + w0 + w];
            Bs[w][r] = B[(size_t)(c0 + r) * nw + w0 + w];
        }
        __syncthreads();
#pragma unroll 4
        for (int w = 0; w < KW4; w++) {
            int a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) { a[i] = As[w][ty + 16 * i]; b[i] = Bs[w][tx + 16 * i]; }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) C[(size_t)(r0 + ty + 16 * i) * N + c0 + tx + 16 * j] = acc[i][j];
}

template <typename F>
float mede(F f, int n = 10) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    f(); CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < n; i++) f();
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / n;
}

int main() {
    srand(0);
    struct F { int M, N, K; const char* nome; } formas[] = {
        {2048, 1024, 256, "modelo fwd Wup"}, {2048, 256, 1024, "modelo fwd Wd"}, {2048, 2048, 2048, "quadrada 2048"}};
    for (auto& f : formas) {
        const int M = f.M, N = f.N, K = f.K;
        std::vector<int8_t> hA((size_t)M * K), hB((size_t)N * K);
        for (auto& x : hA) x = (int8_t)(rand() % 255 - 127);
        for (auto& x : hB) x = (int8_t)(rand() % 255 - 127);
        int8_t *A, *B; uint32_t *Ap, *Bp; int *C1, *C2;
        CK(cudaMalloc(&A, M * K)); CK(cudaMalloc(&B, N * K));
        CK(cudaMalloc(&Ap, (size_t)M * K)); CK(cudaMalloc(&Bp, (size_t)N * K));   // 8 palavras de 4 bytes por 32 elementos = 1 byte por elemento
        CK(cudaMalloc(&C1, (size_t)M * N * 4)); CK(cudaMalloc(&C2, (size_t)M * N * 4));
        CK(cudaMemcpy(A, hA.data(), M * K, cudaMemcpyHostToDevice)); CK(cudaMemcpy(B, hB.data(), N * K, cudaMemcpyHostToDevice));
        dim3 g(N / TN, M / TM);
        float t_dp = mede([&] { k_dp4a<<<g, 256>>>((const int*)A, (const int*)B, C1, M, N, K); });
        float t_cv = mede([&] { k_planos<<<(M * K / 32 + 7) / 8, 256>>>(A, Ap, M, K); k_planos<<<(N * K / 32 + 7) / 8, 256>>>(B, Bp, N, K); });
        float t_b8 = mede([&] { k_bits<8><<<g, 256>>>(Ap, Bp, C2, M, N, K); });
        CK(cudaGetLastError());
        std::vector<int> h1((size_t)M * N), h2((size_t)M * N);
        CK(cudaMemcpy(h1.data(), C1, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2.data(), C2, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
        long long dif = 0, ref_err = 0;
        for (size_t i = 0; i < h1.size(); i++) dif += h1[i] != h2[i];
        for (int i = 0; i < 64; i++) {           // confere amostra contra a conta direta na CPU
            int r = rand() % M, c = rand() % N; long long s = 0;
            for (int k = 0; k < K; k++) s += (int)hA[(size_t)r * K + k] * hB[(size_t)c * K + k];
            ref_err += s != h1[(size_t)r * N + c];
        }
        float t_b4 = mede([&] { k_bits<4><<<g, 256>>>(Ap, Bp, C2, M, N, K); });
        float t_b2 = mede([&] { k_bits<2><<<g, 256>>>(Ap, Bp, C2, M, N, K); });
        float t_b1 = mede([&] { k_bits<1><<<g, 256>>>(Ap, Bp, C2, M, N, K); });
        double op = 2.0 * M * N * K;
        printf("%-16s M=%d N=%d K=%d\n", f.nome, M, N, K);
        printf("  dp4a                      %9.3f ms  %6.2f TOPS\n", t_dp, op / t_dp / 1e9);
        printf("  bitwise 8x8 planos (exato) %8.3f ms  %6.2f TOPS  (%.1fx mais lento)  + conversao p/ planos %.3f ms\n",
               t_b8, op / t_b8 / 1e9, t_b8 / t_dp, t_cv);
        printf("  bitwise 4x4 planos (aprox) %8.3f ms  %6.2f TOPS  (%.1fx)\n", t_b4, op / t_b4 / 1e9, t_b4 / t_dp);
        printf("  bitwise 2x2 planos (aprox) %8.3f ms  %6.2f TOPS  (%.1fx)\n", t_b2, op / t_b2 / 1e9, t_b2 / t_dp);
        printf("  bitwise 1x1 plano  (sinal) %8.3f ms  %6.2f TOPS  (%.1fx)\n", t_b1, op / t_b1 / 1e9, t_b1 / t_dp);
        printf("  conferencia: bitwise 8x8 x dp4a: %lld diferencas de %zu; dp4a x CPU (64 amostras): %lld\n\n", dif, h1.size(), ref_err);
        cudaFree(A); cudaFree(B); cudaFree(Ap); cudaFree(Bp); cudaFree(C1); cudaFree(C2);
    }
    return 0;
}
