// Teste de passa/nao-passa de velocidade: multiplicacao de matriz no formato INT8 bloco-32
// (mantissa INT8 + expoente INT8 por bloco de 32 valores ao longo de K) com __dp4a,
// contra cuBLAS FP32 (Sgemm) e FP16 (Hgemm) nos mesmos tamanhos.
//
// C (M x N) = A (M x K) . B (N x K)^T. Saida tambem no formato: INT8 + expoente por bloco de 32 colunas.
//
// Modos do kernel:
//   0 = so INT8 com acumulador int32, sem expoentes (teto de velocidade do dp4a)
//   1 = alinhamento por limite: shift = (RA_i - ea_ib) + (RB_j - eb_jb), RA/RB = maior expoente da
//       linha/coluna. 3 operacoes inteiras extras por bloco por saida. Truncamento (sem arredondar).
//   2 = alinhamento exato ao maior expoente da saida (Emax_ij), com arredondamento; = jatai_int8.gemm.
//
// Compilar: nvcc -O3 -arch=sm_75 bench_gemm.cu -lcublas -o bench_gemm
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s linha %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

constexpr int BM = 64, BN = 64, KB_MAX = 128, G = 6;

template <int MODO, int KBLK>
__global__ void __launch_bounds__(256) gemm_b32(const int8_t* __restrict__ A, const int8_t* __restrict__ EA,
                                                const int8_t* __restrict__ B, const int8_t* __restrict__ EB,
                                                int8_t* __restrict__ C, int8_t* __restrict__ EC, int M, int N, int K) {
    constexpr int GG = KBLK >= 256 ? 3 : (KBLK >= 128 ? 4 : G);  // guarda menor com bloco maior (cabe em int32)
    constexpr int W = KBLK / 4;                     // palavras de 4 bytes por linha por bloco
    __shared__ int As[BM][W + 1], Bs[BN][W + 1];    // +1 de folga contra conflito de banco
    constexpr int NBM = KB_MAX * 32 / KBLK;          // blocos de expoente que cabem (K ate 4096)
    __shared__ int8_t sEA[NBM][BM], sEB[NBM][BN];  // transposto: lanes vizinhas leem bytes vizinhos
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN, nb = K / KBLK;
    if (MODO > 0)
        for (int i = threadIdx.x; i < BM * nb; i += 256) {
            int r = i / nb, b = i % nb;
            sEA[b][r] = EA[(size_t)(row0 + r) * nb + b];
            sEB[b][r] = EB[(size_t)(col0 + r) * nb + b];
        }
    __syncthreads();
    // linhas desta thread: ty*4+i ; colunas: tx+16*j  (um grupo de 32 colunas = j par/impar, 16 lanes)
    int ra[4], rb[4], emax[4][4];
    if (MODO == 1) {
        for (int i = 0; i < 4; i++) { int m = -128; for (int b = 0; b < nb; b++) m = max(m, (int)sEA[b][ty * 4 + i]); ra[i] = m; }
        for (int j = 0; j < 4; j++) { int m = -128; for (int b = 0; b < nb; b++) m = max(m, (int)sEB[b][tx + 16 * j]); rb[j] = m; }
    }
    if (MODO == 2)
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                int m = -256;
                for (int b = 0; b < nb; b++) m = max(m, sEA[b][ty * 4 + i] + sEB[b][tx + 16 * j]);
                emax[i][j] = m;
            }
    int acc[4][4] = {};
    for (int b = 0; b < nb; b++) {
        for (int t = threadIdx.x; t < BM * W; t += 256) {
            int r = t / W, w = t % W;
            As[r][w] = reinterpret_cast<const int*>(A + (size_t)(row0 + r) * K + b * KBLK)[w];
            Bs[r][w] = reinterpret_cast<const int*>(B + (size_t)(col0 + r) * K + b * KBLK)[w];
        }
        __syncthreads();
        int p[4][4] = {};
#pragma unroll
        for (int w = 0; w < W; w++) {
            int a[4], c[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[ty * 4 + i][w];
#pragma unroll
            for (int j = 0; j < 4; j++) c[j] = Bs[tx + 16 * j][w];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) p[i][j] = __dp4a(a[i], c[j], p[i][j]);
        }
        if (MODO == 0) {
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += p[i][j];
        } else if (MODO == 1) {
            int sa[4], sb[4];
#pragma unroll
            for (int i = 0; i < 4; i++) sa[i] = ra[i] - sEA[b][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; j++) sb[j] = rb[j] - sEB[b][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += (p[i][j] << GG) >> min(sa[i] + sb[j], 31);
        } else {
            int ea[4], eb[4];
#pragma unroll
            for (int i = 0; i < 4; i++) ea[i] = sEA[b][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; j++) eb[j] = sEB[b][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    int s = min(emax[i][j] - ea[i] - eb[j], 31);
                    acc[i][j] += ((p[i][j] << GG) + ((1 << s) >> 1)) >> s;
                }
        }
        __syncthreads();
    }
    // saida: requantiza para INT8 com expoente por grupo de 32 colunas (j = 0,1 -> grupo 0; 2,3 -> grupo 1)
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = row0 + ty * 4 + i;
#pragma unroll
        for (int g = 0; g < 2; g++) {
            int base[2], topo = -1000;
            for (int jj = 0; jj < 2; jj++) {
                int j = 2 * g + jj, v = acc[i][j];
                base[jj] = MODO == 0 ? 0 : (MODO == 1 ? ra[i] + rb[j] : emax[i][j]) - GG;
                if (v != 0) topo = max(topo, base[jj] + 32 - __clz(abs(v)));
            }
            for (int o = 1; o < 16; o <<= 1) topo = max(topo, __shfl_xor_sync(0xffffffff, topo, o));
            int E = topo - 7;
            for (int jj = 0; jj < 2; jj++) {
                int j = 2 * g + jj, s = E - base[jj], v = acc[i][j], m;
                if (s > 0) m = (v + (1 << (min(s, 31) - 1))) >> min(s, 31);
                else m = v << min(-s, 31);
                C[(size_t)r * N + col0 + tx + 16 * j] = (int8_t)max(-127, min(127, m));
            }
            if (tx == 0) EC[(size_t)r * (N / 32) + col0 / 32 + g] = (int8_t)E;
        }
    }
}

template <int MODO, int KBLK = 32>
void roda(const int8_t* A, const int8_t* EA, const int8_t* B, const int8_t* EB, int8_t* C, int8_t* EC, int M, int N, int K) {
    gemm_b32<MODO, KBLK><<<dim3(N / BN, M / BM), 256>>>(A, EA, B, EB, C, EC, M, N, K);
}

template <class F>
float tempo_ms(F f, int reps) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; i++) f();
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) f();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

int main() {
    srand(1);
    cublasHandle_t h; cublasCreate(&h);
    // --- conferencia numerica (modo 2, exato) contra conta exata em double, para cada tamanho de bloco
    for (int kb : {32, 128, 256}) {
        int M = 128, N = 128, K = 512, nb = K / kb;
        std::vector<int8_t> a(M * K), b(N * K), ea(M * nb), eb(N * nb), c(M * N), ec(M * N / 32);
        for (auto& x : a) x = rand() % 255 - 127;
        for (auto& x : b) x = rand() % 255 - 127;
        for (auto& x : ea) x = -12 + rand() % 5;
        for (auto& x : eb) x = -12 + rand() % 5;
        int8_t *dA, *dB, *dEA, *dEB, *dC, *dEC;
        CK(cudaMalloc(&dA, M * K)); CK(cudaMalloc(&dB, N * K)); CK(cudaMalloc(&dEA, M * nb)); CK(cudaMalloc(&dEB, N * nb));
        CK(cudaMalloc(&dC, M * N)); CK(cudaMalloc(&dEC, M * N / 32));
        cudaMemcpy(dA, a.data(), M * K, cudaMemcpyHostToDevice); cudaMemcpy(dB, b.data(), N * K, cudaMemcpyHostToDevice);
        cudaMemcpy(dEA, ea.data(), M * nb, cudaMemcpyHostToDevice); cudaMemcpy(dEB, eb.data(), N * nb, cudaMemcpyHostToDevice);
        {
            if (kb == 32) roda<2, 32>(dA, dEA, dB, dEB, dC, dEC, M, N, K);
            if (kb == 128) roda<2, 128>(dA, dEA, dB, dEB, dC, dEC, M, N, K);
            if (kb == 256) roda<2, 256>(dA, dEA, dB, dEB, dC, dEC, M, N, K);
            CK(cudaDeviceSynchronize());
            cudaMemcpy(c.data(), dC, M * N, cudaMemcpyDeviceToHost); cudaMemcpy(ec.data(), dEC, M * N / 32, cudaMemcpyDeviceToHost);
            double num = 0, den = 0;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    double ref = 0;
                    for (int k = 0; k < K; k++) ref += a[i * K + k] * ldexp(1.0, ea[i * nb + k / kb]) * b[j * K + k] * ldexp(1.0, eb[j * nb + k / kb]);
                    double got = c[i * N + j] * ldexp(1.0, ec[i * (N / 32) + j / 32]);
                    num += (got - ref) * (got - ref); den += ref * ref;
                }
            printf("conferencia bloco K=%3d: erro relativo vs conta exata = %.3f%%\n", kb, 100 * sqrt(num / den));
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dEA); cudaFree(dEB); cudaFree(dC); cudaFree(dEC);
    }
    // --- velocidade
    int formas[][3] = {{1024, 256, 256}, {1024, 1024, 256}, {1024, 256, 1024}, {4096, 1024, 1024},
                       {8192, 4096, 1024}, {4096, 4096, 4096}};
    printf("\n%-18s %7s %7s %9s %7s %7s %7s | %9s %9s\n", "M x N x K", "FP32", "FP16", "INT8 puro", "b32", "b128", "b256",
           "b128/FP32", "b256/FP32");
    for (auto& f : formas) {
        int M = f[0], N = f[1], K = f[2], nb = K / 32;
        float *fA, *fB, *fC; __half *hA, *hB, *hC;
        int8_t *iA, *iB, *iC, *eA, *eB, *eC;
        CK(cudaMalloc(&fA, 4ull * M * K)); CK(cudaMalloc(&fB, 4ull * N * K)); CK(cudaMalloc(&fC, 4ull * M * N));
        CK(cudaMalloc(&hA, 2ull * M * K)); CK(cudaMalloc(&hB, 2ull * N * K)); CK(cudaMalloc(&hC, 2ull * M * N));
        CK(cudaMalloc(&iA, 1ull * M * K)); CK(cudaMalloc(&iB, 1ull * N * K)); CK(cudaMalloc(&iC, 1ull * M * N));
        CK(cudaMalloc(&eA, 1ull * M * nb)); CK(cudaMalloc(&eB, 1ull * N * nb)); CK(cudaMalloc(&eC, 1ull * M * N / 32));
        cudaMemset(fA, 0, 4ull * M * K); cudaMemset(fB, 0, 4ull * N * K); cudaMemset(hA, 0, 2ull * M * K); cudaMemset(hB, 0, 2ull * N * K);
        std::vector<int8_t> r(std::max((size_t)M, (size_t)N) * K);
        for (auto& x : r) x = rand() % 255 - 127;
        cudaMemcpy(iA, r.data(), 1ull * M * K, cudaMemcpyHostToDevice); cudaMemcpy(iB, r.data(), 1ull * N * K, cudaMemcpyHostToDevice);
        for (auto& x : r) x = -12 + rand() % 5;
        cudaMemcpy(eA, r.data(), 1ull * M * nb, cudaMemcpyHostToDevice); cudaMemcpy(eB, r.data(), 1ull * N * nb, cudaMemcpyHostToDevice);
        float um = 1, zero = 0; __half hum = __float2half(1.f), hzero = __float2half(0.f);
        int reps = (double)M * N * K > 1e10 ? 10 : 100;
        double ops = 2.0 * M * N * K;
        float t32 = tempo_ms([&] { cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &um, fB, K, fA, K, &zero, fC, N); }, reps);
        float t16 = tempo_ms([&] { cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &hum, hB, K, hA, K, &hzero, hC, N); }, reps);
        float t0 = tempo_ms([&] { roda<0>(iA, eA, iB, eB, iC, eC, M, N, K); }, reps);
        float t1 = tempo_ms([&] { roda<2, 32>(iA, eA, iB, eB, iC, eC, M, N, K); }, reps);
        float t2 = tempo_ms([&] { roda<2, 128>(iA, eA, iB, eB, iC, eC, M, N, K); }, reps);
        float t3 = tempo_ms([&] { roda<2, 256>(iA, eA, iB, eB, iC, eC, M, N, K); }, reps);
        CK(cudaDeviceSynchronize());
        char nome[32]; snprintf(nome, 32, "%dx%dx%d", M, N, K);
        printf("%-18s %7.2f %7.2f %9.2f %7.2f %7.2f %7.2f | %8.2fx %8.2fx\n", nome, ops / t32 / 1e9, ops / t16 / 1e9, ops / t0 / 1e9,
               ops / t1 / 1e9, ops / t2 / 1e9, ops / t3 / 1e9, t32 / t2, t32 / t3);
        cudaFree(fA); cudaFree(fB); cudaFree(fC); cudaFree(hA); cudaFree(hB); cudaFree(hC);
        cudaFree(iA); cudaFree(iB); cudaFree(iC); cudaFree(eA); cudaFree(eB); cudaFree(eC);
    }
    return 0;
}
