// Multiplicacao de matriz no formato do Jatai (mantissa INT8 + expoente INT8 por bloco de KBLK valores
// ao longo de K), escrita do zero com __dp4a, contra um kernel FP32 escrito do zero com a MESMA
// estrutura (mesmo tile, mesmas leituras; so troca dp4a por FMA). cuBLAS FP32/FP16 aparecem so como
// regua do "treino tradicional" (e o que o PyTorch usa), nao fazem parte do metodo.
//
// C (M x N) = A (M x K) . B (N x K)^T. Saida no mesmo formato: INT8 + expoente por grupo de 32 colunas.
//
// Estrutura: bloco de 256 threads calcula 128 x 128 saidas; cada thread 8 x 8. A cada passo o tile de
// K tem 32 bytes (8 palavras de 4 INT8). Cada leitura de 16 bytes da memoria compartilhada alimenta
// 32 dp4a. Os produtos de um bloco de expoente somam exatos em int32 (p); no fim do bloco, p e
// alinhado no acumulador: acc += arred(p * 2^G / 2^s), s = (RA_i - ea_ib) + (RB_j - eb_jb), onde RA/RB
// sao o maior expoente da linha de A / da linha de B. Arredondamento empate-para-par (sem vies).
//
// Compilar: nvcc -O3 -arch=sm_75 gemm_rapido.cu -lcublas -o gemm_rapido
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s linha %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

constexpr int BM = 128, BN = 128, KMAX = 4096;

// x / 2^s com arredondamento empate-para-par (s >= 0; s = 0 devolve x)
__device__ __forceinline__ int arred_shift(int x, int s) {
    s = min(s, 31);
    if (s == 0) return x;
    return (x + ((1 << (s - 1)) - 1) + ((x >> s) & 1)) >> s;
}

template <int KBLK>
__global__ void __launch_bounds__(256, 1) gemm_int8(const int8_t* __restrict__ A, const int8_t* __restrict__ EA,
                                                    const int8_t* __restrict__ B, const int8_t* __restrict__ EB,
                                                    int8_t* __restrict__ C, int8_t* __restrict__ EC, int M, int N, int K) {
    constexpr int G = KBLK >= 128 ? 4 : 6;                // bits de guarda (o acumulador cabe em int32)
    constexpr int CH = KBLK < 64 ? KBLK : 64;             // bytes de K por tile (por __syncthreads)
    constexpr int WT = CH / 4, NV = WT / 8;               // palavras por linha no tile; int4 por thread na carga
    constexpr int PASSOS = KBLK / CH;                     // tiles por bloco de expoente
    constexpr int NBMAX = KMAX / KBLK;
    __shared__ __align__(16) int As[2][WT][BM];           // [buffer][palavra][linha]: thread le 8 linhas seguidas
    __shared__ __align__(16) int Bs[2][WT][BN];           // 2 buffers: carrega o proximo tile enquanto calcula
    __shared__ int8_t sEA[NBMAX][BM], sEB[NBMAX][BN];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN, nb = K / KBLK;
    // colunas desta thread intercaladas (tx*4..+3 e 64+tx*4..+3): lanes vizinhas leem 16 bytes contiguos
#define COL(j) ((j) < 4 ? tx * 4 + (j) : 64 + tx * 4 + (j) - 4)
    for (int i = tid; i < BM * nb; i += 256) {
        int r = i / nb, b = i % nb;
        sEA[b][r] = EA[(size_t)(row0 + r) * nb + b];
        sEB[b][r] = EB[(size_t)(col0 + r) * nb + b];
    }
    __syncthreads();
    int ra[8], rb[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int ma = -128, mb = -128;
        for (int b = 0; b < nb; b++) { ma = max(ma, (int)sEA[b][ty * 8 + i]); mb = max(mb, (int)sEB[b][COL(i)]); }
        ra[i] = ma; rb[i] = mb;
    }
    int acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0;
    // carga: 128 linhas x WT palavras por matriz; cada thread le NV x 16 bytes de uma linha
    const int lr = tid >> 1, lw = (tid & 1) * NV * 4;
    const int4* gA = reinterpret_cast<const int4*>(A + (size_t)(row0 + lr) * K) + (tid & 1) * NV;
    const int4* gB = reinterpret_cast<const int4*>(B + (size_t)(col0 + lr) * K) + (tid & 1) * NV;
    const int ntiles = K / CH;
    auto guarda = [&](int bf, const int4* va, const int4* vb) {
#pragma unroll
        for (int v = 0; v < NV; v++) {
            As[bf][lw + 4 * v + 0][lr] = va[v].x; As[bf][lw + 4 * v + 1][lr] = va[v].y;
            As[bf][lw + 4 * v + 2][lr] = va[v].z; As[bf][lw + 4 * v + 3][lr] = va[v].w;
            Bs[bf][lw + 4 * v + 0][lr] = vb[v].x; Bs[bf][lw + 4 * v + 1][lr] = vb[v].y;
            Bs[bf][lw + 4 * v + 2][lr] = vb[v].z; Bs[bf][lw + 4 * v + 3][lr] = vb[v].w;
        }
    };
    {
        int4 va[NV], vb[NV];
#pragma unroll
        for (int v = 0; v < NV; v++) { va[v] = gA[v]; vb[v] = gB[v]; }
        guarda(0, va, vb);
        __syncthreads();
    }
    for (int b = 0; b < nb; b++) {
        int p[8][8];
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) p[i][j] = 0;
        for (int t = 0; t < PASSOS; t++) {
            const int c = b * PASSOS + t, buf = c & 1;
            const bool tem_prox = c + 1 < ntiles;
            int4 va[NV], vb[NV];
            if (tem_prox)                                   // leitura da VRAM ja a caminho
#pragma unroll
                for (int v = 0; v < NV; v++) { va[v] = gA[(c + 1) * (CH / 16) + v]; vb[v] = gB[(c + 1) * (CH / 16) + v]; }
#pragma unroll
            for (int w = 0; w < WT; w++) {
                int4 a0 = *reinterpret_cast<const int4*>(&As[buf][w][ty * 8]), a1 = *reinterpret_cast<const int4*>(&As[buf][w][ty * 8 + 4]);
                int4 b0 = *reinterpret_cast<const int4*>(&Bs[buf][w][tx * 4]), b1 = *reinterpret_cast<const int4*>(&Bs[buf][w][64 + tx * 4]);
                int a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                int c[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < 8; i++)
#pragma unroll
                    for (int j = 0; j < 8; j++) p[i][j] = __dp4a(a[i], c[j], p[i][j]);
            }
            if (tem_prox) guarda(buf ^ 1, va, vb);
            __syncthreads();
        }
        int sa[8], sb[8];
#pragma unroll
        for (int i = 0; i < 8; i++) { sa[i] = ra[i] - sEA[b][ty * 8 + i]; sb[i] = rb[i] - sEB[b][COL(i)]; }
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) acc[i][j] += arred_shift(p[i][j] << G, sa[i] + sb[j]);
    }
    // saida: INT8 + expoente por grupo de 32 colunas (= 8 threads vizinhas; cada thread tem 2 grupos)
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = row0 + ty * 8 + i;
#pragma unroll
        for (int g = 0; g < 2; g++) {
            int topo = -1000;
#pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                int j = 4 * g + jj;
                if (acc[i][j] != 0) topo = max(topo, ra[i] + rb[j] - G + 32 - __clz(abs(acc[i][j])));
            }
            topo = max(topo, __shfl_xor_sync(0xffffffff, topo, 1));
            topo = max(topo, __shfl_xor_sync(0xffffffff, topo, 2));
            topo = max(topo, __shfl_xor_sync(0xffffffff, topo, 4));
            int E = topo - 7;
            int8_t m[4];
#pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                int j = 4 * g + jj, s = E - (ra[i] + rb[j] - G);
                int v = s >= 0 ? arred_shift(acc[i][j], s) : acc[i][j] << min(-s, 31);
                m[jj] = (int8_t)max(-127, min(127, v));
            }
            int c0 = col0 + COL(4 * g);
            *reinterpret_cast<int*>(C + (size_t)r * N + c0) = *reinterpret_cast<int*>(m);
            if ((tx & 7) == 0) EC[(size_t)r * (N / 32) + c0 / 32] = (int8_t)E;
        }
    }
#undef COL
}

// Mesma estrutura em FP32: tile de K = 8 floats (32 bytes por linha, como no INT8), 8 x 8 por thread.
__global__ void __launch_bounds__(256, 1) gemm_fp32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                                    int M, int N, int K) {
    __shared__ __align__(16) float As[8][BM], Bs[8][BN];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int lr = tid >> 1, lw = (tid & 1) * 4;
    const float4* gA = reinterpret_cast<const float4*>(A + (size_t)(row0 + lr) * K) + (tid & 1);
    const float4* gB = reinterpret_cast<const float4*>(B + (size_t)(col0 + lr) * K) + (tid & 1);
    float acc[8][8] = {};
    for (int kk = 0; kk < K / 4; kk += 2) {
        float4 va = gA[kk], vb = gB[kk];
        As[lw + 0][lr] = va.x; As[lw + 1][lr] = va.y; As[lw + 2][lr] = va.z; As[lw + 3][lr] = va.w;
        Bs[lw + 0][lr] = vb.x; Bs[lw + 1][lr] = vb.y; Bs[lw + 2][lr] = vb.z; Bs[lw + 3][lr] = vb.w;
        __syncthreads();
#pragma unroll
        for (int w = 0; w < 8; w++) {
            float4 a0 = *reinterpret_cast<const float4*>(&As[w][ty * 8]), a1 = *reinterpret_cast<const float4*>(&As[w][ty * 8 + 4]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[w][tx * 8]), b1 = *reinterpret_cast<const float4*>(&Bs[w][tx * 8 + 4]);
            float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float c[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], c[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        float* cr = C + (size_t)(row0 + ty * 8 + i) * N + col0 + tx * 8;
        *reinterpret_cast<float4*>(cr) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(cr + 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
}

template <int KBLK>
void roda_int8(const int8_t* A, const int8_t* EA, const int8_t* B, const int8_t* EB, int8_t* C, int8_t* EC, int M, int N, int K) {
    gemm_int8<KBLK><<<dim3(N / BN, M / BM), 256>>>(A, EA, B, EB, C, EC, M, N, K);
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
    // --- conferencia: INT8 contra conta exata em double; FP32 proprio contra double
    printf("conferencia (M=N=128, K=512, mantissas aleatorias, expoentes por bloco variando 5 valores):\n");
    for (int kb : {32, 64, 128, 256}) {
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
        if (kb == 32) roda_int8<32>(dA, dEA, dB, dEB, dC, dEC, M, N, K);
        if (kb == 64) roda_int8<64>(dA, dEA, dB, dEB, dC, dEC, M, N, K);
        if (kb == 128) roda_int8<128>(dA, dEA, dB, dEB, dC, dEC, M, N, K);
        if (kb == 256) roda_int8<256>(dA, dEA, dB, dEB, dC, dEC, M, N, K);
        CK(cudaDeviceSynchronize());
        cudaMemcpy(c.data(), dC, M * N, cudaMemcpyDeviceToHost); cudaMemcpy(ec.data(), dEC, M * N / 32, cudaMemcpyDeviceToHost);
        double num = 0, den = 0, num_q = 0;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j += 32) {
                double ref[32], mx = 0;
                for (int jj = 0; jj < 32; jj++) {
                    ref[jj] = 0;
                    for (int k = 0; k < K; k++)
                        ref[jj] += a[i * K + k] * ldexp(1.0, ea[i * nb + k / kb]) * b[(j + jj) * K + k] * ldexp(1.0, eb[(j + jj) * nb + k / kb]);
                    mx = std::max(mx, fabs(ref[jj]));
                }
                double esc = ldexp(1.0, (int)floor(log2(mx)) + 1 - 7);   // regua: so arredondar o exato p/ INT8
                for (int jj = 0; jj < 32; jj++) {
                    double got = c[i * N + j + jj] * ldexp(1.0, ec[i * (N / 32) + j / 32]);
                    double q = std::max(-127.0, std::min(127.0, nearbyint(ref[jj] / esc))) * esc;
                    num += (got - ref[jj]) * (got - ref[jj]); num_q += (q - ref[jj]) * (q - ref[jj]); den += ref[jj] * ref[jj];
                }
            }
        printf("  INT8 bloco K=%3d: erro %.3f%%   (regua, so arredondar a saida exata: %.3f%%)\n", kb, 100 * sqrt(num / den),
               100 * sqrt(num_q / den));
        cudaFree(dA); cudaFree(dB); cudaFree(dEA); cudaFree(dEB); cudaFree(dC); cudaFree(dEC);
    }
    {
        int M = 128, N = 128, K = 512;
        std::vector<float> a(M * K), b(N * K), c(M * N);
        for (auto& x : a) x = (rand() % 2001 - 1000) / 1000.f;
        for (auto& x : b) x = (rand() % 2001 - 1000) / 1000.f;
        float *dA, *dB, *dC;
        CK(cudaMalloc(&dA, 4 * M * K)); CK(cudaMalloc(&dB, 4 * N * K)); CK(cudaMalloc(&dC, 4 * M * N));
        cudaMemcpy(dA, a.data(), 4 * M * K, cudaMemcpyHostToDevice); cudaMemcpy(dB, b.data(), 4 * N * K, cudaMemcpyHostToDevice);
        gemm_fp32<<<dim3(N / BN, M / BM), 256>>>(dA, dB, dC, M, N, K);
        CK(cudaDeviceSynchronize());
        cudaMemcpy(c.data(), dC, 4 * M * N, cudaMemcpyDeviceToHost);
        double num = 0, den = 0;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                double ref = 0;
                for (int k = 0; k < K; k++) ref += (double)a[i * K + k] * b[j * K + k];
                num += (c[i * N + j] - ref) * (c[i * N + j] - ref); den += ref * ref;
            }
        printf("  FP32 proprio:     erro %.5f%%\n", 100 * sqrt(num / den));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    // --- velocidade
    int formas[][3] = {{1024, 256, 256}, {1024, 1024, 256}, {1024, 256, 1024}, {4096, 1024, 1024},
                       {8192, 4096, 1024}, {4096, 4096, 4096}};
    printf("\nvelocidade (TFLOPS para float, TOPS para INT8; 1 multiplicacao + 1 soma = 2 operacoes)\n");
    printf("%-17s | %7s %7s | %7s | %7s %7s %7s %7s | %10s %10s\n", "M x N x K", "cuBLAS", "cuBLAS", "FP32", "INT8", "INT8",
           "INT8", "INT8", "INT8 b128", "INT8 b128");
    printf("%-17s | %7s %7s | %7s | %7s %7s %7s %7s | %10s %10s\n", "", "FP32", "FP16", "proprio", "b32", "b64", "b128", "b256",
           "/FP32 prop", "/cuBLAS32");
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
        float tc32 = tempo_ms([&] { cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &um, fB, K, fA, K, &zero, fC, N); }, reps);
        float tc16 = tempo_ms([&] { cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &hum, hB, K, hA, K, &hzero, hC, N); }, reps);
        float tf = tempo_ms([&] { gemm_fp32<<<dim3(N / BN, M / BM), 256>>>(fA, fB, fC, M, N, K); }, reps);
        float t32 = tempo_ms([&] { roda_int8<32>(iA, eA, iB, eB, iC, eC, M, N, K); }, reps);
        float t64 = tempo_ms([&] { roda_int8<64>(iA, eA, iB, eB, iC, eC, M, N, K); }, reps);
        float t128 = tempo_ms([&] { roda_int8<128>(iA, eA, iB, eB, iC, eC, M, N, K); }, reps);
        float t256 = K >= 256 ? tempo_ms([&] { roda_int8<256>(iA, eA, iB, eB, iC, eC, M, N, K); }, reps) : 0;
        CK(cudaDeviceSynchronize());
        char nome[32]; snprintf(nome, 32, "%dx%dx%d", M, N, K);
        auto T = [&](float t) { return t > 0 ? ops / t / 1e9 : 0.0; };
        printf("%-17s | %7.2f %7.2f | %7.2f | %7.2f %7.2f %7.2f %7.2f | %9.2fx %9.2fx\n", nome, T(tc32), T(tc16), T(tf), T(t32), T(t64),
               T(t128), T(t256), tf / t128, tc32 / t128);
        cudaFree(fA); cudaFree(fB); cudaFree(fC); cudaFree(hA); cudaFree(hB); cudaFree(hC);
        cudaFree(iA); cudaFree(iB); cudaFree(iC); cudaFree(eA); cudaFree(eB); cudaFree(eC);
    }
    return 0;
}
