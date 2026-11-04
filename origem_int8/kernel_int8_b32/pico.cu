// Teto de cada unidade da GTX 1650, so registradores (sem memoria): dp4a, FMA FP32, FMA FP16 (half2).
// Contagem: 1 multiplicacao + 1 soma = 2 operacoes. dp4a = 4 multiplicacoes-soma = 8 operacoes.
// Compilar: nvcc -O3 -arch=sm_75 pico.cu -o pico
#include <cstdio>
#include <cuda_fp16.h>

constexpr int ITER = 4096, CAD = 16;   // 16 cadeias independentes por thread (esconde a latencia)

__global__ void k_dp4a(int* out, int s) {
    int a = threadIdx.x + s, b = a * 7 + 1, acc[CAD];
    for (int c = 0; c < CAD; c++) acc[c] = c;
    for (int i = 0; i < ITER; i++)
#pragma unroll
        for (int c = 0; c < CAD; c++) acc[c] = __dp4a(a, b, acc[c]);
    int t = 0;
    for (int c = 0; c < CAD; c++) t += acc[c];
    if (t == 0x7fffffff) out[0] = t;
}
__global__ void k_fp32(float* out, float s) {
    float a = threadIdx.x * s, b = a + 1.f, acc[CAD];
    for (int c = 0; c < CAD; c++) acc[c] = c;
    for (int i = 0; i < ITER; i++)
#pragma unroll
        for (int c = 0; c < CAD; c++) acc[c] = fmaf(a, b, acc[c]);
    float t = 0;
    for (int c = 0; c < CAD; c++) t += acc[c];
    if (t == 1234.5f) out[0] = t;
}
__global__ void k_fp16(__half2* out, float s) {
    __half2 a = __float2half2_rn(threadIdx.x * s), b = __float2half2_rn(1.f), acc[CAD];
    for (int c = 0; c < CAD; c++) acc[c] = __float2half2_rn(c);
    for (int i = 0; i < ITER; i++)
#pragma unroll
        for (int c = 0; c < CAD; c++) acc[c] = __hfma2(a, b, acc[c]);
    __half2 t = acc[0];
    for (int c = 1; c < CAD; c++) t = __hadd2(t, acc[c]);
    if (__low2float(t) == 1234.5f) out[0] = t;
}

template <class F>
float ms(F f) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    f(); cudaEventRecord(a); for (int i = 0; i < 5; i++) f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float t; cudaEventElapsedTime(&t, a, b); return t / 5;
}

int main() {
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int blocos = sms * 8, thr = 256;
    void* buf; cudaMalloc(&buf, 64);
    double n = (double)blocos * thr * ITER * CAD;
    float t1 = ms([&] { k_dp4a<<<blocos, thr>>>((int*)buf, 3); });
    float t2 = ms([&] { k_fp32<<<blocos, thr>>>((float*)buf, 0.5f); });
    float t3 = ms([&] { k_fp16<<<blocos, thr>>>((__half2*)buf, 0.5f); });
    printf("SMs: %d\n", sms);
    printf("dp4a  (INT8):        %6.2f TOPS\n", n * 8 / t1 / 1e9);
    printf("FMA   FP32:          %6.2f TFLOPS\n", n * 2 / t2 / 1e9);
    printf("HFMA2 FP16 (half2):  %6.2f TFLOPS\n", n * 4 / t3 / 1e9);
    return 0;
}
