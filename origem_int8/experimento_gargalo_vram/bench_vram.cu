// Mede o custo de "tocar na VRAM": a mesma cadeia de S etapas inteiras,
// fundida (um kernel, valores intermediarios so em registrador) vs separada
// (S kernels, cada etapa le e grava a VRAM).
// Compilar: nvcc -O3 -arch=sm_75 bench_vram.cu -o bench_vram
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA erro %s linha %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__device__ __forceinline__ int etapa(int x) {
    x = (x * 37 + 11) >> 5;
    return max(-127, min(127, x));
}

__global__ void cadeia(const char4* __restrict__ in, char4* __restrict__ out, size_t n4, int passos) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        char4 v = in[i];
        int a = v.x, b = v.y, c = v.z, d = v.w;
        for (int s = 0; s < passos; ++s) { a = etapa(a); b = etapa(b); c = etapa(c); d = etapa(d); }
        out[i] = make_char4(a, b, c, d);
    }
}

__global__ void pico_dp4a(int* out, int iters) {
    int x = threadIdx.x * 0x01020304, w = 0x05060708;
    int a0 = 0, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7;
    for (int i = 0; i < iters; ++i) {
        a0 = __dp4a(x, w, a0); a1 = __dp4a(x, w, a1); a2 = __dp4a(x, w, a2); a3 = __dp4a(x, w, a3);
        a4 = __dp4a(x, w, a4); a5 = __dp4a(x, w, a5); a6 = __dp4a(x, w, a6); a7 = __dp4a(x, w, a7);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

float cronometrar(void (*f)(void*), void* ctx, int reps) {
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    f(ctx); CK(cudaDeviceSynchronize());
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r) f(ctx);
    cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    return ms / reps;
}

struct Ctx { char4 *a, *b; size_t n4; int passos; int grid; };

void fundido(void* p) {
    Ctx* c = (Ctx*)p;
    cadeia<<<c->grid, 256>>>(c->a, c->b, c->n4, c->passos);
}
void separado(void* p) {
    Ctx* c = (Ctx*)p;
    char4 *src = c->a, *dst = c->b;
    for (int s = 0; s < c->passos; ++s) {
        cadeia<<<c->grid, 256>>>(src, dst, c->n4, 1);
        char4* tmp = src; src = dst; dst = tmp;
    }
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sms = prop.multiProcessorCount;
    printf("GPU: %s | SMs %d | L2 %d KB | barramento %d bits | clock mem %d MHz\n",
           prop.name, sms, prop.l2CacheSize / 1024, prop.memoryBusWidth, prop.memoryClockRate / 1000);
    double banda_teorica = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
    printf("Banda teorica da VRAM (pelo driver): %.0f GB/s\n\n", banda_teorica);

    int* dout; CK(cudaMalloc(&dout, sms * 8 * 256 * sizeof(int)));
    int iters = 1 << 16;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    pico_dp4a<<<sms * 8, 256>>>(dout, 16); CK(cudaDeviceSynchronize());
    cudaEventRecord(t0); pico_dp4a<<<sms * 8, 256>>>(dout, iters); cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    double ops = (double)sms * 8 * 256 * iters * 8 * 8;  // 8 dp4a por iteracao, 8 ops (4 mul + 4 add) cada
    printf("Pico __dp4a (so registradores): %.2f TOPS INT8\n\n", ops / (ms * 1e-3) / 1e12);

    size_t tamanhos[] = {(size_t)256 << 20, (size_t)512 << 10};
    const char* nomes[] = {"256 MB (nao cabe no L2 -> VRAM)", "512 KB (cabe no L2 de 1 MB)"};
    for (int t = 0; t < 2; ++t) {
        size_t n = tamanhos[t], n4 = n / 4;
        Ctx c; c.n4 = n4; c.grid = sms * 16;
        CK(cudaMalloc(&c.a, n)); CK(cudaMalloc(&c.b, n));
        std::vector<int8_t> h(n);
        srand(1); for (size_t i = 0; i < n; ++i) h[i] = (int8_t)(rand() % 255 - 127);
        int reps = (t == 0) ? 3 : 200;

        c.passos = 0;
        CK(cudaMemcpy(c.a, h.data(), n, cudaMemcpyHostToDevice));
        float ms_copia = cronometrar(fundido, &c, reps);
        printf("== %s | so ler+gravar uma vez: %.3f ms = %.0f GB/s efetivos\n",
               nomes[t], ms_copia, 2.0 * n / (ms_copia * 1e-3) / 1e9);
        printf("  etapas | fundido (ms) | separado (ms) | separado/fundido | trafego VRAM separado\n");
        int lista[] = {1, 2, 4, 8, 16, 32, 64, 128};
        for (int k = 0; k < 8; ++k) {
            c.passos = lista[k];
            CK(cudaMemcpy(c.a, h.data(), n, cudaMemcpyHostToDevice));
            float mf = cronometrar(fundido, &c, reps);
            CK(cudaMemcpy(c.a, h.data(), n, cudaMemcpyHostToDevice));
            float msep = cronometrar(separado, &c, reps);
            printf("  %6d | %12.3f | %13.3f | %15.1fx | %6.1f GB\n",
                   lista[k], mf, msep, msep / mf, 2.0 * n * lista[k] / 1e9);
        }

        // conferencia: fundido e separado tem que dar o mesmo resultado
        c.passos = 16;
        CK(cudaMemcpy(c.a, h.data(), n, cudaMemcpyHostToDevice));
        fundido(&c); CK(cudaDeviceSynchronize());
        std::vector<int8_t> r1(n), r2(n);
        CK(cudaMemcpy(r1.data(), c.b, n, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(c.a, h.data(), n, cudaMemcpyHostToDevice));
        char4 *src = c.a, *dst = c.b;
        for (int s = 0; s < 16; ++s) { cadeia<<<c.grid, 256>>>(src, dst, n4, 1); char4* tmp = src; src = dst; dst = tmp; }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(r2.data(), src, n, cudaMemcpyDeviceToHost));
        size_t dif = 0; for (size_t i = 0; i < n; ++i) dif += (r1[i] != r2[i]);
        printf("  conferencia (16 etapas): %zu diferencas entre fundido e separado\n\n", dif);
        cudaFree(c.a); cudaFree(c.b);
    }
    return 0;
}
