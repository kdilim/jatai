// libbit: kernels do projeto Jataí-bit (separado do Jataí INT8). Versão 0: o treino do "observador" (pesos como
// probabilidade, média de campo inteira) de uma rede 100% binária, e o forward binário para avaliação.
//
// Representação "Q7" (especificação executável em ref_bit.py, que estes kernels têm de reproduzir bit a bit):
//   theta  : int16 em [-1024, 1024], o contador-probabilidade de cada peso; média do peso m8 = clamp(rnd(theta, 3))
//            em int8 Q7 (+-127 ~ +-1) e m2 = rnd(m8 * m8, 7) em [0, 126] (o quadrado, Q7).
//   mu     : média de cada bit de ativação, int8 Q7; mu2 = rnd(mu * mu, 7).
//   camada : M = sum mu * m8 (Q14), S2 = sum mu2 * m2 (Q14), V = K * 2^14 - S2, sv = isqrt(V) (Q7),
//            r = (M << 3) / sv (Q10), mu' = tanh(0,8 r) em Q7 (tanh por exp inteira: shift + polinômio).
//   saída  : so = (M << 3) / sv (logits normalizados, Q10); softmax inteira; d = p - 1[alvo] em Q15.
//   backward: dM = d * 128 / sv; operandos int8 por escala de tensor (shift escolhido no device pelo maior |valor|);
//             dM_camada = dmu * (2^14 - mu^2) * 0,8 / sv; G = dM^T . mu_entrada; theta -= rnd(G normalizado, shift).
// Todas as multiplicações de matriz: C (int32) = A (int8, M x K) . B (int8, N x K)^T por dp4a, K múltiplo de 64
// (quem chama completa com zeros), M e N quaisquer.
//
// Compilar: nvcc -O3 -arch=sm_75 -shared -Xcompiler -fPIC libbit.cu -o libbit.so
#include <cstdint>
#include <cuda_runtime.h>

static cudaStream_t g_stream = 0;

__device__ __forceinline__ int rnd_d(int x, int s) {          // x * 2^-s, empate para o par, s >= 0
    s = min(s, 31);
    if (s == 0) return x;
    return (x + ((1 << (s - 1)) - 1) + ((x >> s) & 1)) >> s;
}
__device__ __forceinline__ long long rnd64(long long x, int s) {
    if (s <= 0) return x << (-s);
    if (s > 62) return 0;
    return (x + ((1LL << (s - 1)) - 1) + ((x >> s) & 1)) >> s;
}
__device__ __forceinline__ int bitlen(long long x) { x = x < 0 ? -x : x; int b = 0; while (x) { b++; x >>= 1; } return b; }

__device__ __forceinline__ int isqrt_i(int n) {
    n = max(n, 1);
    int x = 1 << ((bitlen(n) + 1) / 2);
    for (int i = 0; i < 20; i++) x = min(x, (x + n / max(x, 1)) >> 1);
    return max(x, 1);
}
__device__ __forceinline__ int exp_int(int d) {               // exp(-d/256) em Q15
    int t = min(max(d, 0), 6000) * 23637;
    int n = t >> 22, f = (t & ((1 << 22) - 1)) >> 7;
    int p = -1295;
    p = ((p * f) >> 15) + 7556;
    p = ((p * f) >> 15) - 22645;
    p = ((p * f) >> 15) + 32765;
    return n > 30 ? 0 : p >> n;
}
__device__ __forceinline__ int tanh_q7(int z) {               // tanh(z / 1024) em Q7 (entrada Q10, saída Q7)
    const int e = exp_int(abs(z) >> 1);                       // exp(-2|z|/1024) = exp(-(|z|/2)/256)
    const int t = ((32768 - e) * 127 + ((32768 + e) >> 1)) / (32768 + e);
    return z >= 0 ? t : -t;
}

// ============================================================================ GEMM int8 -> int32 (dp4a)
// C [M x N] = A [M x K] . B [N x K]^T. Tile 64 x 128, 256 threads, 4 x 8 saídas por thread, dois buffers.
constexpr int BM = 64, BN = 128, CH = 64, WA = CH / 4;
__global__ void __launch_bounds__(256, 2) k_gemm(const int8_t* __restrict__ A, const int8_t* __restrict__ B, int* __restrict__ C,
                                                 int M, int N, int K) {
    __shared__ __align__(16) int As[2][WA][BM];
    __shared__ __align__(16) int Bs[2][WA][BN];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4, row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
#define COL(j) ((j) < 4 ? tx * 4 + (j) : 64 + tx * 4 + (j) - 4)
    const int la = tid >> 2, pa = tid & 3;
    const bool okA = row0 + la < M;
    const int4* gA = reinterpret_cast<const int4*>(A + (size_t)(okA ? row0 + la : 0) * K) + pa;
    const int4* gB[2];
    int lb[2], pb[2];
    bool okB[2];
    for (int v = 0; v < 2; v++) {
        const int idx = tid + 256 * v;
        lb[v] = idx >> 2; pb[v] = idx & 3;
        okB[v] = col0 + lb[v] < N;
        gB[v] = reinterpret_cast<const int4*>(B + (size_t)(okB[v] ? col0 + lb[v] : 0) * K) + pb[v];
    }
    auto carrega = [&](int c, int4& va, int4* vb) {
        va = okA ? gA[c * 4] : make_int4(0, 0, 0, 0);
        for (int v = 0; v < 2; v++) vb[v] = okB[v] ? gB[v][c * 4] : make_int4(0, 0, 0, 0);
    };
    auto guarda = [&](int bf, const int4& va, const int4* vb) {
        As[bf][pa * 4][la] = va.x; As[bf][pa * 4 + 1][la] = va.y; As[bf][pa * 4 + 2][la] = va.z; As[bf][pa * 4 + 3][la] = va.w;
        for (int v = 0; v < 2; v++) {
            Bs[bf][pb[v] * 4][lb[v]] = vb[v].x; Bs[bf][pb[v] * 4 + 1][lb[v]] = vb[v].y;
            Bs[bf][pb[v] * 4 + 2][lb[v]] = vb[v].z; Bs[bf][pb[v] * 4 + 3][lb[v]] = vb[v].w;
        }
    };
    int acc[4][8];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0;
    const int net = K / CH;
    {
        int4 va, vb[2];
        carrega(0, va, vb); guarda(0, va, vb);
        __syncthreads();
    }
    for (int c = 0; c < net; c++) {
        const int buf = c & 1;
        const bool prox = c + 1 < net;
        int4 va, vb[2];
        if (prox) carrega(c + 1, va, vb);
#pragma unroll
        for (int w = 0; w < WA; w++) {
            const int4 a4 = *reinterpret_cast<const int4*>(&As[buf][w][ty * 4]);
            const int4 b0 = *reinterpret_cast<const int4*>(&Bs[buf][w][tx * 4]), b1 = *reinterpret_cast<const int4*>(&Bs[buf][w][64 + tx * 4]);
            const int a[4] = {a4.x, a4.y, a4.z, a4.w};
            const int b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
        }
        if (prox) guarda(buf ^ 1, va, vb);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = row0 + ty * 4 + i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int cc = col0 + COL(j);
            if (cc < N) C[(size_t)r * N + cc] = acc[i][j];
        }
    }
#undef COL
}

// ============================================================================ variância aproximada / sem variância
// soma de cada linha de uma matriz int8 (para a variância aproximada: S2_ij ~ (sum_k mu2_ik)(sum_k m2_jk) / K)
__global__ void k_soma_linhas8(const int8_t* __restrict__ X, int* __restrict__ s, int R, int K) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= R) return;
    int t = 0;
    for (int k = lane; k < K; k += 32) t += X[(size_t)r * K + k];
    for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (lane == 0) s[r] = t;
}
// r = (M << 3) / sv com V = K * 2^14 - smu2_i * sm2_j / K  (sem a 2a multiplicação de matriz); guarda sv
__global__ void k_epilogo_aprox(const int* M, const int* smu2, const int* sm2, int K, int N, int* r_o, int* sv_o, long long n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const long long s2 = (long long)smu2[i / N] * sm2[i % N] / K;
    const int V = (int)max((long long)K * 16384 - s2, 1LL), sv = isqrt_i(V);
    sv_o[i] = sv;
    r_o[i] = (M[i] << 3) / sv;
}

// ============================================================================ normalização entre camadas (só no treino)
// Antes da tanh, a razão r de cada amostra é dividida pela RMS da linha e multiplicada por T (Q10):
//   r' = r * T / rms(r).  Multiplicar por um positivo não muda o sinal: a rede binária observada é a mesma. Serve para
// as médias não morrerem (-> 0) nem saturarem (-> +-127) com a profundidade (diagnóstico em diag_medias.py).
__device__ __forceinline__ long long isqrt64(long long n) {
    if (n < 1) return 1;
    long long x = 1LL << ((bitlen(n) + 1) / 2);
    for (int i = 0; i < 40; i++) { const long long y = (x + n / x) >> 1; if (y >= x) break; x = y; }
    return x < 1 ? 1 : x;
}
__global__ void k_norm_tanh(const int* __restrict__ r, int N, int T, int8_t* mu, int8_t* mu2, int* rms_o, int quatro_bits) {
    __shared__ long long red[256];
    const int b = blockIdx.x;
    const int* rl = r + (size_t)b * N;
    long long t = 0;
    for (int i = threadIdx.x; i < N; i += 256) t += (long long)rl[i] * rl[i];
    red[threadIdx.x] = t;
    __syncthreads();
    for (int o = 128; o; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    const long long rms = isqrt64(red[0] / N);
    if (threadIdx.x == 0) rms_o[b] = (int)rms;
    for (int i = threadIdx.x; i < N; i += 256) {
        const int rn = (int)(((long long)rl[i] * T) / rms);
        int m = tanh_q7((rn * 205) >> 8);
        if (quatro_bits) m = ((m * 7 + (m >= 0 ? 63 : -63)) / 127) * 18;
        mu[(size_t)b * N + i] = (int8_t)m;
        mu2[(size_t)b * N + i] = (int8_t)rnd_d(m * m, 7);
    }
}
// backward da camada normalizada: dM = dmu * (1 - mu^2) * 0,8 * (T / rms) / sv   (int64; duas divisões que truncam)
__global__ void k_dtanh_norm(const int* dmu, const int8_t* mu, const int* sv, const int* rms, int T, int N, int* dm, long long n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const long long m = mu[i];
    const long long t = rnd64(rnd64((long long)dmu[i] * (16384 - m * m), 14) * 205, 8);
    dm[i] = (int)(((t * 128 * T) / rms[i / N]) / sv[i]);
}

// versões com recíproco por linha (sem divisão int64 por elemento): inv = 2^k / rms, k = bitlen(rms) + 20
//   forward : r' = (r * T * inv) >> k
//   backward: dM = (((t * inv) >> 10) * 128 * T) >> (k - 10)        [sem variância: sv = 1]
// guarda inv (int64) e k por linha para o backward.
__global__ void k_norm_tanh_r(const int* __restrict__ r, int N, int T, int8_t* mu, int8_t* mu2, long long* inv_o, int* k_o, int quatro_bits) {
    __shared__ long long red[256];
    const int b = blockIdx.x;
    const int* rl = r + (size_t)b * N;
    long long t = 0;
    for (int i = threadIdx.x; i < N; i += 256) t += (long long)rl[i] * rl[i];
    red[threadIdx.x] = t;
    __syncthreads();
    for (int o = 128; o; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    const long long rms = isqrt64(red[0] / N);
    const int k = bitlen(rms) + 20;
    const long long inv = (1LL << k) / rms;
    if (threadIdx.x == 0) { inv_o[b] = inv; k_o[b] = k; }
    for (int i = threadIdx.x; i < N; i += 256) {
        const int rn = (int)(((long long)rl[i] * T * inv) >> k);
        int m = tanh_q7((rn * 205) >> 8);
        if (quatro_bits) m = ((m * 7 + (m >= 0 ? 63 : -63)) / 127) * 18;
        mu[(size_t)b * N + i] = (int8_t)m;
        mu2[(size_t)b * N + i] = (int8_t)rnd_d(m * m, 7);
    }
}
__global__ void k_dtanh_norm_r(const int* dmu, const int8_t* mu, const long long* inv, const int* kk, int T, int N, int* dm, long long n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const long long m = mu[i];
    const long long t = rnd64(rnd64((long long)dmu[i] * (16384 - m * m), 14) * 205, 8);
    const int b = (int)(i / N);
    dm[i] = (int)(((((t * inv[b]) >> 10) * 128 * T) >> (kk[b] - 10)));
}

// ============================================================================ GLU binário (a "tabela de combinações" resumida)
// saída = produto de dois bits (XNOR em +-1); na média de campo: mu = muA * muB (exato para bits independentes)
__global__ void k_glu_mul(const int8_t* a, const int8_t* b, int8_t* mu, int8_t* mu2, long long n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const int m = rnd_d((int)a[i] * b[i], 7);
    mu[i] = (int8_t)m;
    mu2[i] = (int8_t)rnd_d(m * m, 7);
}
// backward do produto: dA = rnd(d * muB, 7), dB = rnd(d * muA, 7)
__global__ void k_glu_back(const int* d, const int8_t* a, const int8_t* b, int* da, int* db, long long n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    da[i] = (int)rnd64((long long)d[i] * b[i], 7);
    db[i] = (int)rnd64((long long)d[i] * a[i], 7);
}

// ============================================================================ peças do modelo de texto (fase 2)
// Mistura temporal: soma com decaimento por canal sobre as médias (linear no tempo; média de campo exata):
//   s_t = s_{t-1} - rnd(s_{t-1}, d_c) + v_t,  d_c = 1 + (c & 7)  (memórias de ~2 a ~256 tokens),  s_{-1} = 0.
// Uma thread por (sequência, canal); x [B*T x D] em ordem de linha (linha = b*T + t).
__device__ __forceinline__ int decai(int c) { return 1 + (c & 7); }
__global__ void k_scan_fwd(const int8_t* __restrict__ v, int* __restrict__ S, int B, int T, int D) {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= B * D) return;
    const int b = idx / D, c = idx % D, d = decai(c);
    int s = 0;
    for (int t = 0; t < T; t++) {
        const size_t o = ((size_t)b * T + t) * D + c;
        s = s - rnd_d(s, d) + v[o];
        S[o] = s;
    }
}
// backward da soma (derivada linear 1 - 2^-d): g_t = dS_t + g_{t+1} - rnd(g_{t+1}, d), percorrido de trás para frente
__global__ void k_scan_back(const int* __restrict__ dS, int* __restrict__ dv, int B, int T, int D) {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= B * D) return;
    const int b = idx / D, c = idx % D, d = decai(c);
    long long g = 0;
    for (int t = T - 1; t >= 0; t--) {
        const size_t o = ((size_t)b * T + t) * D + c;
        g = dS[o] + g - rnd64(g, d);
        dv[o] = (int)g;
    }
}
// Residual em binário ("voto"): o bit novo é o sinal de (soma do bloco normalizada + bit anterior com o mesmo peso T):
//   u = M * T * inv >> k  +  mu_ant * T / 128 ;  mu = tanh(0,8 u)   (na rede final: bit = sinal(M/rms + bit_ant))
// inv, k: recíproco da RMS da linha de M (como k_norm_tanh_r). Uma linha por bloco.
__global__ void k_norm_res_r(const int* __restrict__ M, const int8_t* __restrict__ ant, int N, int T, int a, int8_t* mu, int8_t* mu2,
                             long long* inv_o, int* k_o) {
    __shared__ long long red[256];
    const int b = blockIdx.x;
    const int* ml = M + (size_t)b * N;
    long long t = 0;
    for (int i = threadIdx.x; i < N; i += 256) t += (long long)ml[i] * ml[i];
    red[threadIdx.x] = t;
    __syncthreads();
    for (int o = 128; o; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    const long long rms = isqrt64(red[0] / N);
    const int k = bitlen(rms) + 20;
    const long long inv = (1LL << k) / rms;
    if (threadIdx.x == 0) { inv_o[b] = inv; k_o[b] = k; }
    for (int i = threadIdx.x; i < N; i += 256) {
        const size_t o = (size_t)b * N + i;
        const int u = (int)(((long long)ml[i] * T * inv) >> (k + a)) + ((int)ant[o] * T) / 128;
        const int m = tanh_q7((u * 205) >> 8);
        mu[o] = (int8_t)m;
        mu2[o] = (int8_t)rnd_d(m * m, 7);
    }
}
// backward do voto: dt = dmu (1 - mu^2) 0,8 ; dM = ((dt * inv) >> 10) * 128 * T >> (k - 10) ; d_ant = dt * T / 128
__global__ void k_res_back(const int* dmu, const int8_t* mu, const long long* inv, const int* kk, int T, int a, int N, int* dM, int* dant, long long n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const long long m = mu[i];
    const long long dt = rnd64(rnd64((long long)dmu[i] * (16384 - m * m), 14) * 205, 8);
    const int b = (int)(i / N);
    dM[i] = (int)((((dt * inv[b]) >> 10) * 128 * T) >> (kk[b] - 10 + a));
    dant[i] = (int)((dt * T) / 128);
}
// soma de dois caminhos do gradiente na mesma escala: out = a * 2^-s + b, s com sinal (lido do device; s < 0 = shift à
// esquerda). s = (shifts das quantizações que o caminho b sofreu) - 17: o caminho direto do voto (d_ant = dt T / 128) sai
// em 8 sigma, o que passou por dtanh_norm + GEMM sai em 2^20 sigma / 2^(s1 + s2). Sentinela: |out| >= 2^30 -> *alerta.
__global__ void k_soma_escala(const int* a, const int* b, const int* sh, int* out, int* alerta, long long n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const int s = *sh;
    // s >= 0: o direto desce até a escala do outro; s < 0: o outro desce (nunca se multiplica, para não estourar)
    const long long v = s >= 0 ? (long long)rnd_d(a[i], min(s, 30)) + b[i] : (long long)a[i] + rnd_d(b[i], min(-s, 30));
    if (v >= (1LL << 30) || v <= -(1LL << 30)) atomicCAS(alerta, 0, (int)min(i + 1, 2147483647LL));
    out[i] = (int)max(min(v, (long long)(1 << 30) - 1), -(long long)(1 << 30) + 1);
}
// Voto residual com variância (observador exato): o bit novo é sinal(M + bit_ant * rms * 2^a) = sinal(M/rms * 2^-a + bit_ant).
// Com pesos/ativações incertos, M tem média M e variância V = K 2^14 - S2 (como na saída); a média do bit novo é
//   P(+1) - P(-1) ~ tanh(0,8 u),  u = (M + x_ant rms 2^a) / sigma,  sigma = 128 sv (Q14), sv = isqrt(V).
// Em Q10: u = (M << 3)/sv + ant * rms * 2^a / (16 sv). Enquanto o bloco é incerto (sigma grande) o voto é "macio" e o
// gradiente passa; o modelo sem variância saturava o fluxo residual já na inicialização (visto nos logs do modelo).
__global__ void k_voto_var(const int* __restrict__ M, const int* __restrict__ S2, const int8_t* __restrict__ ant, int N, int K, int a,
                           int8_t* mu, int8_t* mu2, int* sv_o, long long* rf_o) {
    __shared__ long long red[256];
    const int b = blockIdx.x;
    const int* ml = M + (size_t)b * N;
    long long t = 0;
    for (int i = threadIdx.x; i < N; i += 256) t += (long long)ml[i] * ml[i];
    red[threadIdx.x] = t;
    __syncthreads();
    for (int o = 128; o; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    const long long rf = isqrt64(red[0] / N) << a;
    if (threadIdx.x == 0) rf_o[b] = rf;
    for (int i = threadIdx.x; i < N; i += 256) {
        const size_t o = (size_t)b * N + i;
        const int sv = isqrt_i(max(K * 16384 - S2[o], 1));
        long long u = ((long long)ml[i] << 3) / sv + ((long long)ant[o] * rf) / (16LL * sv);
        u = max(min(u, 1LL << 20), -(1LL << 20));
        const int m = tanh_q7(((int)u * 205) >> 8);
        sv_o[o] = sv;
        mu[o] = (int8_t)m;
        mu2[o] = (int8_t)rnd_d(m * m, 7);
    }
}
// backward: dt = dmu (1 - mu^2) 0,8 ; dM = dt 2^10 / sv (escala sigma 2^17, como res_back) ; d_ant = dt rms 2^a / (16 sv)
__global__ void k_voto_var_back(const int* dmu, const int8_t* mu, const int* sv, const long long* rf, int N, int* dM, int* dant, long long n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const long long m = mu[i];
    const long long dt = rnd64(rnd64((long long)dmu[i] * (16384 - m * m), 14) * 205, 8);
    dM[i] = (int)((dt << 10) / sv[i]);
    dant[i] = (int)((dt * rf[i / N]) / (16LL * sv[i]));
}
// sentinela genérica para int32: maior |x| e primeiro índice com |x| >= 2^30
__global__ void k_vigia32(const int* x, int* mx, int* alerta, long long n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const int v = x[i];
    atomicMax(mx, abs(v));
    if (v >= (1 << 30) || v <= -(1 << 30)) atomicCAS(alerta, 0, (int)min(i + 1, 2147483647LL));
}

// ============================================================================ elemento a elemento
#define GRID(n) (unsigned)(((n) + 255) / 256), 256, 0, g_stream
#define IDX const long long i = (long long)blockIdx.x * 256 + threadIdx.x; if (i >= n) return;

__global__ void k_transpoe8(const int8_t* X, int8_t* Y, int R, int Cc) {        // [R x C] -> [C x R]
    __shared__ int8_t t[32][33];
    const int r0 = blockIdx.y * 32, c0 = blockIdx.x * 32, tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    for (int k = ty; k < 32; k += 8) if (r0 + k < R && c0 + tx < Cc) t[k][tx] = X[(size_t)(r0 + k) * Cc + c0 + tx];
    __syncthreads();
    for (int k = ty; k < 32; k += 8) if (c0 + k < Cc && r0 + tx < R) Y[(size_t)(c0 + k) * R + r0 + tx] = t[tx][k];
}

__global__ void k_theta_m(const int16_t* th, int8_t* m8, int8_t* m2, long long n) {
    IDX
    const int m = max(-127, min(127, rnd_d(th[i], 3)));
    m8[i] = (int8_t)m;
    m2[i] = (int8_t)rnd_d(m * m, 7);
}
// hash inteiro (lowbias32): sorteio determinístico e reproduzível para o arredondamento estocástico, sem float
__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
// "pressão de observação": com probabilidade prob/256 (sorteio por hash32(i * 0x9E3779B1 + semente)), o contador anda
// 1 nível na direção do lado que já é o mais provável (theta != 0), limitado a [-lim, lim]. A probabilidade cresce ao
// longo do treino (quem chama decide): o modelo termina "decidido", e a rede de médias se aproxima da rede observada.
__global__ void k_pressao(int8_t* th, int prob, int lim, uint32_t semente, long long n) {
    IDX
    const int t = th[i];
    if (t == 0) return;
    if ((int)(hash32((uint32_t)i * 0x9E3779B1u + semente) & 255u) < prob)
        th[i] = (int8_t)max(-lim, min(lim, t + (t > 0 ? 1 : -1)));
}
// contador de 4 bits (theta em [-7, 7], como a SBC de ../transformer_bit/Doc.md): média do peso = theta * 18 (Q7)
__global__ void k_theta_m_c4(const int8_t* th, int8_t* m8, int8_t* m2, long long n) {
    IDX
    const int m = th[i] * 18;
    m8[i] = (int8_t)m;
    m2[i] = (int8_t)rnd_d(m * m, 7);
}
// theta4 -= g * 2^-shift com arredondamento ESTOCÁSTICO (g normalizado a |g| <= 2^8): parte inteira + 1 com
// probabilidade = resto / 2^shift, sorteada por hash32(i * 0x9E3779B1 + semente). Com só 15 níveis, o arredondamento
// determinístico zeraria quase todos os passos.
__global__ void k_update_c4(int8_t* th, const int* G, const int* mx, int shift, uint32_t semente, long long n) {
    IDX
    const int s0 = max(bitlen(*mx) - 8, 0);
    const int g = rnd_d(G[i], s0);
    const int q = g >> shift, resto = g - (q << shift);
    const int u = (int)(hash32((uint32_t)i * 0x9E3779B1u + semente) & ((1u << shift) - 1));
    const int t = th[i] - (q + (u < resto ? 1 : 0));
    th[i] = (int8_t)max(-7, min(7, t));
}
// contador em int8 (regra do projeto: nada guardado maior que int8): theta8 em [-127, 127] já é a média do peso em Q7
__global__ void k_theta_m_c8(const int8_t* th, int8_t* m8, int8_t* m2, long long n) {
    IDX
    const int m = th[i];
    m8[i] = (int8_t)m;
    m2[i] = (int8_t)rnd_d(m * m, 7);
}
// theta8 -= rnd(rnd(G, s0), shift), s0 = max(bitlen(max|G|) - 8, 0): maior passo = 2^(8 - shift) de 127 (mesma
// fração da faixa que o contador int16 com s0 = bitlen - 11 e faixa 1024)
__global__ void k_update_c8(int8_t* th, const int* G, const int* mx, int shift, long long n) {
    IDX
    const int s0 = max(bitlen(*mx) - 8, 0);
    const int t = th[i] - rnd_d(rnd_d(G[i], s0), shift);
    th[i] = (int8_t)max(-127, min(127, t));
}
__global__ void k_quad8(const int8_t* mu, int8_t* mu2, long long n) {
    IDX
    const int m = mu[i];
    mu2[i] = (int8_t)rnd_d(m * m, 7);
}
// M, S2 -> (modo 0) mu' = tanh(0,8 M/sv), mu2' ; (modo 1) so = M / sv (logits). sv sempre guardado.
__global__ void k_epilogo(const int* M, const int* S2, int K, int8_t* mu, int8_t* mu2, int* so, int* sv_o, int modo, long long n) {
    IDX
    // r = M / sv com 10 bits de fração (M em Q14, sv em Q7 -> (M << 3) / sv em Q10): com só 7 bits, médias pequenas
    // (pesos ainda incertos) viravam 0 já na 2a camada e o gradiente da saída morria (visto na 1a conferência)
    const int V = max(K * 16384 - S2[i], 1), sv = isqrt_i(V), r = (M[i] << 3) / sv;
    sv_o[i] = sv;
    if (modo == 1) { so[i] = r; return; }
    int m = tanh_q7((r * 205) >> 8);
    if (modo == 2) {                                     // média de 4 bits: nível = arredonda(m * 7 / 127) em [-7, 7]
        const int nivel = (m * 7 + (m >= 0 ? 63 : -63)) / 127;
        m = nivel * 18;
    }
    mu[i] = (int8_t)m;
    mu2[i] = (int8_t)rnd_d(m * m, 7);
}
// softmax + entropia cruzada por linha (1 thread por linha; C pequeno): d = p - 1[alvo] (Q15); conta erros.
// ld = largura da linha (C completado até múltiplo de 64 para as multiplicações; as colunas extras recebem d = 0)
__global__ void k_softmax_ce(const int* so, const long long* y, int* d, int* erros, int Bn, int Cc, int ld) {
    const int b = blockIdx.x * 256 + threadIdx.x;
    if (b >= Bn) return;
    const int* s = so + (size_t)b * ld;
    for (int c = Cc; c < ld; c++) d[(size_t)b * ld + c] = 0;
    int mx = s[0], am = 0;
    for (int c = 1; c < Cc; c++) if (s[c] > mx) { mx = s[c]; am = c; }
    long long tot = 0;
    for (int c = 0; c < Cc; c++) tot += exp_int((mx - s[c]) >> 2);      // Q10 -> unidades de 2^-8: >> 2
    for (int c = 0; c < Cc; c++) {
        const long long p = ((long long)exp_int((mx - s[c]) >> 2) << 15) / tot;
        d[(size_t)b * ld + c] = (int)p - (c == (int)y[b] ? 32768 : 0);
    }
    if (am != (int)y[b]) atomicAdd(erros, 1);
}
// dM da saída: d * 128 / sv
__global__ void k_dmo(const int* d, const int* sv, int* dm, long long n) {
    IDX
    dm[i] = (int)(((long long)d[i] << 7) / sv[i]);
}
// dM de uma camada escondida: dmu * (2^14 - mu^2) * 0,8 / sv   (int64 por dentro)
__global__ void k_dtanh(const int* dmu, const int8_t* mu, const int* sv, int* dm, long long n) {
    IDX
    const long long m = mu[i];
    const long long t = rnd64((long long)dmu[i] * (16384 - m * m), 14);
    dm[i] = (int)(rnd64(t * 205, 8) * 128 / sv[i]);
}
__global__ void k_maxabs(const int* x, int* mx, long long n) {
    IDX
    atomicMax(mx, abs(x[i]));
}
// int32 -> int8 por escala de tensor: shift = max(bitlen(max|x|) - 7, 0) (lido do device)
__global__ void k_quant8(const int* x, const int* mx, int8_t* q, long long n) {
    IDX
    const int s = max(bitlen(*mx) - 7, 0);
    q[i] = (int8_t)max(-127, min(127, rnd_d(x[i], s)));
}
// theta -= rnd(rnd(G, s0), shift), s0 = max(bitlen(max|G|) - 11, 0); theta em [-1024, 1024]
__global__ void k_update(int16_t* th, const int* G, const int* mx, int shift, long long n) {
    IDX
    const int s0 = max(bitlen(*mx) - 11, 0);
    const int t = th[i] - rnd_d(rnd_d(G[i], s0), shift);
    th[i] = (int16_t)max(-1024, min(1024, t));
}
__global__ void k_sinal8(const int* x, int8_t* y, long long n) {          // ativação binária: sinal (0 -> +1)
    IDX
    y[i] = x[i] >= 0 ? 1 : -1;
}
__global__ void k_sinal_theta(const int16_t* th, int8_t* w, long long n) {  // "observação": peso = sinal(theta)
    IDX
    w[i] = th[i] >= 0 ? 1 : -1;
}
__global__ void k_sinal_theta_c8(const int8_t* th, int8_t* w, long long n) {
    IDX
    w[i] = th[i] >= 0 ? 1 : -1;
}

// Arredondamento estocástico (sem viés) para os gradientes: o diagnóstico do modelo de texto mostrou 92-100% dos
// gradientes zerados pelo arredondamento determinístico (valores < 1/2 do degrau somem sempre). Aqui o valor sobe um
// degrau com probabilidade resto / 2^s, sorteada por hash32(i * 0x9E3779B1 + semente): a média é exata.
__global__ void k_quant8s(const int* x, const int* mx, int8_t* q, uint32_t semente, long long n) {
    IDX
    const int s = max(bitlen(*mx) - 7, 0);
    int v = x[i];
    if (s > 0) {
        const int base = v >> s, resto = v - (base << s);
        const int u = (int)(hash32((uint32_t)i * 0x9E3779B1u + semente) & ((1u << s) - 1));
        v = base + (u < resto ? 1 : 0);
    }
    q[i] = (int8_t)max(-127, min(127, v));
}
// update do contador int8 com arredondamento estocástico do passo (o determinístico só movia os pesos com |G| > 1/8 do maior)
__global__ void k_update_c8s(int8_t* th, const int* G, const int* mx, int shift, uint32_t semente, long long n) {
    IDX
    const int s0 = max(bitlen(*mx) - 8, 0);
    const int g = rnd_d(G[i], s0);
    const int q = g >> shift, resto = g - (q << shift);
    const int u = (int)(hash32((uint32_t)i * 0x9E3779B1u + semente) & ((1u << shift) - 1));
    const int t = th[i] - (q + (u < resto ? 1 : 0));
    th[i] = (int8_t)max(-127, min(127, t));
}
// renormaliza um tensor int32 para o maior |x| ter 'alvo' bits (desce com arredondamento par, sobe com shift)
__global__ void k_renorm(const int* x, const int* mx, int* out, int alvo, long long n) {
    IDX
    const int d = bitlen(*mx) - alvo;
    out[i] = d >= 0 ? rnd_d(x[i], min(d, 30)) : (x[i] << min(-d, 30));
}
// ============================================================================ pesos e ativações de k bits (planos de bits)
// Um valor de k bits é o índice j em [0, 2^k) dos níveis igualmente espaçados de [-127, 127] (como niveis() no Python).
// Código ímpar o = 2j - (2^k - 1) = soma_p 2^p s_p, com s_p = +-1 o bit p de j: o número é LIDO de k planos de bits.
// Produto de dois vetores de códigos: soma_pq 2^(p+q) (K - 2 popc(A_p xor B_q)): só XOR e popcount, sem multiplicação.
__global__ void k_empacota(const int8_t* __restrict__ x, uint32_t* __restrict__ out, int R, int K, int kb) {
    const long long n = (long long)R * (K >> 5);
    IDX
    const int W = K >> 5, r = (int)(i / W), w = (int)(i % W), nv = (1 << kb) - 1;
    uint32_t pl[4] = {0, 0, 0, 0};
    for (int b = 0; b < 32; b++) {
        const int j = (((int)x[(size_t)r * K + w * 32 + b] + 127) * nv + 127) / 254;
        for (int p = 0; p < kb; p++) pl[p] |= (uint32_t)((j >> p) & 1) << b;
    }
    for (int p = 0; p < kb; p++) out[((size_t)p * R + r) * W + w] = pl[p];
}
// C [M x N] = códigos(A) . códigos(B)^T; A: [PA][M][W] uint32, B: [PB][N][W]; K = 32 W. Tile 64 x 64, 4 x 4 por thread.
constexpr int PT = 64, PW = 8;
template <int PA, int PB>
__global__ void __launch_bounds__(256) k_gemm_planos(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B, int* __restrict__ C,
                                                     int M, int N, int W) {
    __shared__ uint32_t As[PA][PW][PT], Bs[PB][PW][PT];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4, row0 = blockIdx.y * PT, col0 = blockIdx.x * PT;
    int acc[4][4] = {};
    for (int w0 = 0; w0 < W; w0 += PW) {
        for (int e = tid; e < PT * PW; e += 256) {            // linha e / PW, palavra e % PW (leitura contígua por linha)
            const int l = e / PW, c = e % PW, w = w0 + c;
            for (int p = 0; p < PA; p++) As[p][c][l] = (row0 + l < M && w < W) ? A[((size_t)p * M + row0 + l) * W + w] : 0u;
            for (int q = 0; q < PB; q++) Bs[q][c][l] = (col0 + l < N && w < W) ? B[((size_t)q * N + col0 + l) * W + w] : 0u;
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < PW; c++) {
            uint32_t a[PA][4], b[PB][4];
#pragma unroll
            for (int p = 0; p < PA; p++)
#pragma unroll
                for (int i = 0; i < 4; i++) a[p][i] = As[p][c][ty * 4 + i];
#pragma unroll
            for (int q = 0; q < PB; q++)
#pragma unroll
                for (int j = 0; j < 4; j++) b[q][j] = Bs[q][c][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
#pragma unroll
                    for (int p = 0; p < PA; p++)
#pragma unroll
                        for (int q = 0; q < PB; q++) acc[i][j] += __popc(a[p][i] ^ b[q][j]) << (p + q);
        }
        __syncthreads();
    }
    const int base = ((1 << PA) - 1) * ((1 << PB) - 1) * (W * 32);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            const int r = row0 + ty * 4 + i, c = col0 + tx * 4 + j;
            if (r < M && c < N) C[(size_t)r * N + c] = base - 2 * acc[i][j];
        }
}
// ============================================================================ pesos fatorados (tabela linha x coluna)
// W[o][i] = rnd(soma_r A[o][r] B[i][r], s), sem completar o posto com zeros
__global__ void k_compor(const int8_t* __restrict__ A, const int8_t* __restrict__ B, int8_t* __restrict__ Wt, int O, int I, int R, int s) {
    const long long n = (long long)O * I;
    IDX
    const int o = (int)(i / I), c = (int)(i % I);
    int acc = 0;
    for (int r = 0; r < R; r++) acc += (int)A[(size_t)o * R + r] * B[(size_t)c * R + r];
    Wt[i] = (int8_t)max(-127, min(127, rnd_d(acc, s)));
}
// dA[o][r] = soma_i G[o][i] B[i][r]
__global__ void k_grad_a(const int8_t* __restrict__ G, const int8_t* __restrict__ B, int* __restrict__ dA, int O, int I, int R) {
    const long long n = (long long)O * R;
    IDX
    const int o = (int)(i / R), r = (int)(i % R);
    int acc = 0;
    for (int c = 0; c < I; c++) acc += (int)G[(size_t)o * I + c] * B[(size_t)c * R + r];
    dA[i] = acc;
}
// dB[c][r] = soma_o G[o][c] A[o][r]
__global__ void k_grad_b(const int8_t* __restrict__ G, const int8_t* __restrict__ A, int* __restrict__ dB, int O, int I, int R) {
    const long long n = (long long)I * R;
    IDX
    const int c = (int)(i / R), r = (int)(i % R);
    int acc = 0;
    for (int o = 0; o < O; o++) acc += (int)G[(size_t)o * I + c] * A[(size_t)o * R + r];
    dB[i] = acc;
}
// ============================================================================ derivada completa da normalização por rms
// m = tanh(c M / rms(M)): dM_j = (c/rms)(dt_j - M_j sum(dt M) / sum(M^2)). O kernel dtanh_norm_r dá o primeiro termo
// (rms constante); este tira a projeção sobre M: dM_j -= M_j sum(dM M) / sum(M^2) (somas em int128 no registrador).
__global__ void k_proj_norm(int* __restrict__ dM, const int* __restrict__ M, int D) {
    __shared__ __int128 sp[256], sq[256];
    const int row = blockIdx.x, t = threadIdx.x;
    const int* m = M + (size_t)row * D;
    int* d = dM + (size_t)row * D;
    __int128 p = 0, q = 0;
    for (int j = t; j < D; j += 256) { p += (__int128)((long long)d[j] * m[j]); q += (__int128)((long long)m[j] * m[j]); }
    sp[t] = p; sq[t] = q;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (t < o) { sp[t] += sp[t + o]; sq[t] += sq[t + o]; }
        __syncthreads();
    }
    const __int128 P = sp[0], Q = sq[0];
    if (Q == 0) return;
    for (int j = t; j < D; j += 256) {
        const __int128 v = (__int128)d[j] - ((__int128)m[j] * P) / Q;      // divisão truncada para zero
        d[j] = (int)(v > 2147483647 ? (__int128)2147483647 : (v < -2147483647 ? (__int128)-2147483647 : v));
    }
}
// ============================================================================ derivada do rms no voto residual
// u_o = (M_o << 3) / sv_o + ant_o rf / (16 sv_o), rf = rms(M) << a: o peso do caminho residual depende de M pelo rms.
// dM_j += 8 2^(2a) M_j Z / (D rf), Z = soma_o dt_o ant_o / sv_o (Z em Q16, somas em int128 no registrador).
__global__ void k_voto_rms_back(int* __restrict__ dM, const int* __restrict__ dmu, const int8_t* __restrict__ mu, const int8_t* __restrict__ ant,
                                const int* __restrict__ sv, const long long* __restrict__ rf, const int* __restrict__ M, int D, int a) {
    __shared__ __int128 sz[256];
    const int row = blockIdx.x, t = threadIdx.x;
    const size_t base = (size_t)row * D;
    __int128 z = 0;
    for (int j = t; j < D; j += 256) {
        const long long m = mu[base + j];
        const long long dt = rnd64(rnd64((long long)dmu[base + j] * (16384 - m * m), 14) * 205, 8);
        z += (__int128)((dt * ant[base + j]) * 65536LL / sv[base + j]);
    }
    sz[t] = z;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (t < o) sz[t] += sz[t + o];
        __syncthreads();
    }
    const __int128 Z = sz[0], den = (__int128)D * rf[row] * 65536;
    if (den == 0) return;
    for (int j = t; j < D; j += 256) {
        const __int128 v = (__int128)dM[base + j] + ((__int128)M[base + j] * Z * ((__int128)8 << (2 * a))) / den;
        dM[base + j] = (int)(v > 2147483647 ? (__int128)2147483647 : (v < -2147483647 ? (__int128)-2147483647 : v));
    }
}
extern "C" {
void bt_stream(void* s) { g_stream = (cudaStream_t)s; }
int bt_gemm(const void* A, const void* B, void* C, int M, int N, int K) {
    if (K % CH) return (int)cudaErrorInvalidValue;
    k_gemm<<<dim3((N + BN - 1) / BN, (M + BM - 1) / BM), 256, 0, g_stream>>>((const int8_t*)A, (const int8_t*)B, (int*)C, M, N, K);
    return (int)cudaGetLastError();
}
int bt_transpoe8(const void* X, void* Y, int R, int C) {
    k_transpoe8<<<dim3((C + 31) / 32, (R + 31) / 32), 256, 0, g_stream>>>((const int8_t*)X, (int8_t*)Y, R, C);
    return (int)cudaGetLastError();
}
int bt_theta_m(const void* th, void* m8, void* m2, long long n) { k_theta_m<<<GRID(n)>>>((const int16_t*)th, (int8_t*)m8, (int8_t*)m2, n); return (int)cudaGetLastError(); }
int bt_theta_m_c8(const void* th, void* m8, void* m2, long long n) { k_theta_m_c8<<<GRID(n)>>>((const int8_t*)th, (int8_t*)m8, (int8_t*)m2, n); return (int)cudaGetLastError(); }
int bt_update_c8(void* th, const void* G, const void* mx, int shift, long long n) {
    k_update_c8<<<GRID(n)>>>((int8_t*)th, (const int*)G, (const int*)mx, shift, n); return (int)cudaGetLastError();
}
int bt_theta_m_c4(const void* th, void* m8, void* m2, long long n) { k_theta_m_c4<<<GRID(n)>>>((const int8_t*)th, (int8_t*)m8, (int8_t*)m2, n); return (int)cudaGetLastError(); }
int bt_update_c4(void* th, const void* G, const void* mx, int shift, unsigned semente, long long n) {
    k_update_c4<<<GRID(n)>>>((int8_t*)th, (const int*)G, (const int*)mx, shift, semente, n); return (int)cudaGetLastError();
}
int bt_norm_tanh(const void* r, int B, int N, int T, void* mu, void* mu2, void* rms, int quatro_bits) {
    k_norm_tanh<<<B, 256, 0, g_stream>>>((const int*)r, N, T, (int8_t*)mu, (int8_t*)mu2, (int*)rms, quatro_bits);
    return (int)cudaGetLastError();
}
int bt_dtanh_norm(const void* dmu, const void* mu, const void* sv, const void* rms, int T, int N, void* dm, long long n) {
    k_dtanh_norm<<<GRID(n)>>>((const int*)dmu, (const int8_t*)mu, (const int*)sv, (const int*)rms, T, N, (int*)dm, n);
    return (int)cudaGetLastError();
}
int bt_pressao(void* th, int prob, int lim, unsigned semente, long long n) {
    k_pressao<<<GRID(n)>>>((int8_t*)th, prob, lim, semente, n); return (int)cudaGetLastError();
}
int bt_soma_linhas8(const void* X, void* s, int R, int K) {
    k_soma_linhas8<<<(R + 7) / 8, 256, 0, g_stream>>>((const int8_t*)X, (int*)s, R, K); return (int)cudaGetLastError();
}
int bt_epilogo_aprox(const void* M, const void* smu2, const void* sm2, int K, int N, void* r, void* sv, long long n) {
    k_epilogo_aprox<<<GRID(n)>>>((const int*)M, (const int*)smu2, (const int*)sm2, K, N, (int*)r, (int*)sv, n); return (int)cudaGetLastError();
}
int bt_norm_tanh_r(const void* r, int B, int N, int T, void* mu, void* mu2, void* inv, void* k, int quatro_bits) {
    k_norm_tanh_r<<<B, 256, 0, g_stream>>>((const int*)r, N, T, (int8_t*)mu, (int8_t*)mu2, (long long*)inv, (int*)k, quatro_bits);
    return (int)cudaGetLastError();
}
int bt_dtanh_norm_r(const void* dmu, const void* mu, const void* inv, const void* k, int T, int N, void* dm, long long n) {
    k_dtanh_norm_r<<<GRID(n)>>>((const int*)dmu, (const int8_t*)mu, (const long long*)inv, (const int*)k, T, N, (int*)dm, n);
    return (int)cudaGetLastError();
}
int bt_glu_mul(const void* a, const void* b, void* mu, void* mu2, long long n) {
    k_glu_mul<<<GRID(n)>>>((const int8_t*)a, (const int8_t*)b, (int8_t*)mu, (int8_t*)mu2, n); return (int)cudaGetLastError();
}
int bt_glu_back(const void* d, const void* a, const void* b, void* da, void* db, long long n) {
    k_glu_back<<<GRID(n)>>>((const int*)d, (const int8_t*)a, (const int8_t*)b, (int*)da, (int*)db, n); return (int)cudaGetLastError();
}
int bt_scan_fwd(const void* v, void* S, int B, int T, int D) {
    k_scan_fwd<<<(B * D + 255) / 256, 256, 0, g_stream>>>((const int8_t*)v, (int*)S, B, T, D); return (int)cudaGetLastError();
}
int bt_scan_back(const void* dS, void* dv, int B, int T, int D) {
    k_scan_back<<<(B * D + 255) / 256, 256, 0, g_stream>>>((const int*)dS, (int*)dv, B, T, D); return (int)cudaGetLastError();
}
int bt_norm_res_r(const void* M, const void* ant, int R, int N, int T, int a, void* mu, void* mu2, void* inv, void* k) {
    k_norm_res_r<<<R, 256, 0, g_stream>>>((const int*)M, (const int8_t*)ant, N, T, a, (int8_t*)mu, (int8_t*)mu2, (long long*)inv, (int*)k);
    return (int)cudaGetLastError();
}
int bt_res_back(const void* dmu, const void* mu, const void* inv, const void* k, int T, int a, int N, void* dM, void* dant, long long n) {
    k_res_back<<<GRID(n)>>>((const int*)dmu, (const int8_t*)mu, (const long long*)inv, (const int*)k, T, a, N, (int*)dM, (int*)dant, n);
    return (int)cudaGetLastError();
}
int bt_voto_var(const void* M, const void* S2, const void* ant, int R, int N, int K, int a, void* mu, void* mu2, void* sv, void* rf) {
    k_voto_var<<<R, 256, 0, g_stream>>>((const int*)M, (const int*)S2, (const int8_t*)ant, N, K, a, (int8_t*)mu, (int8_t*)mu2, (int*)sv, (long long*)rf);
    return (int)cudaGetLastError();
}
int bt_voto_var_back(const void* dmu, const void* mu, const void* sv, const void* rf, int N, void* dM, void* dant, long long n) {
    k_voto_var_back<<<GRID(n)>>>((const int*)dmu, (const int8_t*)mu, (const int*)sv, (const long long*)rf, N, (int*)dM, (int*)dant, n);
    return (int)cudaGetLastError();
}
int bt_quant8s(const void* x, const void* mx, void* q, unsigned semente, long long n) {
    k_quant8s<<<GRID(n)>>>((const int*)x, (const int*)mx, (int8_t*)q, semente, n); return (int)cudaGetLastError();
}
int bt_update_c8s(void* th, const void* G, const void* mx, int shift, unsigned semente, long long n) {
    k_update_c8s<<<GRID(n)>>>((int8_t*)th, (const int*)G, (const int*)mx, shift, semente, n); return (int)cudaGetLastError();
}
int bt_renorm(const void* x, const void* mx, void* out, int alvo, long long n) {
    k_renorm<<<GRID(n)>>>((const int*)x, (const int*)mx, (int*)out, alvo, n); return (int)cudaGetLastError();
}
int bt_soma_escala(const void* a, const void* b, const void* mx, void* out, void* alerta, long long n) {
    k_soma_escala<<<GRID(n)>>>((const int*)a, (const int*)b, (const int*)mx, (int*)out, (int*)alerta, n); return (int)cudaGetLastError();
}
int bt_vigia32(const void* x, void* mx, void* alerta, long long n) {
    cudaMemsetAsync(mx, 0, 4, g_stream);
    k_vigia32<<<GRID(n)>>>((const int*)x, (int*)mx, (int*)alerta, n); return (int)cudaGetLastError();
}
int bt_quad8(const void* mu, void* mu2, long long n) { k_quad8<<<GRID(n)>>>((const int8_t*)mu, (int8_t*)mu2, n); return (int)cudaGetLastError(); }
int bt_epilogo(const void* M, const void* S2, int K, void* mu, void* mu2, void* so, void* sv, int modo, long long n) {
    k_epilogo<<<GRID(n)>>>((const int*)M, (const int*)S2, K, (int8_t*)mu, (int8_t*)mu2, (int*)so, (int*)sv, modo, n);
    return (int)cudaGetLastError();
}
int bt_softmax_ce(const void* so, const void* y, void* d, void* erros, int B, int C, int ld) {
    k_softmax_ce<<<(B + 255) / 256, 256, 0, g_stream>>>((const int*)so, (const long long*)y, (int*)d, (int*)erros, B, C, ld);
    return (int)cudaGetLastError();
}
int bt_dmo(const void* d, const void* sv, void* dm, long long n) { k_dmo<<<GRID(n)>>>((const int*)d, (const int*)sv, (int*)dm, n); return (int)cudaGetLastError(); }
int bt_dtanh(const void* dmu, const void* mu, const void* sv, void* dm, long long n) {
    k_dtanh<<<GRID(n)>>>((const int*)dmu, (const int8_t*)mu, (const int*)sv, (int*)dm, n); return (int)cudaGetLastError();
}
int bt_maxabs(const void* x, void* mx, long long n) {
    cudaMemsetAsync(mx, 0, 4, g_stream);
    k_maxabs<<<GRID(n)>>>((const int*)x, (int*)mx, n); return (int)cudaGetLastError();
}
int bt_quant8(const void* x, const void* mx, void* q, long long n) { k_quant8<<<GRID(n)>>>((const int*)x, (const int*)mx, (int8_t*)q, n); return (int)cudaGetLastError(); }
int bt_update(void* th, const void* G, const void* mx, int shift, long long n) {
    k_update<<<GRID(n)>>>((int16_t*)th, (const int*)G, (const int*)mx, shift, n); return (int)cudaGetLastError();
}
int bt_sinal8(const void* x, void* y, long long n) { k_sinal8<<<GRID(n)>>>((const int*)x, (int8_t*)y, n); return (int)cudaGetLastError(); }
int bt_sinal_theta_c8(const void* th, void* w, long long n) { k_sinal_theta_c8<<<GRID(n)>>>((const int8_t*)th, (int8_t*)w, n); return (int)cudaGetLastError(); }
int bt_sinal_theta(const void* th, void* w, long long n) { k_sinal_theta<<<GRID(n)>>>((const int16_t*)th, (int8_t*)w, n); return (int)cudaGetLastError(); }
int bt_empacota(const void* x, void* out, int R, int K, int kb) {
    const long long n = (long long)R * (K >> 5);
    k_empacota<<<GRID(n)>>>((const int8_t*)x, (uint32_t*)out, R, K, kb); return (int)cudaGetLastError();
}
int bt_gemm_planos(const void* A, const void* B, void* C, int M, int N, int K, int pa, int pb) {
    const dim3 g((N + PT - 1) / PT, (M + PT - 1) / PT);
    const uint32_t *a = (const uint32_t*)A, *b = (const uint32_t*)B;
    int* c = (int*)C;
    const int W = K >> 5;
#define PL(x, y) if (pa == x && pb == y) { k_gemm_planos<x, y><<<g, 256, 0, g_stream>>>(a, b, c, M, N, W); return (int)cudaGetLastError(); }
    PL(1, 1) PL(1, 2) PL(2, 1) PL(2, 2) PL(2, 3) PL(3, 2) PL(3, 3) PL(1, 3) PL(3, 1) PL(4, 4) PL(2, 4) PL(4, 2) PL(3, 4) PL(4, 3)
#undef PL
    return -1;
}
int bt_compor(const void* A, const void* B, void* W, int O, int I, int R, int s) {
    const long long n = (long long)O * I;
    k_compor<<<GRID(n)>>>((const int8_t*)A, (const int8_t*)B, (int8_t*)W, O, I, R, s); return (int)cudaGetLastError();
}
int bt_grad_fatores(const void* G, const void* A, const void* B, void* dA, void* dB, int O, int I, int R) {
    long long n = (long long)O * R;
    k_grad_a<<<GRID(n)>>>((const int8_t*)G, (const int8_t*)B, (int*)dA, O, I, R);
    n = (long long)I * R;
    k_grad_b<<<GRID(n)>>>((const int8_t*)G, (const int8_t*)A, (int*)dB, O, I, R); return (int)cudaGetLastError();
}
int bt_proj_norm(void* dM, const void* M, int N, int D) {
    k_proj_norm<<<N, 256, 0, g_stream>>>((int*)dM, (const int*)M, D); return (int)cudaGetLastError();
}
int bt_voto_rms_back(void* dM, const void* dmu, const void* mu, const void* ant, const void* sv, const void* rf, const void* M, int R, int D, int a) {
    k_voto_rms_back<<<R, 256, 0, g_stream>>>((int*)dM, (const int*)dmu, (const int8_t*)mu, (const int8_t*)ant, (const int*)sv,
                                            (const long long*)rf, (const int*)M, D, a);
    return (int)cudaGetLastError();
}
}
