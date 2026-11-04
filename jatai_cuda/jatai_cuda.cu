// Jatai: kernels do treino 100% inteiro (mantissa INT8 + expoente INT8 por bloco de 128 valores).
//
// Formato guardado na memoria:
//   tensor em bloco  : m int8 [R x C], E int8 [R x C/128]; valor = m * 2^E. Bloco todo zero tem E = -127.
//   tensor "cru"     : acc int32 [R x C], e int16 [R x C] (expoente por elemento); so sai da GEMM de
//                      gradiente de peso e dos logits, e entra direto no otimizador / softmax.
// Registradores: int32. Todo valor intermediario e mantido abaixo de 2^30 por construcao, e cada kernel
// confere isso (sentinela): se passar, grava o erro no registro do device com o id do lancamento (kid),
// o indice do elemento e o valor. Arredondamento sempre empate-para-par.
//
// Compilar: nvcc -O3 -arch=sm_75 -shared -Xcompiler -fPIC jatai_cuda.cu -o libjatai.so
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h>

#define BLK 128
#define E_ZERO (-127)
#define LIM (1 << 30)
#define SEM_TOPO (-100000)
#define U_MIN (-2147483647)   // 'menos infinito' para maximos de scores/logits

enum { E_ACUM = 1, E_EXP = 2, E_SAT = 3, E_DIV = 4, E_SHL = 5, E_ARG = 6 };

struct Erro {
    int n, codigo, kid, pad;
    long long idx, valor, aux;
};
__device__ Erro g_erro;

// fora de linha: o caminho de erro e frio e nao pode inchar os lacos quentes (cache de instrucoes)
__device__ __noinline__ void reporta(int codigo, int kid, long long idx, long long valor, long long aux) {
    if (atomicAdd(&g_erro.n, 1) == 0) {
        g_erro.codigo = codigo; g_erro.kid = kid; g_erro.idx = idx; g_erro.valor = valor; g_erro.aux = aux;
        __threadfence();
    }
}

// ============================================================================ primitivas
__device__ __forceinline__ int bitlen(int x) { return x == 0 ? 0 : 32 - __clz(x < 0 ? -x : x); }

// x * 2^-s, empate para o par. s <= 0: shift a esquerda (sentinela se o resultado passar de 2^30).
__device__ __forceinline__ int rnd(int x, int s, int kid, long long idx) {
    if (s <= 0) {
        if (s == 0 || x == 0) return x;
        s = -s;
        if (s > 30 || bitlen(x) + s > 30) { reporta(E_SHL, kid, idx, x, -s); return 0; }
        return x << s;
    }
    if (s > 30) return 0;                       // |x| < 2^30  ->  |x| / 2^31 < 1/2
    return (x + ((1 << (s - 1)) - 1) + ((x >> s) & 1)) >> s;
}

// x * 2^-s para s >= 0 conhecido (sem o ramo do shift a esquerda). |x| < 2^30: com s >= 31 a formula ja da 0
// (x + 2^30 - 1 + bit fica em [0, 2^31) para x > -2^30), entao basta limitar s em 31, sem um 2o ramo.
__device__ __forceinline__ int rnd_d(int x, int s) {
    s = min(s, 31);
    if (s == 0) return x;
    return (x + ((1 << (s - 1)) - 1) + ((x >> s) & 1)) >> s;
}

__device__ __forceinline__ int warp_max(int v) {
    for (int o = 16; o; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ int warp_sum(int v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// exp(-d/256) em Q15 (d >= 0): shift + polinomio de grau 3, sem tabela
__device__ __forceinline__ int exp_int(int d) {
    int t = min(d, 6000) * 23637;
    int n = t >> 22, f = (t & ((1 << 22) - 1)) >> 7;
    int p = -1295;
    p = ((p * f) >> 15) + 7556;
    p = ((p * f) >> 15) - 22645;
    p = ((p * f) >> 15) + 32765;
    return n > 30 ? 0 : p >> n;
}

// exp(-(mx - u)/256), com a diferenca limitada antes de subtrair (mx - u poderia passar de 2^31)
__device__ __forceinline__ int exp_dif(int mx, int u) { return exp_int(u <= mx - 6000 ? 6000 : mx - u); }

__device__ __forceinline__ int isqrt_i(int n) {
    n = max(n, 0);
    int x = 1 << ((bitlen(n) + 1) / 2);
    for (int i = 0; i < 10; i++) x = min(x, (x + n / max(x, 1)) >> 1);
    return max(x, 1);
}

// Quantiza um bloco de 128 valores (4 por lane) com expoente por elemento -> mantissa INT8 + 1 expoente.
__device__ void quant_warp(const int v[4], const int e[4], int8_t m[4], int& Eo, int kid, long long idx0) {
    const int lane = threadIdx.x & 31;
    int topo = SEM_TOPO;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        if (v[i] >= LIM || v[i] <= -LIM) reporta(E_ACUM, kid, idx0 + lane * 4 + i, v[i], e[i]);
        if (v[i] != 0) topo = max(topo, e[i] + bitlen(v[i]));
    }
    topo = warp_max(topo);
    if (topo == SEM_TOPO) {
#pragma unroll
        for (int i = 0; i < 4; i++) m[i] = 0;
        Eo = E_ZERO;
        return;
    }
    int E = topo - 7, q[4], est = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) { q[i] = rnd(v[i], E - e[i], kid, idx0 + lane * 4 + i); est |= q[i] > 127 || q[i] < -127; }
    if (__any_sync(0xffffffffu, est)) {
        E += 1;
#pragma unroll
        for (int i = 0; i < 4; i++) q[i] = rnd(v[i], E - e[i], kid, idx0 + lane * 4 + i);
    }
    if (E < -126) {          // underflow: bloco todo abaixo de 2^-119 vira zero (como o float faz)
#pragma unroll
        for (int i = 0; i < 4; i++) m[i] = 0;
        Eo = E_ZERO;
        return;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        if (q[i] > 127 || q[i] < -127) reporta(E_SAT, kid, idx0 + lane * 4 + i, q[i], E);
        m[i] = (int8_t)max(-127, min(127, q[i]));
    }
    if (E > 126) reporta(E_EXP, kid, idx0, E, topo);
    Eo = E;
}

__device__ __forceinline__ void le4(const int8_t* p, int v[4]) {
    char4 c = *reinterpret_cast<const char4*>(p);
    v[0] = c.x; v[1] = c.y; v[2] = c.z; v[3] = c.w;
}
__device__ __forceinline__ void grava4(int8_t* p, const int8_t m[4]) {
    *reinterpret_cast<char4*>(p) = make_char4(m[0], m[1], m[2], m[3]);
}

// x * 2^-s para qualquer sinal de s, onde o shift a esquerda comprovadamente nao perde bits (sem sentinela)
__device__ __forceinline__ int desloca(int x, int s) { return s >= 0 ? rnd_d(x, s) : x << min(-s, 31); }

// Quantiza um grupo de 128 valores = 16 lanes x 8 valores, todos com o mesmo expoente e (mesma regra de quant_warp:
// E = topo - 7, sobe 1 se algum passar de 127, bloco abaixo de 2^-119 vira zero). Devolve E (E_ZERO se zerado);
// quem chama confere E > 126. Chamada por todas as lanes do warp (usa ballot e shuffles).
// soma residual com arredondamento estocastico (jt_res_estocastico): uma atualizacao menor que meio passo da mantissa
// do residuo deixava de ser zerada sempre e passa a sobreviver na media (sem vies). Ver RELATORIO 7.7 (absorcao).
__device__ int g_res_est = 0;
__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
// x * 2^-s com arredondamento estocastico (s > 0); a semente mistura a posicao e o proprio valor (muda a cada passo
// junto com as ativacoes, e funciona dentro do CUDA Graph sem contador no host)
__device__ __forceinline__ int desloca_est(int x, int s, uint32_t h) {
    if (s <= 0) return x << min(-s, 31);
    s = min(s, 31);
    const long long base = (long long)x >> s, resto = (long long)x - (base << s);
    const long long u = (long long)(hash32(h ^ (uint32_t)x * 0x9E3779B1u) & (uint32_t)((1ull << s) - 1));
    return (int)(base + (u < resto ? 1 : 0));
}

template <bool EST = false>
__device__ __forceinline__ int quant_grupo16(const int v[8], int e, int8_t m[8], uint32_t h = 0) {
    int bl = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) bl = max(bl, 32 - __clz(abs(v[j])));
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) bl = max(bl, __shfl_xor_sync(0xffffffffu, bl, o));
    int E = e + bl - 7, q[8], est = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        q[j] = EST ? desloca_est(v[j], bl - 7, h + j) : desloca(v[j], bl - 7);
        est |= q[j] > 127 || q[j] < -127;
    }
    est = (__ballot_sync(0xffffffffu, est) >> (threadIdx.x & 16)) & 0xffff;
    if (est) {
        E += 1;
#pragma unroll
        for (int j = 0; j < 8; j++) q[j] = EST ? desloca_est(v[j], bl - 6, h + j) : desloca(v[j], bl - 6);
    }
    if (bl == 0 || E < -126) {
#pragma unroll
        for (int j = 0; j < 8; j++) q[j] = 0;
        E = E_ZERO;
    }
#pragma unroll
    for (int j = 0; j < 8; j++) m[j] = (int8_t)q[j];
    return E;
}

// ============================================================================ GEMM
// C (M x N) = A (M x K) . B (N x K)^T; A e B em bloco de 128 ao longo de K.
// Saida modo 0: em bloco (128 colunas). Modo 1: crua (acc int32 + expoente int16 por elemento).
constexpr int BN = 128, KMAX = 4096, GG = 4;

// Tile 128 x 128, 256 threads, cada thread faz 8 x 8 saidas. MODO e parametro de compilacao (com as saidas escolhidas
// em tempo de execucao o laco principal ficou mais lento):
//   0 bloco | 1 cru | 2 bloco + soma residual: C = soma(X, A.B^T) | 3 bloco + ReLU^2: C = h = A.B^T e O2 = relu2(h)
//   4 ReLU^2 backward: C = relu2_back(A.B^T, X), com X = h do forward.
// As operacoes fundidas fazem exatamente a conta dos kernels separados (k_soma, k_relu2, k_relu2_back), sobre a saida
// ja quantizada da GEMM; so poupam gravar/ler a saida intermediaria e um lancamento.
template <int MODO, int BM>
__global__ void __launch_bounds__(256, BM == 128 ? 1 : 2) k_gemm(const int8_t* __restrict__ A, const int8_t* __restrict__ EA,
                                                 const int8_t* __restrict__ B, const int8_t* __restrict__ EB,
                                                 int8_t* __restrict__ Cm, int8_t* __restrict__ CE, int* __restrict__ C32,
                                                 int16_t* __restrict__ Ce, const int8_t* __restrict__ Xm, const int8_t* __restrict__ XE,
                                                 int8_t* __restrict__ Om2, int8_t* __restrict__ OE2, int M, int N, int K, int kpart, int kid) {
    // BM = linhas por bloco (128 ou 64); RM = linhas por thread (8 ou 4); cada thread faz RM x 8 saidas.
    constexpr int RM = BM / 16, CH = 64, WT = CH / 4, NT = 256, NVA = BM * 4 / NT, NVB = BN * 4 / NT, NBMAX = KMAX / BLK;
    __shared__ __align__(16) int As[2][WT][BM];
    __shared__ __align__(16) int Bs[2][WT][BN];
    __shared__ int8_t sEA[NBMAX][BM], sEB[NBMAX][BN];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN, nb = K / BLK;
    // split-K (so saida crua): o bloco z soma os blocos de K [b_ini, b_fim) e acumula com atomicAdd. A soma inteira
    // e associativa: o resultado e o mesmo bit a bit em qualquer ordem. ra/rb continuam o maximo sobre todo o K.
    const int b_ini = blockIdx.z * kpart, b_fim = min(nb, b_ini + kpart), parte = gridDim.z > 1;
#define COL(j) ((j) < 4 ? tx * 4 + (j) : 64 + tx * 4 + (j) - 4)
    for (int i = tid; i < BM * nb; i += NT) { int r = i / nb, b = i % nb; sEA[b][r] = EA[(size_t)(row0 + r) * nb + b]; }
    for (int i = tid; i < BN * nb; i += NT) { int r = i / nb, b = i % nb; sEB[b][r] = EB[(size_t)(col0 + r) * nb + b]; }
    __syncthreads();
    int ra[RM], rb[8];
#pragma unroll
    for (int i = 0; i < RM; i++) {
        int ma = -128;
        for (int b = 0; b < nb; b++) ma = max(ma, (int)sEA[b][ty * RM + i]);
        ra[i] = ma;
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        int mb = -128;
        for (int b = 0; b < nb; b++) mb = max(mb, (int)sEB[b][COL(j)]);
        rb[j] = mb;
    }
    int acc[RM][8];
#pragma unroll
    for (int i = 0; i < RM; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0;
    const int c_fim = b_fim * (BLK / CH);
    // carga: cada linha do tile tem 64 bytes = 4 int4; cada thread le NV int4 seguidos de uma linha (4 / NV threads por linha)
    constexpr int TPA = 4 / NVA, TPB = 4 / NVB;
    const int lra = tid / TPA, lwa = (tid % TPA) * NVA * 4, lr = tid / TPB, lw = (tid % TPB) * NVB * 4;
    const int4* gA = reinterpret_cast<const int4*>(A + (size_t)(row0 + lra) * K) + (tid % TPA) * NVA;
    const int4* gB = reinterpret_cast<const int4*>(B + (size_t)(col0 + lr) * K) + (tid % TPB) * NVB;
    auto carrega = [&](int c, int4* va, int4* vb) {
#pragma unroll
        for (int v = 0; v < NVA; v++) va[v] = gA[c * (CH / 16) + v];
#pragma unroll
        for (int v = 0; v < NVB; v++) vb[v] = gB[c * (CH / 16) + v];
    };
    auto guarda = [&](int bf, const int4* va, const int4* vb) {
#pragma unroll
        for (int v = 0; v < NVA; v++) {
            As[bf][lwa + 4 * v][lra] = va[v].x; As[bf][lwa + 4 * v + 1][lra] = va[v].y; As[bf][lwa + 4 * v + 2][lra] = va[v].z; As[bf][lwa + 4 * v + 3][lra] = va[v].w;
        }
#pragma unroll
        for (int v = 0; v < NVB; v++) {
            Bs[bf][lw + 4 * v][lr] = vb[v].x; Bs[bf][lw + 4 * v + 1][lr] = vb[v].y; Bs[bf][lw + 4 * v + 2][lr] = vb[v].z; Bs[bf][lw + 4 * v + 3][lr] = vb[v].w;
        }
    };
    {
        int4 va[NVA], vb[NVB];
        carrega(b_ini * (BLK / CH), va, vb);
        guarda(0, va, vb);
        __syncthreads();
    }
    for (int b = b_ini; b < b_fim; b++) {
        int p[RM][8];
#pragma unroll
        for (int i = 0; i < RM; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) p[i][j] = 0;
        for (int t = 0; t < BLK / CH; t++) {
            const int c = b * (BLK / CH) + t, buf = (c - b_ini * (BLK / CH)) & 1;
            const bool tem_prox = c + 1 < c_fim;
            int4 va[NVA], vb[NVB];
            if (tem_prox) carrega(c + 1, va, vb);
#pragma unroll
            for (int w = 0; w < WT; w++) {
                int a[RM];
#pragma unroll
                for (int i = 0; i < RM; i += 4) {
                    const int4 a4 = *reinterpret_cast<const int4*>(&As[buf][w][ty * RM + i]);
                    a[i] = a4.x; a[i + 1] = a4.y; a[i + 2] = a4.z; a[i + 3] = a4.w;
                }
                int4 b0 = *reinterpret_cast<const int4*>(&Bs[buf][w][tx * 4]), b1 = *reinterpret_cast<const int4*>(&Bs[buf][w][64 + tx * 4]);
                int cc[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < RM; i++)
#pragma unroll
                    for (int j = 0; j < 8; j++) p[i][j] = __dp4a(a[i], cc[j], p[i][j]);
            }
            if (tem_prox) guarda(buf ^ 1, va, vb);
            __syncthreads();
        }
        int sa[RM], sb[8];
#pragma unroll
        for (int i = 0; i < RM; i++) sa[i] = ra[i] - sEA[b][ty * RM + i];
#pragma unroll
        for (int j = 0; j < 8; j++) sb[j] = rb[j] - sEB[b][COL(j)];
#pragma unroll
        for (int i = 0; i < RM; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) acc[i][j] += rnd_d(p[i][j] << GG, sa[i] + sb[j]);   // sa, sb >= 0
    }
    // Saida. Duas sentinelas sao impossiveis por construcao e so ficam onde a prova nao cobre:
    //  - acumulador: |p| <= 128 * 127^2 < 2^21, << GG = 4 -> < 2^25 por bloco de K; com nb <= 32 (K <= 4096) a soma
    //    fica < 32 * 2^25 = 2^30. So com mantissa -128 (que nenhum kernel produz) e K = 4096 chegaria a 2^30:
    //    conferido so quando nb == 32.
    //  - saturacao: sem subir o expoente |q| <= 127 por construcao; depois de subir, |q| <= 64.
    // (A saida antiga, com conferencia por elemento, custava ~40% do kernel com K = 256.)
    const bool confere_acum = nb == KMAX / BLK;
#pragma unroll
    for (int i = 0; i < RM; i++) {
        const int r = row0 + ty * RM + i;
        int base[8], topo = SEM_TOPO, mal = -1, mal_v = 0, mal_b = 0;   // indices so por selecao: sem array indexado
#pragma unroll
        for (int j = 0; j < 8; j++) {
            base[j] = ra[i] + rb[j] - GG;
            if (confere_acum && mal < 0 && (acc[i][j] >= LIM || acc[i][j] <= -LIM)) { mal = COL(j); mal_v = acc[i][j]; mal_b = base[j]; }
            if (acc[i][j] != 0) topo = max(topo, base[j] + 32 - __clz(abs(acc[i][j])));
        }
        if (mal >= 0) reporta(E_ACUM, kid, (long long)r * N + col0 + mal, mal_v, mal_b);
        if constexpr (MODO == 1) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                size_t o = (size_t)r * N + col0 + COL(j);
                if (parte) atomicAdd(C32 + o, acc[i][j]);       // a soma final e conferida por quem consome (k_momento)
                else C32[o] = acc[i][j];
                if (blockIdx.z == 0) Ce[o] = (int16_t)max(-30000, base[j]);
            }
        } else {
        // saida em bloco: 128 colunas = as 16 lanes desta linha (xor 1..8) x 8 valores por lane.
        // Shift a esquerda aqui nunca perde bits: bitlen(acc) + base <= topo, entao |acc << (base - E)| < 2^8.
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) topo = max(topo, __shfl_xor_sync(0xffffffffu, topo, o));
        int E = topo - 7, q[8], est = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int s = E - base[j];
            q[j] = s >= 0 ? rnd_d(acc[i][j], s) : acc[i][j] << min(-s, 31);
            est |= q[j] > 127 || q[j] < -127;
        }
        est = (__ballot_sync(0xffffffffu, est) >> (threadIdx.x & 16)) & 0xffff;   // alguma lane das 16 desta linha
        if (est) {
            E += 1;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int s = E - base[j];
                q[j] = s >= 0 ? rnd_d(acc[i][j], s) : acc[i][j] << min(-s, 31);
            }
        }
        int8_t m[8];
#pragma unroll
        for (int j = 0; j < 8; j++) m[j] = (int8_t)q[j];
        if (topo != SEM_TOPO && E > 126) reporta(E_EXP, kid, (long long)r * N + col0, E, topo);
        if (topo == SEM_TOPO || E < -126) {      // bloco todo zero, ou underflow -> bloco zero
#pragma unroll
            for (int j = 0; j < 8; j++) m[j] = 0;
            E = E_ZERO;
        }
        const size_t o1 = (size_t)r * N + col0 + tx * 4, o2 = o1 + 64, oe = (size_t)r * (N / BLK) + col0 / BLK;
        if constexpr (MODO == 0 || MODO == 3) {
            grava4(Cm + o1, m);
            grava4(Cm + o2, m + 4);
            if (tx == 0) CE[oe] = (int8_t)E;
        }
        if constexpr (MODO == 2) {             // soma residual (k_soma): ec = max(xE, yE) - 21; shift <= 21 em 7 bits
            int x[8], v[8];
            le4(Xm + o1, x); le4(Xm + o2, x + 4);
            const int ex = XE[oe], ec = max(ex, E) - 21;
#pragma unroll
            for (int j = 0; j < 8; j++) v[j] = desloca(x[j], ec - ex) + desloca(m[j], ec - E);
            int8_t z[8];
            const int Ez = g_res_est ? quant_grupo16<true>(v, ec, z, (uint32_t)o1 * 8u) : quant_grupo16(v, ec, z);
            if (Ez > 126 && tx == 0) reporta(E_EXP, kid, (long long)r * N + col0, Ez, -6);
            grava4(Cm + o1, z);
            grava4(Cm + o2, z + 4);
            if (tx == 0) CE[oe] = (int8_t)Ez;
        }
        if constexpr (MODO == 3) {             // ReLU^2 (k_relu2): a = max(h, 0)^2, expoente 2E
            int v[8];
#pragma unroll
            for (int j = 0; j < 8; j++) { const int hr = max((int)m[j], 0); v[j] = hr * hr; }
            int8_t a2[8];
            const int Ea = quant_grupo16(v, 2 * E, a2);
            if (Ea > 126 && tx == 0) reporta(E_EXP, kid, (long long)r * N + col0, Ea, -7);
            grava4(Om2 + o1, a2);
            grava4(Om2 + o2, a2 + 4);
            if (tx == 0) OE2[oe] = (int8_t)Ea;
        }
        if constexpr (MODO == 4) {             // ReLU^2 backward (k_relu2_back): gh = g * 2 max(h, 0), expoente gE + hE
            int hv[8], v[8];
            le4(Xm + o1, hv); le4(Xm + o2, hv + 4);
            const int eh = XE[oe];
#pragma unroll
            for (int j = 0; j < 8; j++) v[j] = (int)m[j] * max(hv[j], 0) * 2;
            int8_t g2[8];
            const int Eg = quant_grupo16(v, E + eh, g2);
            if (Eg > 126 && tx == 0) reporta(E_EXP, kid, (long long)r * N + col0, Eg, -8);
            grava4(Cm + o1, g2);
            grava4(Cm + o2, g2 + 4);
            if (tx == 0) CE[oe] = (int8_t)Eg;
        }
        }
    }
#undef COL
}

// ============================================================================ elemento a elemento (1 warp = 1 bloco)
#define WARP_BLOCO                                          \
    const int lane = threadIdx.x & 31;                      \
    const long long w = (long long)blockIdx.x * 8 + (threadIdx.x >> 5); \
    if (w >= nblocos) return;                               \
    const long long o = w * BLK + lane * 4;

// cru (acc int32, e int16 por elemento) -> bloco
__global__ void k_quant(const int* __restrict__ acc, const int16_t* __restrict__ e, int8_t* __restrict__ m, int8_t* __restrict__ E,
                        long long nblocos, int kid) {
    WARP_BLOCO
    int v[4], ee[4];
    int4 a = *reinterpret_cast<const int4*>(acc + o);
    short4 s = *reinterpret_cast<const short4*>(e + o);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; ee[0] = s.x; ee[1] = s.y; ee[2] = s.z; ee[3] = s.w;
    int8_t q[4]; int Eo;
    quant_warp(v, ee, q, Eo, kid, o - lane * 4);
    grava4(m + o, q);
    if (lane == 0) E[w] = (int8_t)Eo;
}

// soma residual z = x + y (mesmo formato)
__global__ void k_soma(const int8_t* __restrict__ xm, const int8_t* __restrict__ xE, const int8_t* __restrict__ ym,
                       const int8_t* __restrict__ yE, int8_t* __restrict__ zm, int8_t* __restrict__ zE, long long nblocos, int kid) {
    WARP_BLOCO
    int a[4], b[4], v[4], ee[4];
    le4(xm + o, a); le4(ym + o, b);
    const int ex = xE[w], ey = yE[w], ec = max(ex, ey) - 21;
#pragma unroll
    for (int i = 0; i < 4; i++) { v[i] = rnd(a[i], ec - ex, kid, o + i) + rnd(b[i], ec - ey, kid, o + i); ee[i] = ec; }
    int8_t q[4]; int Eo;
    quant_warp(v, ee, q, Eo, kid, o - lane * 4);
    grava4(zm + o, q);
    if (lane == 0) zE[w] = (int8_t)Eo;
}

// ============================================================================ esteira residual de 16 bits (E1)
// Formato R16: mantissa int16 (|m| <= 32767) + expoente int8 por bloco de 128, como o INT8 mas com 8 bits a mais de
// mantissa. So o fluxo residual (forward x e backward gx) usa R16; toda multiplicacao continua em INT8 (dp4a), lendo
// uma copia de 8 bits (k_para8). Motivo: no INT8 as atualizacoes pequenas sumiam ao somar no residuo grande
// (RELATORIO 7.7); com 15 bits de mantissa sobra 2^8 = 256x mais folga.
__device__ void quant_warp16(const int v[4], const int e[4], int16_t m[4], int& Eo, int kid, long long idx0) {
    int topo = SEM_TOPO;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        if (v[i] >= LIM || v[i] <= -LIM) reporta(E_ACUM, kid, idx0 + (threadIdx.x & 31) * 4 + i, v[i], e[i]);
        if (v[i] != 0) topo = max(topo, e[i] + bitlen(v[i]));
    }
    topo = warp_max(topo);
    if (topo == SEM_TOPO) {
#pragma unroll
        for (int i = 0; i < 4; i++) m[i] = 0;
        Eo = E_ZERO;
        return;
    }
    int E = topo - 15, q[4], est = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) { q[i] = rnd(v[i], E - e[i], kid, idx0); est |= q[i] > 32767 || q[i] < -32767; }
    if (__any_sync(0xffffffffu, est)) {
        E += 1;
#pragma unroll
        for (int i = 0; i < 4; i++) q[i] = rnd(v[i], E - e[i], kid, idx0);
    }
    if (E < -126) {
#pragma unroll
        for (int i = 0; i < 4; i++) m[i] = 0;
        Eo = E_ZERO;
        return;
    }
    if (E > 126) reporta(E_EXP, kid, idx0, E, topo);
#pragma unroll
    for (int i = 0; i < 4; i++) m[i] = (int16_t)max(-32767, min(32767, q[i]));
    Eo = E;
}
__device__ __forceinline__ void le4_16(const int16_t* p, int v[4]) {
    const short4 s = *reinterpret_cast<const short4*>(p);
    v[0] = s.x; v[1] = s.y; v[2] = s.z; v[3] = s.w;
}
__device__ __forceinline__ void grava4_16(int16_t* p, const int16_t m[4]) {
    *reinterpret_cast<short4*>(p) = make_short4(m[0], m[1], m[2], m[3]);
}
// INT8 -> R16 (exato: m << 7, E - 7)
__global__ void k_para16(const int8_t* __restrict__ xm, const int8_t* __restrict__ xE, int16_t* __restrict__ zm,
                         int8_t* __restrict__ zE, long long nblocos, int kid) {
    WARP_BLOCO
    int a[4];
    le4(xm + o, a);
    const int16_t z[4] = {(int16_t)(a[0] << 7), (int16_t)(a[1] << 7), (int16_t)(a[2] << 7), (int16_t)(a[3] << 7)};
    grava4_16(zm + o, z);
    if (lane == 0) zE[w] = xE[w] == E_ZERO ? (int8_t)E_ZERO : (int8_t)(xE[w] - 7);
}
// R16 -> INT8 (arredondamento par, mesma regra do quant_warp): a copia que entra no dp4a
__global__ void k_para8(const int16_t* __restrict__ xm, const int8_t* __restrict__ xE, int8_t* __restrict__ ym,
                        int8_t* __restrict__ yE, long long nblocos, int kid) {
    WARP_BLOCO
    int v[4], ee[4];
    le4_16(xm + o, v);
#pragma unroll
    for (int i = 0; i < 4; i++) ee[i] = xE[w];
    int8_t q[4]; int Eo;
    quant_warp(v, ee, q, Eo, kid, o - lane * 4);
    grava4(ym + o, q);
    if (lane == 0) yE[w] = (int8_t)Eo;
}
// z (R16) = x (R16) + y (INT8). Alinhamento: ec = max(ex + 15, ey + 7) - 28 -> as duas parcelas ficam < 2^28 e a
// parcela pequena guarda ate 13 bits abaixo do ultimo bit de x antes de arredondar.
__global__ void k_soma16(const int16_t* __restrict__ xm, const int8_t* __restrict__ xE, const int8_t* __restrict__ ym,
                         const int8_t* __restrict__ yE, int16_t* __restrict__ zm, int8_t* __restrict__ zE, long long nblocos, int kid) {
    WARP_BLOCO
    int a[4], b[4], v[4], ee[4];
    le4_16(xm + o, a); le4(ym + o, b);
    const int ex = xE[w], ey = yE[w], ec = max(ex + 15, ey + 7) - 28;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = (ex == E_ZERO ? 0 : rnd(a[i], ec - ex, kid, o + i)) + (ey == E_ZERO ? 0 : rnd(b[i], ec - ey, kid, o + i));
        ee[i] = ec;
    }
    int16_t q[4]; int Eo;
    quant_warp16(v, ee, q, Eo, kid, o - lane * 4);
    grava4_16(zm + o, q);
    if (lane == 0) zE[w] = (int8_t)Eo;
}

// ReLU^2 forward: a = max(h,0)^2
__global__ void k_relu2(const int8_t* __restrict__ hm, const int8_t* __restrict__ hE, int8_t* __restrict__ am,
                        int8_t* __restrict__ aE, long long nblocos, int kid) {
    WARP_BLOCO
    int h[4], v[4], ee[4];
    le4(hm + o, h);
#pragma unroll
    for (int i = 0; i < 4; i++) { int r = max(h[i], 0); v[i] = r * r; ee[i] = 2 * hE[w]; }
    int8_t q[4]; int Eo;
    quant_warp(v, ee, q, Eo, kid, o - lane * 4);
    grava4(am + o, q);
    if (lane == 0) aE[w] = (int8_t)Eo;
}

// ReLU^2 backward: gh = ga * 2 * max(h,0)
__global__ void k_relu2_back(const int8_t* __restrict__ gm, const int8_t* __restrict__ gE, const int8_t* __restrict__ hm,
                             const int8_t* __restrict__ hE, int8_t* __restrict__ om, int8_t* __restrict__ oE, long long nblocos, int kid) {
    WARP_BLOCO
    int g[4], h[4], v[4], ee[4];
    le4(gm + o, g); le4(hm + o, h);
#pragma unroll
    for (int i = 0; i < 4; i++) { v[i] = g[i] * max(h[i], 0) * 2; ee[i] = gE[w] + hE[w]; }
    int8_t q[4]; int Eo;
    quant_warp(v, ee, q, Eo, kid, o - lane * 4);
    grava4(om + o, q);
    if (lane == 0) oE[w] = (int8_t)Eo;
}

// otimizador: momento m = 7/8 m + g   (g cru: acc int32 + expoente int16 por elemento)
__global__ void k_momento(const int* __restrict__ gacc, const int16_t* __restrict__ ge, int8_t* __restrict__ Mm,
                          int8_t* __restrict__ ME, long long nblocos, int kid) {
    WARP_BLOCO
    int mm[4], v[4], ee[4];
    le4(Mm + o, mm);
    int4 g4 = *reinterpret_cast<const int4*>(gacc + o);
    short4 e4 = *reinterpret_cast<const short4*>(ge + o);
    int g[4] = {g4.x, g4.y, g4.z, g4.w}, eg[4] = {e4.x, e4.y, e4.z, e4.w};
    const int em = ME[w] - 3;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        if (g[i] >= LIM || g[i] <= -LIM) reporta(E_ACUM, kid, o + i, g[i], eg[i]);
        int ec = max(em + 10, eg[i] + bitlen(g[i])) - 28;
        v[i] = rnd(mm[i] * 7, ec - em, kid, o + i) + rnd(g[i], ec - eg[i], kid, o + i);
        ee[i] = ec;
    }
    int8_t q[4]; int Eo;
    quant_warp(v, ee, q, Eo, kid, o - lane * 4);
    grava4(Mm + o, q);
    if (lane == 0) ME[w] = (int8_t)Eo;
}

// otimizador: W <- W - m * 2^-k (+ residuo do error feedback; o que nao cabe no INT8 de W volta para o residuo)
__global__ void k_peso(int8_t* __restrict__ Wm, int8_t* __restrict__ WE, const int8_t* __restrict__ Mm, const int8_t* __restrict__ ME,
                       int8_t* __restrict__ Rm, int8_t* __restrict__ RE, int k_lr, int ef, long long nblocos, int kid) {
    WARP_BLOCO
    int wv[4], mv[4], rv[4], alvo[4], ee[4];
    le4(Wm + o, wv); le4(Mm + o, mv);
    if (ef) le4(Rm + o, rv); else rv[0] = rv[1] = rv[2] = rv[3] = 0;
    const int ew = WE[w], em = ME[w] - k_lr, er = ef ? (int)RE[w] : E_ZERO;
    const int ec = max(max(ew + 7, em + 7), er + 7) - 27;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        alvo[i] = rnd(wv[i], ec - ew, kid, o + i) - rnd(mv[i], ec - em, kid, o + i) + (ef ? rnd(rv[i], ec - er, kid, o + i) : 0);
        ee[i] = ec;
    }
    int8_t q[4]; int Ew;
    quant_warp(alvo, ee, q, Ew, kid, o - lane * 4);
    grava4(Wm + o, q);
    if (lane == 0) WE[w] = (int8_t)Ew;
    if (!ef) return;
    int dif[4];
#pragma unroll
    for (int i = 0; i < 4; i++) dif[i] = alvo[i] - (q[i] == 0 ? 0 : rnd(q[i], ec - Ew, kid, o + i));
    int8_t r[4]; int Er;
    quant_warp(dif, ee, r, Er, kid, o - lane * 4);
    grava4(Rm + o, r);
    if (lane == 0) RE[w] = (int8_t)Er;
}

// ============================================================================ transposicao
// X [R x C] em bloco ao longo de C  ->  Y [C x R] em bloco ao longo de R (e conta: o expoente passa a ser por 128
// linhas de cada coluna). Tile: 128 linhas x 64 colunas, 256 threads. Cada warp faz 8 colunas em 2 grupos de 4: a lane
// l segura as linhas l, l+32, l+64, l+96 das 4 colunas (uma palavra de 4 bytes por linha) e as 4 reducoes andam
// juntas. Linha do tile com 17 palavras (impar): as 32 lanes leem bancos diferentes.
// Shift a esquerda aqui nunca perde bits: bitlen(m) + e <= topo, entao |m << (e - E)| < 2^8 (sem sentinela).
__global__ void __launch_bounds__(256) k_transpoe(const int8_t* __restrict__ X, const int8_t* __restrict__ EX, int8_t* __restrict__ Y,
                                                  int8_t* __restrict__ EY, int R, int C, int kid) {
    __shared__ int tile[128][17];
    __shared__ int er[128];
    const int r0 = blockIdx.y * 128, c0 = blockIdx.x * 64, tid = threadIdx.x;
    {
        const int row = tid >> 1, half = tid & 1;
        const int4* src = reinterpret_cast<const int4*>(X + (size_t)(r0 + row) * C + c0 + half * 32);
        const int4 v0 = src[0], v1 = src[1];
        int* d = &tile[row][half * 8];
        d[0] = v0.x; d[1] = v0.y; d[2] = v0.z; d[3] = v0.w; d[4] = v1.x; d[5] = v1.y; d[6] = v1.z; d[7] = v1.w;
        if (tid < 128) er[tid] = EX[(size_t)(r0 + tid) * (C / BLK) + c0 / BLK];
    }
    __syncthreads();
    const int lane = tid & 31, warp = tid >> 5;
    int e[4];
#pragma unroll
    for (int k = 0; k < 4; k++) e[k] = er[lane + 32 * k];
#pragma unroll
    for (int g = 0; g < 2; g++) {
        const int w = warp * 2 + g;                         // palavra w = colunas c0 + 4w .. 4w + 3
        int m[4][4], topo[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int word = tile[lane + 32 * k][w];
#pragma unroll
            for (int cc = 0; cc < 4; cc++) m[k][cc] = (word << (24 - 8 * cc)) >> 24;
        }
#pragma unroll
        for (int cc = 0; cc < 4; cc++) {
            topo[cc] = SEM_TOPO;
#pragma unroll
            for (int k = 0; k < 4; k++)
                if (m[k][cc] != 0) topo[cc] = max(topo[cc], e[k] + 32 - __clz(abs(m[k][cc])));
        }
#pragma unroll
        for (int o = 16; o; o >>= 1)
#pragma unroll
            for (int cc = 0; cc < 4; cc++) topo[cc] = max(topo[cc], __shfl_xor_sync(0xffffffffu, topo[cc], o));
#pragma unroll
        for (int cc = 0; cc < 4; cc++) {
            const int col = c0 + 4 * w + cc;
            int E = topo[cc] - 7, q[4], est = 0;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int sft = E - e[k];
                q[k] = sft >= 0 ? rnd_d(m[k][cc], sft) : m[k][cc] << min(-sft, 31);
                est |= q[k] > 127 || q[k] < -127;
            }
            if (__any_sync(0xffffffffu, est)) {
                E += 1;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int sft = E - e[k];
                    q[k] = sft >= 0 ? rnd_d(m[k][cc], sft) : m[k][cc] << min(-sft, 31);
                }
            }
            if (topo[cc] == SEM_TOPO || E < -126) {        // bloco todo zero, ou underflow -> bloco zero
#pragma unroll
                for (int k = 0; k < 4; k++) q[k] = 0;
                E = E_ZERO;
            } else if (E > 126 && lane == 0) reporta(E_EXP, kid, (long long)col * R + r0, E, topo[cc]);
#pragma unroll
            for (int k = 0; k < 4; k++) Y[(size_t)col * R + r0 + lane + 32 * k] = (int8_t)q[k];
            if (lane == 0) EY[(size_t)col * (R / BLK) + r0 / BLK] = (int8_t)E;
        }
    }
}

// ============================================================================ embedding
__global__ void k_embed(const int8_t* __restrict__ Wm, const int8_t* __restrict__ WE, const int* __restrict__ tok,
                        int8_t* __restrict__ Xm, int8_t* __restrict__ XE, int M, int D, int V, int kid) {
    const int t = blockIdx.x;
    const int v = tok[t];
    if (v < 0 || v >= V) { if (threadIdx.x == 0) reporta(E_ARG, kid, (long long)t * D, v, V); return; }   // indice = inicio da linha t da saida
    for (int i = threadIdx.x; i < D; i += blockDim.x) Xm[(size_t)t * D + i] = Wm[(size_t)v * D + i];
    for (int i = threadIdx.x; i < D / BLK; i += blockDim.x) XE[(size_t)t * (D / BLK) + i] = WE[(size_t)v * (D / BLK) + i];
}

// one-hot dos tokens (para o gradiente do embedding): linha t = 1 na coluna tok[t]. Token fora do vocabulario vira
// sentinela (E_ARG com a posicao) e a linha fica zerada, em vez de derrubar o contexto do CUDA (como o scatter_ do
// PyTorch fazia com um assert).
__global__ void k_um_quente(const int* __restrict__ tok, int8_t* __restrict__ U, int8_t* __restrict__ UE, int M, int V, int kid) {
    const int t = blockIdx.x, v = tok[t];
    const bool ok = v >= 0 && v < V;
    if (!ok && threadIdx.x == 0) reporta(E_ARG, kid, (long long)t * V, v, V);   // indice = inicio da linha t da saida
    for (int i = threadIdx.x; i < V; i += blockDim.x) U[(size_t)t * V + i] = ok && i == v;
    for (int i = threadIdx.x; i < V / BLK; i += blockDim.x) UE[(size_t)t * (V / BLK) + i] = 0;
}

// ============================================================================ RMSNorm (1 warp = 1 linha; D <= 1024)
__global__ void k_rms(const int8_t* __restrict__ Xm, const int8_t* __restrict__ XE, int8_t* __restrict__ Ym, int8_t* __restrict__ YE,
                      int* __restrict__ cr, int* __restrict__ cemax, int M, int D, int log2D, int G, int kid) {
    const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= M) return;
    const int nb = D / BLK;
    int x[8][4], eb[8], emax = SEM_TOPO;
    for (int b = 0; b < nb; b++) { le4(Xm + (size_t)row * D + b * BLK + lane * 4, x[b]); eb[b] = XE[(size_t)row * nb + b]; emax = max(emax, eb[b]); }
    int ss = 0;
    for (int b = 0; b < nb; b++) {
        int sq = warp_sum(x[b][0] * x[b][0] + x[b][1] * x[b][1] + x[b][2] * x[b][2] + x[b][3] * x[b][3]);
        ss += rnd(sq << G, 2 * (emax - eb[b]), kid, (long long)row * D);
    }
    if (ss >= LIM) reporta(E_ACUM, kid, (long long)row * D, ss, -1);
    const int r = isqrt_i(ss >> log2D), k = bitlen(r) + 10, inv = (1 << k) / r;
    for (int b = 0; b < nb; b++) {
        int v[4], ee[4];
#pragma unroll
        for (int i = 0; i < 4; i++) { v[i] = x[b][i] * inv; ee[i] = eb[b] - k - emax + G / 2; }
        int8_t q[4]; int Eo;
        quant_warp(v, ee, q, Eo, kid, (long long)row * D + b * BLK);
        grava4(Ym + (size_t)row * D + b * BLK + lane * 4, q);
        if (lane == 0) YE[(size_t)row * nb + b] = (int8_t)Eo;
    }
    if (lane == 0) { cr[row] = r; cemax[row] = emax; }
}

// RMSNorm backward: dx = (g - y * mean(g*y)) / rms
__global__ void k_rms_back(const int8_t* __restrict__ gm, const int8_t* __restrict__ gE, const int8_t* __restrict__ ym,
                           const int8_t* __restrict__ yE, const int* __restrict__ cr, const int* __restrict__ cemax,
                           int8_t* __restrict__ om, int8_t* __restrict__ oE, int M, int D, int log2D, int G, int GB, int kid) {
    const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= M) return;
    const int nb = D / BLK;
    int g[8][4], y[8][4], eg[8], ey[8], smax = SEM_TOPO;
    for (int b = 0; b < nb; b++) {
        le4(gm + (size_t)row * D + b * BLK + lane * 4, g[b]); le4(ym + (size_t)row * D + b * BLK + lane * 4, y[b]);
        eg[b] = gE[(size_t)row * nb + b]; ey[b] = yE[(size_t)row * nb + b]; smax = max(smax, eg[b] + ey[b]);
    }
    int S = 0;
    for (int b = 0; b < nb; b++)
#pragma unroll
        for (int i = 0; i < 4; i++) S += rnd((g[b][i] * y[b][i]) << GB, smax - (eg[b] + ey[b]), kid, (long long)row * D);
    S = warp_sum(S);
    if (S >= LIM || S <= -LIM) reporta(E_ACUM, kid, (long long)row * D, S, -2);
    int eS = smax - GB - log2D;
    const int sh = max(bitlen(S) - 15, 0);
    S = rnd(S, sh, kid, (long long)row * D); eS += sh;
    const int r = cr[row], k = bitlen(r) + 10, inv = (1 << k) / r, emax = cemax[row];
    for (int b = 0; b < nb; b++) {
        const int ec = max(eg[b] + 7, ey[b] + eS + 22) - 28;
        int v[4], ee[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            long long idx = (long long)row * D + b * BLK + lane * 4 + i;
            int dentro = rnd(g[b][i], ec - eg[b], kid, idx) - rnd(y[b][i] * S, ec - (ey[b] + eS), kid, idx);
            int sh2 = max(bitlen(dentro) - 18, 0);
            dentro = rnd(dentro, sh2, kid, idx);
            v[i] = dentro * inv;
            ee[i] = ec + sh2 - k - (emax - G / 2);
        }
        int8_t q[4]; int Eo;
        quant_warp(v, ee, q, Eo, kid, (long long)row * D + b * BLK);
        grava4(om + (size_t)row * D + b * BLK + lane * 4, q);
        if (lane == 0) oE[(size_t)row * nb + b] = (int8_t)Eo;
    }
}

// ============================================================================ atencao (1 bloco = 1 (lote, cabeca))
// NR = linhas por thread. NR = 1: 1 thread por token. NR = 2: cada thread cuida de i e T-1-i (triangulo causal
// balanceado). Medido (A/B alternado, GTX 1650): NR = 1 e mais rapido nos dois kernels; NR = 2 fica para registro.
__device__ int bloco_max(int v, int* red) {
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    int r = SEM_TOPO;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) r = max(r, red[i]);
    __syncthreads();
    return r;
}

// q, k, v, dO da cabeca h alinhados ao maior expoente da cabeca no lote, com GA bits de guarda:
//   a = rnd(m << GA, emx - e).
// Truque de bits: o deslocamento s = emx - e e o mesmo para os 4 valores de um token, entao a = m8 << L
// exatamente, com m8 int8: (m8 = m, L = GA - s) se s <= GA; (m8 = rnd(m, s - GA), L = 0) se s > GA.
// Os 4 m8 vao empacotados numa palavra (pw) e o produto de 4 termos a.b vira dp4a(pa, pb) << (La + Lb), exato.
// al (opcional): os valores alinhados a, para os lacos que somam ao longo dos tokens.
template <int NR>
__device__ void empacota(const int8_t* m, const int8_t* E, int ld, int col, int b0, int ga, int* pw, int* pL, int (*al)[4],
                         int* red, int& eo) {
    const int t = threadIdx.x, T = NR * blockDim.x;
    const int rows[2] = {t, T - 1 - t};
    int e[2] = {SEM_TOPO, SEM_TOPO};
#pragma unroll
    for (int r = 0; r < NR; r++) e[r] = E[(size_t)(b0 + rows[r]) * (ld / BLK) + col / BLK];
    const int emx = bloco_max(max(e[0], e[1]), red);
#pragma unroll
    for (int r = 0; r < NR; r++) {
        const int i = rows[r], s = emx - e[r], L = s <= ga ? ga - s : 0;
        const char4 c = *reinterpret_cast<const char4*>(m + (size_t)(b0 + i) * ld + col);
        int v[4] = {c.x, c.y, c.z, c.w}, w = 0;
#pragma unroll
        for (int d = 0; d < 4; d++) {
            if (s > ga) v[d] = rnd_d(v[d], s - ga);
            w |= (v[d] & 255) << (8 * d);
            if (al) al[i][d] = v[d] << L;
        }
        pw[i] = w; pL[i] = L;
    }
    eo = emx - ga;
}

#define PROD(pa, La, pb, Lb) (__dp4a((pa), (pb), 0) << ((La) + (Lb)))

template <int NR>
__global__ void __launch_bounds__(128 / NR) k_atencao(const int8_t* __restrict__ qm, const int8_t* __restrict__ qE, int* __restrict__ Oacc,
                                                int16_t* __restrict__ Oe, uint8_t* __restrict__ P, int T, int H, int D, int GA, int kid) {
    __shared__ int qw[128], qL[128], kw[128], kL[128], vw[128], vL[128], va[128][4], qa[128][4], ka[128][4];
    __shared__ int red[4];
    const int b = blockIdx.x / H, h = blockIdx.x % H, ld = 3 * D, b0 = b * T;
    int eq, ek, ev;
    empacota<NR>(qm, qE, ld, h * 4, b0, GA, qw, qL, qa, red, eq);
    empacota<NR>(qm, qE, ld, D + h * 4, b0, GA, kw, kL, ka, red, ek);
    empacota<NR>(qm, qE, ld, 2 * D + h * 4, b0, GA, vw, vL, va, red, ev);
    __syncthreads();
    const int s_u = -(eq + ek - 1 + 8), incl = 8 - h / 8;
    // aqui o produto q.k usa 4 multiplicacoes dos valores alinhados: medido, o dp4a + shift (PROD) deixou o forward
    // 1,8x mais lento (a cadeia dependente LDS -> dp4a -> shift variavel -> rnd -> exp ficou mais longa); no backward o
    // dp4a ganha. Os dois caminhos dao o mesmo inteiro.
#define QK(j) (qa[i][0] * ka[j][0] + qa[i][1] * ka[j][1] + qa[i][2] * ka[j][2] + qa[i][3] * ka[j][3])
    uint8_t* Pb = P + (size_t)blockIdx.x * T * T;          // P guardada transposta [j][i]: threads vizinhas, bytes vizinhos
    // Score u = rnd(q.k, s_u). s_u e o mesmo para o bloco inteiro e |q.k| <= 4 * 2032^2 < 2^24 (q, k alinhados com
    // |valor| <= 127 << 4): um shift a esquerda so pode perder bits se for maior que 6. A decisao sai do laco: com
    // s_u >= -6 o laco usa a conta sem sentinela (sem a chamada de relatorio, que gerava barreiras de convergencia
    // BSSY/BSYNC por par); com s_u < -6 (raro) usa a conta conferida. Os dois dao o mesmo inteiro.
    auto passadas = [&](auto conferida) {
        constexpr bool CONF = decltype(conferida)::value;
        const int nsh = s_u > 0 ? 0 : -s_u;
#define SC(j) (CONF ? rnd(QK(j), s_u, kid, (long long)row * D + h * 4) : (s_u > 0 ? rnd_d(QK(j), s_u) : QK(j) << nsh))
#pragma unroll 1
        for (int r = 0; r < NR; r++) {
            const int i = r ? T - 1 - (int)threadIdx.x : threadIdx.x, row = b0 + i;
            int mx = U_MIN;
            for (int j = 0; j <= i; j++) mx = max(mx, SC(j) - ((i - j) << incl));
            int soma = 0;
            for (int j = 0; j <= i; j++) soma += exp_dif(mx, SC(j) - ((i - j) << incl)) >> 3;
            if (soma <= 0) { reporta(E_DIV, kid, (long long)row * D + h * 4, soma, mx); soma = 1; }
            const int inv = (1 << 28) / soma;
            int o[4] = {0, 0, 0, 0};
            // um laco so ate T, como P e escrita: todas as lanes no mesmo j (escrita coalescida; zeros acima da diagonal)
            for (int j = 0; j < T; j++) {
                int p = 0;
                if (j <= i) {
                    const int ex = exp_dif(mx, SC(j) - ((i - j) << incl)) >> 3;
                    p = min((ex * inv + (1 << 19)) >> 20, 255);
#pragma unroll
                    for (int d = 0; d < 4; d++) o[d] += p * va[j][d];
                }
                Pb[(size_t)j * T + i] = (uint8_t)p;
            }
#pragma unroll
            for (int d = 0; d < 4; d++) {
                size_t idx = (size_t)row * D + h * 4 + d;
                if (o[d] >= LIM || o[d] <= -LIM) reporta(E_ACUM, kid, idx, o[d], -3);
                Oacc[idx] = o[d];
                Oe[idx] = (int16_t)(ev - 8);
            }
        }
#undef SC
    };
    if (s_u < -6) passadas(std::true_type{});
    else passadas(std::false_type{});
}
#undef QK

// Backward da atencao. P (guardada transposta, [j][i]) e lida da memoria global (cache L1/L2): sem ela na
// memoria compartilhada cabem muito mais blocos por SM. Todos os lacos percorrem so a parte causal (j <= i).
// LIMITE = 1 (variante de especificacao, 3 passadas em vez de 5): os shifts de gP e dS saem de limites superiores em
// vez dos maximos exatos, que exigiam uma passada inteira cada:
//   |gP_ij| <= max_i ||dO_i||_inf * max_j ||v_j||_1          (calculado em O(T), antes de qualquer passada)
//   |P (gp - s_lin)| <= max P * (max |gp| + max |s_lin|)      (os tres maximos saem da passada que calcula s_lin)
// Limite maior so pode custar bits (nunca estoura); o efeito no treino e medido.
template <int NR, int LIMITE>
__global__ void __launch_bounds__(128 / NR) k_atencao_back(const int8_t* __restrict__ gom, const int8_t* __restrict__ goE,
                                                     const int8_t* __restrict__ qm, const int8_t* __restrict__ qE,
                                                     const uint8_t* __restrict__ P, int* __restrict__ Gacc, int16_t* __restrict__ Ge,
                                                     int T, int H, int D, int GA, int kid) {
    __shared__ int gw[128], gL[128], vw[128], vL[128], xw[128], xL[128];
    __shared__ int qa[128][4], ka[128][4], goa[128][4], slin[128];
    __shared__ int red[4];
    const int b = blockIdx.x / H, h = blockIdx.x % H, ld = 3 * D, b0 = b * T, t = threadIdx.x;
    const uint8_t* Pb = P + (size_t)blockIdx.x * T * T;
    int eq, ek, ev, ego;
    empacota<NR>(gom, goE, D, h * 4, b0, GA, gw, gL, goa, red, ego);
    empacota<NR>(qm, qE, ld, h * 4, b0, GA, xw, xL, qa, red, eq);           // q e k: so os alinhados sao usados
    empacota<NR>(qm, qE, ld, D + h * 4, b0, GA, xw, xL, ka, red, ek);
    empacota<NR>(qm, qE, ld, 2 * D + h * 4, b0, GA, vw, vL, nullptr, red, ev);
    __syncthreads();
    const int rows[2] = {t, T - 1 - t};
    // gP = dO . v^T (dp4a + shift); encolhe para 15 bits pelo maior |gP| causal da matriz (b, h) (ou pelo limite)
    int mg = 0;
    if constexpr (LIMITE) {
        int b1 = 0, b2 = 0;
#pragma unroll
        for (int r = 0; r < NR; r++) {
            const int i = rows[r], w = vw[i];
            b1 = max(b1, max(max(abs(goa[i][0]), abs(goa[i][1])), max(abs(goa[i][2]), abs(goa[i][3]))));
            b2 = max(b2, (abs((w << 24) >> 24) + abs((w << 16) >> 24) + abs((w << 8) >> 24) + abs(w >> 24)) << vL[i]);
        }
        b1 = bloco_max(b1, red);
        mg = b1 * bloco_max(b2, red);
    } else {
#pragma unroll 1
        for (int r = 0; r < NR; r++) {
            const int i = rows[r], gwi = gw[i], gLi = gL[i];
            for (int j = 0; j <= i; j++) mg = max(mg, abs(PROD(gwi, gLi, vw[j], vL[j])));
        }
        mg = bloco_max(mg, red);
    }
    const int sh = max(bitlen(mg) - 15, 0), egp = ego + ev + sh;
    // s_lin = sum_j P gP / 256
    int sl[2], pm = 0, gm = 0;
#pragma unroll 1
    for (int r = 0; r < NR; r++) {
        const int i = rows[r], gwi = gw[i], gLi = gL[i];
        int s = 0;
        for (int j = 0; j <= i; j++) {
            const int p = Pb[(size_t)j * T + i], gp = rnd_d(PROD(gwi, gLi, vw[j], vL[j]), sh);
            s += p * gp;
            if constexpr (LIMITE) { pm = max(pm, p); gm = max(gm, abs(gp)); }
        }
        if (s >= LIM || s <= -LIM) reporta(E_ACUM, kid, (long long)(b0 + i) * ld + h * 4, s, -4);
        sl[r] = rnd_d(s, 8);
        slin[i] = sl[r];
    }
    // dS = P (gP - s_lin), encolhido para 12 bits pelo maior |dS| da matriz (ou pelo limite)
    int md = 0;
    if constexpr (LIMITE) {
        int sm = 0;
#pragma unroll
        for (int r = 0; r < NR; r++) sm = max(sm, abs(sl[r]));
        pm = bloco_max(pm, red);
        gm = bloco_max(gm, red);
        md = pm * (gm + bloco_max(sm, red));
    } else {
#pragma unroll 1
        for (int r = 0; r < NR; r++) {
            const int i = rows[r], gwi = gw[i], gLi = gL[i], s = sl[r];
            for (int j = 0; j <= i; j++) md = max(md, abs((int)Pb[(size_t)j * T + i] * (rnd_d(PROD(gwi, gLi, vw[j], vL[j]), sh) - s)));
        }
        md = bloco_max(md, red);
    }
    const int sh2 = max(bitlen(md) - 12, 0), eds = egp - 8 - 1 + sh2;   // bloco_max sincroniza: slin pronto
    // linhas: gq_i = sum_j dS_ij k_j
#pragma unroll 1
    for (int r = 0; r < NR; r++) {
        const int i = rows[r], gwi = gw[i], gLi = gL[i], s = sl[r];
        int gq[4] = {0, 0, 0, 0};
        for (int j = 0; j <= i; j++) {
            const int ds = rnd_d((int)Pb[(size_t)j * T + i] * (rnd_d(PROD(gwi, gLi, vw[j], vL[j]), sh) - s), sh2);
#pragma unroll
            for (int d = 0; d < 4; d++) gq[d] += ds * ka[j][d];
        }
#pragma unroll
        for (int d = 0; d < 4; d++) {
            const size_t o = (size_t)(b0 + i) * ld + h * 4 + d;
            if (gq[d] >= LIM || gq[d] <= -LIM) reporta(E_ACUM, kid, o, gq[d], -5);
            Gacc[o] = gq[d]; Ge[o] = (int16_t)(eds + ek);
        }
    }
    // colunas: gk_j = sum_{i>=j} dS_ij q_i ; gv_j = sum_{i>=j} P_ij dO_i. A linha j de P (transposta) e contigua: 4 bytes por leitura.
#pragma unroll 1
    for (int r = 0; r < NR; r++) {
        const int jj = rows[r], vwj = vw[jj], vLj = vL[jj];
        const unsigned* Pr = reinterpret_cast<const unsigned*>(Pb + (size_t)jj * T);
        int gk[4] = {0, 0, 0, 0}, gv[4] = {0, 0, 0, 0};
        for (int w = jj >> 2; w < T / 4; w++) {
            const unsigned pw = Pr[w];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                // sem 'if (ii < jj) continue': acima da diagonal P = 0 (o forward grava os zeros), e com p = 0 as duas
                // contribuicoes sao exatamente 0. O desvio gerava 4 barreiras de convergencia por palavra (medido no SASS).
                const int ii = 4 * w + q;
                const int p = (pw >> (8 * q)) & 255;
                const int ds = rnd_d(p * (rnd_d(PROD(gw[ii], gL[ii], vwj, vLj), sh) - slin[ii]), sh2);
#pragma unroll
                for (int d = 0; d < 4; d++) { gk[d] += ds * qa[ii][d]; gv[d] += p * goa[ii][d]; }
            }
        }
#pragma unroll
        for (int d = 0; d < 4; d++) {
            const size_t o = (size_t)(b0 + jj) * ld + h * 4 + d;
            if (gk[d] >= LIM || gk[d] <= -LIM) reporta(E_ACUM, kid, o + D, gk[d], -5);
            if (gv[d] >= LIM || gv[d] <= -LIM) reporta(E_ACUM, kid, o + 2 * D, gv[d], -5);
            Gacc[o + D] = gk[d]; Ge[o + D] = (int16_t)(eds + eq);
            Gacc[o + 2 * D] = gv[d]; Ge[o + 2 * D] = (int16_t)(ego - 8);
        }
    }
}
#undef PROD

// ============================================================================ softmax + entropia cruzada (1 warp = 1 linha; V <= 1024)
// logits crus -> gradiente g = (P - 1[alvo]) / M em bloco. Para o relatorio da loss: d_alvo = umax - u_alvo e soma.
__global__ void k_softmax_ce(const int* __restrict__ acc, const int16_t* __restrict__ e, const int* __restrict__ alvo,
                             int8_t* __restrict__ gm, int8_t* __restrict__ gE, int* __restrict__ d_alvo, int* __restrict__ soma_o,
                             int M, int V, int log2M, int kid) {
    const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= M) return;
    const int nb = V / BLK, t = alvo[row];
    if (t < 0 || t >= V) { if (lane == 0) reporta(E_ARG, kid, (long long)row * V, t, V); return; }   // indice = inicio da linha
    int u[8][4], umax = U_MIN;
    for (int b = 0; b < nb; b++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
            size_t o = (size_t)row * V + b * BLK + lane * 4 + i;
            u[b][i] = rnd(acc[o], -(e[o] + 8), kid, o);
            umax = max(umax, u[b][i]);
        }
    umax = warp_max(umax);
    int ex[8][4], s = 0;
    for (int b = 0; b < nb; b++)
#pragma unroll
        for (int i = 0; i < 4; i++) { ex[b][i] = exp_dif(umax, u[b][i]) >> 3; s += ex[b][i]; }
    s = warp_sum(s);
    if (s <= 0) { reporta(E_DIV, kid, (long long)row * V, s, umax); s = 1; }
    const int inv = (1 << 28) / s;
    for (int b = 0; b < nb; b++) {
        int v[4], ee[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int col = b * BLK + lane * 4 + i;
            v[i] = ((ex[b][i] * inv + (1 << 15)) >> 16) - (col == t ? 4096 : 0);
            ee[i] = -12 - log2M;
            if (col == t) d_alvo[row] = umax - u[b][i];
        }
        int8_t q[4]; int Eo;
        quant_warp(v, ee, q, Eo, kid, (long long)row * V + b * BLK);
        grava4(gm + (size_t)row * V + b * BLK + lane * 4, q);
        if (lane == 0) gE[(size_t)row * nb + b] = (int8_t)Eo;
    }
    if (lane == 0) soma_o[row] = s;
}

// ============================================================================ interface C
static int fim() { return (int)cudaGetLastError(); }
static int g_linhas = 1, g_alvo_blocos = 42, g_bm = 0;
// stream de todos os lancamentos (0 = padrao). Gravar o passo num CUDA Graph exige lancar na stream que esta gravando.
static cudaStream_t g_stream = 0;
// tile da GEMM: 64 linhas (128 registradores, 30 KB: 2 blocos = 16 warps por SM) venceu o de 128 (209 registradores,
// 40 KB: 1 bloco = 8 warps por SM) em todas as formas do modelo, 1,2-1,5x (A/B alternado).
static int bm_auto(int M) { return M % 64 == 0 ? 64 : 128; }
static unsigned nblk(long long n) { return (unsigned)((n + 7) / 8); }

extern "C" {
int jt_erro_le(long long* out) {
    Erro e;
    cudaError_t c = cudaMemcpyFromSymbol(&e, g_erro, sizeof(Erro));
    if (c != cudaSuccess) return -(int)c;
    out[0] = e.n; out[1] = e.codigo; out[2] = e.kid; out[3] = e.idx; out[4] = e.valor; out[5] = e.aux;
    return e.n;
}
int jt_erro_limpa() {
    Erro e = {};
    return (int)cudaMemcpyToSymbol(g_erro, &e, sizeof(Erro));
}
int jt_sincroniza() { return (int)cudaDeviceSynchronize(); }
int jt_para16(const void* xm, const void* xE, void* zm, void* zE, long long nblocos, int kid) {
    k_para16<<<nblk(nblocos), 256, 0, g_stream>>>((const int8_t*)xm, (const int8_t*)xE, (int16_t*)zm, (int8_t*)zE, nblocos, kid);
    return (int)cudaGetLastError();
}
int jt_para8(const void* xm, const void* xE, void* ym, void* yE, long long nblocos, int kid) {
    k_para8<<<nblk(nblocos), 256, 0, g_stream>>>((const int16_t*)xm, (const int8_t*)xE, (int8_t*)ym, (int8_t*)yE, nblocos, kid);
    return (int)cudaGetLastError();
}
int jt_soma16(const void* xm, const void* xE, const void* ym, const void* yE, void* zm, void* zE, long long nblocos, int kid) {
    k_soma16<<<nblk(nblocos), 256, 0, g_stream>>>((const int16_t*)xm, (const int8_t*)xE, (const int8_t*)ym, (const int8_t*)yE,
                                                  (int16_t*)zm, (int8_t*)zE, nblocos, kid);
    return (int)cudaGetLastError();
}
int jt_res_estocastico(int ligado) { return (int)cudaMemcpyToSymbol(g_res_est, &ligado, sizeof(int)); }
void jt_stream(void* st) { g_stream = (cudaStream_t)st; }
void jt_gemm_split(int alvo_blocos) { g_alvo_blocos = alvo_blocos; }
void jt_gemm_bm(int bm) { g_bm = bm; }   // 0 automatico, 64 ou 128 forcado (medicao)
void jt_atencao_linhas(int n) { g_linhas = n; }   // linhas por thread na atencao: 1 (padrao, mais rapido medido) ou 2 (i e T-1-i)

int jt_gemm_f(const void* A, const void* EA, const void* B, const void* EB, void* Cm, void* CE, void* C32, void* Ce,
              const void* Xm, const void* XE, void* Om2, void* OE2, int M, int N, int K, int modo, int kid) {
    // tile de 128 linhas (um de 32 linhas perdeu em toda forma medida e foi retirado).
    // Saida crua com poucos blocos: divide K (split-K) ate ter ~g_alvo_blocos blocos, em potencias de 2.
    const int nb = K / BLK;
    if (M % 64 || N % BN || K % BLK || modo < 0 || modo > 4) return (int)cudaErrorInvalidValue;
    // BM = 64 (2 blocos por SM) ou 128 (1 bloco por SM): g_bm forca um dos dois (medicao); 0 = regra medida.
    const int bm = g_bm ? g_bm : bm_auto(M);
    int partes = 1;
    if (modo == 1)
        while (partes * 2 <= nb && (long long)(M / bm) * (N / BN) * partes * 2 <= g_alvo_blocos && nb % (partes * 2) == 0) partes *= 2;
    if (partes > 1) {
        cudaError_t c = cudaMemsetAsync(C32, 0, (size_t)M * N * sizeof(int), g_stream);
        if (c != cudaSuccess) return (int)c;
    }
    const dim3 grade(N / BN, M / bm, partes);
#define ARGS (const int8_t*)A, (const int8_t*)EA, (const int8_t*)B, (const int8_t*)EB, (int8_t*)Cm, (int8_t*)CE, (int*)C32, (int16_t*)Ce, \
             (const int8_t*)Xm, (const int8_t*)XE, (int8_t*)Om2, (int8_t*)OE2, M, N, K
#define LANCA(MD) (bm == 64 ? (k_gemm<MD, 64><<<grade, 256, 0, g_stream>>>(ARGS, MD == 1 ? nb / partes : nb, kid), 0) \
                            : (k_gemm<MD, 128><<<grade, 256, 0, g_stream>>>(ARGS, MD == 1 ? nb / partes : nb, kid), 0))
    switch (modo) {
        case 0: LANCA(0); break;
        case 1: LANCA(1); break;
        case 2: LANCA(2); break;
        case 3: LANCA(3); break;
        case 4: LANCA(4); break;
    }
#undef LANCA
#undef ARGS
    return fim();
}
int jt_gemm(const void* A, const void* EA, const void* B, const void* EB, void* Cm, void* CE, void* C32, void* Ce,
            int M, int N, int K, int modo, int kid) {
    return jt_gemm_f(A, EA, B, EB, Cm, CE, C32, Ce, nullptr, nullptr, nullptr, nullptr, M, N, K, modo, kid);
}
int jt_quant(const void* acc, const void* e, void* m, void* E, long long nblocos, int kid) {
    k_quant<<<nblk(nblocos), 256, 0, g_stream>>>((const int*)acc, (const int16_t*)e, (int8_t*)m, (int8_t*)E, nblocos, kid);
    return fim();
}
int jt_soma(const void* xm, const void* xE, const void* ym, const void* yE, void* zm, void* zE, long long nblocos, int kid) {
    k_soma<<<nblk(nblocos), 256, 0, g_stream>>>((const int8_t*)xm, (const int8_t*)xE, (const int8_t*)ym, (const int8_t*)yE, (int8_t*)zm,
                                   (int8_t*)zE, nblocos, kid);
    return fim();
}
int jt_relu2(const void* hm, const void* hE, void* am, void* aE, long long nblocos, int kid) {
    k_relu2<<<nblk(nblocos), 256, 0, g_stream>>>((const int8_t*)hm, (const int8_t*)hE, (int8_t*)am, (int8_t*)aE, nblocos, kid);
    return fim();
}
int jt_relu2_back(const void* gm, const void* gE, const void* hm, const void* hE, void* om, void* oE, long long nblocos, int kid) {
    k_relu2_back<<<nblk(nblocos), 256, 0, g_stream>>>((const int8_t*)gm, (const int8_t*)gE, (const int8_t*)hm, (const int8_t*)hE, (int8_t*)om,
                                         (int8_t*)oE, nblocos, kid);
    return fim();
}
int jt_momento(const void* gacc, const void* ge, void* Mm, void* ME, long long nblocos, int kid) {
    k_momento<<<nblk(nblocos), 256, 0, g_stream>>>((const int*)gacc, (const int16_t*)ge, (int8_t*)Mm, (int8_t*)ME, nblocos, kid);
    return fim();
}
int jt_peso(void* Wm, void* WE, const void* Mm, const void* ME, void* Rm, void* RE, int k_lr, int ef, long long nblocos, int kid) {
    k_peso<<<nblk(nblocos), 256, 0, g_stream>>>((int8_t*)Wm, (int8_t*)WE, (const int8_t*)Mm, (const int8_t*)ME, (int8_t*)Rm, (int8_t*)RE, k_lr, ef,
                                   nblocos, kid);
    return fim();
}
int jt_transpoe(const void* X, const void* EX, void* Y, void* EY, int R, int C, int kid) {
    k_transpoe<<<dim3(C / 64, R / 128), 256, 0, g_stream>>>((const int8_t*)X, (const int8_t*)EX, (int8_t*)Y, (int8_t*)EY, R, C, kid);
    return fim();
}
int jt_embed(const void* Wm, const void* WE, const void* tok, void* Xm, void* XE, int M, int D, int V, int kid) {
    k_embed<<<M, 128, 0, g_stream>>>((const int8_t*)Wm, (const int8_t*)WE, (const int*)tok, (int8_t*)Xm, (int8_t*)XE, M, D, V, kid);
    return fim();
}
int jt_um_quente(const void* tok, void* U, void* UE, int M, int V, int kid) {
    k_um_quente<<<M, 128, 0, g_stream>>>((const int*)tok, (int8_t*)U, (int8_t*)UE, M, V, kid);
    return fim();
}
int jt_rms(const void* Xm, const void* XE, void* Ym, void* YE, void* cr, void* cemax, int M, int D, int log2D, int G, int kid) {
    k_rms<<<nblk(M), 256, 0, g_stream>>>((const int8_t*)Xm, (const int8_t*)XE, (int8_t*)Ym, (int8_t*)YE, (int*)cr, (int*)cemax, M, D, log2D, G, kid);
    return fim();
}
int jt_rms_back(const void* gm, const void* gE, const void* ym, const void* yE, const void* cr, const void* cemax, void* om, void* oE,
                int M, int D, int log2D, int G, int GB, int kid) {
    k_rms_back<<<nblk(M), 256, 0, g_stream>>>((const int8_t*)gm, (const int8_t*)gE, (const int8_t*)ym, (const int8_t*)yE, (const int*)cr,
                                 (const int*)cemax, (int8_t*)om, (int8_t*)oE, M, D, log2D, G, GB, kid);
    return fim();
}
int jt_atencao(const void* qm, const void* qE, void* Oacc, void* Oe, void* P, int Bt, int T, int H, int D, int GA, int kid) {
    if (g_linhas == 1) k_atencao<1><<<Bt * H, T, 0, g_stream>>>((const int8_t*)qm, (const int8_t*)qE, (int*)Oacc, (int16_t*)Oe, (uint8_t*)P, T, H, D, GA, kid);
    else k_atencao<2><<<Bt * H, T / 2, 0, g_stream>>>((const int8_t*)qm, (const int8_t*)qE, (int*)Oacc, (int16_t*)Oe, (uint8_t*)P, T, H, D, GA, kid);
    return fim();
}
int jt_atencao_back(const void* gom, const void* goE, const void* qm, const void* qE, const void* P, void* Gacc, void* Ge,
                    int Bt, int T, int H, int D, int GA, int limite, int kid) {
#define ARGS (const int8_t*)gom, (const int8_t*)goE, (const int8_t*)qm, (const int8_t*)qE, (const uint8_t*)P, (int*)Gacc, (int16_t*)Ge, T, H, D, GA, kid
    if (g_linhas == 1) {
        if (limite) k_atencao_back<1, 1><<<Bt * H, T, 0, g_stream>>>(ARGS);
        else k_atencao_back<1, 0><<<Bt * H, T, 0, g_stream>>>(ARGS);
    } else {
        if (limite) k_atencao_back<2, 1><<<Bt * H, T / 2, 0, g_stream>>>(ARGS);
        else k_atencao_back<2, 0><<<Bt * H, T / 2, 0, g_stream>>>(ARGS);
    }
#undef ARGS
    return fim();
}
int jt_softmax_ce(const void* acc, const void* e, const void* alvo, void* gm, void* gE, void* d_alvo, void* soma, int M, int V,
                  int log2M, int kid) {
    k_softmax_ce<<<nblk(M), 256, 0, g_stream>>>((const int*)acc, (const int16_t*)e, (const int*)alvo, (int8_t*)gm, (int8_t*)gE, (int*)d_alvo,
                                   (int*)soma, M, V, log2M, kid);
    return fim();
}
}
