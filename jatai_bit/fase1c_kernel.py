"""Fase 1c com kernels próprios (libbit.so): o "observador" (pesos como probabilidade) treinando uma rede 100% binária.

Mesma tarefa professor-aluno de fase1c_binario.py (professor binário 256 -> 512 -> 512 -> 16 classes, avaliação da
rede final 100% binária em 8192 entradas fixas), agora com a representação Q7 da libbit (médias em int8, um dp4a por
produto, contadores em int16) e tudo calculado em kernels próprios. O Python só aloca e chama.

Uso:
  python fase1c_kernel.py conferir                      # um passo de treino: kernels x referência int64, bit a bit
  python fase1c_kernel.py treinar <semente> [passos] [shift]
Saída do treino: logs/fase1ck_ebp_sh<shift>_s<semente>.csv
"""
import ctypes
import os
import sys
import time

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
N_IN, H, C, CP = 256, 512, 16, 64           # CP: classes completadas até múltiplo de 64 (as multiplicações exigem)
B, B_EVAL = 1024, 8192
EVAL_CADA = 250
I8, I16, I32, I64 = torch.int8, torch.int16, torch.int32, torch.int64

lib = ctypes.CDLL(os.path.join(AQUI, "libbit.so"))
P = lambda t: ctypes.c_void_p(t.data_ptr())
I = ctypes.c_int
L = ctypes.c_longlong


def ok(r, nome):
    if r != 0:
        raise RuntimeError(f"libbit: {nome} devolveu cudaError {r}")


def gemm(A, Bm):
    """A [M,K] int8, Bm [N,K] int8 -> [M,N] int32."""
    M, K = A.shape
    Cm = torch.empty(M, Bm.shape[0], dtype=I32, device=DEV)
    ok(lib.bt_gemm(P(A), P(Bm), P(Cm), I(M), I(Bm.shape[0]), I(K)), "gemm")
    return Cm


def tr8(X):
    Y = torch.empty(X.shape[1], X.shape[0], dtype=I8, device=DEV)
    ok(lib.bt_transpoe8(P(X), P(Y), I(X.shape[0]), I(X.shape[1])), "transpoe8")
    return Y


def n_(t):
    return L(t.numel())


# ============================================================================ professor e dados (iguais à fase 1c)
def sgn(x):
    return torch.where(x >= 0, 1, -1)


def professor():
    g = torch.Generator(device=DEV).manual_seed(12345)
    W1 = sgn(torch.randint(0, 2, (H, N_IN), generator=g, device=DEV) * 2 - 1).to(I8)
    W2 = sgn(torch.randint(0, 2, (H, H), generator=g, device=DEV) * 2 - 1).to(I8)
    Wo = sgn(torch.randint(0, 2, (C, H), generator=g, device=DEV) * 2 - 1).to(I8)
    return W1, W2, Wo


def fwd_bin(x8, W1, W2, Wo):
    """rede 100% binária: entradas, pesos e ativações em +-1 (int8); classe = argmax da soma."""
    h1 = torch.empty(x8.shape[0], H, dtype=I8, device=DEV)
    a = gemm(x8, W1); ok(lib.bt_sinal8(P(a), P(h1), n_(a)), "sinal8")
    h2 = torch.empty_like(h1)
    a = gemm(h1, W2); ok(lib.bt_sinal8(P(a), P(h2), n_(a)), "sinal8")
    return gemm(h2, Wo)[:, :C]


def dados(g, n, Wt):
    x = sgn(torch.randint(0, 2, (n, N_IN), generator=g, device=DEV) * 2 - 1).to(I8)
    Wo = Wt[2]
    Wo_p = torch.zeros(CP, H, dtype=I8, device=DEV); Wo_p[:C] = Wo
    y = fwd_bin(x, Wt[0], Wt[1], Wo_p).argmax(1)
    return x, y


# ============================================================================ o observador em kernels
class Observador:
    def __init__(self, g, shift):
        self.shift = shift
        # contadores começam "meio decididos" (média do peso +-0,5): com +-64 (média +-0,06) as médias das ativações
        # ficavam abaixo da resolução int8 já na 2a camada
        th = [torch.randint(-512, 513, s, generator=g, device=DEV) for s in ((H, N_IN), (H, H), (C, H))]
        tho = torch.zeros(CP, H, dtype=I64, device=DEV); tho[:C] = th[2]        # classes extras: theta 0 (média 0)
        self.T = [th[0].to(I16).contiguous(), th[1].to(I16).contiguous(), tho.to(I16).contiguous()]
        self.erros = torch.zeros(1, dtype=I32, device=DEV)
        self.mx = torch.zeros(1, dtype=I32, device=DEV)

    def passo(self, x8, y, devolve=False):
        n = x8.shape[0]
        mu = [(x8.to(I32) * 127).to(I8).contiguous()]
        mu2 = [torch.empty_like(mu[0])]
        ok(lib.bt_quad8(P(mu[0]), P(mu2[0]), n_(mu[0])), "quad8")
        m8s, svs = [], []
        so = None
        for l, th in enumerate(self.T):
            m8, m2 = torch.empty(th.shape, dtype=I8, device=DEV), torch.empty(th.shape, dtype=I8, device=DEV)
            ok(lib.bt_theta_m(P(th), P(m8), P(m2), n_(th)), "theta_m")
            M, S2 = gemm(mu[-1], m8), gemm(mu2[-1], m2)
            K = mu[-1].shape[1]
            sv = torch.empty_like(M)
            if l < 2:
                mo, mo2 = torch.empty(M.shape, dtype=I8, device=DEV), torch.empty(M.shape, dtype=I8, device=DEV)
                ok(lib.bt_epilogo(P(M), P(S2), I(K), P(mo), P(mo2), None, P(sv), I(0), n_(M)), "epilogo")
                mu.append(mo); mu2.append(mo2)
            else:
                so = torch.empty_like(M)
                ok(lib.bt_epilogo(P(M), P(S2), I(K), None, None, P(so), P(sv), I(1), n_(M)), "epilogo")
            m8s.append(m8); svs.append(sv)
        d = torch.empty_like(so)
        self.erros.zero_()
        ok(lib.bt_softmax_ce(P(so), P(y), P(d), P(self.erros), I(n), I(C), I(CP)), "softmax_ce")
        dM = torch.empty_like(d)
        ok(lib.bt_dmo(P(d), P(svs[2]), P(dM), n_(d)), "dmo")
        grads = [None, None, None]
        for l in (2, 1, 0):
            ok(lib.bt_maxabs(P(dM), P(self.mx), n_(dM)), "maxabs")
            dq = torch.empty(dM.shape, dtype=I8, device=DEV)
            ok(lib.bt_quant8(P(dM), P(self.mx), P(dq), n_(dM)), "quant8")
            grads[l] = gemm(tr8(dq), tr8(mu[l]))                                  # [N, K] = dq^T . mu
            if l > 0:
                dmu = gemm(dq, tr8(m8s[l]))                                       # [B, K] = dq . m8
                dM = torch.empty_like(dmu)
                ok(lib.bt_dtanh(P(dmu), P(mu[l]), P(svs[l - 1]), P(dM), n_(dmu)), "dtanh")
        for th, G in zip(self.T, grads):
            ok(lib.bt_maxabs(P(G), P(self.mx), n_(G)), "maxabs")
            ok(lib.bt_update(P(th), P(G), P(self.mx), I(self.shift), n_(th)), "update")
        if devolve:
            return d, grads
        return self.erros

    def pesos(self):
        W = []
        for th in self.T:
            w = torch.empty(th.shape, dtype=I8, device=DEV)
            ok(lib.bt_sinal_theta(P(th), P(w), n_(th)), "sinal_theta")
            W.append(w)
        return W


# ============================================================================ referência int64 (especificação)
def r_rnd(x, s):
    s = min(s, 31) if isinstance(s, int) else s
    if s <= 0:
        return x << (-s)
    return (x + ((1 << (s - 1)) - 1) + ((x >> s) & 1)) >> s


def r_bitlen(v):
    return int(abs(int(v))).bit_length()


def r_isqrt(n):
    n = n.clamp(min=1)
    bl = torch.zeros_like(n); t = n.clone()
    for s in (32, 16, 8, 4, 2, 1):
        m = (t >> s) > 0; bl = bl + s * m; t = torch.where(m, t >> s, t)
    bl = bl + (t > 0)                                       # bitlen
    x = torch.ones_like(n) << ((bl + 1) // 2)
    for _ in range(20):
        x = torch.minimum(x, (x + n // x.clamp(min=1)) >> 1)
    return x.clamp(min=1)


def r_exp(d):
    t = d.clamp(min=0, max=6000) * 23637
    n, f = t >> 22, (t & ((1 << 22) - 1)) >> 7
    p = torch.full_like(f, -1295)
    for c in (7556, -22645, 32765):
        p = ((p * f) >> 15) + c
    return torch.where(n > 30, torch.zeros_like(p), p >> n.clamp(max=30))


def r_tanh(z):
    e = r_exp(z.abs() >> 1)
    t = ((32768 - e) * 127 + ((32768 + e) >> 1)) // (32768 + e)
    return torch.where(z >= 0, t, -t)


def tdiv(a, b):
    return torch.div(a, b, rounding_mode="trunc")


def ref_passo(T, x8, y, shift):
    """um passo inteiro na especificação (int64, CPU). Devolve (d, grads, thetas novos)."""
    T = [t.cpu().to(I64) for t in T]
    x8, y = x8.cpu().to(I64), y.cpu()
    mu = [x8 * 127]
    mu2 = [r_rnd(mu[0] * mu[0], 7)]
    m8s, svs = [], []
    for l, th in enumerate(T):
        m8 = r_rnd(th, 3).clamp(-127, 127); m2 = r_rnd(m8 * m8, 7)
        M, S2 = mu[-1] @ m8.t(), mu2[-1] @ m2.t()
        K = mu[-1].shape[1]
        V = (K * 16384 - S2).clamp(min=1)
        sv = r_isqrt(V)
        r = tdiv(M << 3, sv)
        if l < 2:
            m = r_tanh((r * 205) >> 8)
            mu.append(m); mu2.append(r_rnd(m * m, 7))
        else:
            so = r
        m8s.append(m8); svs.append(sv)
    n = x8.shape[0]
    mx = so[:, :C].max(1, keepdim=True).values
    ex = r_exp((mx - so[:, :C]) >> 2)
    p = (ex << 15) // ex.sum(1, keepdim=True)
    d = torch.zeros_like(so)
    d[:, :C] = p
    d[torch.arange(n), y] -= 32768
    dM = tdiv(d << 7, svs[2])
    grads = [None, None, None]
    for l in (2, 1, 0):
        s = max(r_bitlen(dM.abs().max()) - 7, 0)
        dq = r_rnd(dM, s).clamp(-127, 127)
        grads[l] = dq.t() @ mu[l]
        if l > 0:
            dmu = dq @ m8s[l]
            t = r_rnd(dmu * (16384 - mu[l] * mu[l]), 14)
            dM = tdiv(r_rnd(t * 205, 8) * 128, svs[l - 1])
    novos = []
    for th, G in zip(T, grads):
        s0 = max(r_bitlen(G.abs().max()) - 11, 0)
        novos.append((th - r_rnd(r_rnd(G, s0), shift)).clamp(-1024, 1024))
    return d, grads, novos


def conferir():
    Wt = professor()
    x, y = dados(torch.Generator(device=DEV).manual_seed(5), B, Wt)
    obs = Observador(torch.Generator(device=DEV).manual_seed(1), 6)
    T0 = [t.clone() for t in obs.T]
    d, grads = obs.passo(x, y, devolve=True)
    rd, rg, rT = ref_passo(T0, x, y, 6)
    tudo_ok = True
    for nome, a, b in [("d", d, rd)] + [(f"grad camada {l}", grads[l], rg[l]) for l in range(3)] + \
                      [(f"theta camada {l} depois do update", obs.T[l], rT[l]) for l in range(3)]:
        a = a.cpu().to(I64)
        dif = int((a != b).sum())
        tudo_ok &= dif == 0
        print(f"  {nome:34s} {'ok' if dif == 0 else f'{dif} de {a.numel()} DIFERENTES'}  (max |valor| {int(b.abs().max())})")
    print("conferência bit a bit:", "TUDO IGUAL" if tudo_ok else "FALHOU")
    return tudo_ok


def treinar(semente, passos, shift):
    Wt = professor()
    xe, ye = dados(torch.Generator(device=DEV).manual_seed(999), B_EVAL, Wt)
    g = torch.Generator(device=DEV).manual_seed(semente)
    obs = Observador(g, shift)
    nome = f"fase1ck_ebp_sh{shift}_s{semente}"
    csv = open(os.path.join(AQUI, "logs", nome + ".csv"), "w")
    csv.write("passo,acuracia_binaria,erros_treino_lote,segundos,certos_c0,certos_c1,certos_c2,acima_900_c0,acima_900_c1,acima_900_c2\n")
    torch.cuda.synchronize()
    t0, ult = time.time(), 0
    for p in range(passos + 1):
        if p % EVAL_CADA == 0:
            acc = float((fwd_bin(xe, *obs.pesos()).argmax(1) == ye).float().mean())
            torch.cuda.synchronize()
            # pesos "certos": contador no limite (|theta| = 1024) e acima de 900 (média do peso > 0,88), por camada
            certos = [float((t[:C] if l == 2 else t).abs().eq(1024).float().mean()) for l, t in enumerate(obs.T)]
            altos = [float(((t[:C] if l == 2 else t).abs() >= 900).float().mean()) for l, t in enumerate(obs.T)]
            csv.write(f"{p},{acc:.4f},{ult},{time.time() - t0:.1f}," + ",".join(f"{v:.4f}" for v in certos + altos) + "\n"); csv.flush()
            if p % 1000 == 0:
                print(f"{nome} passo {p:5d} | acurácia da rede binária {acc:6.1%} | erros no lote {ult} | "
                      f"pesos certos (|theta|=1024) por camada {' '.join(f'{v:.1%}' for v in certos)} | {time.time() - t0:.1f}s", flush=True)
        if p == passos:
            break
        x, y = dados(g, B, Wt)
        ult = int(obs.passo(x, y)) if p % EVAL_CADA == EVAL_CADA - 1 else (obs.passo(x, y), ult)[1]
    csv.close()


if __name__ == "__main__":
    if sys.argv[1] == "conferir":
        sys.exit(0 if conferir() else 1)
    treinar(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 2000, int(sys.argv[4]) if len(sys.argv) > 4 else 6)
