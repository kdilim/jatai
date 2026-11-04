"""Fase 1d do Jataí-bit: o observador em redes 100% binárias de profundidade e largura variáveis, com contador de
16 ou 8 bits, contra uma régua float. Kernels próprios (libbit.so); o Python só aloca e chama.

Professor: rede binária sorteada e fixa (não aprende) N_IN -> [Ht] * Lt -> 16 classes. Aluno: N_IN -> [H] * L -> 16.
Avaliação: rede final 100% binária (pesos e ativações +-1) em 8192 entradas fixas.

Uso:
  python fase1d_escala.py conferir <L> <c16|c8|c4|c8mu4|c4mu4>      # um passo: kernels x referência int64, bit a bit
  python fase1d_escala.py binario <semente> <Lt> <Ht> <L> <H> <c16|c8|c4|c8mu4|c4mu4> <passos> [shift]
  python fase1d_escala.py float   <semente> <Lt> <Ht> <L> <H> <passos>          # régua (não é método do projeto)
Saída: logs/fase1d_<modo>_Lt<Lt>Ht<Ht>_L<L>H<H>[_c8|_c16_sh<shift>]_s<semente>.csv
"""
import os
import sys
import time

import ctypes

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fase1c_kernel as fk                # gemm, tr8, sinal, ok, lib, P, I, n_ e a referência int64
from fase1c_kernel import gemm, tr8, ok, lib, P, I, n_, I8, I16, I32, I64, DEV

N_IN, C, CP = 256, 16, 64
B, B_EVAL, EVAL_CADA = 1024, 8192, 250
AQUI = fk.AQUI


def sgn(x):
    return torch.where(x >= 0, 1, -1)


def professor(Lt, Ht):
    """camadas de largura Ht guardadas completadas até múltiplo de 64 (as multiplicações exigem K múltiplo de 64):
    os neurônios extras têm peso de entrada 0 e peso de saída 0, então não mudam a função do professor."""
    g = torch.Generator(device=DEV).manual_seed(12345)
    dims = [N_IN] + [Ht] * Lt
    W = []
    for i in range(Lt):
        w = torch.zeros(pad64(dims[i + 1]), pad64(dims[i]), dtype=I8, device=DEV)
        w[:dims[i + 1], :dims[i]] = sgn(torch.randint(0, 2, (dims[i + 1], dims[i]), generator=g, device=DEV) * 2 - 1).to(I8)
        W.append(w)
    Wo = torch.zeros(CP, pad64(Ht), dtype=I8, device=DEV)
    Wo[:C, :Ht] = sgn(torch.randint(0, 2, (C, Ht), generator=g, device=DEV) * 2 - 1).to(I8)
    return W + [Wo]


def fwd_bin(x8, W):
    h = x8
    for w in W[:-1]:
        a = gemm(h, w)
        h = torch.empty(a.shape, dtype=I8, device=DEV)
        ok(lib.bt_sinal8(P(a), P(h), n_(a)), "sinal8")
    return gemm(h, W[-1])[:, :C]


def pad64(Ht):
    return (Ht + 63) // 64 * 64


def dados(g, n, Wt):
    x = sgn(torch.randint(0, 2, (n, N_IN), generator=g, device=DEV) * 2 - 1).to(I8)
    return x, fwd_bin(x, Wt).argmax(1)


class Observador:
    def __init__(self, g, L, H, c8, shift, c4=False, mu4=False, norm_T=0, passos=None):
        # norm_T > 0: normalização entre camadas (r' = r * T / rms da linha, Q10; não muda a rede binária observada)
        # passos: lista com o passo (shift) de cada tensor, da 1a camada à saída; None = o mesmo shift em todas
        self.norm_T, self.passos_l = norm_T, passos
        # variância das camadas escondidas (só com normalização): exata | aprox (somas de linha/coluna, sem a 2a
        # multiplicação) | sem (r = M direto na normalização: sem raiz e sem divisão por elemento)
        self.var = os.environ.get("JB_VAR", "exata")
        assert self.var == "exata" or norm_T, "JB_VAR != exata exige JB_NORM"
        # pressão de observação: probabilidade máxima (de 256) no fim do treino; cresce linearmente com o passo
        self.pressao, self.passos_total = int(os.environ.get("JB_PRESSAO", "0")), 1
        # c4: contador de 4 bits (+-7) com update estocástico; mu4: médias das ativações com 4 bits (+-7 níveis)
        self.c8, self.c4, self.mu4, self.shift, self.n_passo = c8 or c4, c4, mu4, shift, 0
        dims = [N_IN] + [H] * L
        faixa = 4 if c4 else (64 if c8 else 512)        # metade da faixa do contador (peso médio +-0,5)
        tipo = I8 if (c8 or c4) else I16
        self.T = [torch.randint(-faixa, faixa + 1, (dims[i + 1], dims[i]), generator=g, device=DEV).to(tipo).contiguous()
                  for i in range(L)]
        to = torch.zeros(CP, H, dtype=I64, device=DEV)
        to[:C] = torch.randint(-faixa, faixa + 1, (C, H), generator=g, device=DEV)
        self.T.append(to.to(tipo).contiguous())
        self.erros = torch.zeros(1, dtype=I32, device=DEV)
        self.mx = torch.zeros(1, dtype=I32, device=DEV)

    def passo(self, x8, y, devolve=False):
        n, nl = x8.shape[0], len(self.T)
        mu = [(x8.to(I32) * 127).to(I8).contiguous()]
        mu2 = [torch.empty_like(mu[0])]
        ok(lib.bt_quad8(P(mu[0]), P(mu2[0]), n_(mu[0])), "quad8")
        m8s, svs, so, rmss = [], [], None, []
        for l, th in enumerate(self.T):
            m8, m2 = torch.empty(th.shape, dtype=I8, device=DEV), torch.empty(th.shape, dtype=I8, device=DEV)
            ok((lib.bt_theta_m_c4 if self.c4 else lib.bt_theta_m_c8 if self.c8 else lib.bt_theta_m)(P(th), P(m8), P(m2), n_(th)), "theta_m")
            M = gemm(mu[-1], m8)
            escondida = l < nl - 1
            S2 = gemm(mu2[-1], m2) if (self.var == "exata" or not escondida) else None
            K = mu[-1].shape[1]
            sv = torch.empty_like(M)
            if escondida:
                mo, mo2 = torch.empty(M.shape, dtype=I8, device=DEV), torch.empty(M.shape, dtype=I8, device=DEV)
                if self.norm_T:
                    if self.var == "sem":
                        r = M
                        sv.fill_(1)
                    elif self.var == "aprox":
                        r = torch.empty_like(M)
                        smu2 = torch.empty(M.shape[0], dtype=I32, device=DEV); sm2 = torch.empty(M.shape[1], dtype=I32, device=DEV)
                        ok(lib.bt_soma_linhas8(P(mu2[-1]), P(smu2), I(M.shape[0]), I(K)), "soma_linhas8")
                        ok(lib.bt_soma_linhas8(P(m2), P(sm2), I(M.shape[1]), I(K)), "soma_linhas8")
                        ok(lib.bt_epilogo_aprox(P(M), P(smu2), P(sm2), I(K), I(M.shape[1]), P(r), P(sv), n_(M)), "epilogo_aprox")
                    else:
                        r = torch.empty_like(M)
                        ok(lib.bt_epilogo(P(M), P(S2), I(K), None, None, P(r), P(sv), I(1), n_(M)), "epilogo")
                    if self.var == "sem":         # recíproco por linha (sem divisão por elemento)
                        inv = torch.empty(M.shape[0], dtype=I64, device=DEV); kk = torch.empty(M.shape[0], dtype=I32, device=DEV)
                        ok(lib.bt_norm_tanh_r(P(r), I(M.shape[0]), I(M.shape[1]), I(self.norm_T), P(mo), P(mo2), P(inv), P(kk), I(int(self.mu4))), "norm_tanh_r")
                        rmss.append((inv, kk))
                    else:
                        rms = torch.empty(M.shape[0], dtype=I32, device=DEV)
                        ok(lib.bt_norm_tanh(P(r), I(M.shape[0]), I(M.shape[1]), I(self.norm_T), P(mo), P(mo2), P(rms), I(int(self.mu4))), "norm_tanh")
                        rmss.append(rms)
                else:
                    ok(lib.bt_epilogo(P(M), P(S2), I(K), P(mo), P(mo2), None, P(sv), I(2 if self.mu4 else 0), n_(M)), "epilogo")
                mu.append(mo); mu2.append(mo2)
            else:
                so = torch.empty_like(M)
                ok(lib.bt_epilogo(P(M), P(S2), I(K), None, None, P(so), P(sv), I(1), n_(M)), "epilogo")
            m8s.append(m8); svs.append(sv)
        d = torch.empty_like(so)
        self.erros.zero_()
        ok(lib.bt_softmax_ce(P(so), P(y), P(d), P(self.erros), I(n), I(C), I(CP)), "softmax_ce")
        dM = torch.empty_like(d)
        ok(lib.bt_dmo(P(d), P(svs[-1]), P(dM), n_(d)), "dmo")
        grads = [None] * nl
        for l in reversed(range(nl)):
            ok(lib.bt_maxabs(P(dM), P(self.mx), n_(dM)), "maxabs")
            dq = torch.empty(dM.shape, dtype=I8, device=DEV)
            ok(lib.bt_quant8(P(dM), P(self.mx), P(dq), n_(dM)), "quant8")
            grads[l] = gemm(tr8(dq), tr8(mu[l]))
            if l > 0:
                dmu = gemm(dq, tr8(m8s[l]))
                dM = torch.empty_like(dmu)
                if self.norm_T and self.var == "sem":
                    inv, kk = rmss[l - 1]
                    ok(lib.bt_dtanh_norm_r(P(dmu), P(mu[l]), P(inv), P(kk), I(self.norm_T), I(dmu.shape[1]), P(dM), n_(dmu)), "dtanh_norm_r")
                elif self.norm_T:
                    ok(lib.bt_dtanh_norm(P(dmu), P(mu[l]), P(svs[l - 1]), P(rmss[l - 1]), I(self.norm_T), I(dmu.shape[1]), P(dM), n_(dmu)), "dtanh_norm")
                else:
                    ok(lib.bt_dtanh(P(dmu), P(mu[l]), P(svs[l - 1]), P(dM), n_(dmu)), "dtanh")
        for l, (th, G) in enumerate(zip(self.T, grads)):
            ok(lib.bt_maxabs(P(G), P(self.mx), n_(G)), "maxabs")
            sh = self.passos_l[l] if self.passos_l else self.shift
            if self.c4:
                ok(lib.bt_update_c4(P(th), P(G), P(self.mx), I(sh), ctypes.c_uint(self.semente_update(l)), n_(th)), "update_c4")
            else:
                ok((lib.bt_update_c8 if self.c8 else lib.bt_update)(P(th), P(G), P(self.mx), I(sh), n_(th)), "update")
        if self.pressao and (self.c8 or self.c4):
            prob = self.pressao * self.n_passo // max(self.passos_total, 1)
            for l, th in enumerate(self.T):
                ok(lib.bt_pressao(P(th), I(prob), I(7 if self.c4 else 127), ctypes.c_uint((self.semente_update(l) ^ 0x5A5A5A5A) & 0xFFFFFFFF), n_(th)), "pressao")
        self.n_passo += 1
        return (d, grads) if devolve else self.erros

    def semente_update(self, l):
        return (self.n_passo * 16 + l) & 0xFFFFFFFF

    def pesos(self):
        W = []
        for th in self.T:
            w = torch.empty(th.shape, dtype=I8, device=DEV)
            ok((lib.bt_sinal_theta_c8 if self.c8 else lib.bt_sinal_theta)(P(th), P(w), n_(th)), "sinal_theta")
            W.append(w)
        return W


# ============================================================================ referência int64 (genérica)
M32 = 0xFFFFFFFF


def r_isqrt64(n):
    """igual ao isqrt64 do kernel: chute 2^ceil(bits/2) e Newton até parar de descer."""
    out = torch.empty_like(n)
    for idx, v in enumerate(n.view(-1).tolist()):
        if v < 1:
            out.view(-1)[idx] = 1
            continue
        x = 1 << ((v.bit_length() + 1) // 2)
        for _ in range(40):
            y = (x + v // x) >> 1
            if y >= x:
                break
            x = y
        out.view(-1)[idx] = max(x, 1)
    return out


def r_hash32(x):
    x = x & M32
    x = x ^ (x >> 16); x = (x * 0x7feb352d) & M32
    x = x ^ (x >> 15); x = (x * 0x846ca68b) & M32
    return x ^ (x >> 16)


def ref_passo(T, x8, y, shift, c8, c4=False, mu4=False, semente_l=None, norm_T=0, passos=None, var="exata"):
    T = [t.cpu().to(I64) for t in T]
    x8, y = x8.cpu().to(I64), y.cpu()
    mu = [x8 * 127]
    mu2 = [fk.r_rnd(mu[0] * mu[0], 7)]
    m8s, svs, nl, rmss = [], [], len(T), []
    for l, th in enumerate(T):
        m8 = th * 18 if c4 else (th if c8 else fk.r_rnd(th, 3).clamp(-127, 127))
        m2 = fk.r_rnd(m8 * m8, 7)
        M, S2 = mu[-1] @ m8.t(), mu2[-1] @ m2.t()
        K = mu[-1].shape[1]
        if l < nl - 1 and var == "sem":
            sv = torch.ones_like(M)
            r = M
        else:
            if l < nl - 1 and var == "aprox":
                S2 = fk.tdiv(mu2[-1].sum(1, keepdim=True) * m2.sum(1)[None, :], torch.full_like(M, K))
            sv = fk.r_isqrt((K * 16384 - S2).clamp(min=1))
            r = fk.tdiv(M << 3, sv)
        if l < nl - 1:
            if norm_T:
                rms = r_isqrt64(fk.tdiv((r * r).sum(1, keepdim=True), torch.full((r.shape[0], 1), r.shape[1], dtype=I64)))
                if var == "sem":
                    k = torch.tensor([[int(v).bit_length() + 20] for v in rms.view(-1).tolist()], dtype=I64)
                    inv = torch.tensor([[(1 << int(kv)) // int(rv)] for kv, rv in zip(k.view(-1).tolist(), rms.view(-1).tolist())], dtype=I64)
                    rmss.append((inv, k))
                    r = (r * norm_T * inv) >> k
                else:
                    rmss.append(rms)
                    r = fk.tdiv(r * norm_T, rms)
            m = fk.r_tanh((r * 205) >> 8)
            if mu4:
                m = fk.tdiv(m * 7 + torch.where(m >= 0, 63, -63), torch.full_like(m, 127)) * 18
            mu.append(m); mu2.append(fk.r_rnd(m * m, 7))
        else:
            so = r
        m8s.append(m8); svs.append(sv)
    n = x8.shape[0]
    mx = so[:, :C].max(1, keepdim=True).values
    ex = fk.r_exp((mx - so[:, :C]) >> 2)
    d = torch.zeros_like(so)
    d[:, :C] = (ex << 15) // ex.sum(1, keepdim=True)
    d[torch.arange(n), y] -= 32768
    dM = fk.tdiv(d << 7, svs[-1])
    grads = [None] * nl
    for l in reversed(range(nl)):
        dq = fk.r_rnd(dM, max(fk.r_bitlen(dM.abs().max()) - 7, 0)).clamp(-127, 127)
        grads[l] = dq.t() @ mu[l]
        if l > 0:
            t = fk.r_rnd((dq @ m8s[l]) * (16384 - mu[l] * mu[l]), 14)
            if norm_T and var == "sem":
                inv, k = rmss[l - 1]
                dM = ((((fk.r_rnd(t * 205, 8) * inv) >> 10) * 128 * norm_T) >> (k - 10))
            elif norm_T:
                dM = fk.tdiv(fk.tdiv(fk.r_rnd(t * 205, 8) * 128 * norm_T, rmss[l - 1]), svs[l - 1])
            else:
                dM = fk.tdiv(fk.r_rnd(t * 205, 8) * 128, svs[l - 1])
    novos = []
    for l, (th, G) in enumerate(zip(T, grads)):
        if c4:
            g = fk.r_rnd(G, max(fk.r_bitlen(G.abs().max()) - 8, 0))
            q = g >> shift
            resto = g - (q << shift)
            idx = torch.arange(th.numel(), dtype=I64).reshape(th.shape)
            u = r_hash32(idx * 0x9E3779B1 + semente_l(l)) & ((1 << shift) - 1)
            novos.append((th - (q + (u < resto).to(I64))).clamp(-7, 7))
            continue
        s0 = max(fk.r_bitlen(G.abs().max()) - (8 if c8 else 11), 0)
        lim = 127 if c8 else 1024
        novos.append((th - fk.r_rnd(fk.r_rnd(G, s0), passos[l] if passos else shift)).clamp(-lim, lim))
    return d, grads, novos


def conferir(L, modo):
    c8, c4, mu4 = modo == "c8", modo.startswith("c4"), modo.endswith("mu4")
    Wt = professor(L, 64)
    x, y = dados(torch.Generator(device=DEV).manual_seed(5), B, Wt)
    nT, pl = opcoes(L)
    obs = Observador(torch.Generator(device=DEV).manual_seed(1), L, 128, c8, 6, c4, mu4, nT, pl)
    T0 = [t.clone() for t in obs.T]
    sem = obs.semente_update
    d, grads = obs.passo(x, y, devolve=True)
    obs.n_passo -= 1                                  # a referência usa as sementes do mesmo passo
    rd, rg, rT = ref_passo(T0, x, y, 6, c8, c4, mu4, sem, nT, pl, obs.var)
    tudo = True
    for nome, a, b in [("d", d, rd)] + [(f"grad {l}", grads[l], rg[l]) for l in range(len(grads))] + \
                      [(f"theta {l}", obs.T[l], rT[l]) for l in range(len(grads))]:
        dif = int((a.cpu().to(I64) != b).sum())
        tudo &= dif == 0
        print(f"  {nome:10s} {'ok' if dif == 0 else f'{dif} DIFERENTES'} (max |valor| {int(b.abs().max())})")
    print(f"conferência L={L} {modo}:", "TUDO IGUAL" if tudo else "FALHOU")
    return tudo


def opcoes(L):
    """JB_NORM = T (Q10) liga a normalização entre camadas; JB_PASSOS = lista 'a,b,...' com o passo de cada tensor
    (L camadas escondidas + saída) ou 'escada:<base>:<delta>' (passo = base + delta * distância da saída)."""
    nT = int(os.environ.get("JB_NORM", "0"))
    p = os.environ.get("JB_PASSOS", "")
    if not p:
        return nT, None
    if p.startswith("escada:"):
        _, base, delta = p.split(":")
        return nT, [int(base) + int(delta) * (L - l) for l in range(L + 1)]
    lista = [int(v) for v in p.split(",")]
    assert len(lista) == L + 1, f"JB_PASSOS precisa de {L + 1} valores"
    return nT, lista


def treinar_binario(semente, Lt, Ht, L, H, modo, passos, shift):
    """modo: c16 | c8 | c4 | c8mu4 | c4mu4 (contador de 16/8/4 bits; mu4 = médias das ativações com 4 bits)"""
    Wt = professor(Lt, Ht)
    xe, ye = dados(torch.Generator(device=DEV).manual_seed(999), B_EVAL, Wt)
    g = torch.Generator(device=DEV).manual_seed(semente)
    nT, pl = opcoes(L)
    obs = Observador(g, L, H, modo.startswith("c8"), shift, modo.startswith("c4"), modo.endswith("mu4"), nT, pl)
    obs.passos_total = passos
    pr = int(os.environ.get("JB_PRESSAO", "0"))
    extra = (f"_var{obs.var}" if obs.var != "exata" else "") + (f"_norm{nT}" if nT else "") + ("_passos" + "-".join(map(str, pl)) if pl else "") + (f"_pressao{pr}" if pr else "")
    nome = f"fase1d_bin_Lt{Lt}Ht{Ht}_L{L}H{H}_{modo}_sh{shift}{extra}_s{semente}"
    return roda(nome, passos, lambda: (fwd_bin(xe, obs.pesos()).argmax(1) == ye).float().mean(),
                lambda: obs.passo(*dados(g, B, Wt)), ye)


def treinar_float(semente, Lt, Ht, L, H, passos):
    Wt = professor(Lt, Ht)
    xe, ye = dados(torch.Generator(device=DEV).manual_seed(999), B_EVAL, Wt)
    g = torch.Generator(device=DEV).manual_seed(semente)
    torch.manual_seed(semente)
    camadas, d0 = [], N_IN
    for _ in range(L):
        camadas += [torch.nn.Linear(d0, H), torch.nn.Tanh()]; d0 = H
    rede = torch.nn.Sequential(*camadas, torch.nn.Linear(d0, C)).to(DEV)
    opt = torch.optim.Adam(rede.parameters(), lr=1e-3)

    def passo():
        x, y = dados(g, B, Wt)
        perda = torch.nn.functional.cross_entropy(rede(x.float()), y)
        opt.zero_grad(); perda.backward(); opt.step()

    def avalia():
        with torch.no_grad():
            return (rede(xe.float()).argmax(1) == ye).float().mean()
    return roda(f"fase1d_float_Lt{Lt}Ht{Ht}_L{L}H{H}_s{semente}", passos, avalia, passo, ye)


def roda(nome, passos, avalia, passo, ye):
    maior = float(torch.bincount(ye, minlength=C).max()) / B_EVAL
    csv = open(os.path.join(AQUI, "logs", nome + ".csv"), "w")
    csv.write("passo,acuracia,segundos\n")
    torch.cuda.synchronize()
    t0, acc = time.time(), 0.0
    for p in range(passos + 1):
        if p % EVAL_CADA == 0:
            acc = float(avalia())
            csv.write(f"{p},{acc:.4f},{time.time() - t0:.1f}\n"); csv.flush()
        if p == passos:
            break
        passo()
    csv.close()
    print(f"{nome}: acurácia final {acc:6.1%} (chute constante {maior:.1%}) | {time.time() - t0:.1f}s", flush=True)
    return acc


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "conferir":
        sys.exit(0 if conferir(int(sys.argv[2]), sys.argv[3]) else 1)
    a = sys.argv
    if m == "binario":
        treinar_binario(int(a[2]), int(a[3]), int(a[4]), int(a[5]), int(a[6]), a[7], int(a[8]), int(a[9]) if len(a) > 9 else 6)
    elif m == "float":
        treinar_float(int(a[2]), int(a[3]), int(a[4]), int(a[5]), int(a[6]), int(a[7]))
