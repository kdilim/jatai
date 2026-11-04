"""Operacoes do treino inteiro em tres versoes com a mesma interface:

  OpsCuda       chama os kernels de libjatai.so (o metodo).
  OpsRef        referencia em Python/PyTorch com inteiros de 64 bits: a especificacao de cada conta,
                escrita do jeito mais direto possivel. O CUDA tem que bater bit a bit com ela.
  OpsConferido  roda as duas em cada operacao (mesmas entradas) e para no primeiro bit diferente,
                dizendo operacao, trecho, posicao e os dois valores.

Formatos: Bloco = (m int8 [R,C], E int8 [R,C/128]); Cru = (acc int32 [R,C], e int16 [R,C]).
"""
import ctypes
import math
import os

import torch

BLK = 128
E_ZERO = -127
SEM = -100000
GA = 4                 # bits de guarda de q/k/v/dO alinhados dentro da cabeca
# Variante de especificacao do backward da atencao: shifts de gP e dS por limites superiores (3 passadas em vez
# de 5). Lida pelas duas versoes (CUDA e referencia), entao a conferencia bit a bit vale nos dois modos.
ATENCAO_LIMITE = False
I64 = torch.int64
AQUI = os.path.dirname(os.path.abspath(__file__))


def g_rms(D):
    nb = D // BLK
    return 2 * ((8 - int(math.log2(nb))) // 2), 14 - int(math.log2(D))   # (G da RMSNorm, G do backward)


# ============================================================================ referencia (int64)
def bitlen(x):
    ax = x.abs().to(I64)
    bl = torch.zeros_like(ax)
    for s in (32, 16, 8, 4, 2, 1):
        bl = bl + s * ((ax >> (bl + s)) > 0)
    return bl + (ax > 0)


def rnd(x, s):
    """x * 2^-s, empate para o par; s <= 0 e shift a esquerda."""
    x = x.to(I64)
    s = torch.as_tensor(s, dtype=I64, device=x.device).expand_as(x)
    pos, neg = s.clamp(0, 62), (-s).clamp(0, 62)
    meio = torch.where(pos > 0, torch.bitwise_left_shift(torch.ones_like(pos), (pos - 1).clamp(min=0)), 0)
    q = (x + meio) >> pos
    resto = x + meio - (q << pos)
    q = q - ((resto == 0) & (pos > 0) & ((q & 1) == 1)).to(I64)
    return torch.where(pos > 0, q, x << neg)


def expande(E, b=BLK):
    return E.to(I64).repeat_interleave(b, dim=-1)


def quant(v, e):
    """valores int64 com expoente por elemento -> (m int8, E int8 por bloco de 128 da ultima dimensao)."""
    v = v.to(I64)
    e = torch.as_tensor(e, dtype=I64, device=v.device).expand_as(v)
    forma = v.shape[:-1] + (v.shape[-1] // BLK, BLK)
    topo = torch.where(v != 0, e + bitlen(v), torch.full_like(v, SEM)).reshape(forma).amax(-1)
    zero = topo == SEM
    E = topo - 7
    q = rnd(v, expande(E) - e)
    E = E + (q.abs().reshape(forma).amax(-1) > 127).to(I64)
    q = rnd(v, expande(E) - e)
    zero = zero | (E < -126)                 # underflow: bloco abaixo de 2^-119 vira zero
    E = torch.where(zero, torch.full_like(E, E_ZERO), E)
    q = torch.where(expande(zero.to(I64)) == 1, 0, q)
    return q.clamp(-127, 127).to(torch.int8).contiguous(), E.to(torch.int8).contiguous()


def exp_int(d):
    t = d.clamp(max=6000) * 23637
    n, f = t >> 22, (t & ((1 << 22) - 1)) >> 7
    p = torch.full_like(f, -1295)
    for c in (7556, -22645, 32765):
        p = ((p * f) >> 15) + c
    return torch.where(n > 30, torch.zeros_like(p), p >> n.clamp(max=30))


def isqrt(n):
    n = n.clamp(min=0)
    x = torch.bitwise_left_shift(torch.ones_like(n), (bitlen(n) + 1) // 2)
    for _ in range(10):
        x = torch.minimum(x, (x + n // x.clamp(min=1)) >> 1)
    return x.clamp(min=1)


class OpsRef:
    nome = "ref"

    def gemm(self, A, B, cru=False):
        (a, ea), (b, eb) = A, B
        M, K = a.shape
        RA, RB = ea.to(I64).amax(1), eb.to(I64).amax(1)
        acc = torch.zeros(M, b.shape[0], dtype=I64, device=a.device)
        for j in range(K // BLK):
            sl = slice(j * BLK, (j + 1) * BLK)
            p = torch._int_mm(a[:, sl].contiguous(), b[:, sl].t().contiguous()).to(I64)
            s = (RA - ea[:, j].to(I64))[:, None] + (RB - eb[:, j].to(I64))[None, :]
            acc += rnd(p << 4, s)
        base = RA[:, None] + RB[None, :] - 4
        if cru:
            return acc.to(torch.int32), base.clamp(min=-30000).to(torch.int16)
        return quant(acc, base)

    # operacoes fundidas: por definicao, a composicao das operacoes separadas (o CUDA faz numa GEMM so)
    def gemm_soma(self, A, B, X):
        return self.soma(X, self.gemm(A, B))

    def gemm_relu2(self, A, B):
        h = self.gemm(A, B)
        return h, self.relu2(h)

    def gemm_relu2_back(self, A, B, Hh):
        return self.relu2_back(self.gemm(A, B), Hh)

    def transpoe(self, X):
        m, E = X
        return quant(m.to(I64).t().contiguous(), expande(E).t().contiguous())

    def quant(self, C):
        return quant(C[0], C[1].to(I64))

    def soma(self, X, Y):
        (xm, xE), (ym, yE) = X, Y
        ec = torch.maximum(xE.to(I64), yE.to(I64)) - 21
        v = rnd(xm, expande(ec) - expande(xE)) + rnd(ym, expande(ec) - expande(yE))
        return quant(v, expande(ec))

    def relu2(self, Hh):
        m, E = Hh
        r = m.to(I64).clamp(min=0)
        return quant(r * r, 2 * expande(E))

    def relu2_back(self, Gg, Hh):
        (g, gE), (h, hE) = Gg, Hh
        return quant(g.to(I64) * h.to(I64).clamp(min=0) * 2, expande(gE) + expande(hE))

    def embed(self, W, tok):
        m, E = W
        return m[tok.long()].contiguous(), E[tok.long()].contiguous()

    def um_quente(self, tok, V):
        M = tok.numel()
        U = torch.zeros(M, V, dtype=torch.int8, device=tok.device)
        U[torch.arange(M, device=tok.device), tok.long()] = 1
        return U, torch.zeros(M, V // BLK, dtype=torch.int8, device=tok.device)

    def rms(self, X):
        m, E = X
        Mr, D = m.shape
        G, _ = g_rms(D)
        x, eb = m.to(I64), E.to(I64)
        sq = (x * x).reshape(Mr, D // BLK, BLK).sum(-1)
        emax = eb.amax(1, keepdim=True)
        ss = rnd(sq << G, 2 * (emax - eb)).sum(1)
        r = isqrt(ss >> int(math.log2(D)))
        k = bitlen(r) + 10
        inv = torch.bitwise_left_shift(torch.ones_like(k), k) // r
        Y = quant(x * inv[:, None], expande(eb) - k[:, None] - emax + G // 2)
        return Y, (r.to(torch.int32), emax[:, 0].to(torch.int32))

    def rms_back(self, Gg, Y, cache):
        (g, gE), (y, yE) = Gg, Y
        r, emax = cache[0].to(I64), cache[1].to(I64)
        Mr, D = g.shape
        G, GB = g_rms(D)
        g, y, gE, yE = g.to(I64), y.to(I64), gE.to(I64), yE.to(I64)
        smax = (gE + yE).amax(1, keepdim=True)
        S = rnd((g * y) << GB, expande(smax - (gE + yE))).sum(1)
        eS = smax[:, 0] - GB - int(math.log2(D))
        sh = (bitlen(S) - 15).clamp(min=0)
        S, eS = rnd(S, sh), eS + sh
        ec = torch.maximum(gE + 7, yE + eS[:, None] + 22) - 28
        dentro = rnd(g, expande(ec - gE)) - rnd(y * S[:, None], expande(ec - (yE + eS[:, None])))
        sh2 = (bitlen(dentro) - 18).clamp(min=0)
        dentro = rnd(dentro, sh2)
        k = bitlen(r) + 10
        inv = torch.bitwise_left_shift(torch.ones_like(k), k) // r
        return quant(dentro * inv[:, None], expande(ec) + sh2 - k[:, None] - (emax[:, None] - G // 2))

    @staticmethod
    def _alinha(m, E, Bt, T, H, ld, col0):
        x = m.to(I64).reshape(Bt, T, ld)[:, :, col0:col0 + 4 * H].reshape(Bt, T, H, 4).permute(0, 2, 1, 3)
        cols = (col0 + torch.arange(H, device=m.device) * 4) // BLK
        eh = E.to(I64).reshape(Bt, T, -1)[:, :, cols].permute(0, 2, 1)
        emx = eh.amax(-1, keepdim=True)
        return rnd(x << GA, (emx - eh)[..., None]), emx - GA          # (Bt,H,T,4), (Bt,H,1)

    @staticmethod
    def _prod(a, b):  # (..., T, 4) x (..., T, 4) -> (..., T, T) exato
        return sum(a[..., :, None, d] * b[..., None, :, d] for d in range(4))

    def atencao(self, QKV, Bt, T, H):
        m, E = QKV
        D = m.shape[1] // 3
        qa, eq = self._alinha(m, E, Bt, T, H, 3 * D, 0)
        ka, ek = self._alinha(m, E, Bt, T, H, 3 * D, D)
        va, ev = self._alinha(m, E, Bt, T, H, 3 * D, 2 * D)
        dev = m.device
        i = torch.arange(T, device=dev)
        dist = i[:, None] - i[None, :]
        mascara = dist < 0
        incl = (8 - torch.arange(H, device=dev) // 8)[:, None, None]
        u = rnd(self._prod(qa, ka), -(eq + ek - 1 + 8)[..., None]) - (dist.clamp(min=0)[None] << incl)
        u = torch.where(mascara, -(2**31 - 1), u)
        mx = u.amax(-1, keepdim=True)
        ex = torch.where(mascara, 0, exp_int(mx - u) >> 3)
        inv = (1 << 28) // ex.sum(-1, keepdim=True)
        P = torch.where(mascara, 0, ((ex * inv + (1 << 19)) >> 20).clamp(max=255))
        o = torch.stack([(P * va[..., None, :, d]).sum(-1) for d in range(4)], -1)          # (Bt,H,T,4)
        oacc = o.permute(0, 2, 1, 3).reshape(Bt * T, D)
        oe = (ev - 8)[..., None].expand(Bt, H, T, 4).permute(0, 2, 1, 3).reshape(Bt * T, D)
        Pt = P.transpose(-1, -2).reshape(Bt * H, T, T)          # guardada transposta: [j][i]
        return (oacc.to(torch.int32).contiguous(), oe.to(torch.int16).contiguous()), Pt.to(torch.uint8).contiguous()

    def atencao_back(self, GO, QKV, P, Bt, T, H):
        (gm, gE), (m, E) = GO, QKV
        D = gm.shape[1]
        goa, ego = self._alinha(gm, gE, Bt, T, H, D, 0)
        qa, eq = self._alinha(m, E, Bt, T, H, 3 * D, 0)
        ka, ek = self._alinha(m, E, Bt, T, H, 3 * D, D)
        va, ev = self._alinha(m, E, Bt, T, H, 3 * D, 2 * D)
        Pl = P.to(I64).reshape(Bt, H, T, T).transpose(-1, -2)    # P[i][j]
        gP = self._prod(goa, va)
        causal = torch.arange(T, device=gm.device)[:, None] >= torch.arange(T, device=gm.device)[None, :]
        if ATENCAO_LIMITE:     # |gP_ij| <= max_i ||dO_i||_inf * max_j ||v_j||_1
            lim = goa.abs().amax((-2, -1)) * va.abs().sum(-1).amax(-1)
            sh = (bitlen(lim)[..., None, None] - 15).clamp(min=0)
        else:
            sh = (bitlen(torch.where(causal, gP.abs(), 0).amax((-2, -1), keepdim=True)) - 15).clamp(min=0)   # so j <= i
        gP = rnd(gP, sh)
        egp = ego + ev + sh[..., 0]
        sl = rnd((Pl * gP).sum(-1, keepdim=True), 8)
        dS = Pl * (gP - sl)
        if ATENCAO_LIMITE:     # |dS| <= max P * (max |gp| + max |s_lin|), maximos causais
            lim2 = Pl.amax((-2, -1)) * (torch.where(causal, gP.abs(), 0).amax((-2, -1)) + sl.abs().amax((-2, -1)))
            sh2 = (bitlen(lim2)[..., None, None] - 12).clamp(min=0)
        else:
            sh2 = (bitlen(dS.abs().amax((-2, -1), keepdim=True)) - 12).clamp(min=0)
        dS = rnd(dS, sh2)
        eds = egp - 9 + sh2[..., 0]
        gq = torch.stack([(dS * ka[..., None, :, d]).sum(-1) for d in range(4)], -1)
        gk = torch.stack([(dS * qa[..., :, None, d]).sum(-2) for d in range(4)], -1)
        gv = torch.stack([(Pl * goa[..., :, None, d]).sum(-2) for d in range(4)], -1)

        def lin(x):
            return x.permute(0, 2, 1, 3).reshape(Bt * T, D)

        def exp_(e):
            return lin(e[..., None].expand(Bt, H, T, 4))
        acc = torch.cat([lin(gq), lin(gk), lin(gv)], 1)
        e = torch.cat([exp_(eds + ek), exp_(eds + eq), exp_(ego - 8)], 1)
        return acc.to(torch.int32).contiguous(), e.to(torch.int16).contiguous()

    def softmax_ce(self, L, alvo):
        acc, e = L
        M = acc.shape[0]
        u = rnd(acc, -(e.to(I64) + 8))
        umax = u.amax(1, keepdim=True)
        ex = exp_int(umax - u) >> 3
        s = ex.sum(1, keepdim=True)
        v = (ex * ((1 << 28) // s) + (1 << 15)) >> 16
        lin = torch.arange(M, device=acc.device)
        v[lin, alvo.long()] -= 4096
        g = quant(v, -12 - int(math.log2(M)))
        return g, (umax[:, 0] - u[lin, alvo.long()]).to(torch.int32), s[:, 0].to(torch.int32)

    def momento(self, Gc, Mom):
        (g, ge), (mm, mE) = Gc, Mom
        g, ge = g.to(I64), ge.to(I64)
        em = expande(mE) - 3
        ec = torch.maximum(em + 10, ge + bitlen(g)) - 28
        return quant(rnd(mm.to(I64) * 7, ec - em) + rnd(g, ec - ge), ec)

    def peso(self, W, Mom, R, k, ef):
        (w, wE), (mm, mE), (r, rE) = W, Mom, R
        ew, em = expande(wE), expande(mE) - k
        er = expande(rE) if ef else torch.full_like(ew, E_ZERO)
        ec = torch.maximum(torch.maximum(ew + 7, em + 7), er + 7) - 27
        alvo = rnd(w, ec - ew) - rnd(mm, ec - em) + (rnd(r, ec - er) if ef else 0)
        w2, wE2 = quant(alvo, ec)
        if not ef:
            return (w2, wE2), R
        dif = alvo - torch.where(w2 == 0, 0, rnd(w2, ec - expande(wE2)))
        return (w2, wE2), quant(dif, ec)


# ============================================================================ CUDA
class OpsCuda:
    nome = "cuda"

    def __init__(self, reg):
        self.reg = reg
        self.lib = ctypes.CDLL(os.path.join(AQUI, "libjatai.so"))
        self.lib.jt_stream.argtypes = [ctypes.c_void_p]
        for f in ("jt_gemm", "jt_gemm_f", "jt_quant", "jt_soma", "jt_relu2", "jt_relu2_back", "jt_momento", "jt_peso", "jt_transpoe",
                  "jt_embed", "jt_um_quente", "jt_rms", "jt_rms_back", "jt_atencao", "jt_atencao_back", "jt_softmax_ce", "jt_erro_le",
                  "jt_erro_limpa", "jt_sincroniza", "jt_para16", "jt_para8", "jt_soma16"):
            getattr(self.lib, f).restype = ctypes.c_int
        self.lib.jt_erro_le.argtypes = [ctypes.c_void_p]
        reg.lib = self.lib
        self.lib.jt_erro_limpa()
        if os.environ.get("JATAI_RES_EST") == "1":      # soma residual com arredondamento estocástico (RELATORIO 7.7)
            assert self.lib.jt_res_estocastico(1) == 0
        OpsCuda.lib_global = self.lib

    @staticmethod
    def _a(x):
        if torch.is_tensor(x):
            return ctypes.c_void_p(x.data_ptr()) if x.numel() else ctypes.c_void_p(0)
        if x is None:
            return ctypes.c_void_p(0)
        return ctypes.c_longlong(x) if isinstance(x, int) and abs(x) > 2**31 - 1 else ctypes.c_int(x)

    def _lanca(self, op, fn, args, entradas, forma, longos=()):
        kid = self.reg.registra(op, entradas, forma)
        conv = [ctypes.c_longlong(a) if i in longos else self._a(a) for i, a in enumerate(args)]
        r = getattr(self.lib, fn)(*conv, ctypes.c_int(kid))
        self.reg.depois(kid, r)

    def _novo_bloco(self, R, C, dev):
        return torch.empty(R, C, dtype=torch.int8, device=dev), torch.empty(R, C // BLK, dtype=torch.int8, device=dev)

    def gemm(self, A, B, cru=False):
        reg = self.reg
        reg.confere_bloco("A", A); reg.confere_bloco("B", B)
        (a, ea), (b, eb) = A, B
        M, K = a.shape
        N = b.shape[0]
        reg.exige(b.shape[1] == K, f"gemm: K diferente, A {tuple(a.shape)} x B {tuple(b.shape)}")
        reg.exige(M % 128 == 0 and N % 128 == 0 and K % 128 == 0 and K <= 4096,
                  f"gemm: M, N, K precisam ser multiplos de 128 e K <= 4096 (M={M}, N={N}, K={K})")
        if cru:
            C = (torch.empty(M, N, dtype=torch.int32, device=a.device), torch.empty(M, N, dtype=torch.int16, device=a.device))
            self._lanca("gemm_cru", "jt_gemm", (a, ea, b, eb, None, None, C[0], C[1], M, N, K, 1), dict(A=a, EA=ea, B=b, EB=eb), (M, N))
        else:
            C = self._novo_bloco(M, N, a.device)
            self._lanca("gemm", "jt_gemm", (a, ea, b, eb, C[0], C[1], None, None, M, N, K, 0), dict(A=a, EA=ea, B=b, EB=eb), (M, N))
        return C

    def _gemm_fundida(self, op, modo, A, B, X=None, saida2=False):
        reg = self.reg
        reg.confere_bloco("A", A); reg.confere_bloco("B", B)
        (a, ea), (b, eb) = A, B
        M, K = a.shape
        N = b.shape[0]
        reg.exige(b.shape[1] == K and M % 128 == 0 and N % 128 == 0 and K % 128 == 0 and K <= 4096,
                  f"{op}: formas A {tuple(a.shape)} x B {tuple(b.shape)} (multiplos de 128, K <= 4096)")
        ent = dict(A=a, EA=ea, B=b, EB=eb)
        if X is not None:
            reg.confere_bloco("X", X)
            reg.exige(tuple(X[0].shape) == (M, N), f"{op}: X com forma {tuple(X[0].shape)}, esperado {(M, N)}")
            ent.update(X=X[0], XE=X[1])
        C = self._novo_bloco(M, N, a.device)
        C2 = self._novo_bloco(M, N, a.device) if saida2 else (None, None)
        self._lanca(op, "jt_gemm_f", (a, ea, b, eb, C[0], C[1], None, None, X[0] if X is not None else None,
                                      X[1] if X is not None else None, C2[0], C2[1], M, N, K, modo), ent, (M, N))
        return (C, C2) if saida2 else C

    def gemm_soma(self, A, B, X):
        return self._gemm_fundida("gemm_soma", 2, A, B, X=X)

    def gemm_relu2(self, A, B):
        return self._gemm_fundida("gemm_relu2", 3, A, B, saida2=True)

    def gemm_relu2_back(self, A, B, Hh):
        return self._gemm_fundida("gemm_relu2_back", 4, A, B, X=Hh)

    def transpoe(self, X):
        self.reg.confere_bloco("X", X)
        m, E = X
        R, C = m.shape
        self.reg.exige(R % 128 == 0, f"transpoe: linhas {R} nao multiplo de 128")
        Y = self._novo_bloco(C, R, m.device)
        self._lanca("transpoe", "jt_transpoe", (m, E, Y[0], Y[1], R, C), dict(X=m, EX=E), (C, R))
        return Y

    def quant(self, Cc):
        self.reg.confere_cru("C", Cc)
        acc, e = Cc
        R, C = acc.shape
        Y = self._novo_bloco(R, C, acc.device)
        self._lanca("quant", "jt_quant", (acc, e, Y[0], Y[1], R * C // BLK), dict(acc=acc, e=e), (R, C), longos=(4,))
        return Y

    def _elem(self, op, fn, ins, nomes):
        for n, t in zip(nomes, ins):
            self.reg.confere_bloco(n, t)
        m0 = ins[0][0]
        for n, t in zip(nomes[1:], ins[1:]):
            self.reg.exige(t[0].shape == m0.shape, f"{op}: '{n}' com forma {tuple(t[0].shape)}, esperado {tuple(m0.shape)}")
        Y = self._novo_bloco(*m0.shape, m0.device)
        args = [x for t in ins for x in t] + [Y[0], Y[1], m0.numel() // BLK]
        ent = {f"{n}{s}": x for n, t in zip(nomes, ins) for s, x in zip(("", "E"), t)}
        self._lanca(op, fn, args, ent, tuple(m0.shape), longos=(len(args) - 1,))
        return Y

    def soma(self, X, Y):
        return self._elem("soma", "jt_soma", (X, Y), ("x", "y"))

    # ---- esteira residual de 16 bits (E1): (mantissa int16, expoente int8 por bloco de 128)
    def _r16(self, op, fn, args, ent, forma, saida16):
        m0 = args[0]
        R, C = forma
        Y = (torch.empty(R, C, dtype=torch.int16 if saida16 else torch.int8, device=m0.device),
             torch.empty(R, C // BLK, dtype=torch.int8, device=m0.device))
        a = list(args) + [Y[0], Y[1], R * C // BLK]
        self._lanca(op, fn, a, ent, forma, longos=(len(a) - 1,))
        return Y

    def para16(self, X):
        self.reg.confere_bloco("x", X)
        return self._r16("para16", "jt_para16", X, dict(x=X[0], xE=X[1]), tuple(X[0].shape), True)

    def para8(self, X16):
        return self._r16("para8", "jt_para8", X16, dict(x16=X16[0], xE=X16[1]), tuple(X16[0].shape), False)

    def soma16(self, X16, Y):
        self.reg.confere_bloco("y", Y)
        self.reg.exige(tuple(Y[0].shape) == tuple(X16[0].shape), f"soma16: formas {tuple(X16[0].shape)} e {tuple(Y[0].shape)}")
        return self._r16("soma16", "jt_soma16", (*X16, *Y), dict(x16=X16[0], y=Y[0]), tuple(X16[0].shape), True)

    def relu2(self, Hh):
        return self._elem("relu2", "jt_relu2", (Hh,), ("h",))

    def relu2_back(self, Gg, Hh):
        return self._elem("relu2_back", "jt_relu2_back", (Gg, Hh), ("g", "h"))

    def embed(self, W, tok):
        self.reg.confere_bloco("W", W)
        m, E = W
        V, D = m.shape
        M = tok.numel()
        X = self._novo_bloco(M, D, m.device)
        self._lanca("embed", "jt_embed", (m, E, tok, X[0], X[1], M, D, V), dict(W=m, tok=tok), (M, D))
        return X

    def um_quente(self, tok, V):
        M = tok.numel()
        U = self._novo_bloco(M, V, tok.device)
        self._lanca("um_quente", "jt_um_quente", (tok, U[0], U[1], M, V), dict(tok=tok), (M, V))
        return U

    def rms(self, X):
        self.reg.confere_bloco("x", X)
        m, E = X
        M, D = m.shape
        self.reg.exige(D <= 1024, f"rms: D={D} > 1024 nao suportado")
        G, _ = g_rms(D)
        Y = self._novo_bloco(M, D, m.device)
        r = torch.empty(M, dtype=torch.int32, device=m.device)
        emax = torch.empty(M, dtype=torch.int32, device=m.device)
        self._lanca("rms", "jt_rms", (m, E, Y[0], Y[1], r, emax, M, D, int(math.log2(D)), G), dict(x=m, xE=E), (M, D))
        return Y, (r, emax)

    def rms_back(self, Gg, Y, cache):
        self.reg.confere_bloco("g", Gg); self.reg.confere_bloco("y", Y)
        (g, gE), (y, yE) = Gg, Y
        M, D = g.shape
        G, GB = g_rms(D)
        O = self._novo_bloco(M, D, g.device)
        self._lanca("rms_back", "jt_rms_back", (g, gE, y, yE, cache[0], cache[1], O[0], O[1], M, D, int(math.log2(D)), G, GB),
                    dict(g=g, gE=gE, y=y, yE=yE, r=cache[0], emax=cache[1]), (M, D))
        return O

    def atencao(self, QKV, Bt, T, H):
        self.reg.confere_bloco("qkv", QKV)
        m, E = QKV
        D = m.shape[1] // 3
        self.reg.exige(T <= 128 and T % 32 == 0 and H * 4 == D and H <= 64, f"atencao: T={T} (<=128, multiplo de 32), H*4={H * 4} (=D={D}), H<=64")
        oacc = torch.empty(Bt * T, D, dtype=torch.int32, device=m.device)
        oe = torch.empty(Bt * T, D, dtype=torch.int16, device=m.device)
        P = torch.empty(Bt * H, T, T, dtype=torch.uint8, device=m.device)
        self._lanca("atencao", "jt_atencao", (m, E, oacc, oe, P, Bt, T, H, D, GA), dict(qkv=m, qkvE=E), (Bt * T, D))
        return (oacc, oe), P

    def atencao_back(self, GO, QKV, P, Bt, T, H):
        self.reg.confere_bloco("go", GO); self.reg.confere_bloco("qkv", QKV)
        (gm, gE), (m, E) = GO, QKV
        D = gm.shape[1]
        acc = torch.empty(Bt * T, 3 * D, dtype=torch.int32, device=m.device)
        e = torch.empty(Bt * T, 3 * D, dtype=torch.int16, device=m.device)
        self._lanca("atencao_back", "jt_atencao_back", (gm, gE, m, E, P, acc, e, Bt, T, H, D, GA, int(ATENCAO_LIMITE)),
                    dict(go=gm, goE=gE, qkv=m, qkvE=E, P=P), (Bt * T, 3 * D))
        return acc, e

    def softmax_ce(self, L, alvo):
        self.reg.confere_cru("logits", L)
        acc, e = L
        M, V = acc.shape
        self.reg.exige(V <= 1024 and V % 128 == 0, f"softmax_ce: V={V}")
        g = self._novo_bloco(M, V, acc.device)
        d = torch.empty(M, dtype=torch.int32, device=acc.device)
        s = torch.empty(M, dtype=torch.int32, device=acc.device)
        self._lanca("softmax_ce", "jt_softmax_ce", (acc, e, alvo, g[0], g[1], d, s, M, V, int(math.log2(M))),
                    dict(acc=acc, e=e, alvo=alvo), (M, V))
        return g, d, s

    def momento(self, Gc, Mom):
        self.reg.confere_cru("grad", Gc); self.reg.confere_bloco("momento", Mom)
        n = Mom[0].numel() // BLK
        self._lanca("momento", "jt_momento", (Gc[0], Gc[1], Mom[0], Mom[1], n), dict(g=Gc[0], ge=Gc[1]), tuple(Mom[0].shape), longos=(4,))
        return Mom

    def peso(self, W, Mom, R, k, ef):
        for n, t in (("W", W), ("momento", Mom), ("residuo", R)):
            self.reg.confere_bloco(n, t)
        n = W[0].numel() // BLK
        self._lanca("peso", "jt_peso", (W[0], W[1], Mom[0], Mom[1], R[0], R[1], k, int(ef), n), dict(), tuple(W[0].shape), longos=(8,))
        return W, R


# ============================================================================ conferido
def _compara(reg, op, a, b, nome):
    if a.shape != b.shape:
        reg.falha(f"conferencia '{op}' / {nome}: forma CUDA {tuple(a.shape)} x referencia {tuple(b.shape)}")
    dif = a != b
    if not dif.any():
        return
    n = int(dif.sum())
    pos = dif.nonzero()[0].tolist()
    reg.falha(f"conferencia '{op}' / {nome}: {n} de {a.numel()} valores diferentes da referencia.\n"
              f"  primeiro em {pos}: CUDA = {a[tuple(pos)].item()}, referencia = {b[tuple(pos)].item()}",
              op=op, n_dif=n, pos=pos)


class OpsConferido:
    """Cada operacao roda no CUDA e na referencia, com as mesmas entradas; tem que bater bit a bit."""
    nome = "conferido"

    def __init__(self, reg):
        self.reg, self.c, self.r = reg, OpsCuda(reg), OpsRef()
        self.n = {}

    def _confere(self, op, out_c, out_r, nomes):
        def plano(x):
            if torch.is_tensor(x):
                return [x]
            return [t for y in x for t in plano(y)]
        for i, (a, b) in enumerate(zip(plano(out_c), plano(out_r))):
            _compara(self.reg, op, a, b, nomes[i] if i < len(nomes) else f"saida {i}")
        self.n[op] = self.n.get(op, 0) + 1
        return out_c

    def __getattr__(self, op):
        fc, fr = getattr(self.c, op), getattr(self.r, op)
        nomes = {"rms": ["mantissa", "expoente", "r", "emax"], "atencao": ["o acc", "o exp", "P"],
                 "softmax_ce": ["g mantissa", "g expoente", "d_alvo", "soma"],
                 "peso": ["W", "W exp", "residuo", "residuo exp"],
                 "gemm_relu2": ["h mantissa", "h expoente", "a mantissa", "a expoente"]}.get(op, ["mantissa/acc", "expoente"])

        def f(*args, **kw):
            if op in ("momento", "peso"):     # operacoes no lugar: a referencia trabalha em copias
                copia = [tuple(t.clone() for t in a) if isinstance(a, tuple) else a for a in args]
                out_r = fr(*copia, **kw)
                out_c = fc(*args, **kw)
                return self._confere(op, out_c, out_r, nomes)
            out_c = fc(*args, **kw)
            out_r = fr(*args, **kw)
            return self._confere(op, out_c, out_r, nomes)
        return f
