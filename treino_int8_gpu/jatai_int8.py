"""Transformer treinado 100% em inteiros na GPU (prototipo em PyTorch).

Formato: todo tensor guardado e (mantissa INT8, expoente inteiro por bloco de BLK valores da ultima
dimensao). Valor real = mantissa * 2^expoente. Contas intermediarias em registrador inteiro largo
(int64 aqui; os maximos medidos cabem em int32). Multiplicacoes de matriz em INT8 de verdade
(torch._int_mm -> int32), um bloco de reducao por vez, alinhando expoentes por shift.

Arquitetura: pre-norm, RMSNorm sem ganho, atencao causal com ALiBi (inclinacoes 2^-s), head_dim 4,
MLP ReLU^2, vocabulario de bytes. Otimizador: SGD com momento 7/8 (m = m - m/8 + g), lr = 2^-k.
O modelo de referencia em float (ModeloFloat) tem a mesma arquitetura, mesma inicializacao e o
mesmo otimizador.
"""
import math
import os

import torch
import torch.nn.functional as F

I64 = torch.int64
NEG = -(1 << 40)
BLK = int(os.environ.get("JATAI_BLK", 32))  # bloco do expoente nas ativacoes/pesos guardados
BLK_TOK = 64      # bloco ao longo dos tokens nos gradientes de peso (reducao sobre tokens)
G = 6             # bits de guarda no alinhamento de blocos
G_RMS = 8
INV_BITS = 10
GA = 4            # bits de guarda de Q/K/V alinhados dentro da atencao
LOG2E_Q14, POLI = 23637, (-1295, 7556, -22645, 32765)
MAX_BITS = {}
CHECAR = {"on": False}


# ============================================================================
# primitivas inteiras
# ============================================================================
def anota(nome, x):
    if CHECAR["on"]:
        b = int(bitlen(x.abs().max().reshape(1))[0])
        MAX_BITS[nome] = max(MAX_BITS.get(nome, 0), b + 1)


def bitlen(x):
    ax = x.abs().to(I64)
    bl = torch.zeros_like(ax)
    for s in (32, 16, 8, 4, 2, 1):
        bl = bl + s * ((ax >> (bl + s)) > 0)
    return bl + (ax > 0)


ARRED_PAR = True  # empate vai para o par (sem vies); False = "metade para cima" (versao antiga, enviesada)


def desloca(x, s):
    """x * 2^-s com arredondamento; s > 0 shift a direita, s < 0 a esquerda.

    Empate (resto exatamente igual a meio) vai para o numero par. "Metade para cima" sempre empurra o
    empate para +infinito, e empates sao frequentes ao deslocar inteiros de poucos bits: o vies se
    acumula a cada quantizacao e ao longo das camadas."""
    x = x.to(I64)
    s = torch.as_tensor(s, dtype=I64, device=x.device).expand_as(x)
    pos, neg = s.clamp(0, 62), (-s).clamp(0, 62)
    meio = torch.where(pos > 0, torch.bitwise_left_shift(torch.ones_like(pos), (pos - 1).clamp(min=0)), 0)
    q = (x + meio) >> pos
    if ARRED_PAR:
        resto = x - (q << pos) + meio            # resto de x em relacao a q*2^pos, deslocado por meio
        q = q - ((resto == 0) & (pos > 0) & ((q & 1) == 1)).to(I64)
    return torch.where(pos > 0, q, x << neg)


def expande(E, b):
    return E.repeat_interleave(b, dim=-1)


def quant(acc, e, b=BLK):
    """acc * 2^e (e por elemento) -> (mantissa INT8, expoente por bloco de b)."""
    acc = acc.to(I64)
    e = torch.as_tensor(e, dtype=I64, device=acc.device).expand_as(acc)
    topo = torch.where(acc != 0, e + bitlen(acc), torch.full_like(acc, NEG))
    forma = acc.shape[:-1] + (acc.shape[-1] // b, b)
    E = topo.reshape(forma).amax(-1) - 7
    E = torch.where(E < NEG // 2, torch.zeros_like(E), E)
    m = desloca(acc, expande(E, b) - e)
    E = E + (m.abs().reshape(forma).amax(-1) > 127)
    m = desloca(acc, expande(E, b) - e)
    return m.clamp(-127, 127).to(torch.int8), E


def bloco_de(m, E):
    return m.shape[-1] // E.shape[-1]


def deq(m, E):
    return m.double() * torch.pow(2.0, expande(E, bloco_de(m, E)).double())


def transpoe(m, E, b):
    """(R, C) em blocos ao longo de C  ->  (C, R) em blocos de b ao longo de R (requantiza)."""
    e = expande(E, bloco_de(m, E))
    return quant(m.to(I64).t().contiguous(), e.t().contiguous(), b)


def soma_exp(a, ea, b, eb):
    """a*2^ea + b*2^eb em inteiro, com o expoente escolhido para caber em ~50 bits."""
    ea = torch.as_tensor(ea, dtype=I64, device=a.device)
    eb = torch.as_tensor(eb, dtype=I64, device=a.device)
    e = torch.maximum(ea + bitlen(a), eb + bitlen(b)) - 50
    return desloca(a, e - ea) + desloca(b, e - eb), e


def isqrt(n):
    n = n.clamp(min=0)
    x = torch.bitwise_left_shift(torch.ones_like(n), (bitlen(n) + 1) // 2)
    for _ in range(10):
        x = torch.minimum(x, (x + n // x.clamp(min=1)) >> 1)
    return x.clamp(min=1)


def exp_int(d):
    """exp(-d/256) em Q15, d >= 0. Shift + polinomio de grau 3, sem tabela."""
    t = d.clamp(max=6000) * LOG2E_Q14
    n, f = t >> 22, (t & ((1 << 22) - 1)) >> 7
    p = torch.full_like(f, POLI[0])
    for c in POLI[1:]:
        p = ((p * f) >> 15) + c
    return torch.where(n > 30, torch.zeros_like(p), p >> n.clamp(max=30))


def gemm(A, EA, B, EB, nome="gemm"):
    """C = A @ B^T, com A (M,K) e B (N,K) em blocos ao longo de K. Devolve (acc int64, expoente por elemento)."""
    M, K = A.shape
    N, nb = B.shape[0], EA.shape[1]
    b = K // nb
    Ab = A.reshape(M, nb, b).permute(1, 0, 2).contiguous()
    Bb = B.reshape(N, nb, b).permute(1, 2, 0).contiguous()
    Emax = (EA[:, None, :] + EB[None, :, :]).amax(-1)
    acc = torch.zeros(M, N, dtype=I64, device=A.device)
    for j in range(nb):
        p = torch._int_mm(Ab[j], Bb[j]).to(I64)
        acc += desloca(p << G, Emax - (EA[:, j, None] + EB[None, :, j]))
    anota(f"{nome}: acumulador", acc)
    return acc, Emax - G


def gemm_peso(gY, gYE, X, XE):
    """dW = gY^T X  (reducao sobre os tokens): requantiza os dois em blocos ao longo dos tokens."""
    gt, gtE = transpoe(gY, gYE, BLK_TOK)
    xt, xtE = transpoe(X, XE, BLK_TOK)
    return gemm(gt, gtE, xt, xtE, "dW")


# ============================================================================
# modelo inteiro
# ============================================================================
def inicial(gerador, linhas, fan_in, dev, std_real=None):
    e = -int(round(math.log2(32 * math.sqrt(fan_in)))) if std_real is None else -5
    m = torch.round(torch.randn(linhas, fan_in, generator=gerador) * 32).clamp(-127, 127).to(torch.int8)
    return m.to(dev), torch.full((linhas, fan_in // BLK), e, dtype=I64, device=dev)


class ModeloInt8:
    def __init__(self, V, D, H, HD, FF, L, T, semente, dev):
        self.V, self.D, self.H, self.HD, self.FF, self.L, self.T, self.dev = V, D, H, HD, FF, L, T, dev
        g = torch.Generator().manual_seed(semente)
        self.p = {"emb": inicial(g, V, D, dev, std_real=1.0)}
        for l in range(L):
            for n in ("q", "k", "v", "o"):
                self.p[f"W{n}{l}"] = inicial(g, D, D, dev)
            self.p[f"Wup{l}"] = inicial(g, FF, D, dev)
            self.p[f"Wd{l}"] = inicial(g, D, FF, dev)
        self.p["out"] = inicial(g, V, D, dev)
        self.mom = {k: (torch.zeros_like(m), torch.zeros_like(E)) for k, (m, E) in self.p.items()}
        self.res = {k: (torch.zeros_like(m), torch.zeros_like(E)) for k, (m, E) in self.p.items()}
        self.error_feedback = False
        pos = torch.arange(T, device=dev)
        dist = pos[:, None] - pos[None, :]
        self.mascara = dist < 0
        self.incl = torch.arange(H, device=dev) // 8
        self.vies = (dist.clamp(min=0)[None] << (8 - self.incl)[:, None, None]).to(I64)
        self.idx_cab = (torch.arange(H, device=dev) * HD) // BLK

    # ---------------- RMSNorm
    def rms(self, xm, xE):
        M, D = xm.shape
        nb, x = xE.shape[1], xm.to(I64)
        sq = (x * x).reshape(M, nb, D // nb).sum(-1)
        emax = xE.amax(-1, keepdim=True)
        ss = desloca(sq << G_RMS, 2 * (emax - xE)).sum(-1, keepdim=True)
        r = isqrt(ss // D)
        k = bitlen(r) + INV_BITS
        inv = torch.bitwise_left_shift(torch.ones_like(k), k) // r
        anota("rms: soma dos quadrados", ss)
        ym, yE = quant(x * inv, expande(xE, D // nb) - k - emax + G_RMS // 2)
        return ym, yE, (r, emax)

    def rms_back(self, gm, gE, ym, yE, cache):
        r, emax = cache
        M, D = gm.shape
        b = D // gE.shape[1]
        g, y = gm.to(I64), ym.to(I64)
        eg, ey = expande(gE, b), expande(yE, b)
        smax = (eg + ey).amax(-1, keepdim=True)
        S = desloca((g * y) << G, smax - (eg + ey)).sum(-1, keepdim=True)
        eS = smax - G - int(math.log2(D))
        sh = (bitlen(S) - 15).clamp(min=0)
        S, eS = desloca(S, sh), eS + sh
        dentro, e = soma_exp(g, eg, -(y * S), ey + eS)
        k = bitlen(r) + INV_BITS
        inv = torch.bitwise_left_shift(torch.ones_like(k), k) // r
        sh = (bitlen(dentro) - 19).clamp(min=0)
        dentro, e = desloca(dentro, sh), e + sh
        anota("rms_back: dentro*inv", dentro * inv)
        return quant(dentro * inv, e - k - (emax - G_RMS // 2))

    # ---------------- atencao
    def alinha_cab(self, m, E, Bt):
        T, H, HD = self.T, self.H, self.HD
        x = m.to(I64).reshape(Bt, T, H, HD).permute(0, 2, 1, 3)
        eh = E.reshape(Bt, T, -1)[:, :, self.idx_cab].permute(0, 2, 1)
        emx = eh.amax(-1, keepdim=True)
        return desloca(x << GA, (emx - eh)[..., None]), (emx - GA)[..., None]

    def junta(self, x, e):
        Bt, H, T, HD = x.shape
        e = e.expand(Bt, H, T, HD)
        return quant(x.permute(0, 2, 1, 3).reshape(Bt * T, H * HD),
                     e.permute(0, 2, 1, 3).reshape(Bt * T, H * HD))

    @staticmethod
    def encolhe(x, e, bits):
        sh = (bitlen(x.abs().amax(dim=(-2, -1), keepdim=True)) - bits).clamp(min=0)
        return desloca(x, sh), e + sh

    def atencao(self, xn, xnE, l, Bt):
        p = self.p
        q, qE = quant(*gemm(xn, xnE, *p[f"Wq{l}"], "Q"))
        k, kE = quant(*gemm(xn, xnE, *p[f"Wk{l}"], "K"))
        v, vE = quant(*gemm(xn, xnE, *p[f"Wv{l}"], "V"))
        qa, eq = self.alinha_cab(q, qE, Bt)
        ka, ek = self.alinha_cab(k, kE, Bt)
        va, ev = self.alinha_cab(v, vE, Bt)
        S = sum(qa[..., :, None, d] * ka[..., None, :, d] for d in range(self.HD))
        anota("scores", S)
        u = desloca(S, -(eq + ek - 1 + 8)) - self.vies
        u = torch.where(self.mascara, NEG, u)
        ex = torch.where(self.mascara, 0, exp_int(u.amax(-1, keepdim=True) - u)) >> 3
        soma = ex.sum(-1, keepdim=True)
        P = ((ex * ((1 << 28) // soma) + (1 << 19)) >> 20).clamp(max=255)
        o = torch.stack([(P * va[..., None, :, d]).sum(-1) for d in range(self.HD)], -1)
        anota("P@V", o)
        om, oE = self.junta(o, -8 + ev)
        ym, yE = quant(*gemm(om, oE, *p[f"Wo{l}"], "O"))
        return ym, yE, dict(xn=xn, xnE=xnE, q=q, qE=qE, k=k, kE=kE, v=v, vE=vE, P=P, om=om, oE=oE)

    def atencao_back(self, gm, gE, c, l, Bt, grads):
        p = self.p
        grads[f"Wo{l}"] = gemm_peso(gm, gE, c["om"], c["oE"])
        Wo_t = transpoe(*p[f"Wo{l}"], BLK)
        go, goE = quant(*gemm(gm, gE, *Wo_t, "dO"))
        goa, ego = self.alinha_cab(go, goE, Bt)
        qa, eq = self.alinha_cab(c["q"], c["qE"], Bt)
        ka, ek = self.alinha_cab(c["k"], c["kE"], Bt)
        va, ev = self.alinha_cab(c["v"], c["vE"], Bt)
        P = c["P"]
        gP = sum(goa[..., :, None, d] * va[..., None, :, d] for d in range(self.HD))
        gP, egp = self.encolhe(gP, ego + ev, 15)
        gv = torch.stack([(P * goa[..., :, None, d]).sum(-2) for d in range(self.HD)], -1)
        s_lin = desloca((P * gP).sum(-1, keepdim=True), 8)
        dS, eds = self.encolhe(P * (gP - s_lin), egp - 8 - 1, 12)
        gq = torch.stack([(dS * ka[..., None, :, d]).sum(-1) for d in range(self.HD)], -1)
        gk = torch.stack([(dS * qa[..., :, None, d]).sum(-2) for d in range(self.HD)], -1)
        anota("atencao_back: gq/gk/gv", torch.stack([gq.abs().max(), gk.abs().max(), gv.abs().max()]))
        dxn, edxn = None, None
        for n, x, e in (("q", gq, eds + ek), ("k", gk, eds + eq), ("v", gv, -8 + ego)):
            xm, xE = self.junta(x, e)
            grads[f"W{n}{l}"] = gemm_peso(xm, xE, c["xn"], c["xnE"])
            acc, ea = gemm(xm, xE, *transpoe(*p[f"W{n}{l}"], BLK), f"d{n}")
            dxn, edxn = (acc, ea) if dxn is None else soma_exp(dxn, edxn, acc, ea)
        return quant(dxn, edxn)

    # ---------------- MLP
    def mlp(self, xn, xnE, l):
        h, hE = quant(*gemm(xn, xnE, *self.p[f"Wup{l}"], "up"))
        r = h.to(I64).clamp(min=0)
        am, aE = quant(r * r, 2 * expande(hE, BLK))
        ym, yE = quant(*gemm(am, aE, *self.p[f"Wd{l}"], "down"))
        return ym, yE, dict(xn=xn, xnE=xnE, h=h, hE=hE, am=am, aE=aE)

    def mlp_back(self, gm, gE, c, l, grads):
        p = self.p
        grads[f"Wd{l}"] = gemm_peso(gm, gE, c["am"], c["aE"])
        ga, gaE = quant(*gemm(gm, gE, *transpoe(*p[f"Wd{l}"], BLK), "da"))
        gh, ghE = quant(ga.to(I64) * c["h"].to(I64).clamp(min=0) * 2, expande(gaE + c["hE"], BLK))
        grads[f"Wup{l}"] = gemm_peso(gh, ghE, c["xn"], c["xnE"])
        return quant(*gemm(gh, ghE, *transpoe(*p[f"Wup{l}"], BLK), "dxn_mlp"))

    # ---------------- passo completo
    def forward_backward(self, tok, alvo, k_lr=None):
        Bt, T = tok.shape
        M = Bt * T
        p = self.p
        idx = tok.reshape(-1)
        xm, xE = p["emb"][0][idx], p["emb"][1][idx]
        caches = []
        for l in range(self.L):
            n1, n1E, c1 = self.rms(xm, xE)
            ym, yE, ca = self.atencao(n1, n1E, l, Bt)
            xm, xE = quant(*soma_exp(xm.to(I64), expande(xE, BLK), ym.to(I64), expande(yE, BLK)))
            n2, n2E, c2 = self.rms(xm, xE)
            ym, yE, cm = self.mlp(n2, n2E, l)
            xm2, xE2 = quant(*soma_exp(xm.to(I64), expande(xE, BLK), ym.to(I64), expande(yE, BLK)))
            caches.append((c1, n1, n1E, ca, c2, n2, n2E, cm))
            xm, xE = xm2, xE2
        nf, nfE, cf = self.rms(xm, xE)
        acc, e = gemm(nf, nfE, *p["out"], "logits")
        logits_real = acc.double() * torch.pow(2.0, e.double())  # so para medir a loss (relatorio)
        perda = F.cross_entropy(logits_real, alvo.reshape(-1)).item()
        if k_lr is None:
            return perda
        u = desloca(acc, -(e + 8))
        ex = exp_int(u.amax(-1, keepdim=True) - u) >> 3
        P12 = (ex * ((1 << 28) // ex.sum(-1, keepdim=True)) + (1 << 15)) >> 16
        g = P12.clone()
        g[torch.arange(M, device=self.dev), alvo.reshape(-1)] -= 4096
        gm, gE = quant(g, -12 - int(math.log2(M)))
        grads = {"out": gemm_peso(gm, gE, nf, nfE)}
        gm, gE = quant(*gemm(gm, gE, *transpoe(*p["out"], BLK), "dxf"))
        gx, gxE = self.rms_back(gm, gE, nf, nfE, cf)
        for l in reversed(range(self.L)):
            c1, n1, n1E, ca, c2, n2, n2E, cm = caches[l]
            gn, gnE = self.mlp_back(gx, gxE, cm, l, grads)
            gb, gbE = self.rms_back(gn, gnE, n2, n2E, c2)
            gx, gxE = quant(*soma_exp(gx.to(I64), expande(gxE, BLK), gb.to(I64), expande(gbE, BLK)))
            gn, gnE = self.atencao_back(gx, gxE, ca, l, Bt, grads)
            gb, gbE = self.rms_back(gn, gnE, n1, n1E, c1)
            gx, gxE = quant(*soma_exp(gx.to(I64), expande(gxE, BLK), gb.to(I64), expande(gbE, BLK)))
        um = torch.zeros(M, self.V, dtype=torch.int8, device=self.dev)
        um[torch.arange(M, device=self.dev), idx] = 1
        grads["emb"] = gemm_peso(um, torch.zeros(M, self.V // BLK, dtype=I64, device=self.dev), gx, gxE)
        self.ultimos_grads = grads
        for n, (gacc, ge) in grads.items():
            self.passo(n, gacc, ge, k_lr)
        return perda

    def passo(self, n, gacc, ge, k_lr):
        Wm, WE = self.p[n]
        Mm, ME = self.mom[n]
        b = bloco_de(Wm, WE)
        Mm, ME = quant(*soma_exp(Mm.to(I64) * 7, expande(ME, b) - 3, gacc, ge), b=b)
        self.mom[n] = (Mm, ME)
        ew = expande(WE, b)
        if not self.error_feedback:
            u = desloca(Mm.to(I64), ew - (expande(ME, b) - k_lr))
            self.p[n] = quant(Wm.to(I64) - u, ew, b=b)
            return
        # error feedback: alvo = W - lr*mom + residuo; o que nao couber no INT8 do peso
        # volta para o residuo (tambem INT8 por bloco) e entra no proximo passo.
        Rm, RE = self.res[n]
        alvo, ea = soma_exp(Wm.to(I64), ew, -Mm.to(I64), expande(ME, b) - k_lr)
        alvo, ea = soma_exp(alvo, ea, Rm.to(I64), expande(RE, b))
        Wm2, WE2 = quant(alvo, ea, b=b)
        self.p[n] = (Wm2, WE2)
        self.res[n] = quant(*soma_exp(alvo, ea, -Wm2.to(I64), expande(WE2, b)), b=b)


# ============================================================================
# referencia em float (mesma arquitetura, mesma inicializacao, mesmo otimizador)
# ============================================================================
def quant_float(x, b=BLK, estocastico=False):
    """Mesmo formato do quant (mantissa INT8, expoente 2^E por bloco), simulado em float."""
    xb = x.reshape(*x.shape[:-1], x.shape[-1] // b, b)
    topo = torch.floor(torch.log2(xb.abs().amax(-1, keepdim=True).clamp(min=1e-38))) + 1
    esc = torch.pow(2.0, topo - 7)
    esc = torch.where(torch.round(xb / esc).abs().amax(-1, keepdim=True) > 127, esc * 2, esc)
    r = xb / esc
    r = torch.floor(r + torch.rand_like(r)) if estocastico else torch.round(r)
    return (r.clamp(-127, 127) * esc).reshape(x.shape)


class _QuantGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, estocastico):
        ctx.estocastico = estocastico
        return x

    @staticmethod
    def backward(ctx, g):
        return quant_float(g, estocastico=ctx.estocastico), None


def ponto_quant(x, flags):
    """Ponto onde o modelo inteiro arredonda para INT8: a = ativacao no forward, g = gradiente no
    backward, s = gradiente com arredondamento estocastico."""
    if "a" in flags:
        x = x + (quant_float(x) - x).detach()
    if "g" in flags or "s" in flags:
        x = _QuantGrad.apply(x, "s" in flags)
    return x


class ModeloFloat(torch.nn.Module):
    def __init__(self, mi: ModeloInt8):
        super().__init__()
        self.cfg = (mi.V, mi.D, mi.H, mi.HD, mi.FF, mi.L, mi.T)
        self.w = torch.nn.ParameterDict({k: torch.nn.Parameter(deq(m, E).float()) for k, (m, E) in mi.p.items()})
        self.register_buffer("mascara", mi.mascara)
        self.register_buffer("vies", -(mi.vies.float() / 256.0))
        self.flags = ""   # ablacao: pontos de arredondamento do modelo inteiro (ver ponto_quant)

    def forward(self, tok, alvo):
        V, D, H, HD, FF, L, T = self.cfg
        Bt = tok.shape[0]
        q8 = lambda x: ponto_quant(x, self.flags)
        rms = lambda x: q8(x * torch.rsqrt((x * x).mean(-1, keepdim=True)))
        cab = lambda x: q8(x).reshape(Bt, T, H, HD).transpose(1, 2)
        x = q8(self.w["emb"][tok])
        for l in range(L):
            xn = rms(x)
            q, k, v = cab(xn @ self.w[f"Wq{l}"].t()), cab(xn @ self.w[f"Wk{l}"].t()), cab(xn @ self.w[f"Wv{l}"].t())
            z = (q @ k.transpose(-1, -2)) / 2.0 + self.vies
            a = torch.softmax(z.masked_fill(self.mascara, float("-inf")), -1)
            if "p" in self.flags:     # P como o modelo inteiro guarda hoje: UINT8 fixo em 1/256
                a = a + ((torch.round(a * 256).clamp(max=255) / 256) - a).detach()
            if "b" in self.flags:     # P no formato do resto: INT8 + expoente por bloco de 32 chaves
                a = a + (quant_float(a) - a).detach()
            x = q8(x + q8(q8((a @ v).transpose(1, 2).reshape(Bt, T, D)) @ self.w[f"Wo{l}"].t()))
            xn = rms(x)
            x = q8(x + q8(q8(torch.relu(xn @ self.w[f"Wup{l}"].t()).square()) @ self.w[f"Wd{l}"].t()))
        logits = rms(x) @ self.w["out"].t()
        return F.cross_entropy(logits.reshape(-1, V), alvo.reshape(-1))
