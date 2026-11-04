"""Analise passo a passo da aritmetica INT8 (configuracao D=256, 64 cabecas, head_dim=4, 12 camadas).

Parte 1 - para cada operacao de uma camada: quantos bits o valor EXATO precisa, onde ele vive
          (registrador ou memoria) e o erro local contra float64 com as mesmas entradas.
Parte 2 - erro acumulado ao longo de 12 camadas para cada combinacao posicao x MLP.
Parte 3 - backward local: quanto do gradiente inteiro concorda com o gradiente float.
Parte 4 - somar so com INT8 (sem nenhum valor intermediario maior que 8 bits): quanto se perde.

Aritmetica dos modelos: inteiros (numpy int64 representa o registrador). Float aparece SO nas
referencias de comparacao (numpy float64 / torch float64) e na tabela de cos/sin do RoPE.
"""
import math

import numpy as np
import torch

D, H, HD, T, LOTE, FF, CAMADAS = 256, 64, 4, 64, 2, 1024, 12
SEMENTE = 0


# ============================================================================
# primitivas inteiras
# ============================================================================
def isqrt(n):
    n = np.maximum(n, 0)
    x = np.maximum(n, 1)
    for _ in range(64):
        y = (x + n // np.maximum(x, 1)) >> 1
        if np.all(y >= x):
            break
        x = np.minimum(x, y)
    return np.maximum(x, 1)


def shr(x, s):
    if s > 0:
        return (x + (1 << (s - 1))) >> s
    if s < 0:
        return x << (-s)
    return x


def q8(x, e):
    mx = int(np.abs(x).max())
    s = max(0, mx.bit_length() - 7)
    return np.clip(shr(x, s), -127, 127), e + s


def soma_alinhada(a, ea, b, eb):
    e = min(ea, eb)
    if max(ea, eb) - e > 40:
        return q8(a, ea) if ea > eb else q8(b, eb)
    x = (a << (ea - e)) + (b << (eb - e))
    return x, e


def mul_recip(a, b, nb):
    return (a * ((1 << nb) // b) + (1 << (nb - 1))) >> nb


LOG2E_Q14, POLI_2NEG = 23637, (-1295, 7556, -22645, 32765)


def exp_int(d):
    t = np.minimum(d, 6000) * LOG2E_Q14
    n, f = t >> 22, (t & ((1 << 22) - 1)) >> 7
    p = POLI_2NEG[0]
    for c in POLI_2NEG[1:]:
        p = ((p * f) >> 15) + c
    return np.where(n > 30, 0, p >> np.minimum(n, 30))


def bits(x):
    return int(np.abs(x).max()).bit_length() + 1


def deq(m, e):
    return m.astype(np.float64) * 2.0 ** e


def erro_rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# ============================================================================
# registro do passo a passo
# ============================================================================
class Registro:
    def __init__(self):
        self.linhas = []

    def add(self, camada, op, exato, onde, erro=None):
        self.linhas.append((camada, op, bits(exato), onde, erro))


def iniciar_pesos(rng, fan_in, forma):
    e = -int(round(math.log2(32 * math.sqrt(fan_in))))
    return np.clip(np.round(rng.normal(0, 32, forma)), -127, 127).astype(np.int64), e


def camada_pesos(rng, mlp):
    p = {n: iniciar_pesos(rng, D, (D, D)) for n in ("Wq", "Wk", "Wv", "Wo")}
    p["Wup"] = iniciar_pesos(rng, D, (D, FF))
    if mlp == "reglu":
        p["Wg"] = iniciar_pesos(rng, D, (D, FF))
    p["Wd"] = iniciar_pesos(rng, FF, (FF, D))
    return p


POS = np.arange(T)
OMEGA = np.array([1.0, 10000.0 ** -0.5])  # head_dim 4 -> 2 pares de frequencia
ANG = POS[:, None] * OMEGA[None, :]
COS_Q14 = np.round(np.cos(ANG) * 16384).astype(np.int64)
SIN_Q14 = np.round(np.sin(ANG) * 16384).astype(np.int64)
DIST = POS[:, None] - POS[None, :]
MASCARA = DIST < 0
INCL = np.arange(H) // 8  # ALiBi com inclinacao 2^-s, s = 0..7 (potencias de 2 -> shift exato)


def heads(x):
    return x.reshape(x.shape[0], T, H, HD).transpose(0, 2, 1, 3)


def junta(x):
    return x.transpose(0, 2, 1, 3).reshape(x.shape[0], T, D)


def rope_int(x):
    a, b = x[..., 0:2], x[..., 2:4]
    c, s = COS_Q14[None, None], SIN_Q14[None, None]
    ra, rb = a * c - b * s, a * s + b * c
    return np.concatenate([ra, rb], -1)


def rope_f(x, inversa=False):
    a, b = x[..., 0:2], x[..., 2:4]
    c, s = np.cos(ANG)[None, None], np.sin(ANG)[None, None]
    if inversa:
        s = -s
    return np.concatenate([a * c - b * s, a * s + b * c], -1)


def alibi_f():
    return -(2.0 ** -INCL)[None, :, None, None] * DIST[None, None]


def softmax_f(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def rms_f(x):
    return x / np.sqrt((x * x).mean(-1, keepdims=True))


# ============================================================================
# forward inteiro de uma camada, com registro e erro local de cada operacao
# ============================================================================
def rms_int(m, e, reg, c, nome):
    ss = (m * m).sum(-1, keepdims=True)
    reg.add(c, f"{nome}: soma dos quadrados", ss, "registrador")
    r = isqrt(ss // D)
    prod = m * 32 * ((1 << 16) // r)
    reg.add(c, f"{nome}: x*32*(1/r)", prod, "registrador")
    xn = np.clip(shr(prod, 16), -127, 127)
    reg.add(c, f"{nome}: saida", xn, "memoria INT8", erro_rel(deq(xn, -5), rms_f(deq(m, e))))
    return xn, -5


def proj(xm, xe, W, reg, c, nome):
    Wm, We = W
    acc = xm @ Wm
    reg.add(c, f"{nome}: acumulador do produto escalar", acc, "registrador")
    m, e = q8(acc, xe + We)
    reg.add(c, f"{nome}: saida", m, "memoria INT8", erro_rel(deq(m, e), deq(xm, xe) @ deq(Wm, We)))
    return m, e


def atencao_int(xn, xe, p, pos, reg, c, guardar=None):
    q, eq = proj(xn, xe, p["Wq"], reg, c, "Q")
    k, ek = proj(xn, xe, p["Wk"], reg, c, "K")
    v, ev = proj(xn, xe, p["Wv"], reg, c, "V")
    qh, kh, vh = heads(q), heads(k), heads(v)
    if pos == "rope":
        for nome, t, et in (("RoPE em Q", "qh", eq), ("RoPE em K", "kh", ek)):
            x = qh if t == "qh" else kh
            rot = rope_int(x)
            reg.add(c, f"{nome}: x*cos - y*sin (Q14)", rot, "registrador")
            m, e = q8(shr(rot, 14), et)
            reg.add(c, f"{nome}: saida", m, "memoria INT8", erro_rel(deq(m, e), rope_f(deq(x, et))))
            if t == "qh":
                qh, eq = m, e
            else:
                kh, ek = m, e
    S = qh @ kh.transpose(0, 1, 3, 2)
    reg.add(c, "scores q.k (head_dim 4)", S, "registrador")
    es = eq + ek - 1  # 1/sqrt(4) = 2^-1: shift exato
    u = shr(S, -(es + 8))
    ref = deq(qh, eq) @ deq(kh, ek).transpose(0, 1, 3, 2) / 2.0
    if pos == "alibi":
        u = u - (np.maximum(DIST, 0)[None, None] << (8 - INCL)[None, :, None, None])
        ref = ref + alibi_f()
        reg.add(c, "score + vies ALiBi (unidades 2^-8 nat)", u, "registrador")
    u = np.where(MASCARA, np.iinfo(np.int64).min // 4, u)
    ex = exp_int(u.max(-1, keepdims=True) - u)
    ex = np.where(MASCARA, 0, ex)
    soma = ex.sum(-1, keepdims=True)
    reg.add(c, "softmax: soma das exp (Q15)", soma, "registrador")
    prod = ex * ((1 << 30) // soma)
    reg.add(c, "softmax: exp*(1/soma)", prod, "registrador")
    P = np.minimum(shr(prod, 22), 255)
    reg.add(c, "softmax: probabilidades", P, "memoria UINT8",
            erro_rel(P / 256.0, softmax_f(np.where(MASCARA, -1e30, ref))))
    acc = P @ vh
    reg.add(c, "P@V: acumulador", acc, "registrador")
    o, eo = q8(acc, -8 + ev)
    reg.add(c, "P@V: saida", o, "memoria INT8", erro_rel(deq(o, eo), (P / 256.0) @ deq(vh, ev)))
    om = junta(o)
    y, ey = proj(om, eo, p["Wo"], reg, c, "O")
    if guardar is not None:
        guardar.update(qh=qh, eq=eq, kh=kh, ek=ek, vh=vh, ev=ev, P=P, om=om, eo=eo)
    return y, ey


def mlp_int(xn, xe, p, mlp, reg, c, guardar=None):
    h, eh = proj(xn, xe, p["Wup"], reg, c, "MLP up")
    if mlp == "relu2":
        r = np.maximum(h, 0)
        a = r * r
        reg.add(c, "ReLU^2: x*x", a, "registrador")
        am, ae = q8(a, 2 * eh)
        ref = np.maximum(deq(h, eh), 0) ** 2
    elif mlp == "reglu":
        g, eg = proj(xn, xe, p["Wg"], reg, c, "MLP gate")
        a = np.maximum(g, 0) * h
        reg.add(c, "ReGLU: relu(gate)*up", a, "registrador")
        am, ae = q8(a, eg + eh)
        ref = np.maximum(deq(g, eg), 0) * deq(h, eh)
    else:
        sig = sigmoide_q15(h, eh, reg, c)
        a = h * sig
        reg.add(c, "SiLU: x*sigmoide", a, "registrador")
        am, ae = q8(a, eh - 15)
        x = deq(h, eh)
        ref = x / (1 + np.exp(-x))
    reg.add(c, f"{mlp}: saida", am, "memoria INT8", erro_rel(deq(am, ae), ref))
    y, ey = proj(am, ae, p["Wd"], reg, c, "MLP down")
    if guardar is not None:
        guardar.update(h=h, eh=eh, am=am, ae=ae)
        if mlp == "reglu":
            guardar.update(g=g, eg=eg)
    return y, ey


def sigmoide_q15(m, e, reg=None, c=None):
    d = np.abs(shr(m, -(e + 8)))  # |x| em unidades de 2^-8 nat
    ex = exp_int(d)  # exp(-|x|) em Q15
    num = np.where(m >= 0, 1 << 30, ex << 15)
    if reg is not None:
        reg.add(c, "SiLU: sigmoide = numerador/(1+exp) (divisao por elemento)", num, "registrador")
    return num // ((1 << 15) + ex)


def camada_int(x, ex, p, pos, mlp, reg, c):
    xn, xne = rms_int(x, ex, reg, c, "RMSNorm 1")
    y, ey = atencao_int(xn, xne, p, pos, reg, c)
    s, es = soma_alinhada(x, ex, y, ey)
    reg.add(c, "residual 1: soma alinhada", s, "registrador")
    x, ex = q8(s, es)
    xn, xne = rms_int(x, ex, reg, c, "RMSNorm 2")
    y, ey = mlp_int(xn, xne, p, mlp, reg, c)
    s, es = soma_alinhada(x, ex, y, ey)
    reg.add(c, "residual 2: soma alinhada", s, "registrador")
    return q8(s, es)


def camada_float(x, p, pos, mlp):
    w = {k: deq(*v) for k, v in p.items()}
    xn = rms_f(x)
    qh, kh, vh = heads(xn @ w["Wq"]), heads(xn @ w["Wk"]), heads(xn @ w["Wv"])
    if pos == "rope":
        qh, kh = rope_f(qh), rope_f(kh)
    z = qh @ kh.transpose(0, 1, 3, 2) / 2.0
    if pos == "alibi":
        z = z + alibi_f()
    P = softmax_f(np.where(MASCARA, -1e30, z))
    x = x + junta(P @ vh) @ w["Wo"]
    xn = rms_f(x)
    h = xn @ w["Wup"]
    if mlp == "relu2":
        a = np.maximum(h, 0) ** 2
    elif mlp == "reglu":
        a = np.maximum(xn @ w["Wg"], 0) * h
    else:
        a = h / (1 + np.exp(-h))
    return x + a @ w["Wd"]


# ============================================================================
# backward local: atencao (ALiBi x RoPE) e MLP (ReLU^2 x ReGLU x SiLU)
# ============================================================================
def cos_sim(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def backward_mlp(rng, mlp):
    p = camada_pesos(rng, mlp)
    xn = np.clip(np.round(rng.normal(0, 32, (LOTE, T, D))), -127, 127).astype(np.int64)
    g_out, ego = q8(np.round(rng.normal(0, 40, (LOTE, T, D))).astype(np.int64), -12)
    s = {}
    mlp_int(xn, -5, p, mlp, Registro(), 0, s)
    h, eh, am, ae = s["h"], s["eh"], s["am"], s["ae"]
    Wd, eWd = p["Wd"]
    gWd = np.einsum("btf,btd->fd", am, g_out)
    ga, ega = q8(g_out @ Wd.T, ego + eWd)
    if mlp == "relu2":
        gh, egh = q8(ga * 2 * np.maximum(h, 0), ega + eh)
    elif mlp == "reglu":
        g, eg = s["g"], s["eg"]
        gh, egh = q8(ga * np.maximum(g, 0), ega + eg)
        gg, egg = q8(ga * h * (g > 0), ega + eh)
    else:
        sig = sigmoide_q15(h, eh)
        xu = shr(h * ((1 << 15) - sig), -eh)  # x*(1-sig) em Q15
        der = shr(sig * ((1 << 15) + xu), 15)  # sig*(1 + x*(1-sig)) em Q15
        gh, egh = q8(ga * der, ega - 15)
    Wup, eWup = p["Wup"]
    gx, egx = q8(gh @ Wup.T, egh + eWup)
    gWup = np.einsum("btd,btf->df", xn, gh)
    if mlp == "reglu":
        Wg, eWg = p["Wg"]
        gx2, egx2 = q8(gg @ Wg.T, egg + eWg)
        gx, egx = q8(*soma_alinhada(gx, egx, gx2, egx2))

    tx = torch.tensor(deq(xn, -5), requires_grad=True)
    tw = {k: torch.tensor(deq(*v), requires_grad=True) for k, v in p.items()}
    th = tx @ tw["Wup"]
    if mlp == "relu2":
        ta = torch.relu(th) ** 2
    elif mlp == "reglu":
        ta = torch.relu(tx @ tw["Wg"]) * th
    else:
        ta = torch.nn.functional.silu(th)
    ((ta @ tw["Wd"]) * torch.tensor(deq(g_out, ego))).sum().backward()
    return {"dX": cos_sim(deq(gx, egx), tx.grad.numpy()),
            "dW_up": cos_sim(gWup.astype(np.float64), tw["Wup"].grad.numpy()),
            "dW_down": cos_sim(gWd.astype(np.float64), tw["Wd"].grad.numpy()),
            "zeros em dX": float(np.mean(gx == 0))}


def backward_atencao(rng, pos):
    p = camada_pesos(rng, "relu2")
    xn = np.clip(np.round(rng.normal(0, 32, (LOTE, T, D))), -127, 127).astype(np.int64)
    g_out, ego = q8(np.round(rng.normal(0, 40, (LOTE, T, D))).astype(np.int64), -12)
    s = {}
    atencao_int(xn, -5, p, pos, Registro(), 0, s)
    Wo, eWo = p["Wo"]
    gWo = np.einsum("btd,bte->de", s["om"], g_out)
    go, ego2 = q8(g_out @ Wo.T, ego + eWo)
    goh = heads(go)
    gP = goh @ s["vh"].transpose(0, 1, 3, 2)
    egp = ego2 + s["ev"]
    gv, egv = q8(s["P"].transpose(0, 1, 3, 2) @ goh, -8 + ego2)
    P = s["P"]
    s_lin = shr((P * gP).sum(-1, keepdims=True), 8)
    dS, eds = q8(P * (gP - s_lin), egp - 8 - 1)
    gq, egq = q8(dS @ s["kh"], eds + s["ek"])
    gk, egk = q8(dS.transpose(0, 1, 3, 2) @ s["qh"], eds + s["eq"])
    if pos == "rope":
        def rope_inv(g, eg):
            a, b = g[..., 0:2], g[..., 2:4]
            c, sn = COS_Q14[None, None], SIN_Q14[None, None]
            return q8(shr(np.concatenate([a * c + b * sn, -a * sn + b * c], -1), 14), eg)
        gq, egq = rope_inv(gq, egq)
        gk, egk = rope_inv(gk, egk)
    gq, gk, gv = junta(gq), junta(gk), junta(gv)
    gWq = np.einsum("btd,bte->de", xn, gq)
    acc_e, acc = None, None
    for gg, eg, W in ((gq, egq, p["Wq"]), (gk, egk, p["Wk"]), (gv, egv, p["Wv"])):
        m, e = q8(gg @ W[0].T, eg + W[1])
        acc, acc_e = (m, e) if acc is None else q8(*soma_alinhada(acc, acc_e, m, e))
    gx, egx = acc, acc_e

    tx = torch.tensor(deq(xn, -5), requires_grad=True)
    tw = {k: torch.tensor(deq(*v), requires_grad=True) for k, v in p.items()}
    th = lambda x: x.reshape(LOTE, T, H, HD).transpose(1, 2)
    q, k, v = th(tx @ tw["Wq"]), th(tx @ tw["Wk"]), th(tx @ tw["Wv"])
    if pos == "rope":
        c, sn = torch.tensor(np.cos(ANG)), torch.tensor(np.sin(ANG))
        rot = lambda x: torch.cat([x[..., 0:2] * c - x[..., 2:4] * sn, x[..., 0:2] * sn + x[..., 2:4] * c], -1)
        q, k = rot(q), rot(k)
    z = q @ k.transpose(-1, -2) / 2.0
    if pos == "alibi":
        z = z + torch.tensor(alibi_f())
    a = torch.softmax(z.masked_fill(torch.tensor(MASCARA), -1e30), -1)
    o = (a @ v).transpose(1, 2).reshape(LOTE, T, D)
    ((o @ tw["Wo"]) * torch.tensor(deq(g_out, ego))).sum().backward()
    return {"dX": cos_sim(deq(gx, egx), tx.grad.numpy()),
            "dW_q": cos_sim(gWq.astype(np.float64), tw["Wq"].grad.numpy()),
            "dW_o": cos_sim(gWo.astype(np.float64), tw["Wo"].grad.numpy()),
            "zeros em dX": float(np.mean(gx == 0))}


# ============================================================================
# Parte 4: somar sem nenhum valor maior que INT8
# ============================================================================
def somas_so_int8(rng, K, testes=2000):
    a = np.clip(np.round(rng.normal(0, 32, (testes, K))), -127, 127).astype(np.int64)
    b = np.clip(np.round(rng.normal(0, 32, (testes, K))), -127, 127).astype(np.int64)
    exato = (a * b).sum(-1).astype(np.float64)
    escala = np.sqrt(np.mean(exato ** 2))
    res = {}

    m, e = q8((a * b).sum(-1), 0)
    res["acumulador int32 (como hoje), so a saida vira INT8"] = (deq(m, e), bits((a * b).sum(-1)))

    p = np.clip(shr(a * b, 7), -127, 127)
    acc = np.zeros(testes, dtype=np.int64)
    for i in range(K):
        acc = np.clip(acc + p[:, i], -127, 127)
    res["acumulador INT8 saturando (produto >>7)"] = (acc * 128.0, 15)

    p = np.clip(shr(a * b, 7), -127, 127)
    n = 0
    while p.shape[1] > 1:
        p = np.clip(shr(p[:, 0::2], 1) + shr(p[:, 1::2], 1), -127, 127)
        n += 1
    res["arvore INT8: (a>>1)+(b>>1) a cada nivel (produto >>7)"] = (p[:, 0] * 128.0 * 2 ** n, 15)

    a4, b4 = shr(a, 4), shr(b, 4)
    p = a4 * b4
    n = 0
    while p.shape[1] > 1:
        p = np.clip(shr(p[:, 0::2], 1) + shr(p[:, 1::2], 1), -127, 127)
        n += 1
    res["tudo <= 8 bits: operandos >>4, produto e arvore em INT8"] = (p[:, 0] * 256.0 * 2 ** n, 8)

    return {k: (float(np.sqrt(np.mean((v - exato) ** 2)) / escala),
                float(np.corrcoef(v, exato)[0, 1]) if np.std(v) > 0 else 0.0, bmax)
            for k, (v, bmax) in res.items()}


# ============================================================================
def main():
    print(f"Configuracao: D={D}, {H} cabecas x head_dim {HD}, FF={FF}, {CAMADAS} camadas, "
          f"T={T}, lote={LOTE}\n")

    print("=" * 100)
    print("PARTE 1 - passo a passo da camada 1 (ALiBi + ReGLU; RoPE e SiLU listados a parte)")
    print("=" * 100)
    for pos, mlp in (("alibi", "reglu"), ("rope", "relu2"), ("rope", "silu")):
        rng = np.random.default_rng(SEMENTE)
        reg = Registro()
        p = camada_pesos(rng, mlp)
        x = np.clip(np.round(rng.normal(0, 40, (LOTE, T, D))), -127, 127).astype(np.int64)
        camada_int(x, -5, p, pos, mlp, reg, 1)
        vistos = set()
        print(f"\n--- {pos} + {mlp} ---")
        print(f"{'operacao':60s} {'bits exatos':>11s}  {'onde vive':14s} {'erro local':>10s}")
        for _, op, b, onde, erro in reg.linhas:
            if pos != "alibi" and not any(t in op for t in ("RoPE", "SiLU", "silu", "ReLU^2", "relu2")):
                continue
            if op in vistos:
                continue
            vistos.add(op)
            marca = "  <-- passa de 8 bits" if b > 8 else ""
            err = f"{erro:10.2e}" if erro is not None else " " * 10
            print(f"{op:60s} {b:11d}  {onde:14s} {err}{marca if onde == 'registrador' else ''}")

    print("\n" + "=" * 100)
    print("PARTE 2 - erro acumulado ao longo de 12 camadas (inteiro vs float64, mesmos pesos)")
    print("=" * 100)
    print(f"{'combinacao':16s} " + " ".join(f"c{c:<5d}" for c in (1, 3, 6, 9, 12)) + "  maior bits num registrador")
    for pos in ("alibi", "rope"):
        for mlp in ("relu2", "reglu", "silu"):
            rng = np.random.default_rng(SEMENTE)
            x0 = np.clip(np.round(rng.normal(0, 40, (LOTE, T, D))), -127, 127).astype(np.int64)
            x, ex, xf = x0, -5, deq(x0, -5)
            reg = Registro()
            erros = []
            for c in range(1, CAMADAS + 1):
                p = camada_pesos(rng, mlp)
                x, ex = camada_int(x, ex, p, pos, mlp, reg, c)
                xf = camada_float(xf, p, pos, mlp)
                erros.append(erro_rel(deq(x, ex), xf))
            maxb = max(b for _, _, b, onde, _ in reg.linhas if onde == "registrador")
            print(f"{pos + '+' + mlp:16s} " + " ".join(f"{erros[c - 1]:.3f}" for c in (1, 3, 6, 9, 12))
                  + f"  {maxb}")

    print("\n" + "=" * 100)
    print("PARTE 3 - backward local: similaridade de cosseno gradiente inteiro x float (1.0 = identico)")
    print("=" * 100)
    for pos in ("alibi", "rope"):
        r = backward_atencao(np.random.default_rng(SEMENTE), pos)
        print(f"atencao {pos:6s}: " + " | ".join(f"{k} {v:.4f}" for k, v in r.items()))
    for mlp in ("relu2", "reglu", "silu"):
        r = backward_mlp(np.random.default_rng(SEMENTE), mlp)
        print(f"MLP {mlp:6s}    : " + " | ".join(f"{k} {v:.4f}" for k, v in r.items()))

    print("\n" + "=" * 100)
    print("PARTE 4 - produto escalar sem nenhum valor acima de 8 bits (2000 testes)")
    print("=" * 100)
    for K in (256, 1024):
        print(f"\nK = {K} termos (dimensao do produto escalar)")
        print(f"{'esquema':62s} {'erro rel':>9s} {'correlacao':>11s} {'maior valor':>12s}")
        for k, (erro, corr, bmax) in somas_so_int8(np.random.default_rng(SEMENTE), K).items():
            print(f"{k:62s} {erro:9.3f} {corr:11.4f} {str(bmax) + ' bits':>12s}")


if __name__ == "__main__":
    main()
