"""Teste de aprendizado 100% inteiro na tarefa de inducao.

Modos:
  ref_adam / ref_sgd  -> referencia em float (PyTorch), so para provar que a tarefa e aprendivel
  A                   -> SKILL.md (pure-integer-engine) ao pe da letra
  A_sim               -> A com shift simetrico no update (ablacao)
  B                   -> INT8 com expoente rastreado (block floating point), LUT fina,
                         arredondamento estocastico no update
  B_det               -> B com arredondamento deterministico no update (ablacao)
  B_calc              -> B_det sem LUT: exp por shift + polinomio inteiro, e
                         divisoes por elemento trocadas por reciproco por linha

Uso: python teste_int8_inducao.py <modo> <lr> [passos]
  lr: lr_shift (A), k de 2^-k (B, ref_sgd), lr do Adam (ref_adam)
"""
import sys
import time

import numpy as np

V, L_BLOCO, MAX_OFF, T, D, N_CAMADAS = 16, 6, 4, 16, 32, 2
B_TREINO, B_EVAL = 32, 512
MODO = sys.argv[1]
LR = float(sys.argv[2])
PASSOS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
EVAL_CADA = 500


def gerar(rng, b):
    tok = np.zeros((b, T), dtype=np.int64)
    alvo_pos, alvo_tok, alvo_b = [], [], []
    for i in range(b):
        perm = rng.permutation(V)
        bloco = perm[:L_BLOCO]
        seq = rng.choice(perm[L_BLOCO:], size=T)
        s = rng.integers(0, MAX_OFF + 1)
        seq[s:s + L_BLOCO] = bloco
        seq[s + L_BLOCO:s + 2 * L_BLOCO] = bloco
        tok[i] = seq
        for p in range(s + L_BLOCO, s + 2 * L_BLOCO - 1):
            alvo_b.append(i); alvo_pos.append(p); alvo_tok.append(seq[p + 1])
    return tok, np.array(alvo_b), np.array(alvo_pos), np.array(alvo_tok)


MASCARA = np.triu(np.ones((T, T), dtype=bool), k=1)


# ============================================================================
# utilitarios inteiros (nenhum float daqui ate o fim dos modelos A e B)
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


def rdiv(a, b):
    return (2 * a + b) // (2 * b)


def shr(x, s):
    if s > 0:
        return (x + (1 << (s - 1))) >> s
    if s < 0:
        return x << (-s)
    return x


# ============================================================================
# MODELO A -- SKILL.md ao pe da letra
# ============================================================================
LUT_EXP_A = np.array([1, 2, 5, 14, 38, 103, 280, 762, 2072], dtype=np.int64)


def ss(x, alvo, eixos):
    mx = np.abs(x).max(axis=eixos, keepdims=True)
    div = np.where(mx > alvo, (mx + alvo - 1) // alvo, 1)
    return x // div


def softmax_lut_a(cal, mascara):
    e = LUT_EXP_A[np.clip(cal, -4, 4) + 4]
    if mascara is not None:
        e = np.where(mascara, 0, e)
    return (e * 250) // np.maximum(e.sum(-1, keepdims=True), 1)


def rms_a(x):
    r = isqrt((x * x).sum(-1, keepdims=True) // D)
    return x // r, r


def rms_back_a(g, xn, r):
    md = (g * xn).sum(-1, keepdims=True) // D
    mx = np.maximum(np.abs(xn).max(-1, keepdims=True), 1)
    return (g - (md * xn) // mx) // r


class ModeloA:
    def __init__(self, rng):
        ri = lambda *s: rng.integers(-8, 9, size=s).astype(np.int64)
        self.p = {"E": ri(V, D), "P": ri(T, D), "Wout": ri(D, V)}
        for l in range(N_CAMADAS):
            for n in "qkvo":
                self.p[f"W{n}{l}"] = ri(D, D)
        self.sqrt_d = int(isqrt(np.array(D)))

    def forward(self, tok):
        p, c = self.p, {}
        x = ss(p["E"][tok] + p["P"][None], 20, (1, 2))
        for l in range(N_CAMADAS):
            xn, r = rms_a(x)
            q = ss(xn @ p[f"Wq{l}"], 20, (1, 2))
            k = ss(xn @ p[f"Wk{l}"], 20, (1, 2))
            v = ss(xn @ p[f"Wv{l}"], 20, (1, 2))
            sc = (q @ k.transpose(0, 2, 1)) // self.sqrt_d
            cal = ss(sc, 2, (1, 2))
            po = softmax_lut_a(cal, MASCARA)
            o = ss((po @ v) // 250, 20, (1, 2))
            y = ss(o @ p[f"Wo{l}"], 20, (1, 2))
            c[l] = (xn, r, q, k, v, po, o)
            x = x + y
        xn, r = rms_a(x)
        logits = xn @ p["Wout"]
        c["out"] = (xn, r, softmax_lut_a(ss(logits, 4, (1, 2)), None))
        return logits, c

    def treinar(self, tok, ab, ap, at, lr_shift):
        p = self.p
        _, c = self.forward(tok)
        xn, r, po = c["out"]
        g = np.zeros_like(po)
        g[ab, ap] = po[ab, ap]
        g[ab, ap, at] -= 250
        gr = {"Wout": np.einsum("btd,btv->dv", xn, g)}
        gx = ss(rms_back_a(ss(g @ p["Wout"].T, 20, (1, 2)), xn, r), 20, (1, 2))
        for l in reversed(range(N_CAMADAS)):
            xn, r, q, k, v, po, o = c[l]
            gr[f"Wo{l}"] = np.einsum("btd,bte->de", o, gx)
            go = ss(gx @ p[f"Wo{l}"].T, 20, (1, 2))
            gpo = go @ v.transpose(0, 2, 1)
            gv = ss(po.transpose(0, 2, 1) @ go, 20, (1, 2))
            dpg = (po * gpo).sum(-1, keepdims=True) // 250
            gs = ss((po * (gpo - dpg)) // 250, 20, (1, 2))
            gq = ss(gs @ k, 20, (1, 2))
            gk = ss(gs.transpose(0, 2, 1) @ q, 20, (1, 2))
            for n, gg in (("q", gq), ("k", gk), ("v", gv)):
                gr[f"W{n}{l}"] = np.einsum("btd,bte->de", xn, gg)
            gxn = ss(gq @ p[f"Wq{l}"].T + gk @ p[f"Wk{l}"].T + gv @ p[f"Wv{l}"].T, 20, (1, 2))
            gx = ss(gx + ss(rms_back_a(gxn, xn, r), 20, (1, 2)), 20, (1, 2))
        gE = np.zeros_like(p["E"])
        np.add.at(gE, tok, gx)
        gr["E"], gr["P"] = gE, gx.sum(0)
        for n, gw in gr.items():
            g = ss(gw, 100, None)
            # A: >> em negativo arredonda para -inf (-1 >> k == -1), entao todo
            # gradiente negativo vira update -1 e os pesos sobem ate +127.
            # A_sim: unica mudanca, shift simetrico (trunca em direcao ao zero).
            u = np.sign(g) * (np.abs(g) >> lr_shift) if MODO == "A_sim" else g >> lr_shift
            p[n] = np.clip(p[n] - u, -128, 127)


# ============================================================================
# MODELO B -- INT8 com expoente rastreado (block floating point)
# ============================================================================
def q8(x, e):
    mx = int(np.abs(x).max())
    s = 0
    while (mx >> s) > 127:
        s += 1
    return np.clip(shr(x, s), -127, 127), e + s


def soma_alinhada(a, ea, b, eb):
    if ea - eb > 40:
        return q8(a, ea)
    if eb - ea > 40:
        return q8(b, eb)
    e = min(ea, eb)
    return q8((a << (ea - e)) + (b << (eb - e)), e)


def lut_exp_b():
    t = [1 << 15]
    for _ in range(159):
        t.append((t[-1] * 28918 + (1 << 14)) >> 15)  # 28918 = 2^15 * e^(-1/8)
    return np.array(t, dtype=np.int64)


LUT_EXP_B = lut_exp_b()
INV_SQRT_D = 181  # 181/1024 = 1/sqrt(32)
E_EMB, E_W = -5, -6
CALC = MODO == "B_calc"

# exp sem LUT: exp(-d) = 2^(-d*log2(e)); parte inteira vira shift, parte
# fracionaria vem de um polinomio de grau 3 para 2^-f em Q15 (erro rel. max
# 1.9e-4; coeficientes ajustados uma vez por minimos quadrados e fixados).
LOG2E_Q14 = 23637
POLI_2NEG = (-1295, 7556, -22645, 32765)


def exp_int(d):
    """d >= 0 em unidades de 2^-8 nat -> exp(-d) em Q15. Cabe em int32."""
    t = np.minimum(d, 6000) * LOG2E_Q14
    n = t >> 22
    f = (t & ((1 << 22) - 1)) >> 7
    p = POLI_2NEG[0]
    for c in POLI_2NEG[1:]:
        p = ((p * f) >> 15) + c
    return np.where(n > 30, 0, p >> np.minimum(n, 30))


def mul_recip(a, b, bits):
    """a / b (b > 0, uma divisao por linha) como multiplicacao + shift."""
    return (a * ((1 << bits) // b) + (1 << (bits - 1))) >> bits


def softmax_calc(z_m, e_z, mascara):
    u = shr(z_m, -(e_z + 8))
    if mascara is not None:
        u = np.where(mascara, np.iinfo(np.int64).min // 4, u)
    ex = exp_int(u.max(-1, keepdims=True) - u)
    if mascara is not None:
        ex = np.where(mascara, 0, ex)
    s = ex.sum(-1, keepdims=True)
    return np.minimum((ex * ((1 << 30) // s) + (1 << 21)) >> 22, 255)


def softmax_b(z_m, e_z, mascara):
    if CALC:
        return softmax_calc(z_m, e_z, mascara)
    u = shr(z_m, -(e_z + 3))
    if mascara is not None:
        u = np.where(mascara, np.iinfo(np.int64).min // 4, u)
    d = u.max(-1, keepdims=True) - u
    ex = LUT_EXP_B[np.minimum(d, 159)]
    if mascara is not None:
        ex = np.where(mascara, 0, ex)
    return np.minimum(rdiv(ex * 256, ex.sum(-1, keepdims=True)), 255)


def rms_b(m):
    r = isqrt((m * m).sum(-1, keepdims=True) // D)
    xn = mul_recip(m * 32, r, 16) if CALC else rdiv(m * 32, r)
    return np.clip(xn, -127, 127), r


def rms_back_b(mg, eg, ym, r, ex):
    S = (mg * ym).sum(-1, keepdims=True)
    interno = mg * 1024 * D - ym * S
    dx = mul_recip(interno, r * D, 24) if CALC else rdiv(interno, r * D)
    return q8(dx, eg - 10 - ex)


class ModeloB:
    def __init__(self, rng):
        self.rng = rng
        self.p = {"E": rng.integers(-55, 56, (V, D)), "P": rng.integers(-55, 56, (T, D)),
                  "Wout": rng.integers(-20, 21, (D, V))}
        for l in range(N_CAMADAS):
            for n in "qkvo":
                self.p[f"W{n}{l}"] = rng.integers(-20, 21, (D, D))
        self.p = {k: v.astype(np.int64) for k, v in self.p.items()}
        self.exp = {k: (E_EMB if k in ("E", "P") else E_W) for k in self.p}

    def forward(self, tok):
        p, c = self.p, {}
        x, ex = q8(p["E"][tok] + p["P"][None], E_EMB)
        for l in range(N_CAMADAS):
            xn, r = rms_b(x)
            q, eq = q8(xn @ p[f"Wq{l}"], -5 + E_W)
            k, ek = q8(xn @ p[f"Wk{l}"], -5 + E_W)
            v, ev = q8(xn @ p[f"Wv{l}"], -5 + E_W)
            S = shr((q @ k.transpose(0, 2, 1)) * INV_SQRT_D, 10)
            P = softmax_b(S, eq + ek, MASCARA)
            o, eo = q8(P @ v, -8 + ev)
            y, ey = q8(o @ p[f"Wo{l}"], eo + E_W)
            c[l] = (x, ex, xn, r, q, eq, k, ek, v, ev, P, o, eo)
            x, ex = soma_alinhada(x, ex, y, ey)
        xn, r = rms_b(x)
        logits = xn @ p["Wout"]
        c["out"] = (x, ex, xn, r, softmax_b(logits, -5 + E_W, None))
        return logits, c

    def treinar(self, tok, ab, ap, at, k_lr):
        p = self.p
        _, c = self.forward(tok)
        x, ex, xn, r, P = c["out"]
        g = np.zeros_like(P)
        g[ab, ap] = P[ab, ap]
        g[ab, ap, at] -= 256
        eg = -8
        gr = {"Wout": (np.einsum("btd,btv->dv", xn, g), -5 + eg)}
        gxn, egxn = q8(g @ p["Wout"].T, eg + E_W)
        gx, egx = rms_back_b(gxn, egxn, xn, r, ex)
        for l in reversed(range(N_CAMADAS)):
            x, ex, xn, r, q, eq, k, ek, v, ev, P, o, eo = c[l]
            gr[f"Wo{l}"] = (np.einsum("btd,bte->de", o, gx), eo + egx)
            go, ego = q8(gx @ p[f"Wo{l}"].T, egx + E_W)
            gP = go @ v.transpose(0, 2, 1)
            egp = ego + ev
            gv, egv = q8(P.transpose(0, 2, 1) @ go, -8 + ego)
            s_lin = shr((P * gP).sum(-1, keepdims=True), 8)
            dS, eds = q8(shr(P * (gP - s_lin) * INV_SQRT_D, 10), egp - 8)
            gq, egq = q8(dS @ k, eds + ek)
            gk, egk = q8(dS.transpose(0, 2, 1) @ q, eds + eq)
            for n, gg, ee in (("q", gq, egq), ("k", gk, egk), ("v", gv, egv)):
                gr[f"W{n}{l}"] = (np.einsum("btd,bte->de", xn, gg), -5 + ee)
            a, ea = q8(gq @ p[f"Wq{l}"].T, egq + E_W)
            b, eb = q8(gk @ p[f"Wk{l}"].T, egk + E_W)
            a, ea = soma_alinhada(a, ea, b, eb)
            b, eb = q8(gv @ p[f"Wv{l}"].T, egv + E_W)
            gxn, egxn = soma_alinhada(a, ea, b, eb)
            gb, egb = rms_back_b(gxn, egxn, xn, r, ex)
            gx, egx = soma_alinhada(gx, egx, gb, egb)
        gE = np.zeros_like(p["E"])
        np.add.at(gE, tok, gx)
        gr["E"], gr["P"] = (gE, egx), (gx.sum(0), egx)
        for n, (gw, egw) in gr.items():
            s = -(egw - k_lr - self.exp[n])
            if s <= 0:
                u = gw << (-s)
            elif s > 62:
                u = np.zeros_like(gw)
            elif MODO in ("B_det", "B_calc"):
                u = shr(gw, s)
            else:
                u = (gw + self.rng.integers(0, 1 << s, size=gw.shape)) >> s
            p[n] = np.clip(p[n] - u, -127, 127)


# ============================================================================
# REFERENCIA EM FLOAT (mesma arquitetura) -- so para calibrar a dificuldade
# ============================================================================
def rodar_ref(rng, dados_eval):
    import torch
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pr = {"E": torch.randn(V, D), "P": torch.randn(T, D), "Wout": torch.randn(D, V) / D ** 0.5}
    for l in range(N_CAMADAS):
        for n in "qkvo":
            pr[f"W{n}{l}"] = torch.randn(D, D) / D ** 0.5
    pr = {k: v.to(dev).requires_grad_() for k, v in pr.items()}
    mask = torch.tensor(MASCARA, device=dev)
    rms = lambda x: x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)

    def fwd(tok):
        x = pr["E"][tok] + pr["P"][None]
        for l in range(N_CAMADAS):
            xn = rms(x)
            q, k, v = xn @ pr[f"Wq{l}"], xn @ pr[f"Wk{l}"], xn @ pr[f"Wv{l}"]
            s = (q @ k.transpose(1, 2)) / D ** 0.5
            a = torch.softmax(s.masked_fill(mask, -1e9), -1)
            x = x + (a @ v) @ pr[f"Wo{l}"]
        return rms(x) @ pr["Wout"]

    opt = (torch.optim.Adam(pr.values(), lr=LR) if MODO == "ref_adam"
           else torch.optim.SGD(pr.values(), lr=2.0 ** -LR))
    for passo in range(PASSOS + 1):
        if passo % EVAL_CADA == 0:
            with torch.no_grad():
                tok, ab, ap, at = dados_eval
                lg = fwd(torch.tensor(tok, device=dev))
                acc = (lg[ab, ap].argmax(-1).cpu().numpy() == at).mean()
            print(f"passo {passo:5d} | acuracia {acc:6.1%}", flush=True)
        tok, ab, ap, at = gerar(rng, B_TREINO)
        lg = fwd(torch.tensor(tok, device=dev))
        perda = torch.nn.functional.cross_entropy(
            lg[ab, ap], torch.tensor(at, device=dev), reduction="mean" if MODO == "ref_adam" else "sum")
        opt.zero_grad(); perda.backward(); opt.step()


def main():
    rng = np.random.default_rng(1)
    dados_eval = gerar(np.random.default_rng(999), B_EVAL)
    print(f"modo={MODO} lr={LR} passos={PASSOS} | acaso={1 / V:.1%}", flush=True)
    t0 = time.time()
    if MODO.startswith("ref"):
        rodar_ref(rng, dados_eval)
    else:
        m = ModeloA(rng) if MODO.startswith("A") else ModeloB(rng)
        for passo in range(PASSOS + 1):
            if passo % EVAL_CADA == 0:
                tok, ab, ap, at = dados_eval
                logits, _ = m.forward(tok)
                assert logits.dtype == np.int64 and all(v.dtype == np.int64 for v in m.p.values())
                acc = (logits[ab, ap].argmax(-1) == at).mean()
                faixa = max(int(np.abs(v).max()) for v in m.p.values())
                print(f"passo {passo:5d} | acuracia {acc:6.1%} | max|peso| {faixa}", flush=True)
            m.treinar(*gerar(rng, B_TREINO), int(LR))
    print(f"tempo {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
