"""INT8 com expoente por bloco de 32 valores (mesma ideia do MXINT8 da especificacao OCP
Microscaling). Mede a deriva em 12 camadas contra float64 e gera a auditoria passo a passo
de uma camada.

Configuracao (igual ao experimento_analise_int8): D=256, 64 cabecas x head_dim 4, FF=1024,
12 camadas, T=64, lote 2, ALiBi + ReLU^2.

Representacao: valor real = mantissa (INT8, -127..127) * 2^E, um E inteiro por bloco de 32
valores consecutivos da ultima dimensao. Tudo que e guardado e INT8 (mais o expoente do bloco).
Valores maiores so existem no registrador, dentro de uma operacao.
"""
import math
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "experimento_analise_int8"))
import analise_passo_a_passo as a  # noqa: E402

D, H, HD, T, LOTE, FF, CAMADAS = a.D, a.H, a.HD, a.T, a.LOTE, a.FF, a.CAMADAS
G_MATMUL = 6   # bits de guarda ao alinhar blocos com expoentes diferentes
G_RMS = 8      # bits de guarda na soma dos quadrados (par, para a raiz)
INV_BITS = 10  # precisao do reciproco 1/r da RMSNorm (mantissa de ~10-11 bits)
CFG = {"B": 32, "LIM": 127, "FUNDIDO": False}
BITS = {}
SAT = {"saturados": 0, "total": 0}
MUITO_NEG = np.iinfo(np.int64).min // 4


# ============================================================================
# primitivas
# ============================================================================
def bitlen(x):
    x = np.abs(x)
    bl = np.zeros(x.shape, dtype=np.int64)
    for k in range(int(x.max()).bit_length()):
        bl = np.where((x >> k) > 0, k + 1, bl)
    return bl


def anota(nome, x):
    BITS[nome] = max(BITS.get(nome, 0), int(np.abs(x).max()).bit_length() + 1)


def desloca(x, s):
    """x * 2^-s com arredondamento (s > 0: shift a direita; s < 0: shift a esquerda)."""
    s = np.broadcast_to(np.asarray(s, dtype=np.int64), x.shape)
    pos = np.minimum(np.maximum(s, 0), 62)
    neg = np.minimum(np.maximum(-s, 0), 62)
    meio = np.where(pos > 0, np.left_shift(np.int64(1), np.maximum(pos - 1, 0)), 0)
    return np.where(pos > 0, (x + meio) >> pos, x << neg)


def expande(E):
    return np.repeat(E, CFG["B"], axis=-1)


def quant_blocos(acc, e, lim=None):
    """acc * 2^e (e por elemento) -> mantissa em [-LIM, LIM] + expoente por bloco de B valores.
    O expoente do bloco e o menor que faz o maior valor do bloco caber em LIM."""
    B, LIM = CFG["B"], (lim or CFG["LIM"])
    e = np.broadcast_to(np.asarray(e, dtype=np.int64), acc.shape)
    topo = np.where(acc != 0, e + bitlen(acc), MUITO_NEG)
    forma = (*acc.shape[:-1], acc.shape[-1] // B, B)
    E = topo.reshape(forma).max(-1) - (LIM.bit_length())
    E = np.where(E < MUITO_NEG // 2, 0, E)
    for _ in range(4):
        m = desloca(acc, expande(E) - e)
        passa = np.abs(m).reshape(forma).max(-1) > LIM
        if not passa.any():
            break
        E = E + passa
    m = desloca(acc, expande(E) - e)
    SAT["saturados"] += int(np.sum(np.abs(m) > LIM))
    SAT["total"] += m.size
    return np.clip(m, -LIM, LIM), E


def deqb(m, E):
    return m.astype(np.float64) * 2.0 ** expande(E).astype(np.float64)


# ============================================================================
# operacoes da camada
# ============================================================================
def matmul_blocos(xm, xE, W, nome, R=None, lim=None):
    Wm, We = W
    B, nb = CFG["B"], xm.shape[-1] // CFG["B"]
    parc = np.einsum("...bk,bkn->...bn", xm.reshape(*xm.shape[:-1], nb, B), Wm.reshape(nb, B, -1))
    emax = xE.max(-1, keepdims=True)
    al = desloca(parc << G_MATMUL, (emax - xE)[..., None])
    acc = al.sum(-2)
    base = emax + We - G_MATMUL
    anota(f"{nome}: parcial de 32 termos", parc)
    anota(f"{nome}: parcial com guarda (<<{G_MATMUL})", parc << G_MATMUL)
    anota(f"{nome}: acumulador alinhado", acc)
    m, E = quant_blocos(acc, base, lim)
    if R is not None:
        R[nome] = dict(xm=xm, xE=xE, Wm=Wm, We=We, parc=parc, emax=emax, al=al, acc=acc, base=base, m=m, E=E)
    return m, E


def rms_blocos(xm, xE, nome, R=None):
    B, nb = CFG["B"], D // CFG["B"]
    sq = (xm.reshape(*xm.shape[:-1], nb, B) ** 2).sum(-1)
    emax = xE.max(-1, keepdims=True)
    al = desloca(sq << G_RMS, 2 * (emax - xE))
    ss = al.sum(-1, keepdims=True)
    r = a.isqrt(ss // D)
    # 1/r como mantissa inteira de ~INV_BITS bits + expoente: inv = 2^k // r, k = bits(r) + INV_BITS
    k = bitlen(r) + INV_BITS
    inv = np.left_shift(np.int64(1), k) // r
    prod = xm * inv
    e = expande(xE) - k - emax + G_RMS // 2
    anota(f"{nome}: soma dos quadrados do bloco", sq)
    anota(f"{nome}: soma alinhada (ss)", ss)
    anota(f"{nome}: m*inv", prod)
    m, E = quant_blocos(prod, e)
    if R is not None:
        R[nome] = dict(xm=xm, xE=xE, sq=sq, emax=emax, al=al, ss=ss, r=r, k=k, inv=inv, prod=prod, e=e, m=m, E=E)
    return m, E


def cab_exp(E):
    return E[:, :, (np.arange(H) * HD) // CFG["B"]].transpose(0, 2, 1)


def atencao_blocos(xm, xE, p, R=None):
    lim_qk = 2047 if CFG["FUNDIDO"] else None  # fundido: Q e K ficam com 12 bits no registrador
    PB, G_PV = (12, 3) if CFG["FUNDIDO"] else (8, G_MATMUL)  # fundido: P com 12 bits
    q, Eq = matmul_blocos(xm, xE, p["Wq"], "Q", R, lim_qk)
    k, Ek = matmul_blocos(xm, xE, p["Wk"], "K", R, lim_qk)
    v, Ev = matmul_blocos(xm, xE, p["Wv"], "V", R)
    qh, kh, vh = a.heads(q), a.heads(k), a.heads(v)
    eq, ek, ev = cab_exp(Eq), cab_exp(Ek), cab_exp(Ev)
    S = qh @ kh.transpose(0, 1, 3, 2)
    es = eq[..., :, None] + ek[..., None, :] - 1
    u = desloca(S, -(es + 8))
    vies = np.maximum(a.DIST, 0)[None, None] << (8 - a.INCL)[None, :, None, None]
    u = np.where(a.MASCARA, MUITO_NEG, u - vies)
    dd = u.max(-1, keepdims=True) - u
    ex = np.where(a.MASCARA, 0, a.exp_int(dd))
    ex12 = ex >> 3
    soma = ex12.sum(-1, keepdims=True)
    inv = (1 << 28) // soma
    P = np.minimum((ex12 * inv + (1 << (27 - PB))) >> (28 - PB), (1 << PB) - 1)
    anota("scores q.k", S)
    anota("score em unidades de 2^-8 nat (com ALiBi)", np.where(a.MASCARA, 0, u))
    anota("softmax: soma das exp (Q12)", soma)
    anota("softmax: exp*(2^28/soma)", ex12 * inv)
    emv = ev.max(-1, keepdims=True)
    vl = desloca(vh << G_PV, (emv - ev)[..., None])
    acc = P @ vl
    eacc = -PB + emv - G_PV
    anota("P@V: acumulador", acc)
    accm = a.junta(acc)
    e_elem = a.junta(np.broadcast_to(eacc[..., None], acc.shape).copy())
    om, Eo = quant_blocos(accm, e_elem)
    if R is not None:
        R["att"] = dict(qh=qh, kh=kh, vh=vh, eq=eq, ek=ek, ev=ev, S=S, es=es, u=u, dd=dd, ex=ex, ex12=ex12,
                        soma=soma, inv=inv, P=P, emv=emv, vl=vl, acc=acc, eacc=eacc, accm=accm, e_elem=e_elem,
                        om=om, Eo=Eo)
    return matmul_blocos(om, Eo, p["Wo"], "O", R)


def mlp_blocos(xm, xE, p, R=None):
    h, Eh = matmul_blocos(xm, xE, p["Wup"], "MLP up", R)
    r = np.maximum(h, 0)
    a2 = r * r
    anota("ReLU^2: x*x", a2)
    am, Ea = quant_blocos(a2, 2 * expande(Eh))
    if R is not None:
        R["relu2"] = dict(h=h, Eh=Eh, a2=a2, am=am, Ea=Ea)
    return matmul_blocos(am, Ea, p["Wd"], "MLP down", R)


def soma_blocos(xm, xE, ym, yE, nome, R=None):
    ex, ey = expande(xE), expande(yE)
    e = np.minimum(ex, ey)
    dx, dy = ex - e, ey - e
    BITS["somas com diferenca de expoente > 40 (cortadas)"] = (
        BITS.get("somas com diferenca de expoente > 40 (cortadas)", 0) + int(np.sum(np.maximum(dx, dy) > 40)))
    s = (xm << np.minimum(dx, 40)) + (ym << np.minimum(dy, 40))
    anota(f"{nome}: soma alinhada", s)
    m, E = quant_blocos(s, e)
    if R is not None:
        R[nome] = dict(xm=xm, xE=xE, ym=ym, yE=yE, e=e, s=s, m=m, E=E)
    return m, E


def camada_blocos(xm, xE, p, R=None):
    if R is not None:
        R["entrada"] = dict(xm=xm, xE=xE)
    nm, nE = rms_blocos(xm, xE, "RMSNorm 1", R)
    ym, yE = atencao_blocos(nm, nE, p, R)
    xm, xE = soma_blocos(xm, xE, ym, yE, "residual 1", R)
    nm, nE = rms_blocos(xm, xE, "RMSNorm 2", R)
    ym, yE = mlp_blocos(nm, nE, p, R)
    return soma_blocos(xm, xE, ym, yE, "residual 2", R)


# ============================================================================
# deriva em 12 camadas
# ============================================================================
def entrada(rng):
    return np.clip(np.round(rng.normal(0, 40, (LOTE, T, D))), -127, 127).astype(np.int64)


def deriva(bloco):
    rng = np.random.default_rng(a.SEMENTE)
    x0 = entrada(rng)
    xf = a.deq(x0, -5)
    if bloco is None:
        x, ex = x0, -5
    else:
        CFG["B"] = bloco
        x, E = x0, np.full((LOTE, T, D // bloco), -5, dtype=np.int64)
    erros = []
    for _ in range(CAMADAS):
        p = a.camada_pesos(rng, "relu2")
        if bloco is None:
            x, ex = a.camada_int(x, ex, p, "alibi", "relu2", a.Registro(), 0)
            real = a.deq(x, ex)
        else:
            x, E = camada_blocos(x, E, p)
            real = deqb(x, E)
        xf = a.camada_float(xf, p, "alibi", "relu2")
        erros.append(a.erro_rel(real, xf))
    return erros


# ============================================================================
# auditoria: refaz as contas de um token / uma cabeca em aritmetica escalar e confere
# ============================================================================
def shr_s(x, s):
    x, s = int(x), int(s)
    if s > 0:
        return (x + (1 << (s - 1))) >> s
    return x << -s


def exp_int_s(d, linhas):
    t = min(int(d), 6000) * a.LOG2E_Q14
    n, f = t >> 22, (t & ((1 << 22) - 1)) >> 7
    p = a.POLI_2NEG[0]
    for c in a.POLI_2NEG[1:]:
        p = ((p * f) >> 15) + c
    ex = 0 if n > 30 else p >> n
    linhas.append(f"    t = min(d,6000)·23637 = {t};  n = t>>22 = {n};  f = (t mod 2^22)>>7 = {f};  "
                  f"polinômio(f) = {p};  exp = p>>n = {ex}  (real {ex / 32768:.6f}, float {math.exp(-d / 256):.6f})")
    return ex


def real(m, e):
    return f"{int(m)}·2^{int(e)} = {int(m) * 2.0 ** int(e):.6g}"


def auditoria(R, xf_saida, b=0, t=5, h=0):
    L = []
    w = L.append
    B = CFG["B"]
    blk = (h * HD) // B
    dims = range(h * HD, h * HD + HD)
    w("# Auditoria passo a passo — camada 1\n")
    w(f"Lote {b}, token t={t}, cabeça {h} (dimensões {dims.start}–{dims.stop - 1}, bloco {blk}). "
      f"Bloco = {B} valores. Valor real = mantissa × 2^expoente. Cada passo mostra a fórmula, os "
      "inteiros, o valor real e a referência em float64 (calculada das mesmas entradas inteiras). "
      "Toda conta abaixo foi refeita em Python escalar e conferida contra o cálculo vetorizado.\n")

    # ---------------- entrada
    x, xE = R["entrada"]["xm"][b, t], R["entrada"]["xE"][b, t]
    w("## 1. Entrada\n")
    w(f"Expoentes dos {D // B} blocos do token: {xE.tolist()}")
    w(f"Mantissas do bloco {blk}: {x[blk * B:(blk + 1) * B].tolist()}\n")

    # ---------------- RMSNorm 1
    r1 = R["RMSNorm 1"]
    w("## 2. RMSNorm 1\n")
    w("Fórmulas: sq_b = Σ m² no bloco b;  al_b = (sq_b << 8) >> 2·(Emax − E_b);  ss = Σ al_b;  "
      "r = isqrt(ss // D)  →  rms real = r·2^(Emax − 4).\n")
    emax = int(r1["emax"][b, t, 0])
    ss = 0
    for bb in range(D // B):
        sq = sum(int(v) ** 2 for v in x[bb * B:(bb + 1) * B])
        assert sq == r1["sq"][b, t, bb]
        al = shr_s(sq << G_RMS, 2 * (emax - int(xE[bb])))
        assert al == r1["al"][b, t, bb]
        ss += al
        w(f"- bloco {bb}: E={int(xE[bb])}, sq={sq}, deslocamento 2·({emax}−{int(xE[bb])})={2 * (emax - int(xE[bb]))}, al={al}")
    assert ss == r1["ss"][b, t, 0]
    r = math.isqrt(ss // D)
    assert r == r1["r"][b, t, 0]
    xreal = a.deq(x, 0) * 2.0 ** expande(xE).astype(np.float64)
    w(f"\nEmax={emax};  ss={ss};  ss//D={ss // D};  r=isqrt={r};  rms real = {r}·2^{emax - G_RMS // 2} = "
      f"{r * 2.0 ** (emax - G_RMS // 2):.6g}  (float64: {np.sqrt(np.mean(xreal ** 2)):.6g})\n")
    k = r.bit_length() + INV_BITS
    inv = (1 << k) // r
    assert inv == r1["inv"][b, t, 0]
    e_blk = int(xE[blk]) - k - emax + G_RMS // 2
    w(f"Recíproco por linha (uma divisão só): k = bits(r) + {INV_BITS} = {k};  inv = 2^k // r = {inv}  "
      f"(1/r ≈ inv·2^−k).  Cada elemento: produto = m·inv (≤ 18 bits);  expoente = E_b − k − Emax + 4.\n")
    for d in dims:
        prod = int(x[d]) * inv
        assert prod == r1["prod"][b, t, d]
        w(f"- dim {d}: m={int(x[d])}, produto = {int(x[d])}·{inv} = {prod}, expoente {e_blk} → real {prod * 2.0 ** e_blk:.6g}  "
          f"(float64: {xreal[d] / np.sqrt(np.mean(xreal ** 2)):.6g})")
    pb = r1["prod"][b, t, blk * B:(blk + 1) * B]
    topo = max(e_blk + int(abs(v)).bit_length() for v in pb if v != 0)
    E0 = int(r1["E"][b, t, blk])
    w(f"\nBloco {blk} vira INT8 (mantissa em ±{CFG['LIM']}): maior (expoente + bits) do bloco = {topo}  →  E = {E0};  "
      f"mantissa = produto >> ({E0} − ({e_blk}))")
    for d in dims:
        m = max(-CFG["LIM"], min(CFG["LIM"], shr_s(int(r1["prod"][b, t, d]), E0 - e_blk)))
        assert m == r1["m"][b, t, d]
        w(f"- dim {d}: {real(m, E0)}")

    # ---------------- projecao Q
    rq = R["Q"]
    w("\n## 3. Projeção Q (produto escalar de 256 termos em 8 blocos de 32)\n")
    w(f"Fórmulas: parcial_b = Σ_(k no bloco b) xn_k·W_kn  (exato);  alinhado_b = (parcial_b << {G_MATMUL}) >> (Emax − E_b);  "
      f"acc = Σ alinhado_b;  expoente de acc = Emax + E_W − {G_MATMUL}.\n")
    xn, xnE, Wm, We = rq["xm"][b, t], rq["xE"][b, t], rq["Wm"], rq["We"]
    em = int(xnE.max())
    w(f"Expoentes dos blocos da entrada normalizada: {xnE.tolist()};  Emax={em};  E_W={We}\n")
    for n in dims:
        acc = 0
        partes = []
        for bb in range(D // B):
            parc = int(sum(int(xn[k]) * int(Wm[k, n]) for k in range(bb * B, (bb + 1) * B)))
            assert parc == rq["parc"][b, t, bb, n]
            al = shr_s(parc << G_MATMUL, em - int(xnE[bb]))
            acc += al
            partes.append(f"{parc}→{al}")
        assert acc == rq["acc"][b, t, n]
        ref = float(deqb(xn[None], xnE[None])[0] @ (Wm[:, n] * 2.0 ** We))
        w(f"- saída {n}: parciais→alinhados [{', '.join(partes)}];  acc={acc} → real {acc * 2.0 ** (em + We - G_MATMUL):.6g}  (float64: {ref:.6g})")
    qE = int(rq["E"][b, t, blk])
    w(f"\nBloco {blk} da saída Q vira INT8 com E={qE}:")
    for n in dims:
        w(f"- q[{n}] = {real(rq['m'][b, t, n], qE)}")
    for nome in ("K", "V"):
        rr = R[nome]
        w(f"\n{nome} da cabeça {h} (mesmo cálculo), tokens 0..{t}:")
        for j in range(t + 1):
            w(f"- token {j}: E={int(rr['E'][b, j, blk])}, mantissas {rr['m'][b, j, dims.start:dims.stop].tolist()}")

    # ---------------- scores e softmax
    at = R["att"]
    w(f"\n## 4. Scores e softmax (cabeça {h}, consulta t={t})\n")
    w("Fórmulas: S = Σ_(4 dims) q·k (exato);  es = E_q + E_k − 1 (o −1 é o ÷√4 = ÷2, exato);  "
      "u = S·2^(es+8) (score em unidades de 1/256 nat);  ALiBi: u' = u − (t−j)·2^(8−s) "
      f"(inclinação 2^−s, s={int(a.INCL[h])} nesta cabeça);  d = max(u') − u';  "
      "exp(−d/256) por shift + polinômio;  P = exp/Σexp em 0..255.\n")
    us = []
    for j in range(t + 1):
        qv, kv = at["qh"][b, h, t], at["kh"][b, h, j]
        S = int(sum(int(qv[i]) * int(kv[i]) for i in range(HD)))
        assert S == at["S"][b, h, t, j]
        es = int(at["eq"][b, h, t]) + int(at["ek"][b, h, j]) - 1
        u = shr_s(S, -(es + 8))
        vies = (t - j) << (8 - int(a.INCL[h]))
        assert u - vies == at["u"][b, h, t, j]
        us.append(u - vies)
        w(f"- j={j}: S = {' + '.join(f'{int(qv[i])}·{int(kv[i])}' for i in range(HD))} = {S};  es={es};  "
          f"u = {S}·2^{es + 8} = {u};  viés = {vies};  u' = {u - vies}")
    mx = max(us)
    w(f"\nmax(u') = {mx}\n")
    exs = []
    for j in range(t + 1):
        dd = mx - us[j]
        w(f"- j={j}: d = {dd}")
        exs.append(exp_int_s(dd, L) >> 3)
        assert exs[-1] == at["ex12"][b, h, t, j]
    soma = sum(exs)
    inv = (1 << 28) // soma
    zf = np.array([float(at["qh"][b, h, t] * 2.0 ** int(at["eq"][b, h, t]) @ at["kh"][b, h, j] * 2.0 ** int(at["ek"][b, h, j])) / 2
                   - 2.0 ** -int(a.INCL[h]) * (t - j) for j in range(t + 1)])
    pf = np.exp(zf - zf.max()) / np.exp(zf - zf.max()).sum()
    w(f"\nexp em Q12 (exp >> 3): {exs};  soma = {soma};  inv = 2^28 // soma = {inv};  P = (exp·inv + 2^19) >> 20\n")
    for j in range(t + 1):
        P = min((exs[j] * inv + (1 << 19)) >> 20, 255)
        assert P == at["P"][b, h, t, j]
        w(f"- P[{j}] = {P} → {P / 256:.4f}  (float64: {pf[j]:.4f})")

    # ---------------- P@V
    emv = int(at["emv"][b, h, 0])
    w(f"\n## 5. P@V (cabeça {h}, token {t})\n")
    w(f"Fórmulas: V é alinhado ao maior expoente da cabeça entre todos os tokens (Emax_V={emv}): "
      f"v' = (v << {G_MATMUL}) >> (Emax_V − E_v(j));  acc = Σ_j P_j·v'_j;  expoente = −8 + Emax_V − {G_MATMUL}.\n")
    for i, d in enumerate(dims):
        acc = 0
        termos = []
        for j in range(t + 1):
            vl = shr_s(int(at["vh"][b, h, j, i]) << G_MATMUL, emv - int(at["ev"][b, h, j]))
            assert vl == at["vl"][b, h, j, i]
            acc += int(at["P"][b, h, t, j]) * vl
            termos.append(f"{int(at['P'][b, h, t, j])}·{vl}")
        assert acc == at["acc"][b, h, t, i]
        ref = sum(int(at["P"][b, h, t, j]) / 256 * int(at["vh"][b, h, j, i]) * 2.0 ** int(at["ev"][b, h, j]) for j in range(t + 1))
        e = -8 + emv - G_MATMUL
        w(f"- dim {d}: acc = {' + '.join(termos)} = {acc} → {real(acc, e)}  (float64: {ref:.6g})")
    w(f"\nO bloco {blk} da saída (cabeças {blk * B // HD}–{(blk + 1) * B // HD - 1}) vira INT8 com E={int(at['Eo'][b, t, blk])}: "
      f"mantissas {at['om'][b, t, dims.start:dims.stop].tolist()}")

    # ---------------- O, residual, MLP
    ro, res1 = R["O"], R["residual 1"]
    w("\n## 6. Projeção O e residual 1\n")
    w("Projeção O: mesmo cálculo da seção 3.  Residual: e = min(E_x, E_y);  s = (m_x << (E_x − e)) + (m_y << (E_y − e));  vira INT8 por bloco.\n")
    for d in dims:
        ex_, ey_ = int(res1["xE"][b, t, blk]), int(res1["yE"][b, t, blk])
        e = min(ex_, ey_)
        s = (int(res1["xm"][b, t, d]) << (ex_ - e)) + (int(res1["ym"][b, t, d]) << (ey_ - e))
        assert s == res1["s"][b, t, d]
        w(f"- dim {d}: x = {real(res1['xm'][b, t, d], ex_)};  y (saída de O) = {real(res1['ym'][b, t, d], ey_)};  "
          f"s = {s}·2^{e} = {s * 2.0 ** e:.6g}  →  {real(res1['m'][b, t, d], res1['E'][b, t, blk])}")

    r2, up, rl, dn, res2 = R["RMSNorm 2"], R["MLP up"], R["relu2"], R["MLP down"], R["residual 2"]
    w(f"\n## 7. RMSNorm 2 e MLP (ReLU²)\n")
    w(f"RMSNorm 2 (mesmo cálculo da seção 2): r = {int(r2['r'][b, t, 0])};  saída nas dims {dims.start}–{dims.stop - 1}: "
      + ", ".join(real(r2["m"][b, t, d], r2["E"][b, t, blk]) for d in dims))
    w(f"\nMLP up (seção 3 com 1024 saídas). Unidades ocultas 0–3: "
      + ", ".join(real(up["m"][b, t, n], up["E"][b, t, 0]) for n in range(4)))
    w("\nReLU²: a = max(m, 0)² (exato, no máximo 14 bits);  expoente = 2·E;  depois vira INT8 por bloco.\n")
    for n in range(4):
        m = int(rl["h"][b, t, n])
        a2 = max(m, 0) ** 2
        assert a2 == rl["a2"][b, t, n]
        w(f"- oculta {n}: m={m}, E={int(rl['Eh'][b, t, 0])} → a = {a2}·2^{2 * int(rl['Eh'][b, t, 0])} → "
          f"{real(rl['am'][b, t, n], rl['Ea'][b, t, 0])}  (float64: {max(m, 0) ** 2 * 2.0 ** (2 * int(rl['Eh'][b, t, 0])):.6g})")
    w(f"\nMLP down (seção 3 com 1024 entradas em 32 blocos) e residual 2 (seção 6):\n")
    for d in dims:
        w(f"- dim {d}: down = {real(dn['m'][b, t, d], dn['E'][b, t, blk])};  saída da camada = "
          f"{real(res2['m'][b, t, d], res2['E'][b, t, blk])}  (rede toda em float64: {xf_saida[b, t, d]:.6g})")
    w("\nTodas as contas acima conferem exatamente com o cálculo vetorizado (asserts no código).")
    return "\n".join(L)


def main():
    print("Deriva em 12 camadas (ALiBi + ReLU^2) vs float64, bloco de 32, limite da mantissa variando:")
    linhas = ["por tensor (antes, mantissa +-127):   c1  6.49% c3 11.67% c6 15.73% c9 19.25% c12 21.51%"]
    print(linhas[0])
    for lim in (127, 63, 31, 20):
        BITS.clear()
        SAT.update(saturados=0, total=0)
        CFG["LIM"] = lim
        er = deriva(32)
        txt = (f"bloco 32, mantissa +-{lim:<3d}          " + " ".join(f"c{c} {er[c - 1] * 100:5.2f}%" for c in (1, 3, 6, 9, 12))
               + f"  | saturados {SAT['saturados']} de {SAT['total']:,}")
        print(txt, flush=True)
        linhas.append(txt)
        if lim == 127:
            bits32 = dict(BITS)
    linhas.append("referencia BF16:                      c1  0.60% c3  0.90% c6  1.10% c9  1.30% c12  1.50%")
    print(linhas[-1])

    print("\nMaior numero de bits de cada valor intermediario (bloco 32, mantissa +-127, 12 camadas):")
    for k, v in bits32.items():
        print(f"  {k:55s} {v}")

    CFG["B"], CFG["LIM"] = 32, 127
    rng = np.random.default_rng(a.SEMENTE)
    x0 = entrada(rng)
    p = a.camada_pesos(rng, "relu2")
    R = {}
    camada_blocos(x0, np.full((LOTE, T, D // 32), -5, dtype=np.int64), p, R)
    xf = a.camada_float(a.deq(x0, -5), p, "alibi", "relu2")
    with open(os.path.join(AQUI, "auditoria_camada1.md"), "w", encoding="utf-8") as f:
        f.write(auditoria(R, xf))
    with open(os.path.join(AQUI, "resultado_bloco32.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n\nbits maximos (bloco 32):\n"
                + "\n".join(f"{k}: {v}" for k, v in bits32.items()) + "\n")
    print("\nauditoria_camada1.md gerada (todas as contas conferidas)")


if __name__ == "__main__":
    main()
