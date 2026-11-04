"""DIAGNÓSTICO: modelo inteiro (modelo_bit.py) x réplica float exata (float64, derivadas completas), MESMOS pesos
(contador/127) e MESMO lote, etapa por etapa. Mede o cosseno entre as duas versões de cada ativação e de cada
gradiente: a primeira etapa onde o cosseno cai é onde a aritmética inteira se separa da conta que ela aproxima.

O modelo inteiro treina normalmente; nos passos pedidos o estado é congelado, comparado e restaurado.
Uso: JB_VRMS=1 JB_PROJ=1 python comparar_camadas.py <passos separados por vírgula> [L] [D] [F] [shift]
"""
import os
import sys

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import modelo_bit as mb                                        # noqa: E402
from modelo_bit import ModeloBit, Log, V, NB, NORM_T, RES, DEV  # noqa: E402

import treino as dados                                         # noqa: E402  (caminho posto por modelo_bit)


class Gravado(ModeloBit):
    """ModeloBit que guarda as entradas e saídas de cada etapa (em ordem) sem mudar nenhuma conta."""
    def _norm(self, M, nome):
        r = super()._norm(M, nome)
        self.fw.append((nome, M, r[0]))
        return r

    def _voto(self, M, ant, x2, W, nome):
        r = super()._voto(M, ant, x2, W, nome)
        self.fw.append((nome.split()[0], M, r[0]))
        return r

    def _quant(self, X, nome):
        self.bw[nome] = X.clone()
        r = super()._quant(X, nome)
        self.bw[nome + " q"] = r[0].clone()
        return r

    def _renorm(self, X, alvo, nome):
        self.bw[nome] = X.clone()
        return super()._renorm(X, alvo, nome)


def cos(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float(a @ b / (a.norm() * b.norm()).clamp(min=1e-300))


def replica(m, tok, alvo):
    """a mesma conta em float64 com autograd (rms e sigma derivados de verdade)."""
    Bn, T = tok.shape
    f = lambda t: (t.double() / 127).requires_grad_()
    E = f(m.E); out = f(m.out)
    W = [{k: f(v) for k, v in b.items()} for b in m.blocos]
    d = (1 + (torch.arange(m.D, device=DEV) & 7)).double()
    rms = lambda M: M.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
    sig = lambda x, Wt: (x.shape[1] - (x * x) @ (Wt * Wt).t()).clamp(min=1e-6).sqrt()
    norm = lambda M: torch.tanh(0.8 * (NORM_T / 1024) * M / rms(M))
    reter = {}

    def r(nome, t):
        t.retain_grad(); reter[nome] = t
        return t

    def voto(M, ant, xin, Wt):
        return torch.tanh(0.8 * (M + ant * rms(M) * 2 ** RES) / sig(xin, Wt))

    fw = []
    x = r("x0", E[tok.reshape(-1)])
    for l, b in enumerate(W):
        x = r(f"b{l} x", x) if l else x
        Mv = r(f"b{l} Mv", x @ b["Wv"].t()); mv = norm(Mv); fw.append(("v", Mv, mv))
        vv = mv.view(Bn, T, -1); s = torch.zeros_like(vv[:, 0]); Ss = []
        for t in range(T):
            s = s - s / 2 ** d + vv[:, t]; Ss.append(s)
        S = r(f"b{l} S", torch.stack(Ss, 1).view(Bn * T, -1)); ms = norm(S); fw.append(("s", S, ms))
        Mo = r(f"b{l} Mo", ms @ b["Wo"].t()); xa = r(f"b{l} xa", voto(Mo, x, ms, b["Wo"])); fw.append(("xa", Mo, xa))
        Mu = r(f"b{l} Mu", xa @ b["Wu"].t()); mh = norm(Mu); fw.append(("h", Mu, mh))
        Md = r(f"b{l} Md", mh @ b["Wd"].t()); x = voto(Md, xa, mh, b["Wd"]); fw.append(("xb", Md, x))
    x = r("x final", x)
    xo = torch.cat([x, torch.ones(x.shape[0], NB, device=DEV, dtype=x.dtype)], 1)
    Mz = r("Mz", xo @ out.t())
    lg = Mz / sig(xo, out)
    loss = torch.nn.functional.cross_entropy(lg, alvo.reshape(-1))
    loss.backward()
    G = {"E": E.grad, "saida": out.grad}
    for l, b in enumerate(W):
        for k, v in b.items():
            G[f"bloco {l} {k}"] = v.grad
    val = {k: v.detach() for k, v in reter.items()}
    val["W"] = [{k: v.detach() for k, v in b.items()} for b in W]
    val["d"], val["BnT"] = d, (Bn, T)
    return fw, lg.detach(), float(loss), {k: v.grad for k, v in reter.items()}, G, val


def locais(m, val, L):
    """cada kernel do backward isolado: a MESMA entrada inteira passa pela derivada exata (vjp em float64) daquela
    etapa só. cos(kernel) = saída do kernel x vjp exato; cos(quant) = vjp com o gradiente já em int8 x vjp sem quantizar
    (quanto o arredondamento para int8 sozinho tira da direção)."""
    from torch.autograd.functional import vjp
    rms = lambda M: M.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
    sig = lambda x, Wt: (x.shape[1] - (x * x) @ (Wt * Wt).t()).clamp(min=1e-6).sqrt()
    norm = lambda M: torch.tanh(0.8 * (NORM_T / 1024) * M / rms(M))
    voto = lambda M, ant, xin, Wt: torch.tanh(0.8 * (M + ant * rms(M) * 2 ** RES) / sig(xin, Wt))
    Bn, T = val["BnT"]; d = val["d"]

    def scan(v):
        vv = v.view(Bn, T, -1); s = torch.zeros_like(vv[:, 0]); Ss = []
        for t in range(T):
            s = s - s / 2 ** d + vv[:, t]; Ss.append(s)
        return torch.stack(Ss, 1).view(Bn * T, -1)

    def renorm12(X):
        sh = max(int(X.abs().max()).bit_length() - mb.BITS_DS, 0)
        return torch.round(X.double() / 2 ** sh)

    bw, f = m.bw, lambda t: t.double()
    linhas = ["KERNELS ISOLADOS (cos kernel x derivada exata da mesma entrada | cos que o int8 sozinho deixa)"]
    for l in reversed(range(L)):
        W = val["W"][l]
        xa, x = val[f"b{l} xa"], val[f"b{l} x" if l else "x0"]
        mh, ms = norm(val[f"b{l} Mu"]), norm(val[f"b{l} S"])
        xb_up = bw[f"b{l + 1} dx"] if l + 1 < L else m.inter["dx saida"]
        # voto xb: entrada dx (gradiente em xb) -> dMd
        e = vjp(lambda Md: voto(Md, xa, mh, W["Wd"]), val[f"b{l} Md"], f(xb_up))[1]
        linhas.append(f"  b{l} voto xb  -> dMd   cos(kernel) {cos(bw[f'b{l} dMd'], e):.4f}")
        # norm h: entrada dMd -> dMu
        fn = lambda Mu: norm(Mu) @ W["Wd"].t()
        eq = vjp(fn, val[f"b{l} Mu"], f(bw[f"b{l} dMd q"]))[1]
        ec = vjp(fn, val[f"b{l} Mu"], f(bw[f"b{l} dMd"]))[1]
        linhas.append(f"  b{l} norm h   -> dMu   cos(kernel) {cos(bw[f'b{l} dMu'], eq):.4f} | cos(quant dMd) {cos(eq, ec):.4f}")
        # voto xa: entrada dxa -> dMo
        e = vjp(lambda Mo: voto(Mo, x, ms, W["Wo"]), val[f"b{l} Mo"], f(bw[f"b{l} dxa"]))[1]
        linhas.append(f"  b{l} voto xa  -> dMo   cos(kernel) {cos(bw[f'b{l} dMo'], e):.4f}")
        # norm s: entrada dMo -> dS
        fn = lambda S: norm(S) @ W["Wo"].t()
        eq = vjp(fn, val[f"b{l} S"], f(bw[f"b{l} dMo q"]))[1]
        ec = vjp(fn, val[f"b{l} S"], f(bw[f"b{l} dMo"]))[1]
        linhas.append(f"  b{l} norm s   -> dS    cos(kernel) {cos(bw[f'b{l} dS'], eq):.4f} | cos(quant dMo) {cos(eq, ec):.4f}")
        # soma reversa + norm v: entrada dS (12 bits) -> dMv
        fn = lambda Mv: scan(norm(Mv))
        eq = vjp(fn, val[f"b{l} Mv"], renorm12(bw[f"b{l} dS"]))[1]
        ec = vjp(fn, val[f"b{l} Mv"], f(bw[f"b{l} dS"]))[1]
        linhas.append(f"  b{l} scan+v   -> dMv   cos(kernel) {cos(bw[f'b{l} dMv'], eq):.4f} | cos(12 bits dS) {cos(eq, ec):.4f}")
    return "\n".join(linhas)


def comparar(m, tok, alvo):
    estado = [t.clone() for _, t in m.tensores()]
    t0 = m.t
    m.fw, m.bw, m.guardar, m.inter = [], {}, True, {}
    so, _ = m.passo(tok, alvo)
    for (_, t), e in zip(m.tensores(), estado):                # desfaz o update: só a medida
        t.copy_(e)
    m.t, m.guardar = t0, False
    fw, lg, loss_f, gr, G, val = replica(m, tok, alvo)
    loss_i = mb.nats(so, alvo)
    linhas = [f"loss do lote: inteiro {loss_i:.4f} | réplica float {loss_f:.4f} | cos(logits) {cos(so, lg):.4f}"]
    L = m.L
    linhas.append("FORWARD (cos da soma M e da saída da etapa)")
    for i, ((nome, Mi, oi), (_, Mf, of)) in enumerate(zip(m.fw, fw)):
        linhas.append(f"  bloco {i // 5} {nome:3s} cos(M) {cos(Mi, Mf):.4f}  cos(saída) {cos(oi, of):.4f}")
    linhas.append("BACKWARD (cos do gradiente inteiro antes de quantizar x derivada exata)")
    pares = [("dZ", "Mz"), ("dx saida", "x final")]
    for l in reversed(range(L)):
        pares += [(f"b{l} dMd", f"b{l} Md"), (f"b{l} dMu", f"b{l} Mu"), (f"b{l} dxa", f"b{l} xa"),
                  (f"b{l} dMo", f"b{l} Mo"), (f"b{l} dS", f"b{l} S"), (f"b{l} dMv", f"b{l} Mv")]
        pares.append((f"b{l} dx", f"b{l} x" if l else "x0"))
    for ni, nf in pares:
        a = m.bw.get(ni, m.inter.get(ni))
        linhas.append(f"  {ni:14s} cos {cos(a, gr[nf]):.4f}")
    linhas.append("GRADIENTE DOS PESOS (cos G inteiro x G exato)")
    for nome, _ in m.tensores():
        linhas.append(f"  {nome:14s} cos {cos(m.inter['G ' + nome], G[nome]):.4f}")
    linhas.append(locais(m, val, L))
    if os.environ.get("JC_SALVAR"):
        torch.save({"bw": m.bw, "inter": m.inter, "fw": m.fw, "val": val, "gr": gr,
                    "blocos": m.blocos, "out": m.out, "E": m.E}, os.environ["JC_SALVAR"])
    return "\n".join(linhas)


def main():
    marcos = [int(v) for v in sys.argv[1].split(",")]
    L, D, F, shift = (int(v) for v in (sys.argv[2:6] + ["4", "256", "1024", "6"][len(sys.argv[2:6]):]))
    treino_txt, val_txt = dados.carregar()
    dados.BT = 8
    tok_v, alvo_v = (t.to(DEV) for t in dados.lote(val_txt, torch.Generator().manual_seed(123)))
    g = torch.Generator().manual_seed(2)
    log = Log("comparar_camadas", vigiar=False, eco=False)
    log.vigiar = False
    m = Gravado(L, D, F, shift, 1, log)
    for p in range(max(marcos) + 1):
        log.passo = p
        if p in marcos:
            print(f"\n===== passo {p} (VRMS={mb.VRMS} PROJ={mb.PROJ}) =====\n" + comparar(m, tok_v, alvo_v), flush=True)
        if p == max(marcos):
            break
        tok, alvo = dados.lote(treino_txt, g)
        m.fw, m.bw = [], {}
        m.passo(tok.to(DEV), alvo.to(DEV))


if __name__ == "__main__":
    main()
