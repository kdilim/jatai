"""Rede de PULSOS: cada passada é 100% binária (pesos ±1, ativações ±1, só produtos de bits e somas inteiras); a
nuance vem de repetir N passadas com bits sorteados e somar os logits (contagem de pulsos, como em hardware
estocástico / neurônios de spike).

Por que funciona em princípio: a média de campo do treino é o valor esperado de um bit que liga com probabilidade
(1 + m/127)/2. Na rede final antiga cada bit era congelado em sinal(m) e a nuance se perdia; aqui ele é SORTEADO a
cada passada: bit = +1 se m > U, U inteiro uniforme em [-127, 126].

Variantes: pesos sinal|sorteio  x  ativações sinal|sorteio; N = 1, 4, 16, 64, 256 passadas.
Uso: python pulsos.py [passos_treino]   (treina com as correções JB_SAT/VRMS/PROJ e guarda os contadores)
"""
import os
import sys

os.environ.setdefault("JB_SAT", "3584")
os.environ.setdefault("JB_VRMS", "1")
os.environ.setdefault("JB_PROJ", "1")

import torch  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import modelo_bit as mb                                         # noqa: E402
from modelo_bit import ModeloBit, Log, NB, TRES, RES, DEV, I8, I32, I64, P, I, n_, lib, gemm   # noqa: E402

import treino as dados                                          # noqa: E402


def sorteia(m, g):
    """int8 média em [-127, 127] -> bit ±1 com P(+1) = (m + 127) / 254."""
    U = torch.randint(-127, 127, m.shape, generator=g, device=DEV, dtype=I32)
    return torch.where(m.to(I32) > U, 1, -1).to(I8).contiguous()


def passada(m, tok, pesos, ativ, g):
    lg = m.log
    Bn, T = tok.shape
    N = Bn * T
    W = (lambda t: sorteia(t, g)) if pesos == "sorteio" else (lambda t: torch.where(t >= 0, 1, -1).to(I8).contiguous())

    def bits(M):
        if ativ == "sorteio":
            return sorteia(m._norm(M, "ativ")[0], g)
        return torch.where(M >= 0, 1, -1).to(I8).contiguous()

    def voto(M, ant):
        mo, mo2 = torch.empty(M.shape, dtype=I8, device=DEV), torch.empty(M.shape, dtype=I8, device=DEV)
        inv, kk = torch.empty(M.shape[0], dtype=I64, device=DEV), torch.empty(M.shape[0], dtype=I32, device=DEV)
        a8 = (ant.to(I32) * 127).to(I8)
        lg.ok(lib.bt_norm_res_r(P(M), P(a8), I(M.shape[0]), I(M.shape[1]), I(TRES), I(RES), P(mo), P(mo2), P(inv), P(kk)), "norm_res_r")
        return sorteia(mo, g) if ativ == "sorteio" else torch.where(mo >= 0, 1, -1).to(I8)

    x = W(m.E)[tok.reshape(-1)].contiguous()
    for b in m.blocos:
        v = bits(gemm(x, W(b["Wv"])))
        S = torch.empty(N, m.D, dtype=I32, device=DEV)
        lg.ok(lib.bt_scan_fwd(P(v), P(S), I(Bn), I(T), I(m.D)), "scan_fwd")
        s = bits(S)
        xa = voto(gemm(s, W(b["Wo"])), x)
        h = bits(gemm(xa, W(b["Wu"])))
        x = voto(gemm(h, W(b["Wd"])), xa)
    xo = torch.cat([x, torch.ones(N, NB, dtype=I8, device=DEV)], 1).contiguous()
    return gemm(xo, W(m.out))


def avaliar(m, lotes, pesos, ativ, Ns, semente=7):
    g = torch.Generator(device=DEV).manual_seed(semente)
    soma = [torch.zeros(t.numel(), mb.V, dtype=I64, device=DEV) for t, _ in lotes]
    feitos, res = 0, {}
    for Nalvo in Ns:
        while feitos < Nalvo:
            for i, (tok, _) in enumerate(lotes):
                soma[i] += passada(m, tok, pesos, ativ, g).to(I64)
            feitos += 1
        acc = sum(float((s.argmax(1) == a.reshape(-1)).float().mean()) for s, (_, a) in zip(soma, lotes)) / len(lotes)
        # loss calibrada: melhor escala 2^-k dos logits somados (só relatório)
        loss = min(sum(float(torch.nn.functional.cross_entropy(s.double() / Nalvo / 2 ** (k / 2), a.reshape(-1))) for s, (_, a) in zip(soma, lotes)) / len(lotes)
                   for k in range(0, 24))
        res[Nalvo] = (acc, loss)
    return res


def main():
    passos = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    arq = os.path.join(AQUI, "logs", f"pulsos_modelo_{passos}.pt")
    treino_txt, val_txt = dados.carregar()
    gv = torch.Generator().manual_seed(123)
    dados.BT = 8
    lotes = [tuple(t.to(DEV) for t in dados.lote(val_txt, gv)) for _ in range(8)]
    log = Log("pulsos", vigiar=False, eco=False)
    log.vigiar = False
    m = ModeloBit(4, 256, 1024, 6, 1, log)
    if os.path.exists(arq):
        for (_, t), s in zip(m.tensores(), torch.load(arq)):
            t.copy_(s.to(DEV))
    else:
        g = torch.Generator().manual_seed(2)
        for p in range(passos):
            log.passo = p
            tok, alvo = dados.lote(treino_txt, g)
            m.passo(tok.to(DEV), alvo.to(DEV))
        torch.save([t.cpu() for _, t in m.tensores()], arq)
    lv = sum(mb.nats(m.passo(*b_, treinar=False)[0], b_[1]) for b_ in lotes) / len(lotes)
    am = sum(float((m.passo(*b_, treinar=False)[0].argmax(1) == b_[1].reshape(-1)).float().mean()) for b_ in lotes) / len(lotes)
    print(f"modelo {passos} passos (SAT={mb.SAT} VRMS={mb.VRMS} PROJ={mb.PROJ}) | rede de médias: loss {lv:.3f} acurácia {am:.1%}", flush=True)
    Ns = [1, 4, 16, 64, 256]
    for pesos, ativ in (("sinal", "sinal"), ("sinal", "sorteio"), ("sorteio", "sinal"), ("sorteio", "sorteio")):
        r = avaliar(m, lotes, pesos, ativ, [1] if (pesos, ativ) == ("sinal", "sinal") else Ns)
        print(f"pesos {pesos:7s} ativações {ativ:7s} | " + " | ".join(f"N={n}: {a:.1%} loss {l:.3f}" for n, (a, l) in r.items()), flush=True)


if __name__ == "__main__":
    main()
