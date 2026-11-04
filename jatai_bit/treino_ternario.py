"""Treino que CONVERGE para pesos ternários sem STE: o forward/backward continua o observador honesto (médias θ), e
depois de cada update um puxão explícito leva cada contador ao nível ternário mais próximo (0 se |θ| <= τ, senão
±127), com força crescente na segunda metade do treino. No fim a rede de médias JÁ É a rede ternária, e o treino
viu (e corrigiu) o efeito de cada peso indeciso virar 0.

Puxão (inteiro, estocástico): θ -= arred_estocástico((θ - alvo) * r / 256), r sobe linearmente de 0 a R_MAX.
Uso: python treino_ternario.py <passos> <τ> <R_MAX> [início do puxão, fração]   -> logs/ternario_modelo_*.pt
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
from modelo_bit import ModeloBit, Log, DEV, I8, I32             # noqa: E402

import treino as dados                                          # noqa: E402


def puxar(th, tau, r, g):
    t = th.to(I32)
    alvo = torch.where(t.abs() > tau, torch.where(t > 0, 127, -127), 0)
    v = (t - alvo) * r                                          # em unidades de 1/256
    base = torch.div(v, 256, rounding_mode="floor")
    resto = v - base * 256
    u = torch.randint(0, 256, t.shape, generator=g, device=DEV, dtype=I32)
    th.copy_((t - base - (u < resto).to(I32)).clamp(-127, 127).to(I8))


def main():
    passos, tau, rmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ini = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    treino_txt, val_txt = dados.carregar()
    gv = torch.Generator().manual_seed(123)
    dados.BT = 8
    lotes = [tuple(t.to(DEV) for t in dados.lote(val_txt, gv)) for _ in range(8)]
    log = Log("treino_ternario", vigiar=False, eco=False)
    log.vigiar = False
    m = ModeloBit(4, 256, 1024, 6, 1, log)
    g, gp = torch.Generator().manual_seed(2), torch.Generator(device=DEV).manual_seed(5)
    p0 = int(passos * ini)
    for p in range(passos + 1):
        if p % 500 == 0:
            lv = sum(mb.nats(m.passo(x, y, treinar=False)[0], y) for x, y in lotes) / len(lotes)
            am = sum(float((m.passo(x, y, treinar=False)[0].argmax(1) == y.reshape(-1)).float().mean()) for x, y in lotes) / len(lotes)
            th = torch.cat([t.view(-1) for _, t in m.tensores()])
            print(f"passo {p:5d} | médias: loss {lv:.3f} acurácia {am:.1%} | contadores em 0: {float((th == 0).float().mean()):.0%}, "
                  f"em ±127: {float((th.abs() == 127).float().mean()):.0%}", flush=True)
        if p == passos:
            break
        log.passo = p
        tok, alvo = dados.lote(treino_txt, g)
        m.passo(tok.to(DEV), alvo.to(DEV))
        if p >= p0:
            r = 1 + (rmax - 1) * (p - p0) // max(passos - 1 - p0, 1)
            for _, t in m.tensores():
                puxar(t, tau, r, gp)
    arq = os.path.join(AQUI, "logs", f"ternario_modelo_{passos}_t{tau}_r{rmax}_i{ini}.pt")
    torch.save([t.cpu() for _, t in m.tensores()], arq)
    print("salvo", arq)


if __name__ == "__main__":
    main()
