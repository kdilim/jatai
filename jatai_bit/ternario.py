"""Pesos e ativações TERNÁRIOS {-1, 0, +1} (pulso, sem pulso, pulso invertido) a partir do modelo treinado.

Peso: contador |θ| <= τw vira 0 (o peso indeciso não vota), os outros ±127. Ativação: média |m| <= τa vira 0.
Avalia a rede de médias com pesos ternários (só a perda dos pesos) e a rede final com ativações binárias/ternárias.
Uso: python ternario.py <arquivo de contadores .pt (de pulsos.py)>
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
from modelo_bit import ModeloBit, Log, DEV, I8                  # noqa: E402

import treino as dados                                          # noqa: E402


def tern(t, tau):
    return torch.where(t.abs() > tau, torch.where(t > 0, 127, -127), 0).to(I8).contiguous()


def main():
    arq = sys.argv[1]
    _, val_txt = dados.carregar()
    gv = torch.Generator().manual_seed(123)
    dados.BT = 8
    lotes = [tuple(t.to(DEV) for t in dados.lote(val_txt, gv)) for _ in range(8)]
    log = Log("ternario", vigiar=False, eco=False)
    log.vigiar = False
    m = ModeloBit(4, 256, 1024, 6, 1, log)
    orig = [s.to(DEV) for s in torch.load(arq)]
    acc = lambda f: sum(float((f(x).argmax(1) == y.reshape(-1)).float().mean()) for x, y in lotes) / len(lotes)
    taus_a = (0, 16, 32, 48, 64)
    print(f"contadores de {arq}")
    print("τw (zeros) | médias: loss acurácia | rede final, ativações binárias e ternárias τa = " + " / ".join(map(str, taus_a[1:])))
    for tw in (None, 0, 8, 16, 24, 32, 48, 64):
        for (_, t), t0 in zip(m.tensores(), orig):
            t.copy_(t0 if tw is None else tern(t0, tw))
        zeros = sum(int((t == 0).sum()) for _, t in m.tensores()) / sum(t.numel() for _, t in m.tensores())
        lv = sum(mb.nats(m.passo(x, y, treinar=False)[0], y) for x, y in lotes) / len(lotes)
        am = acc(lambda x: m.passo(x, torch.zeros_like(x), treinar=False)[0])
        pc = tw is not None
        r = [acc(lambda x: m.logits_binarios(x, pesos_crus=pc, ativ_tern=ta)) for ta in taus_a]
        rot = "contador" if tw is None else f"τw={tw:3d}"
        print(f"{rot:9s} ({zeros:4.0%}) | {lv:.3f} {am:5.1%} | " + " / ".join(f"{v:5.1%}" for v in r), flush=True)


if __name__ == "__main__":
    main()
