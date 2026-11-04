"""Quantos bits por peso preservam o que a rede aprendeu?

Treina o modelo_bit (observador, contador int8 por peso) e avalia a MESMA rede guardando k bits por peso:
o contador θ em [-127, 127] vira um de 2^k níveis igualmente espaçados em [-127, 127] (k = 1: sinal, ±127).
Duas avaliações por k:
  ativações suaves (médias int8, como no treino)  -> mede só a perda dos pesos
  ativações com 1, 2, 3, 4 bits (2^k níveis da média) -> mede pesos + ativações
Uso: python precisao_bits.py [passos] [L] [D] [F]
"""
import sys
import torch
import modelo_bit as mb
from modelo_bit import ModeloBit, Log, nats, niveis, DEV

sys.path.insert(0, mb.os.path.join(mb.AQUI, "..", "treino_int8_gpu"))
import treino as dados  # noqa: E402


def main():
    a = [int(v) for v in sys.argv[1:]]
    passos, L, D, F = (a + [3000, 4, 256, 1024][len(a):])[:4]
    dados.BT = 8
    treino_txt, val_txt = dados.carregar()
    gv = torch.Generator().manual_seed(123)
    lotes_val = [dados.lote(val_txt, gv) for _ in range(8)]
    g = torch.Generator().manual_seed(2)
    log = Log(f"precisao_bits_L{L}D{D}F{F}_p{mb.POSTO}", vigiar=False, eco=True)
    m = ModeloBit(L, D, F, 6, 1, log)
    for p in range(passos):
        log.passo = p
        tok, alvo = dados.lote(treino_txt, g)
        m.passo(tok.to(DEV), alvo.to(DEV))
    orig = [(n, t.clone()) for n, t in m.tensores()]
    nb = sum(t.numel() for _, t in orig)
    print(f"posto {mb.POSTO}: {nb / 1e6:.2f} M pesos (cada um com k bits na tabela abaixo)")
    print(f"treinado {passos} passos | contadores: |θ| médio "
          f"{float(torch.cat([t.abs().float().view(-1) for _, t in orig]).mean()):.1f} de 127")

    def avalia(rot):
        x_ = [(b[0].to(DEV), b[1].to(DEV)) for b in lotes_val]
        so = [m.passo(x, y, treinar=False)[0] for x, y in x_]
        lv = sum(nats(s, y) for s, (_, y) in zip(so, x_)) / len(x_)
        am = sum(float((s.argmax(1) == y.reshape(-1)).float().mean()) for s, (_, y) in zip(so, x_)) / len(x_)
        ab = [sum(float((m.logits_binarios(x, pesos_crus=True, ativ_bits=ka).argmax(1) == y.reshape(-1)).float().mean()) for x, y in x_) / len(x_)
              for ka in (1, 2, 3, 4)]
        print(f"{rot:>28} | ativações suaves: loss {lv:.3f} acurácia {am:5.1%} | ativações com 1/2/3/4 bits: "
              + " / ".join(f"{v:5.1%}" for v in ab), flush=True)

    for k in (1, 2, 3, 4, 8):
        for (_, t), (_, t0) in zip(m.tensores(), orig):
            t.copy_(t0 if k == 8 else niveis(t0, k))
        if mb.POSTO:
            m._compor()
        avalia(f"{k} bit(s) por peso" + (" (contador)" if k == 8 else ""))


if __name__ == "__main__":
    main()
