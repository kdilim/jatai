"""Teste do formato R16 (E1): ida e volta INT8 -> R16 -> INT8 exata, e a soma R16 + INT8 contra a soma exata em
float64, inclusive com a parcela pequena (2^-12 da grande), que a soma INT8 apagava."""
import os
import sys

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import ops as O                       # noqa: E402
from registro import Registro         # noqa: E402

BLK = 128


def deq(m, E):
    e = E.to(torch.float64).repeat_interleave(BLK, 1)
    return m.to(torch.float64) * torch.pow(2.0, e)


def bloco(R, C, emin, emax, g):
    m = torch.randint(-127, 128, (R, C), generator=g, dtype=torch.int8).cuda()
    E = torch.randint(emin, emax + 1, (R, C // BLK), generator=g, dtype=torch.int8).cuda()
    return m, E


def main():
    reg = Registro(os.path.join(AQUI, "logs"), "teste_res16")
    ops = O.OpsCuda(reg)
    g = torch.Generator().manual_seed(0)
    x = bloco(256, 512, -10, 5, g)
    volta = ops.para8(ops.para16(x))
    ok1 = bool((deq(*volta) == deq(*x)).all())
    print(f"ida e volta INT8 -> R16 -> INT8 exata: {ok1}")
    for rot, (emin_y, emax_y) in (("parcelas do mesmo tamanho", (-10, 5)), ("parcela 2^-12 menor", (-22, -7))):
        y = bloco(256, 512, emin_y, emax_y, g)
        x16 = ops.para16(x)
        z16 = ops.soma16(x16, y)
        exato = deq(*x) + deq(*y)
        z = deq(*z16)
        z8 = deq(*ops.soma(x, y))
        err16 = float(((z - exato).norm() / (deq(*y)).norm()))
        err8 = float(((z8 - exato).norm() / (deq(*y)).norm()))
        print(f"{rot}: erro da soma relativo à parcela somada: R16 {err16:.2e} | INT8 {err8:.2e}")
    torch.cuda.synchronize()
    reg.fim_passo() if hasattr(reg, "fim_passo") else None


if __name__ == "__main__":
    main()
