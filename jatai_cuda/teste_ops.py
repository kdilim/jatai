"""Teste de cada kernel contra a referencia em Python: tem que bater bit a bit.

Entradas aleatorias nos tamanhos do modelo + casos de borda (blocos zerados, expoentes espalhados,
mantissas no limite, logits grandes). Uso: python teste_ops.py
"""
import sys
import traceback

import torch

import ops as O
from registro import Registro, ErroJatai

dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(0)


def bloco(R, C, espalha=4, zera=0.0, e0=-10, mmax=127):
    m = torch.randint(-mmax, mmax + 1, (R, C), generator=g, dtype=torch.int32).to(torch.int8)
    E = (e0 + torch.randint(-espalha, espalha + 1, (R, C // 128), generator=g)).to(torch.int8)
    if zera:
        z = torch.rand(R, C // 128, generator=g) < zera
        m = m.reshape(R, C // 128, 128).masked_fill(z[..., None], 0).reshape(R, C)
        E = E.masked_fill(z, O.E_ZERO)
    return m.to(dev).contiguous(), E.to(dev).contiguous()


def cru(R, C, bits=24, e0=-30):
    acc = torch.randint(-(1 << bits), 1 << bits, (R, C), generator=g, dtype=torch.int32)
    e = (e0 + torch.randint(-3, 4, (R, C), generator=g)).to(torch.int16)
    return acc.to(dev).contiguous(), e.to(dev).contiguous()


def tile(bm, f):
    lib = O.OpsCuda.lib_global
    lib.jt_gemm_bm(bm)
    try:
        return f()
    finally:
        lib.jt_gemm_bm(0)


def limite(f):
    O.ATENCAO_LIMITE = True
    try:
        return f()
    finally:
        O.ATENCAO_LIMITE = False


def main():
    reg = Registro("logs", "teste_ops", vigiar=True)
    ops = O.OpsConferido(reg)
    reg.novo_passo(0)
    Bt, T, H, D, FF, V = 8, 128, 64, 256, 1024, 256
    M = Bt * T
    casos = [
        ("gemm Q x Wqkv", lambda: ops.gemm(bloco(M, D), bloco(3 * D, D))),
        ("gemm tile 128 (M=2048, N=1024)", lambda: ops.gemm(bloco(2 * M, D, espalha=6), bloco(FF, D))),
        ("gemm tile 128 cru", lambda: ops.gemm(bloco(2 * M, D), bloco(FF, D), cru=True)),
        ("gemm com blocos zerados", lambda: ops.gemm(bloco(M, D, zera=0.3), bloco(D, D, zera=0.3))),
        ("gemm K=1024 expoentes espalhados", lambda: ops.gemm(bloco(M, FF, espalha=12), bloco(D, FF, espalha=12))),
        ("gemm cru (grad de peso, K=tokens)", lambda: ops.gemm(bloco(D, M), bloco(FF, M), cru=True)),
        ("gemm cru split-K em 8 partes (D x D, K=tokens)", lambda: ops.gemm(bloco(D, M, espalha=8), bloco(D, M, espalha=8), cru=True)),
        ("gemm cru split-K com blocos zerados", lambda: ops.gemm(bloco(D, M, zera=0.4), bloco(3 * D, M, zera=0.4), cru=True)),
        ("gemm + soma residual (fundida)", lambda: ops.gemm_soma(bloco(M, D), bloco(D, D), bloco(M, D, espalha=10, zera=0.1))),
        ("gemm + soma, residuo muito maior/menor", lambda: ops.gemm_soma(bloco(M, FF), bloco(D, FF), bloco(M, D, e0=8, espalha=20, zera=0.2))),
        ("gemm + relu2 (fundida)", lambda: ops.gemm_relu2(bloco(M, D, zera=0.1), bloco(FF, D))),
        ("gemm + relu2 com underflow (expoente 2E < -126)", lambda: ops.gemm_relu2(bloco(M, D, e0=-40), bloco(FF, D, e0=-40))),
        ("gemm + relu2_back (fundida)", lambda: ops.gemm_relu2_back(bloco(M, D), bloco(FF, D), bloco(M, FF, zera=0.1))),
        ("transpoe", lambda: ops.transpoe(bloco(M, 3 * D, espalha=8, zera=0.1))),
        ("quant", lambda: ops.quant(cru(M, D))),
        ("quant com zeros", lambda: ops.quant((torch.zeros(M, D, dtype=torch.int32, device=dev), torch.zeros(M, D, dtype=torch.int16, device=dev)))),
        ("soma", lambda: ops.soma(bloco(M, D, espalha=10, zera=0.1), bloco(M, D, espalha=10, zera=0.1))),
        ("relu2", lambda: ops.relu2(bloco(M, FF, zera=0.1))),
        ("relu2_back", lambda: ops.relu2_back(bloco(M, FF), bloco(M, FF))),
        ("um_quente", lambda: ops.um_quente(torch.randint(0, V, (M,), generator=g).to(torch.int32).to(dev), V)),
        ("embed", lambda: ops.embed(bloco(V, D), torch.randint(0, V, (M,), generator=g).to(torch.int32).to(dev))),
        ("rms", lambda: ops.rms(bloco(M, D, espalha=6, zera=0.2))),
        ("rms com mantissas pequenas", lambda: ops.rms(bloco(M, D, mmax=3))),
        ("rms_back", lambda: (lambda Y: ops.rms_back(bloco(M, D, espalha=6), Y[0], Y[1]))(ops.rms(bloco(M, D)))),
        ("atencao", lambda: ops.atencao(bloco(M, 3 * D, espalha=3), Bt, T, H)),
        ("atencao (q,k grandes: softmax concentrado)", lambda: ops.atencao(bloco(M, 3 * D, e0=-3), Bt, T, H)),
        ("atencao com shift do score a esquerda <= 6 (laco sem sentinela; e0=0 -> s_u = -1)", lambda: ops.atencao(bloco(M, 3 * D, e0=0, espalha=1), Bt, T, H)),
        ("atencao com shift do score a esquerda > 6 (laco conferido; e0=4 -> s_u = -9)", lambda: ops.atencao(bloco(M, 3 * D, e0=4, espalha=1, mmax=3), Bt, T, H)),
        ("atencao expoentes muito espalhados (m8 arredondado)", lambda: ops.atencao(bloco(M, 3 * D, espalha=12, zera=0.1), Bt, T, H)),
        ("atencao_back expoentes muito espalhados", lambda: (lambda q: ops.atencao_back(bloco(M, D, e0=-20, espalha=12, zera=0.1), q, ops.atencao(q, Bt, T, H)[1], Bt, T, H))(bloco(M, 3 * D, espalha=12, zera=0.1))),
        ("atencao_back variante limite (3 passadas)", lambda: limite(lambda: (lambda q: ops.atencao_back(bloco(M, D, e0=-20), q, ops.atencao(q, Bt, T, H)[1], Bt, T, H))(bloco(M, 3 * D)))),
        ("atencao_back variante limite, expoentes espalhados", lambda: limite(lambda: (lambda q: ops.atencao_back(bloco(M, D, e0=-20, espalha=12, zera=0.1), q, ops.atencao(q, Bt, T, H)[1], Bt, T, H))(bloco(M, 3 * D, espalha=12, zera=0.1)))),
        ("atencao_back", lambda: (lambda q: ops.atencao_back(bloco(M, D, e0=-20), q, ops.atencao(q, Bt, T, H)[1], Bt, T, H))(bloco(M, 3 * D))),
        ("softmax_ce", lambda: ops.softmax_ce(cru(M, V, bits=22, e0=-20), torch.randint(0, V, (M,), generator=g).to(torch.int32).to(dev))),
        ("momento", lambda: ops.momento(cru(D, D, bits=28, e0=-50), bloco(D, D, e0=-40))),
        ("momento caindo abaixo de 2^-126 (underflow vira zero)", lambda: ops.momento((torch.zeros(D, D, dtype=torch.int32, device=dev), torch.zeros(D, D, dtype=torch.int16, device=dev)), bloco(D, D, e0=-122, espalha=3))),
        ("momento a partir de zero", lambda: ops.momento(cru(D, D, bits=20), (torch.zeros(D, D, dtype=torch.int8, device=dev), torch.full((D, D // 128), O.E_ZERO, dtype=torch.int8, device=dev)))),
        ("peso com error feedback", lambda: ops.peso(bloco(D, D, e0=-9), bloco(D, D, e0=-15), bloco(D, D, e0=-20, zera=0.3), 4, True)),
        ("peso sem error feedback", lambda: ops.peso(bloco(D, D, e0=-9), bloco(D, D, e0=-15), bloco(D, D), 4, False)),
        ("peso com passo maior que o peso", lambda: ops.peso(bloco(D, D, e0=-12), bloco(D, D, e0=-2), bloco(D, D, e0=-20), 1, True)),
    ]
    # cada GEMM roda com os dois tiles (64 e 128 linhas por bloco), forcados
    gemms = [(n, f) for n, f in casos if n.startswith("gemm")]
    casos = [(n, f) for n, f in casos if not n.startswith("gemm")]
    for bm in (64, 128):
        casos += [(f"{n} [tile {bm}]", (lambda f, bm: lambda: tile(bm, f))(f, bm)) for n, f in gemms]
    falhas = 0
    for nome, f in casos:
        try:
            with reg.trecho(nome):
                f()
            torch.cuda.synchronize()
            print(f"  ok      {nome}")
        except ErroJatai as e:
            falhas += 1
            print(f"  FALHOU  {nome}{e}")
            reg.novo_passo(0)
        except Exception:
            falhas += 1
            print(f"  FALHOU  {nome} (excecao Python)\n{traceback.format_exc()}")
            reg.novo_passo(0)
    print(f"\n{len(casos) - falhas} de {len(casos)} casos bateram bit a bit com a referencia")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
