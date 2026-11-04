"""Tempo de cada kernel nas formas exatas do modelo (lote Bt x 128 tokens), em TOPS de dp4a.

Uso: python bench_formas.py [Bt]
Cada operacao: 3 aquecimentos + 20 repeticoes medidas com eventos CUDA. TOPS = 2 * M * N * K / tempo.
Pico medido do dp4a nesta GPU (GTX 1650): 10,06 TOPS (../kernel_int8_b32/pico.cu).
"""
import sys

import torch

import ops as O
from registro import Registro

dev = "cuda"
Bt = int(sys.argv[1]) if len(sys.argv) > 1 else 16
T, H, D, FF, V = 128, 64, 256, 1024, 256
M = Bt * T
L = 12
g = torch.Generator(device="cpu").manual_seed(0)


def bloco(R, C, e0=-10):
    m = torch.randint(-127, 128, (R, C), generator=g, dtype=torch.int32).to(torch.int8)
    E = (e0 + torch.randint(-3, 4, (R, C // 128), generator=g)).to(torch.int8)
    return m.to(dev).contiguous(), E.to(dev).contiguous()


def mede(f, n=20):
    for _ in range(3):
        f()
    ini, fim = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    ini.record()
    for _ in range(n):
        f()
    fim.record()
    torch.cuda.synchronize()
    return ini.elapsed_time(fim) / n


def main():
    reg = Registro("logs", "bench_formas", eco=True)
    ops = O.OpsCuda(reg)
    reg.novo_passo(0)
    # (nome, M, N, K, cru, vezes por passo)  -- as 12 GEMMs de cada camada + saida
    formas = [
        ("fwd Wqkv", M, 3 * D, D, False), ("fwd Wo", M, D, D, False), ("fwd Wup", M, FF, D, False), ("fwd Wd", M, D, FF, False),
        ("dX Wd (ga)", M, FF, D, False), ("dX Wup (gn)", M, D, FF, False), ("dX Wo (go)", M, D, D, False), ("dX Wqkv (gn)", M, D, 3 * D, False),
        ("dW Wd", D, FF, M, True), ("dW Wup", FF, D, M, True), ("dW Wo", D, D, M, True), ("dW Wqkv", 3 * D, D, M, True),
    ]
    tot_ms = tot_op = 0.0
    linhas = []
    for nome, m_, n_, k_, cru in formas:
        A, B = bloco(m_, k_), bloco(n_, k_)
        ms = mede(lambda: ops.gemm(A, B, cru=cru))
        op = 2.0 * m_ * n_ * k_
        tot_ms += ms * L; tot_op += op * L
        linhas.append(f"  {nome:14s} M={m_:5d} N={n_:5d} K={k_:5d} {'cru ' if cru else 'bloco'} {ms:7.3f} ms  {op / ms / 1e9:6.2f} TOPS")
    qkv = bloco(M, 3 * D)
    go = bloco(M, D, e0=-20)
    (oc, P) = ops.atencao(qkv, Bt, T, H)
    ms_af = mede(lambda: ops.atencao(qkv, Bt, T, H))
    ms_ab = mede(lambda: ops.atencao_back(go, qkv, P, Bt, T, H))
    X = bloco(M, D)
    ms_tr = mede(lambda: ops.transpoe(X))
    reg.fim_passo()
    reg.info(f"lote {Bt} x {T} = {M} tokens\n" + "\n".join(linhas) +
             f"\n  GEMMs do passo (12 camadas): {tot_ms:.2f} ms, media {tot_op / tot_ms / 1e9:.2f} TOPS"
             f"\n  atencao forward {ms_af:.3f} ms (x12 = {12 * ms_af:.2f})"
             f"\n  atencao backward {ms_ab:.3f} ms (x12 = {12 * ms_ab:.2f})"
             f"\n  transpoe {M}x{D} {ms_tr:.3f} ms")


if __name__ == "__main__":
    main()
