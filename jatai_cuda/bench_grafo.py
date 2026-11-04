"""CUDA Graph x modo normal, em A/B alternado no mesmo processo (blocos de 10 passos de treino de cada modo).

Uso: python bench_grafo.py [Bt]
Cada passo inclui o que o laco de treino faz: copia do lote, passo completo, leitura das sentinelas (sincroniza)
e float(loss). Mede tambem o tempo de CPU para enfileirar o passo no modo normal.
"""
import os
import sys
import time

import torch

import treino_cuda as tc
import ops as O
from modelo import Modelo, PassoGravado
from registro import Registro

Bt = int(sys.argv[1]) if len(sys.argv) > 1 else 8
reg = Registro(os.path.join(tc.AQUI, "logs"), f"bench_grafo_bt{Bt}", eco=True)
ops = O.OpsCuda(reg)
treino, _ = tc.carrega(Bt)
ger = torch.Generator().manual_seed(1)
lotes = [tc.dados.lote(treino, ger) for _ in range(10)]
m = Modelo(tc.V, tc.D, tc.H, tc.FF, tc.L, tc.SEMENTE, tc.DEV, ops, reg)
reg.novo_passo(0); m.passo(*lotes[0], k_lr=4); reg.fim_passo()
g = PassoGravado(m, Bt, tc.T, 4)


def bloco(f):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(10):
        reg.novo_passo(i)
        perda = f(*lotes[i])
        reg.fim_passo()
        float(perda)
    return time.perf_counter() - t0


t = {"normal": 0.0, "grafo": 0.0}
for rep in range(8):
    t["normal"] += bloco(lambda a, b: m.passo(a, b, k_lr=4))
    t["grafo"] += bloco(g)
cpu = 0.0
for i in range(10):
    torch.cuda.synchronize(); reg.novo_passo(i)
    t0 = time.perf_counter(); m.passo(*lotes[i], k_lr=4); cpu += time.perf_counter() - t0
    reg.fim_passo()
n = 80 * Bt * tc.T
reg.info(f"lote {Bt} x {tc.T}: normal {n / t['normal']:,.0f} tokens/s | CUDA Graph {n / t['grafo']:,.0f} tokens/s | "
         f"grafo / normal = {t['normal'] / t['grafo']:.3f}x | CPU para enfileirar um passo (normal): {cpu / 10 * 1000:.1f} ms | "
         f"passo normal {t['normal'] / 80 * 1000:.1f} ms, grafo {t['grafo'] / 80 * 1000:.1f} ms",
         normal=n / t["normal"], grafo=n / t["grafo"], cpu_ms=cpu / 10 * 1000)
