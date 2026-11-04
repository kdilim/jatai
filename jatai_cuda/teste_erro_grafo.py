"""Teste do sistema de logs dentro de um CUDA Graph.

Grava um passo de treino, repete uma vez normal e outra com o token 300 (fora do vocabulario de 256) na posicao
(3, 17) do lote. O erro tem que ser detectado na repeticao e apontar: operacao embed, trecho "forward / embedding",
linha 3 * 128 + 17 = 401, valor 300. Uso: python teste_erro_grafo.py
"""
import sys, torch
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import treino_cuda as tc, ops as O
from modelo import Modelo, PassoGravado
from registro import Registro, ErroJatai
reg = Registro("logs", "teste_erro_grafo", eco=True); ops = O.OpsCuda(reg)
treino, _ = tc.carrega(8)
tok, alvo = tc.dados.lote(treino, torch.Generator().manual_seed(1))
m = Modelo(tc.V, tc.D, tc.H, tc.FF, tc.L, tc.SEMENTE, tc.DEV, ops, reg)
reg.novo_passo(0); m.passo(tok, alvo, k_lr=4); reg.fim_passo()
g = PassoGravado(m, 8, 128, 4)
reg.novo_passo(1); g(tok, alvo); reg.fim_passo(); print("repeticao normal: sem erro")
ruim = tok.clone(); ruim[3, 17] = 300
reg.novo_passo(2)
try:
    g(ruim, alvo); reg.fim_passo(); print("FALHOU: erro nao detectado")
except ErroJatai as e:
    msg = str(e)
    ok = "forward / embedding" in msg and "linha 401, coluna 0" in msg and "valor    : 300" in msg
    print(msg)
    print("\nOK: erro detectado dentro do grafo, com trecho, posicao e valor certos" if ok else "\nFALHOU: mensagem sem os dados esperados")
    sys.exit(0 if ok else 1)
sys.exit(1)
