"""Treino do Transformer 100% inteiro com os kernels CUDA (libjatai.so).

Uso:
  python treino_cuda.py conferir <passos>          # modelo inteiro, cada operacao CUDA x referencia bit a bit
  python treino_cuda.py treinar <k> <passos> [Bt]  # treino; lr = 2^-k; log em logs/treino_k<k>.jsonl e .csv
  python treino_cuda.py velocidade [Bt]            # tokens/s: INT8 CUDA x float PyTorch (mesma arquitetura)
Variavel JATAI_GRAFO=0: sem CUDA Graph (cada kernel lancado pelo Python a cada passo).
Variavel JATAI_ATENCAO_LIMITE=1: backward da atencao com shifts por limites superiores (3 passadas; ver ops.py).

Mesma configuracao e mesmos dados de ../treino_int8_gpu (D=256, 64 cabecas x head_dim 4, FF=1024,
12 camadas, T=128, vocabulario de bytes; wiki PT + EN).
"""
import math
import os
import sys
import time

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "treino_int8_gpu"))
import treino as dados        # noqa: E402  (carregar, lote)
import jatai_int8 as ji        # noqa: E402  (ModeloFloat: referencia em float)

import ops as O                # noqa: E402
O.ATENCAO_LIMITE = os.environ.get("JATAI_ATENCAO_LIMITE") == "1"   # variante: shifts do backward da atencao por limites
GRAFO = os.environ.get("JATAI_GRAFO", "1") == "1"                  # passo gravado num CUDA Graph (0 = modo normal)
from modelo import Modelo, PassoGravado  # noqa: E402
from registro import Registro  # noqa: E402

V, D, H, T = 256, 256, 64, 128
FF = int(os.environ.get("JATAI_FF", 1024))   # largura do MLP (padrao dos experimentos: 1024)
L = int(os.environ.get("JATAI_L", 12))       # camadas (padrao: 12)
SEMENTE, DEV = 0, "cuda"
EVAL_CADA, N_EVAL, STAT_CADA = 100, 8, 100


def carrega(Bt):
    dados.BT = Bt
    treino, val = dados.carregar()
    ger_val = torch.Generator().manual_seed(123)
    dados.BT = 8                                   # validacao sempre nos mesmos 8 x 128 tokens de ../treino_int8_gpu
    lotes_val = [dados.lote(val, ger_val) for _ in range(N_EVAL)]
    dados.BT = Bt
    return treino, lotes_val


def conferir(passos):
    reg = Registro(os.path.join(AQUI, "logs"), "conferir", vigiar=True)
    ops = O.OpsConferido(reg)
    treino, lotes_val = carrega(8)
    m = Modelo(V, D, H, FF, L, SEMENTE, DEV, ops, reg)
    ger = torch.Generator().manual_seed(SEMENTE + 1)
    for p in range(passos):
        reg.novo_passo(p)
        tok, alvo = dados.lote(treino, ger)
        perda = float(m.passo(tok, alvo, k_lr=4))
        reg.fim_passo()
        reg.info(f"passo {p}: todas as operacoes bateram bit a bit ({sum(ops.n.values())} conferencias ate aqui), loss {perda:.4f}")
    reg.info("resumo das conferencias: " + ", ".join(f"{k} {v}" for k, v in sorted(ops.n.items())))


def treinar(k, passos, Bt):
    pasta = os.path.join(AQUI, "logs")
    sufixo = "_limite" if O.ATENCAO_LIMITE else ""
    reg = Registro(pasta, f"treino_k{k}_bt{Bt}{sufixo}", perfil=False)
    ops = O.OpsCuda(reg)
    treino, lotes_val = carrega(Bt)
    m = Modelo(V, D, H, FF, L, SEMENTE, DEV, ops, reg)
    reg.info(f"treino INT8 CUDA | lr 2^-{k} | {Bt} x {T} = {Bt * T} tokens por passo | {passos} passos | error feedback")
    csv = open(os.path.join(pasta, f"treino_k{k}_bt{Bt}{sufixo}.csv"), "w")
    csv.write("passo,loss_treino,loss_val,tokens_por_s\n")
    ger = torch.Generator().manual_seed(SEMENTE + 1)
    acum = n_acum = tok_cont = 0
    t_treino = 0.0
    melhor = float("inf")
    grafo = None
    for p in range(passos + 1):
        if p % EVAL_CADA == 0:
            reg.novo_passo(p)
            with reg.trecho("validacao"):
                pv = float(sum(m.passo(*b) for b in lotes_val)) / N_EVAL
            reg.fim_passo()
            tps = tok_cont / max(t_treino, 1e-9)
            pt = acum / max(n_acum, 1)
            reg.info(f"passo {p:5d} | loss treino {pt:.4f} | loss val {pv:.4f} | {tps:,.0f} tokens/s", loss_val=pv, loss_treino=pt, tps=tps)
            csv.write(f"{p},{pt:.5f},{pv:.5f},{tps:.1f}\n"); csv.flush()
            if not math.isfinite(pv):
                reg.falha(f"loss de validacao nao finita: {pv}")
            if pv > 1.5 * melhor:
                reg.aviso(f"loss de validacao subiu de {melhor:.4f} para {pv:.4f}: treino divergindo?")
            melhor = min(melhor, pv)
            acum = n_acum = tok_cont = 0
            t_treino = 0.0
        if p == passos:
            break
        reg.novo_passo(p)
        tok, alvo = dados.lote(treino, ger)
        # passos de estatistica/perfil (1 a cada 100) rodam no modo normal; os outros repetem o passo gravado
        normal = grafo is None or p % STAT_CADA in (0, 1)
        m.estatisticas = p % STAT_CADA == 0
        reg.perfil = p % STAT_CADA == 1
        torch.cuda.synchronize()
        t0 = time.time()
        perda = float(m.passo(tok, alvo, k_lr=k) if normal else grafo(tok, alvo))
        if grafo is None and GRAFO and p == 0:
            reg.fim_passo()
            grafo = PassoGravado(m, Bt, T, k)
        reg.fim_passo()                          # sincroniza e confere as sentinelas dos kernels
        t_treino += time.time() - t0
        if reg.perfil:
            reg.relatorio_tempos(1)
        if not math.isfinite(perda):
            reg.falha(f"loss de treino nao finita: {perda}")
        acum += perda; n_acum += 1; tok_cont += Bt * T
    csv.close()


def velocidade(Bt):
    """Tokens/s do passo completo de treino (forward + backward + update), mesmo lote, GPU so para isso."""
    reg = Registro(os.path.join(AQUI, "logs"), f"velocidade_bt{Bt}", eco=True)
    ops = O.OpsCuda(reg)
    treino, _ = carrega(Bt)
    ger = torch.Generator().manual_seed(1)
    lotes = [dados.lote(treino, ger) for _ in range(6)]
    m = Modelo(V, D, H, FF, L, SEMENTE, DEV, ops, reg)

    def mede(f, n=30):
        f(lotes[0])
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(n):
            f(lotes[1 + i % 5])
        torch.cuda.synchronize()
        return Bt * T * n / (time.time() - t0)

    def passo_int8(b):
        reg.novo_passo(0)
        m.passo(*b, k_lr=4)
        reg.fim_passo()
    t_int8 = mede(passo_int8)
    reg.perfil = True
    passo_int8(lotes[0]); reg.relatorio_tempos(1); reg.perfil = False
    reg.novo_passo(0)
    grafo = PassoGravado(m, Bt, T, 4)

    def passo_grafo(b):
        reg.novo_passo(0)
        grafo(*b)
        reg.fim_passo()
    t_grafo = mede(passo_grafo)

    os.environ["JATAI_BLK"] = "32"
    mi = ji.ModeloInt8(V, D, H, 4, FF, L, T, SEMENTE, DEV)
    mf = ji.ModeloFloat(mi).to(DEV)
    del mi
    opt = torch.optim.SGD(mf.parameters(), lr=2.0 ** -4, momentum=0.875)

    def passo_float(b):
        perda = mf(*b)
        opt.zero_grad(); perda.backward(); opt.step()
    t_f32 = mede(passo_float)
    reg.info(f"lote {Bt} x {T} = {Bt * T} tokens | INT8 CUDA {t_int8:,.0f} tokens/s (CUDA Graph: {t_grafo:,.0f}) | "
             f"float32 PyTorch {t_f32:,.0f} tokens/s | INT8 / float32 = {t_int8 / t_f32:.2f}x (com grafo {t_grafo / t_f32:.2f}x)",
             int8=t_int8, int8_grafo=t_grafo, f32=t_f32)


if __name__ == "__main__":
    modo = sys.argv[1]
    if modo == "conferir":
        conferir(int(sys.argv[2]))
    elif modo == "treinar":
        treinar(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 8)
    elif modo == "velocidade":
        velocidade(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
