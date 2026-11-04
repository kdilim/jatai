"""Treino longo de um modelo maior: INT8 (biblioteca CUDA, CUDA Graph) x float32 (PyTorch), mesmas condicoes.

Uso: python noite.py int8|float <camadas> <ff> <lote> <passos> [eval_cada]

Diferencas em relacao a treino_cuda.py:
  - usa o corpus inteiro (todas as linhas de Portuguese.jsonl e English.jsonl, ~104 MB; os experimentos curtos
    usaram as 3000 primeiras de cada, ~63 MB), com a mesma regra de divisao (1 linha em 20 vai para validacao);
  - camadas e largura do MLP configuraveis (D = 256 e head_dim 4 ficam: os kernels aceitam ate 64 cabecas);
  - salva os pesos a cada CHECK_CADA passos e no fim (logs/noite_<modo>_pesos.pt, sobrescrito).
Mesmos pesos iniciais, mesmos lotes (gerador com semente 1) e mesmos 8 lotes de validacao (semente 123) nos dois
modos. Otimizador: SGD com momento 7/8 e lr 2^-4 nos dois (no INT8, com error feedback).
Saida: logs/noite_<modo>_L<camadas>_ff<ff>.csv (passo, loss treino, loss val, tokens/s, horas) e .jsonl (eventos).
"""
import json
import math
import os
import sys
import time

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "treino_int8_gpu"))
import treino as dados          # noqa: E402  (lote, DADOS)
import jatai_int8 as ji         # noqa: E402

import ops as O                 # noqa: E402
from modelo import Modelo, PassoGravado  # noqa: E402
from registro import Registro, ErroJatai  # noqa: E402

V, D, H, T, K_LR, SEMENTE, DEV = 256, 256, 64, 128, 4, 0, "cuda"
N_EVAL, CHECK_CADA = 8, 10000


def carregar_tudo():
    treino, val = bytearray(), bytearray()
    for lingua in ("Portuguese", "English"):
        with open(os.path.join(dados.DADOS, f"{lingua}.jsonl"), encoding="utf-8") as f:
            for i, linha in enumerate(f):
                b = json.loads(linha)["text"].encode("utf-8") + b"\n\n"
                (val if i % 20 == 0 else treino).extend(b)
    # direto do buffer (uma lista Python de 100 milhoes de inteiros ocuparia ~4 GB de RAM); long: os modelos indexam com ele
    return (torch.frombuffer(bytes(treino), dtype=torch.uint8).long(), torch.frombuffer(bytes(val), dtype=torch.uint8).long())


def main():
    modo, L, FF, Bt, passos = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    eval_cada = int(sys.argv[6]) if len(sys.argv) > 6 else 500
    nome = f"noite_{modo}_L{L}_ff{FF}" + os.environ.get("JATAI_SUFIXO", "")
    pasta = os.path.join(AQUI, "logs")
    reg = Registro(pasta, nome)
    treino, val = carregar_tudo()
    dados.BT = 8
    ger_val = torch.Generator().manual_seed(123)
    lotes_val = [dados.lote(val, ger_val) for _ in range(N_EVAL)]
    dados.BT = Bt
    ger = torch.Generator().manual_seed(SEMENTE + 1)
    reg.info(f"{nome}: {len(treino):,} bytes de treino, {len(val):,} de validacao | L={L} FF={FF} lote {Bt}x{T} | "
             f"{passos} passos ({passos * Bt * T / 1e6:.1f} M tokens) | lr 2^-{K_LR}")

    if modo == "int8":
        ops = O.OpsCuda(reg)
        m = Modelo(V, D, H, FF, L, SEMENTE, DEV, ops, reg)
        n_par = sum(t[0].numel() for t in m.p.values())
        grafo = None

        def passo_treino(p, tok, alvo):
            nonlocal grafo
            normal = grafo is None or p % 1000 in (0, 1)       # estatisticas e perfil 1x a cada 1000 passos
            m.estatisticas = p % 1000 == 0
            reg.perfil = p % 1000 == 1
            perda = m.passo(tok, alvo, k_lr=K_LR) if normal else grafo(tok, alvo)
            reg.fim_passo()
            if reg.perfil:
                reg.relatorio_tempos(1)
            if grafo is None:
                grafo = PassoGravado(m, Bt, T, K_LR)
            return float(perda)

        def avalia():
            s = 0.0
            for b in lotes_val:
                reg.novo_passo(-1)
                s += float(m.passo(*b))
                reg.fim_passo()
            return s / N_EVAL

        def salva():
            torch.save({"modo": "int8", "L": L, "FF": FF, "pesos": {k: (a.cpu(), e.cpu()) for k, (a, e) in m.p.items()},
                        "residuo": {k: (a.cpu(), e.cpu()) for k, (a, e) in m.res.items()}},
                       os.path.join(pasta, f"{nome}_pesos.pt"))
    else:
        os.environ["JATAI_BLK"] = "32"
        mi = ji.ModeloInt8(V, D, H, 4, FF, L, T, SEMENTE, DEV)
        mf = ji.ModeloFloat(mi).to(DEV)
        del mi
        n_par = sum(w.numel() for w in mf.parameters())
        opt = torch.optim.SGD(mf.parameters(), lr=2.0 ** -K_LR, momentum=0.875)

        def passo_treino(p, tok, alvo):
            perda = mf(tok, alvo)
            opt.zero_grad()
            perda.backward()
            opt.step()
            return float(perda)

        def avalia():
            with torch.no_grad():
                return sum(float(mf(*b)) for b in lotes_val) / N_EVAL

        def salva():
            torch.save({"modo": "float", "L": L, "FF": FF, "pesos": mf.state_dict()}, os.path.join(pasta, "noite_float_pesos.pt"))

    reg.info(f"{n_par / 1e6:.1f} M parametros")
    csv = open(os.path.join(pasta, f"{nome}.csv"), "w")
    csv.write("passo,loss_treino,loss_val,tokens_por_s,horas\n")
    t_ini = time.time()
    acum = n_acum = tok_cont = 0
    t_trecho, melhor = 0.0, float("inf")
    for p in range(passos + 1):
        if p % eval_cada == 0:
            reg.passo = p
            pv = avalia()
            tps = tok_cont / max(t_trecho, 1e-9)
            pt = acum / max(n_acum, 1)
            horas = (time.time() - t_ini) / 3600
            reg.info(f"passo {p:6d} | loss treino {pt:.4f} | loss val {pv:.4f} | {tps:,.0f} tokens/s | {horas:.2f} h",
                     loss_val=pv, loss_treino=pt, tps=tps)
            csv.write(f"{p},{pt:.5f},{pv:.5f},{tps:.1f},{horas:.3f}\n"); csv.flush()
            if not math.isfinite(pv):
                reg.aviso(f"loss de validacao nao finita ({pv}): treino interrompido")
                break
            if pv > 1.5 * melhor:
                reg.aviso(f"loss de validacao subiu de {melhor:.4f} para {pv:.4f}: divergindo?")
            melhor = min(melhor, pv)
            acum = n_acum = tok_cont = 0
            t_trecho = 0.0
        if p == passos:
            break
        if p > 0 and p % CHECK_CADA == 0:
            salva()
        tok, alvo = dados.lote(treino, ger)
        if modo == "int8":
            reg.novo_passo(p)
        torch.cuda.synchronize()
        t0 = time.time()
        perda = passo_treino(p, tok, alvo)
        t_trecho += time.time() - t0
        if not math.isfinite(perda):
            reg.aviso(f"loss de treino nao finita no passo {p}: treino interrompido")
            break
        acum += perda; n_acum += 1; tok_cont += Bt * T
    salva()
    csv.close()
    reg.info(f"{nome}: fim em {(time.time() - t_ini) / 3600:.2f} h; melhor loss de validacao {melhor:.4f}")


if __name__ == "__main__":
    try:
        main()
    except ErroJatai as e:      # o registro ja escreveu a mensagem completa no .jsonl; sai com codigo de erro
        print(e, flush=True)
        sys.exit(2)
