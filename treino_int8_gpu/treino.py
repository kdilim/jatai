"""Treino: Transformer 100% INT8 vs referencia em float (mesma arquitetura, init, dados e otimizador).

Uso:
  python treino.py grad                 # um passo: cosseno gradiente INT8 x float e bits maximos
  python treino.py float <k> <passos>   # referencia em float, lr = 2^-k
  python treino.py int8  <k> <passos>   # modelo inteiro,   lr = 2^-k
  python treino.py int8ef <k> <passos>  # modelo inteiro com error feedback no update dos pesos
  python treino.py floatq <k> <passos> <flags>  # ablacao: contas em float, mas o que e guardado
      pelo otimizador vira INT8 bloco 32. flags: w = pesos, e = error feedback nos pesos, m = momento
"""
import json
import math
import os
import sys
import time

import torch

import jatai_int8 as j

V, D, H, HD, FF, L, T, BT = 256, 256, 64, 4, 1024, 12, 128, 8
SEMENTE = 0
DADOS = os.environ.get("JATAI_DADOS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dados"))   # pasta com Portuguese.jsonl e English.jsonl
AQUI = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
EVAL_CADA, N_EVAL = 100, 8


def carregar():
    treino, val = bytearray(), bytearray()
    for lingua in ("Portuguese", "English"):
        with open(os.path.join(DADOS, f"{lingua}.jsonl"), encoding="utf-8") as f:
            for i, linha in enumerate(f):
                if i >= 3000:
                    break
                b = json.loads(linha)["text"].encode("utf-8") + b"\n\n"
                (val if i % 20 == 0 else treino).extend(b)
    return torch.tensor(list(treino), dtype=torch.long), torch.tensor(list(val), dtype=torch.long)


def lote(dados, gerador):
    ini = torch.randint(0, len(dados) - T - 1, (BT,), generator=gerador)
    x = torch.stack([dados[i:i + T + 1] for i in ini.tolist()]).to(DEV)
    return x[:, :-1], x[:, 1:]


quant_float = j.quant_float


def main():
    modo = sys.argv[1]
    treino, val = carregar()
    print(f"dados: {len(treino):,} bytes de treino, {len(val):,} de validacao | modo {modo}", flush=True)
    mi = j.ModeloInt8(V, D, H, HD, FF, L, T, SEMENTE, DEV)
    mi.error_feedback = modo == "int8ef"
    ger_val = torch.Generator().manual_seed(123)
    lotes_val = [lote(val, ger_val) for _ in range(N_EVAL)]

    if modo == "grad":
        mf = j.ModeloFloat(mi).to(DEV)
        tok, alvo = lote(treino, torch.Generator().manual_seed(7))
        j.CHECAR["on"] = True
        perda_i = mi.forward_backward(tok, alvo, k_lr=99)
        perda_f = mf(tok, alvo)
        perda_f.backward()
        ms = j.ModeloFloat(mi).to(DEV)   # régua: formato INT8 simulado em float (ativacoes + gradientes)
        ms.flags = "ag"
        ms(tok, alvo).backward()
        print(f"loss inicial: INT8 {perda_i:.4f} | float {perda_f.item():.4f} | ln(256) = {math.log(256):.4f}")
        print("cosseno do gradiente com o float, por parametro:   INT8 real | INT8 simulado")
        cosf = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-30))
        for n, (gacc, ge) in mi.ultimos_grads.items():
            gi = (gacc.double() * torch.pow(2.0, ge.double())).flatten()
            gf = mf.w[n].grad.double().flatten()
            print(f"  {n:8s} {cosf(gi, gf):.4f} | {cosf(ms.w[n].grad.double().flatten(), gf):.4f}")
        print("maior numero de bits por operacao:")
        for n, b in sorted(j.MAX_BITS.items(), key=lambda x: -x[1]):
            print(f"  {n:30s} {b}")
        return

    k = int(sys.argv[2])
    passos = int(sys.argv[3])
    ger = torch.Generator().manual_seed(SEMENTE + 1)
    flags = sys.argv[4] if modo == "floatq" else ""
    log = open(os.path.join(AQUI, f"log_{modo}{flags}_k{k}.csv"), "w")
    log.write("passo,loss_treino,loss_val,tokens_por_s\n")
    if modo == "floatq":
        modo = "float"
    if modo == "float":
        mf = j.ModeloFloat(mi).to(DEV)
        mf.flags = flags
        opt = torch.optim.SGD(mf.parameters(), lr=2.0 ** -k, momentum=0.875)
        residuo = {id(w): torch.zeros_like(w) for w in mf.parameters()}
    acum, n_acum, t0, tok_cont = 0.0, 0, time.time(), 0
    for passo in range(passos + 1):
        if passo % EVAL_CADA == 0:
            if modo == "float":
                with torch.no_grad():
                    pv = sum(mf(*b).item() for b in lotes_val) / N_EVAL
            else:
                pv = sum(mi.forward_backward(*b, k_lr=None) for b in lotes_val) / N_EVAL
            torch.cuda.synchronize()
            tps = tok_cont / max(time.time() - t0, 1e-9)
            pt = acum / max(n_acum, 1)
            print(f"passo {passo:5d} | loss treino {pt:.4f} | loss val {pv:.4f} | {tps:,.0f} tokens/s", flush=True)
            log.write(f"{passo},{pt:.5f},{pv:.5f},{tps:.1f}\n")
            log.flush()
            if not math.isfinite(pv):
                break
            acum, n_acum, t0, tok_cont = 0.0, 0, time.time(), 0
        if passo == passos:
            break
        tok, alvo = lote(treino, ger)
        if modo == "float":
            perda = mf(tok, alvo)
            opt.zero_grad()
            perda.backward()
            opt.step()
            with torch.no_grad():
                for w in mf.parameters():
                    if "m" in flags:
                        buf = opt.state[w]["momentum_buffer"]
                        buf.copy_(quant_float(buf))
                    if "w" in flags:
                        alvo_w = w + residuo[id(w)] if "e" in flags else w.clone()
                        w.copy_(quant_float(alvo_w))
                        if "e" in flags:
                            residuo[id(w)] = quant_float(alvo_w - w)
            perda = perda.item()
        else:
            perda = mi.forward_backward(tok, alvo, k_lr=k)
        acum += perda
        n_acum += 1
        tok_cont += BT * T
    log.close()


if __name__ == "__main__":
    main()
