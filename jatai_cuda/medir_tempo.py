"""Quanto trabalho dá para CORTAR deixando a nuance chegar com o tempo (mesmos pesos, mesma memória)?

Mede, no modelo de 24 camadas treinado (pesos da noite), três potenciais, sem kernel novo:
  1. delta: fração das entradas INT8 de cada multiplicação que NÃO mudam do token t-1 para o t (essas colunas de W
     podem ser puladas: y_t = y_{t-1} + W (x_t - x_{t-1}), exato), e fração de zeros (pulo trivial);
  2. precisão progressiva: o modelo inteiro com as entradas das multiplicações truncadas aos p bits mais altos da
     magnitude (p = 1..7) -> loss e concordância do próximo byte com o modelo completo;
  3. parar cedo: em que camada a previsão (lente de logits: rms + saída) já é a final; e saída por confiança.
Uso: python medir_tempo.py [arquivo de pesos float]
"""
import os
import sys

import torch
import torch.nn.functional as F

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "treino_int8_gpu"))
import treino as dados          # noqa: E402
import jatai_int8 as ji         # noqa: E402
from noite import carregar_tudo  # noqa: E402

DEV = "cuda"


def q8(x):
    return ji.quant_float(x)                                    # INT8 + expoente por bloco de 32, como o modelo inteiro


def trunca(x, p):
    """mantém os p bits mais altos da mantissa INT8 (7 bits de magnitude) de cada bloco."""
    if p >= 7:
        return q8(x)
    xb = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    topo = torch.floor(torch.log2(xb.abs().amax(-1, keepdim=True).clamp(min=1e-38))) + 1
    esc = torch.pow(2.0, topo - 7)
    m = torch.round(xb / esc).clamp(-127, 127)
    passo = 2.0 ** (7 - p)
    m = torch.sign(m) * torch.floor(m.abs() / passo) * passo
    return (m * esc).reshape(x.shape)


def forward(mf, tok, p=7, gravar=None, lente=False):
    V, D, H, HD, FF, L, T = mf.cfg
    w = mf.w
    Bt = tok.shape[0]
    ent = lambda nome, l, x: (gravar.setdefault((nome, l), []).append(q8(x)) if gravar is not None else None) or trunca(x, p)
    rms = lambda x: x * torch.rsqrt((x * x).mean(-1, keepdim=True))
    cab = lambda x: x.reshape(Bt, T, H, HD).transpose(1, 2)
    x = w["emb"][tok]
    lentes = []
    for l in range(L):
        xn = ent("qkv", l, rms(x))
        q, k, v = cab(xn @ w[f"Wq{l}"].t()), cab(xn @ w[f"Wk{l}"].t()), cab(xn @ w[f"Wv{l}"].t())
        z = (q @ k.transpose(-1, -2)) / 2.0 + mf.vies
        a = torch.softmax(z.masked_fill(mf.mascara, float("-inf")), -1)
        x = x + ent("Wo", l, (a @ v).transpose(1, 2).reshape(Bt, T, D)) @ w[f"Wo{l}"].t()
        xn = ent("Wup", l, rms(x))
        x = x + ent("Wd", l, torch.relu(xn @ w[f"Wup{l}"].t()).square()) @ w[f"Wd{l}"].t()
        if lente:
            lentes.append(rms(x) @ w["out"].t())
    return rms(x) @ w["out"].t(), lentes


@torch.no_grad()
def main():
    arq = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, "logs", "noite_float_pesos.pt")
    ck = torch.load(arq, weights_only=False)
    L, FF = ck["L"], ck["FF"]
    mi = ji.ModeloInt8(256, 256, 64, 4, FF, L, 128, 0, DEV)
    mf = ji.ModeloFloat(mi).to(DEV)
    del mi
    mf.load_state_dict(ck["pesos"])
    _, val = carregar_tudo()
    dados.BT = 8
    g = torch.Generator().manual_seed(123)
    lotes = [tuple(t.to(DEV) for t in dados.lote(val, g)) for _ in range(4)]
    print(f"pesos {arq} | L={L} FF={FF} | {4 * 8 * 128} tokens de validação", flush=True)

    # ---- 1. delta e zeros
    grav = {}
    ref = []
    for tok, alvo in lotes:
        lg, lentes = forward(mf, tok, gravar=grav, lente=True)
        ref.append((lg, lentes, alvo))
    print("\n1) ENTRADAS QUE NÃO MUDAM ENTRE TOKENS SEGUIDOS (colunas de W puláveis) | zeros")
    tot_trab = tot_pul = tot_z = 0
    for nome in ("qkv", "Wo", "Wup", "Wd"):
        linha = []
        for l in range(L):
            X = torch.cat(grav[(nome, l)])                      # (B, T, K)
            igual = (X[:, 1:] == X[:, :-1]).float().mean().item()
            zero = (X == 0).float().mean().item()
            linha.append((igual, zero))
        med = lambda i: sum(v[i] for v in linha) / L
        print(f"  {nome:4s} iguais: média {med(0):.1%} (camadas 0/6/12/18/23: "
              + " ".join(f"{linha[l][0]:.0%}" for l in (0, 6, 12, 18, L - 1)) + f") | zeros: média {med(1):.1%}")
    # trabalho por multiplicação (por token): qkv 3*D*D, Wo D*D, Wup D*FF, Wd FF*D
    custo = {"qkv": 3 * 256 * 256, "Wo": 256 * 256, "Wup": 256 * FF, "Wd": FF * 256}
    for l in range(L):
        for nome in custo:
            X = torch.cat(grav[(nome, l)])
            tot_trab += custo[nome]
            tot_pul += custo[nome] * (X[:, 1:] == X[:, :-1]).float().mean().item()
            tot_z += custo[nome] * (X == 0).float().mean().item()
    print(f"  trabalho das multiplicações por pesos que dá para pular: só zeros {tot_z / tot_trab:.1%} | "
          f"delta (inclui zeros repetidos) {tot_pul / tot_trab:.1%}")

    # ---- 2. precisão progressiva
    print("\n2) PRECISÃO PROGRESSIVA (entradas das multiplicações com p bits de magnitude)")
    base = [r[0].argmax(-1) for r in ref]
    for p in (1, 2, 3, 4, 5, 6, 7):
        loss = conc = 0.0
        for (tok, alvo), b in zip(lotes, base):
            lg, _ = forward(mf, tok, p=p)
            loss += F.cross_entropy(lg.reshape(-1, 256), alvo.reshape(-1)).item()
            conc += (lg.argmax(-1) == b).float().mean().item()
        print(f"  p={p}: loss {loss / len(lotes):.4f} | mesmo próximo byte que o completo: {conc / len(lotes):.1%}", flush=True)

    # ---- 3. parar cedo
    print("\n3) PARAR CEDO (lente de logits por camada)")
    fin = torch.cat([r[0].argmax(-1).reshape(-1) for r in ref])
    alvos = torch.cat([r[2].reshape(-1) for r in ref])
    camadas = [torch.cat([r[1][l].argmax(-1).reshape(-1) for r in ref]) for l in range(L)]
    estavel = torch.full_like(fin, L)                            # primeira camada a partir da qual a previsão não muda mais
    ok = torch.ones_like(fin, dtype=torch.bool)
    for l in reversed(range(L)):
        ok = ok & (camadas[l] == fin)
        estavel = torch.where(ok, l + 1, estavel)
    print(f"  oráculo (sai na 1a camada já estável): média {estavel.float().mean():.1f} de {L} camadas "
          f"({1 - estavel.float().mean().item() / L:.0%} do trabalho cortado)")
    print("  camada | já igual à final")
    print("  " + " ".join(f"{l + 1}:{(camadas[l] == fin).float().mean():.0%}" for l in range(0, L, 3)))
    probs = [torch.cat([F.softmax(r[1][l], -1).reshape(-1, 256) for r in ref]) for l in range(L)]
    acc_full = (fin == alvos).float().mean().item()
    loss_full = sum(F.cross_entropy(r[0].reshape(-1, 256), r[2].reshape(-1)).item() for r in ref) / len(ref)
    print(f"  modelo completo: acurácia {acc_full:.1%} loss {loss_full:.4f}")
    for lim in (0.5, 0.7, 0.8, 0.9, 0.95):
        saiu = torch.zeros_like(fin, dtype=torch.bool)
        pred = fin.clone(); prof = torch.full_like(fin, L)
        P_fim = torch.cat([F.softmax(r[0], -1).reshape(-1, 256) for r in ref])
        Pesc = P_fim.clone()
        for l in range(L):
            novo = (~saiu) & (probs[l].max(-1).values > lim)
            pred = torch.where(novo, probs[l].argmax(-1), pred)
            prof = torch.where(novo, l + 1, prof)
            Pesc = torch.where(novo.unsqueeze(1), probs[l], Pesc)
            saiu = saiu | novo
        loss = -torch.log(Pesc[torch.arange(len(alvos)), alvos].clamp(min=1e-9)).mean().item()
        print(f"  sai quando confiança > {lim:.2f}: média {prof.float().mean():.1f} camadas ({1 - prof.float().mean().item() / L:.0%} cortado) | "
              f"acurácia {(pred == alvos).float().mean():.1%} loss {loss:.4f}")


if __name__ == "__main__":
    main()
