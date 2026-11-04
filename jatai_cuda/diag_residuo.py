"""DIAGNÓSTICO da divergência do INT8 longo: quanto de cada atualização do fluxo residual some no arredondamento INT8
(mantissa de 7 bits + expoente por bloco de 32) quando o resíduo fica grande perto da saída das camadas.
perdido = ||q8(x + d) - x - d|| / ||d||  (x já em INT8; d = saída da atenção ou do MLP, também em INT8).
Uso: python diag_residuo.py <pesos .pt (int8 ou float)>
"""
import os
import sys

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "treino_int8_gpu"))
import treino as dados          # noqa: E402
import jatai_int8 as ji         # noqa: E402
from noite import carregar_tudo  # noqa: E402

DEV = "cuda"
q8 = ji.quant_float


@torch.no_grad()
def main():
    ck = torch.load(sys.argv[1], weights_only=False)
    L, FF = ck["L"], ck["FF"]
    mi = ji.ModeloInt8(256, 256, 64, 4, FF, L, 128, 0, DEV)
    mf = ji.ModeloFloat(mi).to(DEV)
    del mi
    if ck["modo"] == "float":
        mf.load_state_dict(ck["pesos"])
    else:
        for k, (m, E) in ck["pesos"].items():
            W = ji.deq(m.to(DEV), E.to(DEV)).float()
            if k.startswith("Wqkv"):                           # o modelo CUDA guarda Q, K e V numa matriz só
                l = k[4:]
                for n, parte in zip("qkv", W.chunk(3, 0)):
                    mf.w[f"W{n}{l}"].data.copy_(parte)
            else:
                mf.w[k].data.copy_(W.reshape(mf.w[k].shape))
    _, val = carregar_tudo()
    dados.BT = 8
    tok, alvo = (t.to(DEV) for t in dados.lote(val, torch.Generator().manual_seed(123)))
    V, D, H, HD, FF, L, T = mf.cfg
    w, Bt = mf.w, tok.shape[0]
    rms = lambda x: q8(x * torch.rsqrt((x * x).mean(-1, keepdim=True)))
    cab = lambda x: q8(x).reshape(Bt, T, H, HD).transpose(1, 2)
    nr = lambda t: t.pow(2).mean().sqrt().item()
    perd = lambda x, d: ((q8(x + d) - x - d).norm() / d.norm().clamp(min=1e-30)).item()
    x = q8(w["emb"][tok])
    print(f"{sys.argv[1]} | rms do embedding {nr(x):.3g} | |W| rms: emb {nr(w['emb']):.3g}, Wd0 {nr(w['Wd0']):.3g}, Wd{L-1} {nr(w[f'Wd{L-1}']):.3g}")
    print("camada | rms resíduo | rms saída atenção, % perdido | rms saída MLP, % perdido")
    for l in range(L):
        xn = rms(x)
        q, k, v = cab(xn @ w[f"Wq{l}"].t()), cab(xn @ w[f"Wk{l}"].t()), cab(xn @ w[f"Wv{l}"].t())
        a = torch.softmax(((q @ k.transpose(-1, -2)) / 2.0 + mf.vies).masked_fill(mf.mascara, float("-inf")), -1)
        da = q8(q8((a @ v).transpose(1, 2).reshape(Bt, T, D)) @ w[f"Wo{l}"].t())
        pa = perd(x, da)
        x2 = q8(x + da)
        dm = q8(q8(torch.relu(rms(x2) @ w[f"Wup{l}"].t()).square()) @ w[f"Wd{l}"].t())
        pm = perd(x2, dm)
        print(f"{l:6d} | {nr(x):11.3g} | {nr(da):10.3g} {pa:6.1%} | {nr(dm):10.3g} {pm:6.1%}")
        x = q8(x2 + dm)
    lg = rms(x) @ w["out"].t()
    print(f"loss do lote com ativações INT8: {torch.nn.functional.cross_entropy(lg.reshape(-1, V), alvo.reshape(-1)).item():.4f}")


if __name__ == "__main__":
    main()
