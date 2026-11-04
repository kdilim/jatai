"""Complementos da analise: (1) soma so-INT8 com arredondamento simetrico (sem vies);
(2) regua: deriva de FP16 e BF16 vs float64 na mesma rede de 12 camadas."""
import numpy as np
import torch
import analise_passo_a_passo as a


def shr_sim(x, s):
    return np.sign(x) * ((np.abs(x) + (1 << (s - 1))) >> s)


def somas_simetricas(K, testes=2000):
    rng = np.random.default_rng(a.SEMENTE)
    x = np.clip(np.round(rng.normal(0, 32, (testes, K))), -127, 127).astype(np.int64)
    y = np.clip(np.round(rng.normal(0, 32, (testes, K))), -127, 127).astype(np.int64)
    exato = (x * y).sum(-1).astype(np.float64)
    esc = np.sqrt(np.mean(exato ** 2))
    out = {}
    for nome, p, fator in (("arvore INT8 simetrica (produto >>7)", np.clip(shr_sim(x * y, 7), -127, 127), 128.0),
                           ("tudo <= 8 bits simetrico (operandos >>4)", shr_sim(x, 4) * shr_sim(y, 4), 256.0)):
        n = 0
        while p.shape[1] > 1:
            p = np.clip(shr_sim(p[:, 0::2], 1) + shr_sim(p[:, 1::2], 1), -127, 127)
            n += 1
        v = p[:, 0] * fator * 2 ** n
        out[nome] = (np.sqrt(np.mean((v - exato) ** 2)) / esc, np.corrcoef(v, exato)[0, 1])
    return out


def deriva_float(dtype, pos="alibi", mlp="relu2"):
    rng = np.random.default_rng(a.SEMENTE)
    x0 = np.clip(np.round(rng.normal(0, 40, (a.LOTE, a.T, a.D))), -127, 127).astype(np.int64)
    xf, xb = a.deq(x0, -5), torch.tensor(a.deq(x0, -5), dtype=dtype)
    mask = torch.tensor(a.MASCARA)
    alibi = torch.tensor(a.alibi_f(), dtype=dtype)
    erros = []
    for c in range(1, a.CAMADAS + 1):
        p = a.camada_pesos(rng, mlp)
        xf = a.camada_float(xf, p, pos, mlp)
        w = {k: torch.tensor(a.deq(*v), dtype=dtype) for k, v in p.items()}
        rms = lambda t: t * torch.rsqrt((t * t).mean(-1, keepdim=True))
        hh = lambda t: t.reshape(a.LOTE, a.T, a.H, a.HD).transpose(1, 2)
        xn = rms(xb)
        z = hh(xn @ w["Wq"]) @ hh(xn @ w["Wk"]).transpose(-1, -2) / 2 + alibi
        P = torch.softmax(z.masked_fill(mask, -1e4), -1)
        xb = xb + ((P @ hh(xn @ w["Wv"])).transpose(1, 2).reshape(a.LOTE, a.T, a.D)) @ w["Wo"]
        xn = rms(xb)
        xb = xb + (torch.relu(xn @ w["Wup"]) ** 2) @ w["Wd"]
        erros.append(a.erro_rel(xb.double().numpy(), xf))
    return erros


for K in (256, 1024):
    for nome, (e, c) in somas_simetricas(K).items():
        print(f"K={K:5d} {nome:42s} erro rel {e:7.3f} | correlacao {c:.4f}")
for nome, dt in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
    er = deriva_float(dt)
    print(f"{nome} vs float64, alibi+relu2: " + " ".join(f"c{c} {er[c - 1]:.3f}" for c in (1, 3, 6, 9, 12)))
