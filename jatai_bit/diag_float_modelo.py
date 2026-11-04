"""DIAGNÓSTICO (float, não faz parte do método): a mesma arquitetura e a mesma média de campo de modelo_bit.py, em float
com Adam, para separar "limite do método/arquitetura" de "limite da aritmética inteira".

Modos:
  norm  = como modelo_bit.py: camadas escondidas tanh(0,8 * 6 * M / rms(M)) (sem variância), voto com variância
  var   = variância em TODAS as camadas: média = tanh(0,8 * M / sigma), sigma^2 = sum(1 - x^2 w^2) (o observador exato);
          a soma com decaimento propaga média e variância
Pesos = média do bit, limitados a [-1, 1]. Avaliação binária: pesos = sinal(w), ativações = sinal, igual a logits_binarios.

Uso: python diag_float_modelo.py <norm|var> <passos> [L] [D] [F] [lr]

Escada (variável JF_ESCADA, letras combináveis), troca uma peça do float pela do método inteiro:
  u = regra de update do método: dw = -(G / max|G| do tensor) * 3/127 (sem Adam, sem momento)
  g = gradientes das somas (dZ, dMv, dMo, dMu, dMd) quantizados em int8 por tensor, arredondamento estocástico
  c = peso preso na grade do contador (múltiplos de 1/127), arredondamento estocástico após cada update
  a = ativações (saídas de camada, soma temporal normalizada e voto) arredondadas para int8 Q7 (passos de 1/127)
  s = soma temporal em inteiros: s = s - rnd(s / 2^d) + v, em unidades de 1/127 (o erro de arredondamento entra
      no forward; o gradiente segue o decaimento exato)
  d = gradiente da soma temporal reduzido a 12 bits por tensor (como o renorm de dS antes da soma reversa)
  r = rms e sigma tratados como constantes no backward (como os kernels dtanh_norm_r / voto_var_back)
  R = só o rms constante; S = só o sigma constante; H = só o rms das camadas escondidas; V = só o rms do voto
"""
import os
import sys
import time

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "treino_int8_gpu"))
import treino as dados  # noqa: E402

DEV = "cuda"
V, NB, A = 256, 64, 2
ESCADA = os.environ.get("JF_ESCADA", "")


class QGrad(torch.autograd.Function):
    """identidade no forward; no backward o gradiente vira int8 por tensor (estocástico), como no método inteiro."""
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        s = g.abs().max().clamp(min=1e-30) / 127
        return torch.floor(g / s + torch.rand_like(g)) * s


def qg(x):
    return QGrad.apply(x) if "g" in ESCADA else x


class QGrad12(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        s = g.abs().max().clamp(min=1e-30) / 2047
        return torch.round(g / s) * s


def qa(x):
    """ativação na grade int8 (o gradiente passa como se fosse a média exata, igual ao kernel que deriva no valor guardado)."""
    return x + (torch.round(x * 127) / 127 - x).detach() if "a" in ESCADA else x


def rms(M, voto=False):
    r = M.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-6)
    const = "r" in ESCADA or "R" in ESCADA or ("V" in ESCADA if voto else "H" in ESCADA)
    return r.detach() if const else r


class Modelo(torch.nn.Module):
    def __init__(self, L, D, F, modo):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        u = lambda r, c: torch.nn.Parameter((torch.rand(r, c, generator=g) - 0.5).to(DEV))   # média inicial em [-0,5; 0,5]
        self.E, self.out = u(V, D), u(V, D + NB)
        self.W = torch.nn.ParameterList([u(D, D) for _ in range(2 * L)] + [u(F, D) for _ in range(L)] + [u(D, F) for _ in range(L)])
        self.L, self.D, self.modo = L, D, modo
        self.d = (1 + (torch.arange(D, device=DEV) & 7)).float()

    def pesos(self, l):
        L = self.L
        return self.W[2 * l], self.W[2 * l + 1], self.W[2 * L + l], self.W[3 * L + l]

    @staticmethod
    def sig(x, W):
        sg = (x.shape[1] - (x * x) @ (W * W).t()).clamp(min=1e-3).sqrt()   # sigma^2 = soma(1 - x^2 w^2)
        return sg.detach() if ("r" in ESCADA or "S" in ESCADA) else sg

    def camada(self, x, W):
        M = qg(x @ W.t())
        if self.modo == "var":
            return torch.tanh(0.8 * M / self.sig(x, W))
        return qa(torch.tanh(0.8 * 6 * M / rms(M)))

    def scan(self, v, Bn, T):
        vv = v.view(Bn, T, -1)
        k = 1 - 2.0 ** (-self.d)
        s = torch.zeros_like(vv[:, 0]); q = torch.zeros_like(s)
        Ss, Qs = [], []
        dd = 2.0 ** self.d
        for t in range(T):
            if "s" in ESCADA:                                    # em unidades de 1/127, arredondamento par como r_rnd_v
                exato = s * 127 - torch.round(s * 127 / dd) + torch.round(vv[:, t] * 127)
                s = s * k + vv[:, t]
                s = s + (exato / 127 - s).detach()
            else:
                s = s * k + vv[:, t]
            q = q * k * k + (1 - vv[:, t] ** 2)
            Ss.append(s); Qs.append(q)
        S, Q = torch.stack(Ss, 1).view(Bn * T, -1), torch.stack(Qs, 1).view(Bn * T, -1)
        if "d" in ESCADA:
            S = QGrad12.apply(S)
        if self.modo == "var":
            return torch.tanh(0.8 * S / Q.clamp(min=1e-3).sqrt())
        return qa(torch.tanh(0.8 * 6 * S / rms(S)))

    def voto(self, xin, W, ant):
        M = qg(xin @ W.t())
        u = (M + ant * rms(M, voto=True) * 2 ** A) / self.sig(xin, W)
        return qa(torch.tanh(0.8 * u))

    def forward(self, tok):
        Bn, T = tok.shape
        x = self.E[tok.reshape(-1)]
        for l in range(self.L):
            Wv, Wo, Wu, Wd = self.pesos(l)
            s = self.scan(self.camada(x, Wv), Bn, T)
            xa = self.voto(s, Wo, x)
            x = self.voto(self.camada(xa, Wu), Wd, xa)
        xo = torch.cat([x, torch.ones(x.shape[0], NB, device=DEV)], 1)
        return qg(xo @ self.out.t()) / self.sig(xo, self.out)

    @torch.no_grad()
    def binario(self, tok):
        Bn, T = tok.shape
        sg = lambda t: torch.where(t >= 0, 1.0, -1.0)
        x = sg(self.E)[tok.reshape(-1)]
        for l in range(self.L):
            Wv, Wo, Wu, Wd = [sg(w) for w in self.pesos(l)]
            v = sg(x @ Wv.t()).view(Bn, T, -1)
            k = 1 - 2.0 ** (-self.d)
            s = torch.zeros_like(v[:, 0]); Ss = []
            for t in range(T):
                s = s * k + v[:, t]; Ss.append(s)
            s = sg(torch.stack(Ss, 1).view(Bn * T, -1))
            M = s @ Wo.t(); xa = sg(M / rms(M) * 2 ** -A + x)
            M = sg(xa @ Wu.t()) @ Wd.t(); x = sg(M / rms(M) * 2 ** -A + xa)
        xo = torch.cat([x, torch.ones(x.shape[0], NB, device=DEV)], 1)
        return xo @ sg(self.out).t()


def main():
    modo, passos = sys.argv[1], int(sys.argv[2])
    L, D, F = (int(v) for v in (sys.argv[3:6] + ["4", "256", "1024"][len(sys.argv[3:6]):]))
    lr = float(sys.argv[6]) if len(sys.argv) > 6 else 3e-3
    treino_txt, val_txt = dados.carregar()
    gv = torch.Generator().manual_seed(123)
    lotes_val = [dados.lote(val_txt, gv) for _ in range(8)]
    g = torch.Generator().manual_seed(2)
    m = Modelo(L, D, F, modo)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    ce = torch.nn.functional.cross_entropy
    t0 = time.time()
    for p in range(passos + 1):
        if p % 500 == 0:
            with torch.no_grad():
                lv = sum(float(ce(m(x), y.reshape(-1))) for x, y in lotes_val) / 8
                am = sum(float((m(x).argmax(1) == y.reshape(-1)).float().mean()) for x, y in lotes_val) / 8
                ab = sum(float((m.binario(x).argmax(1) == y.reshape(-1)).float().mean()) for x, y in lotes_val) / 8
                w = torch.cat([q.detach().abs().view(-1) for q in m.parameters()])
            print(f"{modo}{'+' + ESCADA if ESCADA else ''} L{L} lr{lr} passo {p:5d} | loss val (médias) {lv:.3f} acurácia {am:.1%} | rede binária: acurácia {ab:.1%} | "
                  f"|w| médio {float(w.mean()):.2f}, certos (|w|>0,8) {float((w > 0.8).float().mean()):.0%} | {time.time() - t0:.0f}s", flush=True)
        if p == passos:
            break
        x, y = dados.lote(treino_txt, g)
        loss = ce(m(x), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            if "u" in ESCADA:
                for q in m.parameters():
                    q -= q.grad / q.grad.abs().max().clamp(min=1e-30) * (3 / 127)
            else:
                opt.step()
            for q in m.parameters():
                q.clamp_(-1, 1)
                if "c" in ESCADA:
                    q.copy_(torch.floor(q * 127 + torch.rand_like(q)) / 127)


if __name__ == "__main__":
    main()
