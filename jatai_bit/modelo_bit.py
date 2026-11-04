"""Jataí-bit: modelo de linguagem 100% binário (pesos e ativações em bits na rede final), treinado do zero com o
"observador" (contador int8 por peso, médias de campo), sem STE e sem float.

Arquitetura (vocabulário de bytes, V = 256; todas as dimensões são parâmetros):
  embedding binário  x0 = E[token]                                          (contador int8 por bit)
  L blocos:
    mistura temporal  v = bits(Wv x) ; S_t = S_{t-1} - S_{t-1} >> d_c + v_t ; s = bits(S) ; xa = voto(Wo s, x)
    mistura de canais h = bits(Wu xa) ; xb = voto(Wd h, xa)
  saída              logits = Wout x                                        (256 classes)
  bits(M) = sinal(M) na rede final; no treino, média = tanh(0,8 · M · T / rms(M)) (normalização por linha).
  voto(M, x_ant): o residual em bits: sinal(M / rms(M) + x_ant); no treino, tanh(0,8 · (M·T/rms + x_ant·T/128)).
  d_c = 1 + (c & 7): canais com memórias de ~2 a ~256 tokens.

Sistema de logs (classe Log):
  - cada etapa roda dentro de um trecho nomeado ("bloco 2 / temporal / soma com decaimento");
  - sentinelas por etapa: médias mortas (|média| ~ 0) ou saturadas (|média| ~ 127), somas int32 perto de estourar
    (>= 2^30, com linha e coluna), gradiente zerado, contadores colados no limite;
  - erro do CUDA em qualquer lançamento vira exceção com o kernel e o trecho;
  - modo conferir: o mesmo passo roda na especificação int64 (RefBit), e a PRIMEIRA etapa que diferir é reportada
    pelo nome, com posição e os dois valores.
Tudo vai para logs/<nome>.jsonl.

Uso:
  python modelo_bit.py conferir                       # um passo completo, config pequena, kernels x especificação
  python modelo_bit.py treinar <passos> [L] [D] [F] [B] [passo_contador] [semente]
"""
import contextlib
import json
import math
import os
import sys
import time
import zlib
import ctypes

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "treino_int8_gpu"))
import fase1c_kernel as fk                                     # noqa: E402  (aritmética da especificação)
import fase1d_escala as fe                                     # noqa: E402  (r_isqrt64)
from fase1c_kernel import gemm, tr8, lib, P, I, n_, I8, I32, I64, DEV   # noqa: E402

V, T_CTX = 256, 128
RES = int(os.environ.get("JB_RES", "2"))                       # peso do voto do bloco: u = M/rms * 2^-RES + bit anterior
VAR = int(os.environ.get("JB_VAR", "1"))                       # 1 = voto com variância (observador exato), 0 = sem variância
EST = int(os.environ.get("JB_EST", "1"))                       # 1 = arredondamento estocástico nos gradientes e no update
BITS_RES = 18                                                  # bits do gradiente do fluxo residual (era int8: zerava)
BITS_DS = 12                                                   # bits de dS antes da soma reversa
NB = 64                                                        # canais constantes (+1) na entrada da saída: o viés binário
NORM_T = int(os.environ.get("JB_NORM", "6144"))                # alvo da normalização entre camadas (Q10)
TRES = int(os.environ.get("JB_TRES", str(NORM_T)))             # temperatura do voto residual (Q10)
PROJ = int(os.environ.get("JB_PROJ", "0"))
VRMS = int(os.environ.get("JB_VRMS", "0"))                     # 1 = derivada do rms no voto residual (exige JB_VAR=1)                     # 1 = derivada completa da normalização (rms não constante)
SAT = int(os.environ.get("JB_SAT", "0"))                       # > 0: derivada zero onde |entrada do tanh| > SAT (Q10)
POSTO = int(os.environ.get("JB_POSTO", "0"))                   # > 0: matrizes dos blocos fatoradas (bits de linha x bits de coluna)
LIM = 1 << 30
BLOCO_K = 64                                                   # o GEMM exige K múltiplo de 64: o posto é completado com zeros


def pad_k(X):
    """completa as colunas com zeros até múltiplo de 64 (zeros não mudam nenhum produto)."""
    r = X.shape[1]
    rp = -(-r // BLOCO_K) * BLOCO_K
    return X if rp == r else torch.cat([X, torch.zeros(X.shape[0], rp - r, dtype=X.dtype, device=X.device)], 1).contiguous()


def niveis(th, k):
    """int8 em [-127, 127] -> o mais próximo de 2^k níveis igualmente espaçados em [-127, 127] (k = 1: ±127)."""
    n = (1 << k) - 1
    j = torch.div((th.to(torch.int32) + 127) * n + 127, 254, rounding_mode="floor")
    return (-127 + torch.div(j * 254 + n // 2, n, rounding_mode="floor")).clamp(-127, 127).to(I8).contiguous()


def shift_posto(R):
    """W = soma de R produtos de bits; dividir por ~sqrt(R) (o desvio da soma) mantém W na escala de um peso int8."""
    return 7 + (R.bit_length()) // 2


# ============================================================================ sistema de logs
class ErroBit(RuntimeError):
    pass


class Log:
    def __init__(self, nome, vigiar=True, eco=True):
        os.makedirs(os.path.join(AQUI, "logs"), exist_ok=True)
        self.arq = open(os.path.join(AQUI, "logs", nome + ".jsonl"), "a")
        self.pilha, self.passo, self.vigiar, self.eco = [], None, vigiar, eco
        self.mx = torch.zeros(1, dtype=I32, device=DEV)
        self.alerta = torch.zeros(1, dtype=I32, device=DEV)
        self.saude = {}

    def onde(self):
        return " / ".join(self.pilha) or "-"

    @contextlib.contextmanager
    def trecho(self, nome):
        self.pilha.append(nome)
        try:
            yield
        finally:
            self.pilha.pop()

    def escreve(self, tipo, **kw):
        self.arq.write(json.dumps({"hora": time.strftime("%H:%M:%S"), "tipo": tipo, "passo": self.passo, "trecho": self.onde(), **kw},
                                  ensure_ascii=False, default=str) + "\n")
        self.arq.flush()

    def info(self, msg, **kw):
        self.escreve("info", msg=msg, **kw)
        if self.eco:
            print(msg, flush=True)

    def aviso(self, msg, **kw):
        self.escreve("aviso", msg=msg, **kw)
        print(f"[AVISO] passo {self.passo} | {self.onde()} | {msg}", flush=True)

    def falha(self, msg, **kw):
        self.escreve("erro", msg=msg, **kw)
        raise ErroBit(f"\n[ERRO] passo {self.passo} | {self.onde()}\n{msg}")

    def ok(self, r, kernel):
        if r != 0:
            self.falha(f"o CUDA recusou o kernel '{kernel}' (cudaError {r})")

    # ------------------------------------------------------------------ sentinelas
    def medias(self, nome, mu):
        """int8 de médias: morto (sinal não passa) ou saturado (tudo 'certo', gradiente ~0)."""
        if not self.vigiar:
            return
        a = mu.abs().float()
        media, sat, zero = float(a.mean()), float((a >= 120).float().mean()), float((mu == 0).float().mean())
        self.saude[f"{self.onde()} / {nome}"] = round(media, 1)
        const = float((mu.float().mean(0).abs() >= 114).float().mean())        # canal com o mesmo valor em ~todos os tokens
        self.escreve("saude", nome=nome, media_abs=round(media, 2), saturados=round(sat, 4), zeros=round(zero, 4), canais_constantes=round(const, 4))
        if const > 0.5:
            self.aviso(f"'{nome}': {const:.0%} dos canais são constantes entre tokens (colapso: a etapa não carrega informação do texto)")
        if media < 3:
            self.aviso(f"'{nome}': médias quase zero (|média| {media:.2f} de 127): o sinal está morrendo nesta etapa")
        elif sat > 0.98:
            self.aviso(f"'{nome}': {sat:.1%} das médias saturadas (|média| >= 120): gradiente quase zero nesta etapa")

    def soma32(self, nome, x):
        """int32: estouro iminente (>= 2^30) vira erro com linha e coluna."""
        if not self.vigiar:
            return
        self.alerta.zero_()
        self.ok(lib.bt_vigia32(P(x), P(self.mx), P(self.alerta), n_(x)), "vigia32")
        a, m = int(self.alerta), int(self.mx)
        self.escreve("saude", nome=nome, max_abs=m, bits=m.bit_length())
        if a:
            i = a - 1
            self.falha(f"'{nome}': soma int32 com |valor| >= 2^30 (estouraria o registrador) na linha {i // x.shape[1]}, coluna "
                       f"{i % x.shape[1]}; maior |valor| = {m}", nome=nome, indice=i)

    def gradiente(self, nome, G):
        if not self.vigiar:
            return
        m = int(G.abs().max())
        self.escreve("saude", nome=nome, grad_max=m)
        if m == 0:
            self.aviso(f"'{nome}': gradiente todo zero (o sinal de erro não chega a esta etapa)")

    def contadores(self, nome, th):
        if not self.vigiar:
            return
        lim = float((th.abs() >= 127).float().mean())
        self.escreve("saude", nome=nome, no_limite=round(lim, 4), certos=round(float((th.abs() >= 100).float().mean()), 4),
                     media_abs_theta=round(float(th.abs().float().mean()), 1))
        if lim > 0.5:
            self.aviso(f"'{nome}': {lim:.0%} dos contadores no limite ±127 (pesos 'decididos' demais: passo grande?)")


# ============================================================================ o modelo nos kernels
class ModeloBit:
    def __init__(self, L, D, F, shift, semente, log):
        self.L, self.D, self.F, self.shift, self.log = L, D, F, shift, log
        g = torch.Generator(device=DEV).manual_seed(semente)
        sorteia = lambda r, c: torch.randint(-64, 65, (r, c), generator=g, device=DEV).to(I8).contiguous()
        self.E = sorteia(V, D)
        self.blocos = [{"Wv": sorteia(D, D), "Wo": sorteia(D, D), "Wu": sorteia(F, D), "Wd": sorteia(D, F)} for _ in range(L)]
        self.out = sorteia(V, D + NB)
        if POSTO:
            # a ideia da tabela: W[i][j] = soma_r bit_linha[i][r] * bit_coluna[j][r]; só os bits A (linhas) e B (colunas)
            # existem e são treinados; W é sempre gerada a partir deles
            for b in self.blocos:
                for k in ("Wv", "Wo", "Wu", "Wd"):
                    b[k + "A"], b[k + "B"] = sorteia(b[k].shape[0], POSTO), sorteia(b[k].shape[1], POSTO)
            self._compor()
        self.inter = {}                                     # etapas guardadas (modo conferir)
        self.guardar = False
        self.t = 0                                          # passos de treino feitos (semente do sorteio)

    def tensores(self):
        t = [("E", self.E)]
        for l, b in enumerate(self.blocos):
            t += [(f"bloco {l} {k}", b[k]) for k in (("WvA", "WvB", "WoA", "WoB", "WuA", "WuB", "WdA", "WdB") if POSTO else ("Wv", "Wo", "Wu", "Wd"))]
        return t + [("saida", self.out)]

    def _compor(self):
        """média de W = rnd(A . B^T, 7 + ~log2(sqrt R)) a partir das médias dos bits de linha e de coluna."""
        s = shift_posto(POSTO)
        for b in self.blocos:
            for k in ("Wv", "Wo", "Wu", "Wd"):
                A, B = b[k + "A"], b[k + "B"]
                if k not in b:
                    b[k] = torch.empty(A.shape[0], B.shape[0], dtype=I8, device=DEV)
                self.log.ok(lib.bt_compor(P(A), P(B), P(b[k]), I(A.shape[0]), I(B.shape[0]), I(POSTO), I(s)), "compor")

    def _gradientes_fatores(self, G):
        """dA = G . B e dB = G^T . A (o gradiente de cada bit de linha/coluna soma a linha/coluna inteira de G)."""
        for l, b in enumerate(self.blocos):
            for k in ("Wv", "Wo", "Wu", "Wd"):
                gq, _ = self._quant(G.pop(f"bloco {l} {k}"), f"b{l} G{k}")
                A, B = b[k + "A"], b[k + "B"]
                dA = torch.empty(A.shape[0], POSTO, dtype=I32, device=DEV)
                dB = torch.empty(B.shape[0], POSTO, dtype=I32, device=DEV)
                self.log.ok(lib.bt_grad_fatores(P(gq), P(A), P(B), P(dA), P(dB), I(A.shape[0]), I(B.shape[0]), I(POSTO)), "grad_fatores")
                G[f"bloco {l} {k}A"], G[f"bloco {l} {k}B"] = dA, dB

    def _g(self, nome, t):
        if self.guardar:
            self.inter[nome] = t.clone()
        return t

    def _norm(self, M, nome):
        lg = self.log
        mo, mo2 = torch.empty(M.shape, dtype=I8, device=DEV), torch.empty(M.shape, dtype=I8, device=DEV)
        inv, kk = torch.empty(M.shape[0], dtype=I64, device=DEV), torch.empty(M.shape[0], dtype=I32, device=DEV)
        lg.ok(lib.bt_norm_tanh_r(P(M), I(M.shape[0]), I(M.shape[1]), I(NORM_T), P(mo), P(mo2), P(inv), P(kk), I(0)), "norm_tanh_r")
        lg.medias(nome, mo)
        return mo, mo2, inv, kk

    def _voto(self, M, ant, x2, W, nome):
        """voto residual; com VAR a variância de M vem de S2 = (média^2 da entrada) . (contador^2)."""
        lg = self.log
        mo, mo2 = torch.empty(M.shape, dtype=I8, device=DEV), torch.empty(M.shape, dtype=I8, device=DEV)
        if VAR:
            w2 = torch.empty_like(W)
            lg.ok(lib.bt_quad8(P(W), P(w2), n_(W)), "quad8")
            S2 = gemm(x2, w2)
            sv, rf = torch.empty(M.shape, dtype=I32, device=DEV), torch.empty(M.shape[0], dtype=I64, device=DEV)
            lg.ok(lib.bt_voto_var(P(M), P(S2), P(ant), I(M.shape[0]), I(M.shape[1]), I(x2.shape[1]), I(RES), P(mo), P(mo2), P(sv), P(rf)), "voto_var")
            p1, p2 = sv, rf
        else:
            p1, p2 = torch.empty(M.shape[0], dtype=I64, device=DEV), torch.empty(M.shape[0], dtype=I32, device=DEV)
            lg.ok(lib.bt_norm_res_r(P(M), P(ant), I(M.shape[0]), I(M.shape[1]), I(TRES), I(RES), P(mo), P(mo2), P(p1), P(p2)), "norm_res_r")
        lg.medias(nome, mo)
        return mo, mo2, p1, p2

    def _voto_back(self, dmu, mu, p1, p2, M=None, ant=None):
        lg = self.log
        dM, dant = torch.empty_like(dmu), torch.empty_like(dmu)
        if VAR:
            lg.ok(lib.bt_voto_var_back(P(dmu), P(mu), P(p1), P(p2), I(self.D), P(dM), P(dant), n_(dmu)), "voto_var_back")
            if VRMS:
                lg.ok(lib.bt_voto_rms_back(P(dM), P(dmu), P(mu), P(ant), P(p1), P(p2), P(M), I(M.shape[0]), I(self.D), I(RES)), "voto_rms_back")
        else:
            lg.ok(lib.bt_res_back(P(dmu), P(mu), P(p1), P(p2), I(TRES), I(RES), I(self.D), P(dM), P(dant), n_(dmu)), "res_back")
        return dM, dant

    def _sat_voto(self, d, M, ant, sv, rf):
        """com SAT: zera o gradiente onde a entrada do tanh do voto passou de SAT (Q10). O tanh inteiro vira 127 já em
        ~2,8 e a derivada 16384 - 127^2 fica num piso de 1,6%, quando a verdadeira é ~0: o piso empurrava os pesos até
        ±127 e saturava o voto inteiro (visto em comparar_camadas.py)."""
        if not SAT:
            return d
        M, a = M.to(I64), ant.to(I64)
        u = (torch.div(M << 3, sv, rounding_mode="trunc") + torch.div(a * rf.view(-1, 1), 16 * sv.to(I64), rounding_mode="trunc"))
        return torch.where(((u.clamp(-(1 << 20), 1 << 20) * 205) >> 8).abs() > SAT, 0, d).contiguous()

    def _sat_norm(self, d, M, inv, k):
        if not SAT:
            return d
        u = ((M.to(I64) * NORM_T * inv.view(-1, 1)) >> k.to(I64).view(-1, 1)) * 205 >> 8
        return torch.where(u.abs() > SAT, 0, d).contiguous()

    def _proj(self, dM, M):
        """com PROJ: tira de dM a componente na direção de M (a derivada do 1/rms(M) que o dtanh_norm_r não tem)."""
        if PROJ:
            self.log.ok(lib.bt_proj_norm(P(dM), P(M), I(M.shape[0]), I(M.shape[1])), "proj_norm")

    def semente(self, nome):
        return ctypes.c_uint(semente(nome, self.t))

    @staticmethod
    def _bitlen(mx):
        return (mx >= torch.tensor([1 << i for i in range(31)], dtype=I32, device=DEV)).sum().to(I32)

    def _maxabs(self, X):
        mx = torch.zeros(1, dtype=I32, device=DEV)
        self.log.ok(lib.bt_maxabs(P(X), P(mx), n_(X)), "maxabs")
        return mx

    def _quant(self, X, nome):
        """int32 -> int8 por escala de tensor (estocástico com EST); devolve (q, shift aplicado, no device)."""
        lg = self.log
        lg.soma32(nome, X)
        mx = self._maxabs(X)
        q = torch.empty(X.shape, dtype=I8, device=DEV)
        if EST:
            lg.ok(lib.bt_quant8s(P(X), P(mx), P(q), self.semente(nome), n_(X)), "quant8s")
        else:
            lg.ok(lib.bt_quant8(P(X), P(mx), P(q), n_(X)), "quant8")
        if lg.vigiar:                                   # quanto do gradiente some no arredondamento para int8
            lg.escreve("saude", nome=nome, zeros_q=round(float((q == 0).float().mean()), 4), zeros_antes=round(float((X == 0).float().mean()), 4))
        return q, (self._bitlen(mx) - 7).clamp(min=0)

    def _renorm(self, X, alvo, nome):
        """int32 -> int32 com o maior |x| em 'alvo' bits; devolve (x, shift aplicado com sinal)."""
        lg = self.log
        lg.soma32(nome, X)
        mx = self._maxabs(X)
        out = torch.empty_like(X)
        lg.ok(lib.bt_renorm(P(X), P(mx), P(out), I(alvo), n_(X)), "renorm")
        if lg.vigiar:
            lg.escreve("saude", nome=nome, zeros_antes=round(float((X == 0).float().mean()), 4), zeros_q=round(float((out == 0).float().mean()), 4))
        return out, self._bitlen(mx) - alvo

    def _soma(self, a, b, shifts, nome, base=17):
        """a (caminho direto do voto) + b (caminho que passou pelo bloco e pelos shifts dados), na escala de b."""
        lg = self.log
        sh = (sum(shifts) - base).to(I32).reshape(1).contiguous()
        out = torch.empty_like(a)
        lg.alerta.zero_()
        lg.ok(lib.bt_soma_escala(P(a), P(b), P(sh), P(out), P(lg.alerta), n_(a)), "soma_escala")
        if lg.vigiar and int(lg.alerta):
            i = int(lg.alerta) - 1
            lg.falha(f"'{nome}': soma dos caminhos do gradiente passou de 2^30 na linha {i // a.shape[1]}, coluna {i % a.shape[1]} "
                     f"(shift {int(sh)})")
        return out

    # ------------------------------------------------------------------ passo de treino
    def passo(self, tok, alvo, treinar=True):
        lg = self.log
        Bn, T = tok.shape
        N = Bn * T
        tokf, alvof = tok.reshape(-1), alvo.reshape(-1)
        cache = []
        with lg.trecho("forward"):
            with lg.trecho("embedding"):
                x = self._g("x0", self.E[tokf].contiguous())
                lg.medias("x0", x)
            for l, b in enumerate(self.blocos):
                c = {"x": x}
                with lg.trecho(f"bloco {l}"):
                    with lg.trecho("temporal"):
                        Mv = gemm(x, b["Wv"]); lg.soma32("Mv", Mv); c["Mv"] = Mv
                        c["mv"], _, c["iv"], c["kv"] = self._norm(Mv, "v")
                        S = torch.empty(N, self.D, dtype=I32, device=DEV)
                        lg.ok(lib.bt_scan_fwd(P(c["mv"]), P(S), I(Bn), I(T), I(self.D)), "scan_fwd")
                        lg.soma32("S (soma com decaimento)", S)
                        self._g(f"b{l} S", S); c["S"] = S
                        c["ms"], c["ms2"], c["is"], c["ks"] = self._norm(S, "s")
                        Mo = gemm(c["ms"], b["Wo"]); lg.soma32("Mo", Mo); c["Mo"] = Mo
                        c["xa"], c["xa2"], c["ia"], c["ka"] = self._voto(Mo, x, c["ms2"], b["Wo"], "xa (voto residual)")
                        self._g(f"b{l} xa", c["xa"])
                    with lg.trecho("canais"):
                        Mu = gemm(c["xa"], b["Wu"]); lg.soma32("Mu", Mu); c["Mu"] = Mu
                        c["mh"], c["mh2"], c["ih"], c["kh"] = self._norm(Mu, "h")
                        Md = gemm(c["mh"], b["Wd"]); lg.soma32("Md", Md); c["Md"] = Md
                        c["xb"], c["xb2"], c["ib"], c["kb"] = self._voto(Md, c["xa"], c["mh2"], b["Wd"], "xb (voto residual)")
                        self._g(f"b{l} xb", c["xb"])
                cache.append(c)
                x = c["xb"]
            with lg.trecho("saida"):
                m2o = torch.empty_like(self.out)
                lg.ok(lib.bt_quad8(P(self.out), P(m2o), n_(self.out)), "quad8")
                x2 = cache[-1]["xb2"] if cache else torch.empty_like(x)
                if not cache:
                    lg.ok(lib.bt_quad8(P(x), P(x2), n_(x)), "quad8")
                xo = torch.cat([x, torch.full((N, NB), 127, dtype=I8, device=DEV)], 1).contiguous()
                xo2 = torch.cat([x2, torch.full((N, NB), 126, dtype=I8, device=DEV)], 1).contiguous()   # rnd(127^2, 7)
                Mz, S2 = gemm(xo, self.out), gemm(xo2, m2o)
                lg.soma32("logits (soma)", Mz)
                so, sv = torch.empty_like(Mz), torch.empty_like(Mz)
                lg.ok(lib.bt_epilogo(P(Mz), P(S2), I(self.D + NB), None, None, P(so), P(sv), I(1), n_(Mz)), "epilogo")
                self._g("logits", so)
                d = torch.empty_like(so)
                erros = torch.zeros(1, dtype=I32, device=DEV)
                lg.ok(lib.bt_softmax_ce(P(so), P(alvof), P(d), P(erros), I(N), I(V), I(V)), "softmax_ce")
                self._g("d", d)
        if not treinar:
            return so, erros
        G = {}
        with lg.trecho("backward"):
            with lg.trecho("saida"):
                dZ = torch.empty_like(d)
                lg.ok(lib.bt_dmo(P(d), P(sv), P(dZ), n_(d)), "dmo")
                dq, _ = self._quant(dZ, "dZ")
                G["saida"] = gemm(tr8(dq), tr8(xo))
                dx = gemm(dq, tr8(self.out[:, :self.D].contiguous()))
                self._g("dx saida", dx)
            for l in reversed(range(self.L)):
                b, c = self.blocos[l], cache[l]
                with lg.trecho(f"bloco {l}"):
                    with lg.trecho("canais"):
                        dx = self._sat_voto(dx, c["Md"], c["xa"], c["ib"], c["kb"])
                        dMd, dxa_dir = self._voto_back(dx, c["xb"], c["ib"], c["kb"], c["Md"], c["xa"])
                        dqd, sd = self._quant(dMd, f"b{l} dMd")
                        G[f"bloco {l} Wd"] = gemm(tr8(dqd), tr8(c["mh"]))
                        dmh = gemm(dqd, tr8(b["Wd"]))
                        dmh = self._sat_norm(dmh, c["Mu"], c["ih"], c["kh"])
                        dMu = torch.empty_like(dmh)
                        lg.ok(lib.bt_dtanh_norm_r(P(dmh), P(c["mh"]), P(c["ih"]), P(c["kh"]), I(NORM_T), I(self.F), P(dMu), n_(dmh)), "dtanh_norm_r")
                        self._proj(dMu, c["Mu"])
                        dqu, su = self._quant(dMu, f"b{l} dMu")
                        G[f"bloco {l} Wu"] = gemm(tr8(dqu), tr8(c["xa"]))
                        dxa = self._renorm(self._soma(dxa_dir, gemm(dqu, tr8(b["Wu"])), [sd, su], "dxa"), BITS_RES, f"b{l} dxa")[0]
                        self._g(f"b{l} dxa", dxa)
                    with lg.trecho("temporal"):
                        dxa = self._sat_voto(dxa, c["Mo"], c["x"], c["ia"], c["ka"])
                        dMo, dx_dir = self._voto_back(dxa, c["xa"], c["ia"], c["ka"], c["Mo"], c["x"])
                        dqo, so_ = self._quant(dMo, f"b{l} dMo")
                        G[f"bloco {l} Wo"] = gemm(tr8(dqo), tr8(c["ms"]))
                        dms = gemm(dqo, tr8(b["Wo"]))
                        dms = self._sat_norm(dms, c["S"], c["is"], c["ks"])
                        dS = torch.empty_like(dms)
                        lg.ok(lib.bt_dtanh_norm_r(P(dms), P(c["ms"]), P(c["is"]), P(c["ks"]), I(NORM_T), I(self.D), P(dS), n_(dms)), "dtanh_norm_r")
                        self._proj(dS, c["S"])
                        dSq32, sS = self._renorm(dS, BITS_DS, f"b{l} dS")   # poucos bits antes da soma reversa (canais lentos ampliam 256x)
                        dv = torch.empty_like(dS)
                        lg.ok(lib.bt_scan_back(P(dSq32), P(dv), I(Bn), I(T), I(self.D)), "scan_back")
                        lg.soma32("dv (soma com decaimento, backward)", dv)
                        self._g(f"b{l} dv", dv)
                        dv = self._sat_norm(dv, c["Mv"], c["iv"], c["kv"])
                        dMv = torch.empty_like(dv)
                        lg.ok(lib.bt_dtanh_norm_r(P(dv), P(c["mv"]), P(c["iv"]), P(c["kv"]), I(NORM_T), I(self.D), P(dMv), n_(dv)), "dtanh_norm_r")
                        self._proj(dMv, c["Mv"])
                        dqv, sv_ = self._quant(dMv, f"b{l} dMv")
                        G[f"bloco {l} Wv"] = gemm(tr8(dqv), tr8(c["x"]))
                        dx = self._renorm(self._soma(dx_dir, gemm(dqv, tr8(b["Wv"])), [so_, sS, sv_], "dx", base=27), BITS_RES, f"b{l} dx")[0]
                        self._g(f"b{l} dx", dx)
            with lg.trecho("embedding"):
                GE = torch.zeros(V, self.D, dtype=I32, device=DEV)
                GE.index_add_(0, tokf, dx)                    # soma dos gradientes de cada token (inteiros)
                G["E"] = GE
            if POSTO:
                with lg.trecho("fatores"):
                    self._gradientes_fatores(G)
        with lg.trecho("update"):
            for nome, th in self.tensores():
                with lg.trecho(nome):
                    g_ = G[nome]
                    lg.gradiente(nome, g_)
                    self._g(f"G {nome}", g_)
                    mx = torch.zeros(1, dtype=I32, device=DEV)
                    lg.ok(lib.bt_maxabs(P(g_), P(mx), n_(g_)), "maxabs")
                    if EST:
                        lg.ok(lib.bt_update_c8s(P(th), P(g_), P(mx), I(self.shift), self.semente("upd " + nome), n_(th)), "update_c8s")
                    else:
                        lg.ok(lib.bt_update_c8(P(th), P(g_), P(mx), I(self.shift), n_(th)), "update_c8")
                    lg.contadores(nome, th)
            if POSTO:
                self._compor()
        self.t += 1
        return so, erros

    # ------------------------------------------------------------------ rede final 100% binária
    def logits_binarios(self, tok, pesos_crus=False, ativ_bits=1, ativ_tern=0):
        """pesos = sinal(contador), ativações = bits (+-1); as somas inteiras ficam no registrador/kernels.
        pesos_crus: usa os contadores como estão (para avaliar pesos de k bits já quantizados).
        ativ_bits = k > 1: cada ativação é a média do treino reduzida a 2^k níveis (grupo de k bits em vez de 1).
        ativ_tern = τ > 0: ativação ternária, média com |m| <= τ vira 0 (sem pulso), as outras ±127."""
        lg = self.log
        Bn, T = tok.shape
        N = Bn * T

        def sinal_w(t):
            w = torch.empty(t.shape, dtype=I8, device=DEV)
            lg.ok(lib.bt_sinal_theta_c8(P(t), P(w), n_(t)), "sinal_theta_c8")
            return w

        def peso(b, k):
            if not POSTO:
                return b[k] if pesos_crus else sinal_w(b[k])
            if pesos_crus:                                      # A e B já reduzidos a k bits: W composta por _compor
                return b[k]
            # W = sinal(A) . sinal(B)^T: inteiro em [-R, R] (cabe em int8 para R <= 127); x.W = soma_r sinal(A_r) (sinal(B_r).x),
            # ou seja, na inferência são só 2 produtos binários de posto R
            return gemm(pad_k(sinal_w(b[k + "A"])), pad_k(sinal_w(b[k + "B"]))).clamp(-127, 127).to(I8).contiguous()

        multi = ativ_bits > 1 or ativ_tern > 0

        def quant(mo):
            if ativ_tern:
                return torch.where(mo.abs() > ativ_tern, torch.where(mo > 0, 127, -127), 0).to(I8).contiguous()
            return niveis(mo, ativ_bits)

        def bits(M):
            if ativ_tern:
                return quant(self._norm(M, "ativ")[0])
            if ativ_bits > 1:
                return niveis(self._norm(M, "ativ")[0], ativ_bits)
            h = torch.empty(M.shape, dtype=I8, device=DEV)
            lg.ok(lib.bt_sinal8(P(M), P(h), n_(M)), "sinal8")
            return h

        def voto(M, ant):
            mo, mo2 = torch.empty(M.shape, dtype=I8, device=DEV), torch.empty(M.shape, dtype=I8, device=DEV)
            inv, kk = torch.empty(M.shape[0], dtype=I64, device=DEV), torch.empty(M.shape[0], dtype=I32, device=DEV)
            a8 = ant if multi else (ant.to(I32) * 127).to(I8)
            lg.ok(lib.bt_norm_res_r(P(M), P(a8), I(M.shape[0]), I(M.shape[1]), I(TRES), I(RES), P(mo), P(mo2), P(inv), P(kk)), "norm_res_r")
            if multi:
                return quant(mo)
            return torch.where(mo >= 0, 1, -1).to(I8)          # o sinal de u (empate -> +1)

        E = self.E if pesos_crus else sinal_w(self.E)
        x = quant(E[tok.reshape(-1)].contiguous()) if multi else bits(E[tok.reshape(-1)].to(I32).contiguous())
        for b in self.blocos:
            v = bits(gemm(x, peso(b, "Wv")))
            S = torch.empty(N, self.D, dtype=I32, device=DEV)
            lg.ok(lib.bt_scan_fwd(P(v), P(S), I(Bn), I(T), I(self.D)), "scan_fwd")
            s = bits(S)
            xa = voto(gemm(s, peso(b, "Wo")), x)
            h = bits(gemm(xa, peso(b, "Wu")))
            x = voto(gemm(h, peso(b, "Wd")), xa)
        xo = torch.cat([x, torch.full((N, NB), 127 if multi else 1, dtype=I8, device=DEV)], 1).contiguous()
        return gemm(xo, self.out if pesos_crus else sinal_w(self.out))


# ============================================================================ especificação int64 (mesmos nomes)
def r_norm(M):
    rms = fe.r_isqrt64(fk.tdiv((M * M).sum(1, keepdim=True), torch.full((M.shape[0], 1), M.shape[1], dtype=I64)))
    k = torch.tensor([[int(v).bit_length() + 20] for v in rms.view(-1).tolist()], dtype=I64)
    inv = torch.tensor([[(1 << int(kv)) // int(rv)] for kv, rv in zip(k.view(-1).tolist(), rms.view(-1).tolist())], dtype=I64)
    return inv, k


def r_tanh_norm(M):
    inv, k = r_norm(M)
    m = fk.r_tanh((((M * NORM_T * inv) >> k) * 205) >> 8)
    return m, inv, k


def r_voto(M, ant):
    inv, k = r_norm(M)
    u = ((M * TRES * inv) >> (k + RES)) + fk.tdiv(ant * TRES, torch.full_like(ant, 128))
    m = fk.r_tanh((u * 205) >> 8)
    return m, inv, k


def r_voto_var(M, S2, ant, K):
    rms = fe.r_isqrt64(fk.tdiv((M * M).sum(1, keepdim=True), torch.full((M.shape[0], 1), M.shape[1], dtype=I64)))
    rf = rms << RES
    sv = fk.r_isqrt((K * 16384 - S2).clamp(min=1))
    u = (fk.tdiv(M << 3, sv) + fk.tdiv(ant * rf, 16 * sv)).clamp(-(1 << 20), 1 << 20)
    return fk.r_tanh((u * 205) >> 8), sv, rf


def r_voto_any(M, ant, x2, W):
    if VAR:
        return r_voto_var(M, x2 @ fk.r_rnd(W * W, 7).t(), ant, x2.shape[1])
    return r_voto(M, ant)


def r_tdiv_py(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def r_voto_back(d, m, p1, p2, M=None, ant=None):
    dt = r_dtanh(d, m)
    if VAR:
        dM = fk.tdiv(dt << 10, p1)
        if VRMS:                                       # especificação de k_voto_rms_back (inteiros exatos do Python)
            dM = dM.clone()
            for r, (dtl, al, svl, Ml, dMl) in enumerate(zip(dt.tolist(), ant.tolist(), p1.tolist(), M.tolist(), dM.tolist())):
                Z = sum(r_tdiv_py(x * y * 65536, z) for x, y, z in zip(dtl, al, svl))
                den = len(Ml) * int(p2.view(-1)[r]) * 65536
                if den == 0:
                    continue
                for j in range(len(Ml)):
                    v = dMl[j] + r_tdiv_py(Ml[j] * Z * (8 << (2 * RES)), den)
                    dM[r, j] = max(-2147483647, min(2147483647, v))
        return dM, fk.tdiv(dt * p2, 16 * p1)
    return (((dt * p1) >> 10) * 128 * TRES) >> (p2 - 10 + RES), fk.tdiv(dt * TRES, torch.full_like(dt, 128))


def r_dtanh(d, m):
    return fk.r_rnd(fk.r_rnd(d * (16384 - m * m), 14) * 205, 8)


def r_dnorm(d, m, inv, k):
    return (((r_dtanh(d, m) * inv) >> 10) * 128 * NORM_T) >> (k - 10)


def r_proj(dM, M):
    """especificação de k_proj_norm: dM_j - tdiv(M_j sum(dM M), sum(M^2)) por linha, inteiros exatos (Python)."""
    if not PROJ:
        return dM
    out = dM.clone()
    for r, (d, m) in enumerate(zip(dM.tolist(), M.tolist())):
        Pp, Q = sum(a * b for a, b in zip(d, m)), sum(b * b for b in m)
        if Q == 0:
            continue
        for j in range(len(d)):
            num = m[j] * Pp
            corr = abs(num) // Q * (1 if num >= 0 else -1)
            out[r, j] = max(-2147483647, min(2147483647, d[j] - corr))
    return out


def r_rnd_v(x, d):
    """arredondamento par com shift por canal (tensor d >= 1)."""
    return (x + ((1 << (d - 1)) - 1) + ((x >> d) & 1)) >> d


def r_scan(v, Bn, T):
    D = v.shape[1]
    d = torch.tensor([1 + (c & 7) for c in range(D)], dtype=I64)
    vv = v.reshape(Bn, T, D)
    S = torch.zeros_like(vv)
    s = torch.zeros(Bn, D, dtype=I64)
    for t in range(T):
        s = s - r_rnd_v(s, d) + vv[:, t]
        S[:, t] = s
    return S.reshape(Bn * T, D)


def r_scan_back(dS, Bn, T):
    D = dS.shape[1]
    d = torch.tensor([1 + (c & 7) for c in range(D)], dtype=I64)
    dd = dS.reshape(Bn, T, D)
    out = torch.zeros_like(dd)
    g = torch.zeros(Bn, D, dtype=I64)
    for t in reversed(range(T)):
        g = dd[:, t] + g - r_rnd_v(g, d)
        out[:, t] = g
    return out.reshape(Bn * T, D)


M32 = 0xFFFFFFFF


def semente(nome, t):
    return (zlib.crc32(nome.encode()) + t * 0x9E3779B1) & M32


def r_sorteio(n, sem, s):
    i = torch.arange(n, dtype=I64)
    return fe.r_hash32((i * 0x9E3779B1 + sem) & M32) & ((1 << s) - 1)


def r_sround(X, s, sem):
    """X * 2^-s com arredondamento estocástico (mesmo sorteio do kernel)."""
    if s <= 0:
        return X
    base = X >> s
    resto = X - (base << s)
    return base + (r_sorteio(X.numel(), sem, s).view(X.shape) < resto).to(I64)


def r_quant(X, nome=None):
    s = max(fk.r_bitlen(X.abs().max()) - 7, 0)
    q = r_sround(X, s, semente(nome, 0)) if EST else fk.r_rnd(X, s)
    return q.clamp(-127, 127), s


def r_renorm(X, alvo):
    d = fk.r_bitlen(X.abs().max()) - alvo
    return (fk.r_rnd(X, min(d, 30)) if d >= 0 else X << min(-d, 30)), d


def r_soma(a, b, s_total, base=17):
    s = s_total - base
    v = fk.r_rnd(a, min(s, 30)) + b if s >= 0 else a + fk.r_rnd(b, min(-s, 30))
    return r_renorm(v.clamp(-(LIM - 1), LIM - 1), BITS_RES)[0]


def ref_passo(m0, tok, alvo, shift):
    """o mesmo passo na especificação (int64, CPU); devolve (etapas, contadores novos)."""
    Bn, T = tok.shape
    tokf, alvof = tok.reshape(-1).cpu(), alvo.reshape(-1).cpu()
    E = m0["E"].cpu().to(I64)
    blocos = [{k: v.cpu().to(I64) for k, v in b.items()} for b in m0["blocos"]]
    out = m0["out"].cpu().to(I64)
    et, cache = {}, []
    x = E[tokf]
    et["x0"] = x
    for l, b in enumerate(blocos):
        c = {"x": x}
        c["Mv"] = x @ b["Wv"].t()
        c["mv"], c["iv"], c["kv"] = r_tanh_norm(c["Mv"])
        S = r_scan(c["mv"], Bn, T)
        et[f"b{l} S"] = S; c["S"] = S
        c["ms"], c["is"], c["ks"] = r_tanh_norm(S)
        c["Mo"] = c["ms"] @ b["Wo"].t()
        c["xa"], c["ia"], c["ka"] = r_voto_any(c["Mo"], x, fk.r_rnd(c["ms"] * c["ms"], 7), b["Wo"])
        et[f"b{l} xa"] = c["xa"]
        c["Mu"] = c["xa"] @ b["Wu"].t()
        c["mh"], c["ih"], c["kh"] = r_tanh_norm(c["Mu"])
        c["Md"] = c["mh"] @ b["Wd"].t()
        c["xb"], c["ib"], c["kb"] = r_voto_any(c["Md"], c["xa"], fk.r_rnd(c["mh"] * c["mh"], 7), b["Wd"])
        et[f"b{l} xb"] = c["xb"]
        cache.append(c)
        x = c["xb"]
    x = torch.cat([x, torch.full((x.shape[0], NB), 127, dtype=I64)], 1)
    x2 = fk.r_rnd(x * x, 7)
    m2o = fk.r_rnd(out * out, 7)
    Mz, S2 = x @ out.t(), x2 @ m2o.t()
    sv = fk.r_isqrt((x.shape[1] * 16384 - S2).clamp(min=1))
    so = fk.tdiv(Mz << 3, sv)
    et["logits"] = so
    mx = so.max(1, keepdim=True).values
    ex = fk.r_exp((mx - so) >> 2)
    d = (ex << 15) // ex.sum(1, keepdim=True)
    d[torch.arange(d.shape[0]), alvof] -= 32768
    et["d"] = d
    G = {}
    dq, _ = r_quant(fk.tdiv(d << 7, sv), "dZ")
    G["saida"] = dq.t() @ x
    dx = dq @ out[:, :out.shape[1] - NB]
    et["dx saida"] = dx
    for l in reversed(range(len(blocos))):
        b, c = blocos[l], cache[l]
        dMd, dxa_dir = r_voto_back(dx, c["xb"], c["ib"], c["kb"], c["Md"], c["xa"])
        dqd, sd = r_quant(dMd, f"b{l} dMd")
        G[f"bloco {l} Wd"] = dqd.t() @ c["mh"]
        dqu, su = r_quant(r_proj(r_dnorm(dqd @ b["Wd"], c["mh"], c["ih"], c["kh"]), c["Mu"]), f"b{l} dMu")
        G[f"bloco {l} Wu"] = dqu.t() @ c["xa"]
        dxa = r_soma(dxa_dir, dqu @ b["Wu"], sd + su)
        et[f"b{l} dxa"] = dxa
        dMo, dx_dir = r_voto_back(dxa, c["xa"], c["ia"], c["ka"], c["Mo"], c["x"])
        dqo, so_ = r_quant(dMo, f"b{l} dMo")
        G[f"bloco {l} Wo"] = dqo.t() @ c["ms"]
        dS = r_proj(r_dnorm(dqo @ b["Wo"], c["ms"], c["is"], c["ks"]), c["S"])
        dSq, sS = r_renorm(dS, BITS_DS)
        dv = r_scan_back(dSq, Bn, T)
        et[f"b{l} dv"] = dv
        dqv, sv_ = r_quant(r_proj(r_dnorm(dv, c["mv"], c["iv"], c["kv"]), c["Mv"]), f"b{l} dMv")
        G[f"bloco {l} Wv"] = dqv.t() @ c["x"]
        dx = r_soma(dx_dir, dqv @ b["Wv"], so_ + sS + sv_, base=27)
        et[f"b{l} dx"] = dx
    GE = torch.zeros_like(E)
    GE.index_add_(0, tokf, dx)
    G["E"] = GE
    novos = {}
    for nome, th in [("E", E)] + [(f"bloco {l} {k}", blocos[l][k]) for l in range(len(blocos)) for k in ("Wv", "Wo", "Wu", "Wd")] + [("saida", out)]:
        g = G[nome]
        et[f"G {nome}"] = g
        g8 = fk.r_rnd(g, max(fk.r_bitlen(g.abs().max()) - 8, 0))
        passo_ = r_sround(g8, shift, semente("upd " + nome, 0)) if EST else fk.r_rnd(g8, shift)
        novos[nome] = (th - passo_).clamp(-127, 127)
    return et, novos


# ============================================================================ conferir e treinar
def conferir():
    """config pequena (2 blocos, D = 128, F = 256, 2 sequências de 64 tokens): um passo completo, etapa por etapa."""
    log = Log("modelo_bit_conferir")
    log.passo = 0
    m = ModeloBit(2, 128, 256, 4, 1, log)
    g = torch.Generator(device=DEV).manual_seed(5)
    tok = torch.randint(0, V, (2, 65), generator=g, device=DEV)
    x, y = tok[:, :-1].contiguous(), tok[:, 1:].contiguous()
    m0 = {"E": m.E.clone(), "blocos": [{k: v.clone() for k, v in b.items()} for b in m.blocos], "out": m.out.clone()}
    m.guardar = True
    m.passo(x, y)
    et, novos = ref_passo(m0, x, y, m.shift)
    ordem = list(et.keys())
    falhou = None
    for nome in ordem:
        a, b_ = m.inter[nome].cpu().to(I64), et[nome]
        dif = (a != b_)
        if dif.any():
            pos = dif.nonzero()[0].tolist()
            falhou = (nome, int(dif.sum()), a.numel(), pos, int(a[tuple(pos)]), int(b_[tuple(pos)]))
            break
        print(f"  ok  {nome}")
    if falhou is None:
        for nome, th in m.tensores():
            if not torch.equal(th.cpu().to(I64), novos[nome]):
                falhou = (f"contador {nome} depois do update", int((th.cpu().to(I64) != novos[nome]).sum()), th.numel(), [], 0, 0)
                break
    if falhou:
        nome, n, tot, pos, vk, vr = falhou
        log.falha(f"PRIMEIRA etapa diferente da especificação: '{nome}': {n} de {tot} valores; posição {pos}: kernel = {vk}, "
                  f"especificação = {vr}. As etapas anteriores bateram bit a bit.")
    log.info(f"conferência do modelo completo: as {len(ordem)} etapas e os contadores batem bit a bit com a especificação")


def nats(so, alvo):
    """loss em nats por byte da rede de médias (float só para o relatório)."""
    z = so.double() / 1024
    return float((torch.logsumexp(z, 1) - z[torch.arange(z.shape[0], device=DEV), alvo.reshape(-1)]).mean())


def treinar(passos, L, D, F, Bt, shift, semente):
    import treino as dados
    dados.BT = Bt
    treino_txt, val_txt = dados.carregar()
    gv = torch.Generator().manual_seed(123)
    dados.BT = 8
    lotes_val = [dados.lote(val_txt, gv) for _ in range(8)]
    dados.BT = Bt
    g = torch.Generator().manual_seed(semente + 1)
    nome = f"modelo_bit_L{L}D{D}F{F}_bt{Bt}_sh{shift}_T{NORM_T}_tr{TRES}_r{RES}_v{VAR}_e{EST}_p{POSTO}_j{PROJ}_vr{VRMS}_sat{SAT}_s{semente}"
    log = Log(nome)
    m = ModeloBit(L, D, F, shift, semente, log)
    npesos = sum(t.numel() for _, t in m.tensores())
    log.info(f"{nome}: {npesos / 1e6:.2f} M pesos binários ({npesos / 8 / 1e6:.2f} MB empacotados) | {passos} passos de {Bt}x{T_CTX} tokens")
    # régua trivial: entropia dos bytes do treino (unigrama)
    cont = torch.bincount(treino_txt, minlength=V).double()
    p = cont / cont.sum()
    unigrama = float(-(p[p > 0] * p[p > 0].log()).sum())
    log.info(f"régua: unigrama de bytes = {unigrama:.3f} nats/byte (chutar pela frequência, sem contexto)")
    csv = open(os.path.join(AQUI, "logs", nome + ".csv"), "w")
    csv.write("passo,loss_treino_medias,loss_val_medias,acuracia_val_binaria,loss_val_binaria_calibrada,tokens_por_s\n")
    acum = n = 0
    t_tr, tok_cont = 0.0, 0
    for passo in range(passos + 1):
        log.passo = passo
        if passo % 100 == 0:
            log.vigiar = False
            lv = sum(nats(m.passo(*b_, treinar=False)[0], b_[1]) for b_ in lotes_val) / len(lotes_val)
            acc_m = sum(float((m.passo(*b_, treinar=False)[0].argmax(1) == b_[1].reshape(-1)).float().mean()) for b_ in lotes_val) / len(lotes_val)
            acc = sum(float((m.logits_binarios(b_[0]).argmax(1) == b_[1].reshape(-1)).float().mean()) for b_ in lotes_val) / len(lotes_val)
            melhor = min(sum(nats(m.logits_binarios(b_[0]).to(I64) << (10 - sh), b_[1]) if sh <= 10 else 1e9 for b_ in lotes_val) / len(lotes_val)
                         for sh in range(0, 11))
            tps = tok_cont / max(t_tr, 1e-9)
            lt = acum / max(n, 1)
            log.info(f"passo {passo:5d} | loss treino {lt:.3f} | loss val (rede de médias) {lv:.3f} acurácia {acc_m:.1%} | rede 100% binária: acurácia {acc:.1%}, "
                     f"loss calibrada {melhor:.3f} | {tps:,.0f} tokens/s", loss_val=lv, acc_bin=acc, loss_bin=melhor)
            csv.write(f"{passo},{lt:.4f},{lv:.4f},{acc:.4f},{melhor:.4f},{tps:.0f}\n"); csv.flush()
            acum = n = 0; t_tr, tok_cont = 0.0, 0
        if passo == passos:
            break
        log.vigiar = passo < 3 or passo % 250 == 0                  # sentinelas nos primeiros passos e a cada 500
        tok, alvo = dados.lote(treino_txt, g)
        torch.cuda.synchronize(); t0 = time.time()
        so, _ = m.passo(tok.to(DEV), alvo.to(DEV))
        acum += nats(so, alvo.to(DEV)); n += 1
        torch.cuda.synchronize(); t_tr += time.time() - t0; tok_cont += tok.numel()
    csv.close()


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "conferir":
        conferir()
    else:
        padrao = [4, 256, 1024, 8, 4, 1]                         # L D F B passo_contador semente
        args = [int(v) for v in a[3:9]] + padrao[len(a[3:9]):]
        treinar(int(a[2]), *args)
