"""Transformer treinado 100% em inteiros: o mesmo codigo roda com OpsCuda, OpsRef ou OpsConferido.

Arquitetura (igual a ../treino_int8_gpu): pre-norm, RMSNorm sem ganho, atencao causal com ALiBi
(inclinacoes 2^-s), head_dim 4, MLP ReLU^2, vocabulario de bytes. Q, K e V numa matriz so (3D x D).
Otimizador: SGD com momento 7/8, lr = 2^-k, error feedback no peso.
Inicializacao identica a de ../treino_int8_gpu (mesmo gerador e ordem), para comparar com o float.
"""
import math
import os

import torch

import ops as O


def inicial(g, linhas, fan_in, dev, emb=False):
    e = -5 if emb else -int(round(math.log2(32 * math.sqrt(fan_in))))
    m = torch.round(torch.randn(linhas, fan_in, generator=g) * 32).clamp(-127, 127).to(torch.int8)
    return m.to(dev).contiguous(), torch.full((linhas, fan_in // O.BLK), e, dtype=torch.int8, device=dev)


def zeros_como(t):
    return torch.zeros_like(t[0]), torch.full_like(t[1], O.E_ZERO)


R16 = os.environ.get("JATAI_RES16") == "1"     # E1: fluxo residual (forward e backward) com mantissa de 16 bits


class Modelo:
    def __init__(self, V, D, H, FF, L, semente, dev, ops, reg):
        self.V, self.D, self.H, self.FF, self.L, self.dev, self.ops, self.reg = V, D, H, FF, L, dev, ops, reg
        g = torch.Generator().manual_seed(semente)
        p = {"emb": inicial(g, V, D, dev, emb=True)}
        for l in range(L):
            q, k, v = inicial(g, D, D, dev), inicial(g, D, D, dev), inicial(g, D, D, dev)
            p[f"Wqkv{l}"] = (torch.cat([q[0], k[0], v[0]]).contiguous(), torch.cat([q[1], k[1], v[1]]).contiguous())
            p[f"Wo{l}"] = inicial(g, D, D, dev)
            p[f"Wup{l}"] = inicial(g, FF, D, dev)
            p[f"Wd{l}"] = inicial(g, D, FF, dev)
        p["out"] = inicial(g, V, D, dev)
        self.p = p
        self.mom = {n: zeros_como(t) for n, t in p.items()}
        self.res = {n: zeros_como(t) for n, t in p.items()}
        self.error_feedback = True
        self.estatisticas = False

    def _st(self, nome, t, zeros_normais=False):
        if self.estatisticas:
            self.reg.estatistica(nome, t, zeros_normais)

    def passo(self, tok, alvo, k_lr=None):
        """forward (+ backward e update se k_lr). Devolve a loss como tensor (so relatorio, em float)."""
        ops, reg, p = self.ops, self.reg, self.p
        Bt, T = tok.shape
        M = Bt * T
        tok, alvo = tok.reshape(-1).to(torch.int32).contiguous(), alvo.reshape(-1).to(torch.int32).contiguous()
        with reg.trecho("forward"):
            with reg.trecho("embedding"):
                x = ops.embed(p["emb"], tok)
                if R16:                                         # E1: a esteira residual anda em 16 bits
                    x16 = ops.para16(x)
            caches = []
            for l in range(self.L):
                with reg.trecho(f"camada {l}"):
                    with reg.trecho("atencao"):
                        n1, c1 = ops.rms(ops.para8(x16) if R16 else x)
                        qkv = ops.gemm(n1, p[f"Wqkv{l}"])
                        oc, P = ops.atencao(qkv, Bt, T, self.H)
                        om = ops.quant(oc)
                        if R16:
                            x16 = ops.soma16(x16, ops.gemm(om, p[f"Wo{l}"]))
                            x = ops.para8(x16) if self.estatisticas else None
                        else:
                            x = ops.gemm_soma(om, p[f"Wo{l}"], x)          # x + om.Wo^T numa GEMM so (medido 1,57x)
                        self._st(f"c{l}.qkv", qkv); self._st(f"c{l}.atencao_saida", om); self._st(f"c{l}.residuo1", x)
                    with reg.trecho("mlp"):
                        n2, c2 = ops.rms(ops.para8(x16) if R16 else x)
                        # fundidas na saida da GEMM. Com o tile de 128 linhas (1 bloco por SM) estas fusoes perdiam
                        # (0,82x / 0,92x); com o de 64 (2 blocos por SM) ganham (1,12x / 1,05x). Ver README.
                        h, a = ops.gemm_relu2(n2, p[f"Wup{l}"])
                        if R16:
                            x16 = ops.soma16(x16, ops.gemm(a, p[f"Wd{l}"]))
                            x = ops.para8(x16) if self.estatisticas else None
                        else:
                            x = ops.gemm_soma(a, p[f"Wd{l}"], x)
                        self._st(f"c{l}.relu2", a, zeros_normais=True); self._st(f"c{l}.residuo2", x)
                caches.append((n1, c1, qkv, P, om, n2, c2, h, a))
            with reg.trecho("saida"):
                nf, cf = ops.rms(ops.para8(x16) if R16 else x)
                logits = ops.gemm(nf, p["out"], cru=True)
                g, d_alvo, soma = ops.softmax_ce(logits, alvo)
        # relatorio (fica na GPU; quem chama converte com float() depois do passo, sem parar a fila)
        perda = (d_alvo.double() / 256 + torch.log(soma.double() / 4096)).mean()
        if k_lr is None:
            return perda
        T_ = ops.transpoe
        grads = {}
        with reg.trecho("backward"):
            with reg.trecho("saida"):
                grads["out"] = ops.gemm(T_(g), T_(nf), cru=True)
                gx = ops.gemm(g, T_(p["out"]))
                gx = ops.rms_back(gx, nf, cf)
                self._st("grad_saida", gx)
                if R16:                                         # E1: o gradiente que volta pela esteira tambem em 16 bits
                    gx16 = ops.para16(gx)
            for l in reversed(range(self.L)):
                n1, c1, qkv, P, om, n2, c2, h, a = caches[l]
                with reg.trecho(f"camada {l}"):
                    with reg.trecho("mlp"):
                        grads[f"Wd{l}"] = ops.gemm(T_(gx), T_(a), cru=True)
                        gh = ops.gemm_relu2_back(gx, T_(p[f"Wd{l}"]), h)   # fundida: 1,14x (era 0,87x com tile 128)
                        grads[f"Wup{l}"] = ops.gemm(T_(gh), T_(n2), cru=True)
                        gn = ops.gemm(gh, T_(p[f"Wup{l}"]))
                        if R16:
                            gx16 = ops.soma16(gx16, ops.rms_back(gn, n2, c2))
                            gx = ops.para8(gx16)
                        else:
                            gx = ops.soma(gx, ops.rms_back(gn, n2, c2))
                        self._st(f"c{l}.grad_mlp", gh, zeros_normais=True)
                    with reg.trecho("atencao"):
                        grads[f"Wo{l}"] = ops.gemm(T_(gx), T_(om), cru=True)
                        go = ops.gemm(gx, T_(p[f"Wo{l}"]))
                        gq = ops.quant(ops.atencao_back(go, qkv, P, Bt, T, self.H))
                        grads[f"Wqkv{l}"] = ops.gemm(T_(gq), T_(n1), cru=True)
                        gn = ops.gemm(gq, T_(p[f"Wqkv{l}"]))
                        if R16:
                            gx16 = ops.soma16(gx16, ops.rms_back(gn, n1, c1))
                            gx = ops.para8(gx16)
                        else:
                            gx = ops.soma(gx, ops.rms_back(gn, n1, c1))
                        self._st(f"c{l}.grad_qkv", gq); self._st(f"c{l}.grad_residuo", gx)
            with reg.trecho("embedding"):
                um = ops.um_quente(tok, self.V)                  # kernel proprio: token invalido vira sentinela, nao assert
                grads["emb"] = ops.gemm(T_(um), T_(gx), cru=True)
        with reg.trecho("otimizador"):
            for n, gr in grads.items():
                with reg.trecho(n):
                    self.mom[n] = ops.momento(gr, self.mom[n])
                    self.p[n], self.res[n] = ops.peso(self.p[n], self.mom[n], self.res[n], k_lr, self.error_feedback)
        return perda


class PassoGravado:
    """Um passo de treino gravado num CUDA Graph e repetido: os ~520 lancamentos viram uma chamada so.

    Gravar nao executa nada (os pesos nao mudam na gravacao). Tudo o que o passo aloca fica no pool do grafo, com os
    mesmos enderecos em toda repeticao; os pesos, o momento e o residuo sao atualizados no lugar pelos kernels, entao
    o grafo continua valido. A entrada vai para buffers fixos (copy_) antes de cada repeticao.
    Logs: a lista de lancamentos da gravacao (id -> operacao, trecho, entradas) fica guardada; os ids (kid) estao
    gravados nos argumentos dos kernels, entao um erro de sentinela numa repeticao aponta o mesmo trecho que apontaria
    no modo normal. As conferencias de formato rodam uma vez, na gravacao (as formas nao mudam).
    """

    def __init__(self, modelo, Bt, T, k_lr):
        import torch
        self.m, self.reg, self.k_lr = modelo, modelo.reg, k_lr
        lib = modelo.ops.lib
        self.tok = torch.zeros(Bt, T, dtype=torch.long, device=modelo.dev)
        self.alvo = torch.zeros(Bt, T, dtype=torch.long, device=modelo.dev)
        self.grafo = torch.cuda.CUDAGraph()
        est, vig = modelo.estatisticas, self.reg.vigiar
        modelo.estatisticas, self.reg.vigiar = False, False     # nada que sincronize a GPU durante a gravacao
        torch.cuda.synchronize()
        self.reg.lancamentos = []
        try:
            with torch.cuda.graph(self.grafo):
                lib.jt_stream(torch.cuda.current_stream().cuda_stream)
                self.perda = modelo.passo(self.tok, self.alvo, k_lr=k_lr)
        finally:
            lib.jt_stream(None)
            modelo.estatisticas, self.reg.vigiar = est, vig
        self.lancamentos = list(self.reg.lancamentos)
        self.reg.info(f"passo gravado num CUDA Graph: {len(self.lancamentos)} lancamentos", lancamentos=len(self.lancamentos))

    def __call__(self, tok, alvo):
        self.tok.copy_(tok)
        self.alvo.copy_(alvo)
        self.reg.lancamentos = self.lancamentos
        self.grafo.replay()
        return self.perda
