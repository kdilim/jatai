"""Registro (log) do treino inteiro: cada lancamento de kernel e anotado, e qualquer problema vira uma
mensagem que ja diz o que aconteceu, onde e com que valores.

Camadas de protecao:
  1. Antes de lancar: confere formato de cada tensor (dtype, forma, contiguo, na GPU, expoentes coerentes).
     Falha -> ErroJatai com o trecho e o nome do argumento.
  2. Depois de lancar: codigo de retorno do CUDA (configuracao invalida, memoria, etc.).
  3. Dentro do kernel (sentinelas): acumulador perto de estourar o int32, shift que perderia bits,
     expoente fora do INT8, saturacao, divisao por zero, token fora do vocabulario. O primeiro evento
     grava (codigo, id do lancamento, indice, valor). O id aponta para o trecho exato (ex.:
     "passo 12 / camada 3 / atencao_back"), e o indice vira (linha, coluna) na forma da saida.
     As entradas daquele kernel sao salvas em erro_*.pt para reproduzir.
     Modo normal: conferido 1x por passo. Modo vigiar: conferido depois de cada kernel (mais lento).
  4. Modo conferir (ver ops.OpsConferido): cada operacao roda no CUDA e na referencia em Python;
     a primeira diferenca de bit gera erro com posicao e valores dos dois lados.
  5. Estatisticas por trecho (zeros, faixa de expoentes) e avisos de anomalia; tempo por operacao.

Tudo vai para <pasta>/<nome>.jsonl (uma linha JSON por evento) e os erros/avisos tambem para a tela.
"""
import contextlib
import ctypes
import json
import os
import time

import torch

CODIGOS = {
    1: "acumulador passou de 2^30 (estouraria o registrador int32)",
    2: "expoente do bloco passou de 126 (estouro para cima do INT8; abaixo de -126 o bloco vira zero)",
    3: "mantissa passou de +-127 mesmo depois de subir o expoente",
    4: "divisao por zero (soma do softmax <= 0)",
    5: "shift a esquerda perderia bits (resultado >= 2^30)",
    6: "argumento invalido (token ou alvo fora do vocabulario)",
}


class ErroJatai(RuntimeError):
    pass


class Registro:
    def __init__(self, pasta, nome, lib=None, vigiar=False, perfil=False, eco=True):
        os.makedirs(pasta, exist_ok=True)
        self.pasta, self.nome, self.lib = pasta, nome, lib
        self.vigiar, self.perfil, self.eco = vigiar, perfil, eco
        self.arq = open(os.path.join(pasta, f"{nome}.jsonl"), "a")
        self.pilha, self.lancamentos, self.passo = [], [], None
        self.eventos, self.tempos = [], {}

    # ------------------------------------------------------------------ escrita
    def escreve(self, tipo, **kw):
        rec = {"hora": time.strftime("%H:%M:%S"), "tipo": tipo, "passo": self.passo, "trecho": self.onde(), **kw}
        self.arq.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self.arq.flush()
        return rec

    def info(self, msg, **kw):
        self.escreve("info", msg=msg, **kw)
        if self.eco:
            print(msg, flush=True)

    def aviso(self, msg, **kw):
        self.escreve("aviso", msg=msg, **kw)
        print(f"[AVISO] passo {self.passo} | {self.onde()} | {msg}", flush=True)

    def falha(self, msg, **kw):
        self.escreve("erro", msg=msg, **kw)
        raise ErroJatai(f"\n[ERRO] passo {self.passo} | {self.onde()}\n{msg}")

    def onde(self):
        return " / ".join(self.pilha) or "-"

    @contextlib.contextmanager
    def trecho(self, nome):
        self.pilha.append(nome)
        try:
            yield
        finally:
            self.pilha.pop()

    # ------------------------------------------------------------------ passos e lancamentos
    def novo_passo(self, passo):
        self.passo = passo
        self.lancamentos = []
        self.eventos = []
        if self.lib is not None:
            self.lib.jt_erro_limpa()

    def registra(self, op, entradas, forma_saida):
        kid = len(self.lancamentos)
        self.lancamentos.append({"op": op, "trecho": self.onde(), "entradas": entradas, "forma": forma_saida})
        if self.perfil:
            ini, fim = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ini.record()
            self.eventos.append((op, ini, fim))
        return kid

    def depois(self, kid, codigo_cuda):
        if self.perfil:
            self.eventos[-1][2].record()
        if codigo_cuda != 0:
            lanc = self.lancamentos[kid]
            self.falha(f"o CUDA recusou o lancamento de '{lanc['op']}' (cudaError {codigo_cuda}). "
                       f"Formas das entradas: {self._formas(lanc['entradas'])}")
        if self.vigiar:
            self.confere_device()

    def confere_device(self):
        """Le o registro de erro do device. Sincroniza a GPU."""
        buf = (ctypes.c_longlong * 6)()
        n = self.lib.jt_erro_le(buf)
        if n < 0:
            self.falha(f"falha do CUDA ao ler o registro de erro (cudaError {-n}); um kernel anterior "
                       f"provavelmente abortou. Ultimo lancamento: {self._resumo(len(self.lancamentos) - 1)}")
        if n == 0:
            return
        _, codigo, kid, idx, valor, aux = list(buf)
        lanc = self.lancamentos[kid] if 0 <= kid < len(self.lancamentos) else None
        if lanc is None:
            self.falha(f"{CODIGOS.get(codigo, codigo)} em kernel de id {kid} desconhecido (idx {idx}, valor {valor})")
        forma = lanc["forma"]
        pos = f"indice {idx}"
        if forma and len(forma) == 2:
            pos = f"linha {idx // forma[1]}, coluna {idx % forma[1]} (forma {tuple(forma)})"
        arq = os.path.join(self.pasta, f"erro_passo{self.passo}_{lanc['op']}.pt")
        try:
            torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in lanc["entradas"].items()}, arq)
        except Exception as e:  # o proprio salvamento nao pode esconder o erro original
            arq = f"(nao salvou: {e})"
        msg = (f"{CODIGOS.get(codigo, f'codigo {codigo}')}\n"
               f"  operacao : {lanc['op']}\n"
               f"  trecho   : {lanc['trecho']}\n"
               f"  posicao  : {pos}\n"
               f"  valor    : {valor}   (dado extra: {aux})\n"
               f"  eventos  : {n} no total neste passo (so o primeiro e detalhado)\n"
               f"  entradas : {self._formas(lanc['entradas'])}\n"
               f"  salvas em: {arq}")
        self.pilha = [lanc["trecho"]]   # o erro e do trecho daquele kernel, nao de onde a conferencia rodou
        self.falha(msg, codigo=codigo, op=lanc["op"], idx=idx, valor=valor, aux=aux)

    def fim_passo(self):
        if self.lib is not None:
            self.confere_device()
        if self.perfil:
            torch.cuda.synchronize()
            for op, ini, fim in self.eventos:
                self.tempos[op] = self.tempos.get(op, 0.0) + ini.elapsed_time(fim)

    def relatorio_tempos(self, passos):
        if not self.tempos:
            return
        tot = sum(self.tempos.values())
        linhas = [f"  {op:16s} {ms / passos:8.2f} ms/passo  {100 * ms / tot:5.1f}%"
                  for op, ms in sorted(self.tempos.items(), key=lambda x: -x[1])]
        self.info("tempo de GPU por operacao:\n" + "\n".join(linhas) + f"\n  {'total':16s} {tot / passos:8.2f} ms/passo",
                  tempos={k: v / passos for k, v in self.tempos.items()})
        self.tempos = {}

    # ------------------------------------------------------------------ conferencias de formato
    def confere_bloco(self, nome, t):
        m, E = t
        if not (torch.is_tensor(m) and torch.is_tensor(E)):
            self.falha(f"'{nome}': esperava (mantissa, expoente) como tensores, veio {type(m)}, {type(E)}")
        prob = []
        if m.dtype != torch.int8: prob.append(f"mantissa dtype {m.dtype} (esperado int8)")
        if E.dtype != torch.int8: prob.append(f"expoente dtype {E.dtype} (esperado int8)")
        if not m.is_cuda or not E.is_cuda: prob.append("tensor fora da GPU")
        if not m.is_contiguous() or not E.is_contiguous(): prob.append("tensor nao contiguo")
        if m.dim() != 2: prob.append(f"mantissa com {m.dim()} dimensoes (esperado 2)")
        elif m.shape[1] % 128: prob.append(f"colunas {m.shape[1]} nao multiplo de 128")
        elif tuple(E.shape) != (m.shape[0], m.shape[1] // 128):
            prob.append(f"expoente com forma {tuple(E.shape)}, esperado {(m.shape[0], m.shape[1] // 128)}")
        if prob:
            self.falha(f"argumento '{nome}' com formato invalido: " + "; ".join(prob))

    def confere_cru(self, nome, t):
        acc, e = t
        prob = []
        if acc.dtype != torch.int32: prob.append(f"acumulador dtype {acc.dtype} (esperado int32)")
        if e.dtype != torch.int16: prob.append(f"expoente dtype {e.dtype} (esperado int16)")
        if acc.shape != e.shape: prob.append(f"formas diferentes {tuple(acc.shape)} x {tuple(e.shape)}")
        if not acc.is_cuda or not acc.is_contiguous() or not e.is_contiguous(): prob.append("fora da GPU ou nao contiguo")
        if prob:
            self.falha(f"argumento '{nome}' (cru) com formato invalido: " + "; ".join(prob))

    def exige(self, cond, msg):
        if not cond:
            self.falha(msg)

    # ------------------------------------------------------------------ estatisticas
    def estatistica(self, nome, t, zeros_normais=False):
        """Resumo de um tensor em bloco (so relatorio; usa contas do PyTorch fora do metodo)."""
        m, E = t
        nz = E != -127
        rec = {
            "nome": nome,
            "zeros": round(float((m == 0).float().mean()), 4),
            "blocos_zerados": round(float((~nz).float().mean()), 4),
            "exp_min": int(E[nz].min()) if nz.any() else None,
            "exp_max": int(E[nz].max()) if nz.any() else None,
            "maior_mantissa_media": round(float(m.abs().reshape(m.shape[0], -1, 128).amax(-1).float().mean()), 1),
        }
        self.escreve("estatistica", **rec)
        if not nz.any():
            self.aviso(f"'{nome}' saiu inteiro zerado")
        elif not zeros_normais and rec["zeros"] > 0.5:
            self.aviso(f"'{nome}' tem {rec['zeros']:.0%} de zeros: valores pequenos sumindo no arredondamento?")
        if rec["exp_min"] is not None and rec["exp_min"] < -110:
            self.aviso(f"'{nome}' tem expoente {rec['exp_min']}, perto do limite -126 do INT8")
        return rec

    @staticmethod
    def _formas(ent):
        return {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in ent.items()}

    def _resumo(self, kid):
        if kid < 0 or kid >= len(self.lancamentos):
            return "-"
        l = self.lancamentos[kid]
        return f"{l['op']} em {l['trecho']}"
