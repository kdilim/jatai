"""Resumo do treino longo: le os dois CSVs de noite.py e escreve logs/noite_resumo.md (tabela INT8 x float32)."""
import csv
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
L, FF = sys.argv[1], sys.argv[2]


def le(modo):
    arq = os.path.join(AQUI, "logs", f"noite_{modo}_L{L}_ff{FF}.csv")
    if not os.path.exists(arq):
        return {}
    return {int(r["passo"]): r for r in csv.DictReader(open(arq))}


i8, fl = le("int8"), le("float")
passos = sorted(set(i8) | set(fl))
marcos = [p for p in passos if p % 5000 == 0 or p == passos[-1]] if passos else []
linhas = [f"# Treino longo: INT8 x float32 (L = {L}, FF = {FF})", "",
          "| passo | INT8 treino | INT8 validação | float32 treino | float32 validação | diferença (validação) |",
          "|---|---|---|---|---|---|"]
for p in marcos:
    a, b = i8.get(p), fl.get(p)
    f = lambda r, k: f"{float(r[k]):.4f}" if r else "—"
    d = f"{float(a['loss_val']) - float(b['loss_val']):+.4f}" if a and b else "—"
    linhas.append(f"| {p} | {f(a, 'loss_treino')} | {f(a, 'loss_val')} | {f(b, 'loss_treino')} | {f(b, 'loss_val')} | {d} |")
for nome, d in (("INT8", i8), ("float32", fl)):
    if d:
        tps = [float(r["tokens_por_s"]) for r in d.values() if float(r["tokens_por_s"]) > 0]
        ult = d[max(d)]
        linhas.append(f"\n{nome}: {max(d)} passos em {float(ult['horas']):.2f} h; média {sum(tps) / len(tps):,.0f} tokens/s; "
                      f"melhor validação {min(float(r['loss_val']) for r in d.values()):.4f}")
open(os.path.join(AQUI, "logs", "noite_resumo.md"), "w").write("\n".join(linhas) + "\n")
print("\n".join(linhas))
