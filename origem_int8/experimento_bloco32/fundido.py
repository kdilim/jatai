"""Compara a deriva em 12 camadas: cada operacao arredondando para INT8 vs kernel fundido
(Q, K e P com 12 bits no registrador; o que vai para a memoria ou para o dp4a continua INT8)."""
import bloco32 as b

for nome, fund in (("separado (tudo INT8)", False), ("fundido (Q,K,P 12 bits)", True)):
    b.BITS.clear(); b.SAT.update(saturados=0, total=0)
    b.CFG.update(LIM=127, FUNDIDO=fund)
    er = b.deriva(32)
    print(f"{nome:26s} " + " ".join(f"c{c} {er[c - 1] * 100:5.2f}%" for c in (1, 3, 6, 9, 12))
          + f" | maior registrador: {max(v for k, v in b.BITS.items() if 'cortadas' not in k)} bits"
          + f" (scores {b.BITS['scores q.k']}, P@V {b.BITS['P@V: acumulador']}) | saturados {b.SAT['saturados']}", flush=True)
