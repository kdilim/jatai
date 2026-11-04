# INT8 com expoente por bloco de 32 valores

Experimento 4 da série "treino em inteiros puros" do projeto Jataí.
Data: 2026-11-02. Mesma configuração e mesmos pesos aleatórios do experimento 3
(D=256, 64 cabeças × head_dim 4, FF=1024, 12 camadas, T=64, lote 2, ALiBi + ReLU²,
semente 0). **Mede fidelidade numérica do forward, não capacidade de aprender.**

## Representação

Valor real = mantissa × 2^E. A mantissa é INT8; cada bloco de 32 valores consecutivos
da última dimensão tem seu próprio expoente inteiro E (a mesma ideia do MXINT8 da
especificação OCP Microscaling). O expoente do bloco é o menor que faz o maior valor
do bloco caber no limite da mantissa (±127 por padrão).

Tudo que é guardado é INT8 mais um expoente por bloco. Valores maiores existem só no
registrador, dentro de uma operação.

## Como cada operação funciona

- **Produto de matriz:** produto escalar exato de 32 termos por bloco de entrada
  (parcial). Os parciais de blocos com expoentes diferentes são alinhados ao maior
  expoente: `(parcial << 6) >> (Emax − E_b)`, com 6 bits de guarda, e somados. O
  resultado vira INT8 por bloco.
- **RMSNorm:** soma dos quadrados por bloco, alinhada com 8 bits de guarda; `r = isqrt`.
  O recíproco é calculado **uma vez por linha** como mantissa de ~10 bits mais expoente
  (`inv = 2^k // r`, `k = bits(r) + 10`). Cada elemento vira `m × inv` (≤ 19 bits) e o
  expoente absorve o resto. Na primeira versão a RMSNorm usava 12 bits fracionários e
  chegava a **36 bits**, o que estouraria o int32 na GPU (a simulação em int64 escondeu
  isso). Corrigido.
- **Scores:** `q·k` exato em 4 termos; o ÷√4 é um −1 no expoente; ALiBi como shift exato.
- **Softmax:** exp por shift + polinômio inteiro, em Q12; um recíproco por linha.
- **P@V:** V alinhado ao maior expoente da cabeça com 6 bits de guarda.
- **Residual:** alinhamento ao menor expoente do par e soma.
- **ReLU²:** `max(m, 0)²` exato; o expoente dobra.

## Resultados

### Deriva em 12 camadas (inteiro vs float64, mesmos pesos)

| representação | camada 1 | 3 | 6 | 9 | 12 | valores saturados |
|---|---|---|---|---|---|---|
| expoente por tensor (exp. 3) | 6,5% | 11,7% | 15,7% | 19,3% | 21,5% | – |
| bloco 64, mantissa ±127 (1ª versão) | 2,4% | 4,5% | 7,0% | 8,8% | 10,2% | – |
| **bloco 32, mantissa ±127** | **2,1%** | **4,1%** | **6,4%** | **8,1%** | **9,4%** | **0 de 7.077.888** |
| bloco 16, mantissa ±127 (1ª versão) | 1,8% | 3,5% | 5,5% | 7,1% | 8,3% | – |
| bloco 32, mantissa ±63 | 4,2% | 8,2% | 12,8% | 16,2% | 18,7% | 0 |
| bloco 32, mantissa ±31 | 8,2% | 16,1% | 25,2% | 32,2% | 37,4% | 0 |
| bloco 32, mantissa ±20 | 12,0% | 23,2% | 37,5% | 48,2% | 56,4% | 0 |
| *referência BF16* | 0,6% | 0,9% | 1,1% | 1,3% | 1,5% | – |

As linhas "1ª versão" usam a RMSNorm antiga (36 bits); a diferença para a versão
corrigida no bloco 32 é de 9,29% para 9,44%.

### Maior número de bits no registrador (bloco 32, ±127, 12 camadas)

| valor intermediário | bits |
|---|---|
| parcial de 32 termos (todas as projeções) | 17 |
| parcial com 6 bits de guarda | 23 |
| acumulador alinhado (todas as projeções) | 24 |
| RMSNorm: soma dos quadrados alinhada | 29 |
| RMSNorm: m × inv | 19 |
| scores q·k | 17 |
| softmax: exp × (2^28 / soma) | 29 |
| P@V: acumulador | 22 |
| ReLU²: x·x | 15 |
| residual: soma alinhada | 12 |

Tudo cabe em int32 com pelo menos 2 bits de folga.

## Conclusões

1. **O bloco de 32 corta a deriva de 21,5% para 9,4%,** sem sair do INT8. Diminuir o
   bloco para 16 ajuda pouco.
2. **Nenhum valor saturou** em 7 milhões de quantizações, e nenhum intermediário passa
   de 29 bits. Mantissas maiores que ±20 não causam estouro: elas só significam mais
   bits de precisão, porque a escala do valor está no expoente.
3. **Limitar a mantissa piora o erro na proporção exata dos bits perdidos.** Cada
   metade da faixa (±127 → ±63 → ±31) dobra a deriva: 9,4% → 18,7% → 37,4%. Em ±20,
   56,4%. O erro restante é de **arredondamento** (precisão), não de saturação.
4. O raio ±20 do `SKILL.md` era necessário naquele desenho, em que não havia expoente
   e o próprio inteiro era o valor. Com expoente por bloco, o papel de "manter os
   valores sob controle" passa a ser do expoente, em todo tensor e não só depois da
   normalização.
5. **Próxima alavanca: menos pontos de arredondamento.** Cada camada arredonda para
   INT8 ~12 vezes (RMSNorm, Q, K, V, P, P@V, O, residual, RMSNorm, up, ReLU², down,
   residual). Num kernel fundido, os valores intermediários ficam no registrador e só
   arredonda o que precisa ser guardado. Pelo experimento de VRAM isso também é o que
   deixa a GPU mais rápida.

## Teste do kernel fundido (`fundido.py`)

Q, K e as probabilidades do softmax ficam com 12 bits no registrador em vez de virar
INT8. O que vai para a memória ou entra num `__dp4a` continua INT8.

| versão | camada 12 | maior registrador |
|---|---|---|
| separado (tudo INT8) | 9,44% | 29 bits |
| fundido (Q, K, P com 12 bits) | 9,29% | 29 bits |

Ganho desprezível: o erro **não** vem dos arredondamentos internos da atenção. Ele vem
dos operandos INT8 das multiplicações grandes (saída da RMSNorm, saída da atenção,
ReLU²) e do residual guardado, que precisam ser INT8 de qualquer jeito.

## Conclusão final

A representação INT8 com expoente por bloco de 32 está **pronta para o teste de treino**:
nenhum estouro, nenhuma saturação, tudo cabe em int32 com folga, e os gradientes
inteiros concordam com os de float em cosseno 0,993–0,9995 (experimento 3). A deriva
de 9,4% contra float64 numa rede de pesos **aleatórios** não é o critério certo para
treino: o modelo é treinado dentro da aritmética INT8 e ajusta os pesos a ela; não existe
uma rede em float para ele imitar. O que decide é a curva de loss no treino real
contra uma referência em float do mesmo tamanho.

## Auditoria

`auditoria_camada1.md` segue o token 5 e a cabeça 0 (dimensões 0–3) por todos os
passos da camada 1: fórmula, inteiros, valor real e referência em float64. Cada conta
foi refeita em Python escalar e conferida com `assert` contra o cálculo vetorizado.

## Como reproduzir

```bash
python bloco32.py   # tabela de deriva, bits, e gera auditoria_camada1.md e resultado_bloco32.txt
```

Depende de `../experimento_analise_int8/analise_passo_a_passo.py` (pesos, referência
float e primitivas). Python 3.12.3, NumPy 2.5.4.
