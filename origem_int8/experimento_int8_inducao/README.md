# Treino 100% INT8 na tarefa de indução

Experimento 1 da série "treino em inteiros puros" do projeto Jataí.
Data: 2026-11-02. Status: **resultado preliminar, uma semente, simulação em CPU.**

## Pergunta

Um transformer consegue **aprender** (não só rodar sem saturar) com aritmética
100% inteira, pesos e ativações guardados em INT8, sem nenhuma operação em
ponto flutuante no forward, no backward ou no update dos pesos?

## Tarefa

Indução: um bloco de 6 tokens distintos aparece duas vezes numa sequência de 16,
começando numa posição aleatória entre 0 e 4. O resto é preenchido com tokens
aleatórios que não estão no bloco. Vocabulário de 16, então o acaso é 6,25%.
O modelo é avaliado nas 5 posições do segundo bloco em que o próximo token é
previsível. Como o deslocamento varia, decorar posições não resolve: é preciso
achar a ocorrência anterior do token atual e copiar o seguinte.

Avaliação: 512 sequências fixas (semente 999), acurácia de argmax.

## Modelo

2 camadas só de atenção, 1 cabeça, D=32, RMSNorm antes de cada bloco e antes da
saída, embeddings de token e de posição aprendidos, máscara causal exata.
Treino: lote de 32 sequências, 4000 passos, semente 1.

## Variantes

| Modo | Descrição |
|---|---|
| `ref_adam`, `ref_sgd` | Referência em float32 (PyTorch), mesma arquitetura. Só prova que a tarefa é aprendível. |
| `A` | O método do `SKILL.md` (`pure-integer-engine`) ao pé da letra: `scale_to_safe` para ±20 após cada projeção, RMSNorm para [−4, 4], LUT de exp de 9 pontos, calibração dinâmica dos scores para ±2, gradientes reescalados para ±20 (±100 nos pesos), update `W -= g >> lr_shift`. Única exceção: máscara causal com exp = 0 em vez de −4 (ver Notas). |
| `A_sim` | Ablação: `A` com shift simétrico no update, `sinal(g) × (abs(g) >> k)`. Todo o resto igual. |
| `B` | INT8 com expoente rastreado ("block floating point"): cada tensor guardado é uma mantissa INT8 mais um expoente inteiro de potência de 2 por tensor. Acumula em inteiro largo e só reduz para INT8 ao guardar. O backward usa as escalas verdadeiras. RMSNorm com saída em ponto fixo (2⁻⁵). Softmax com subtração do máximo e LUT de exp com passo de 1/8 nat (160 entradas, geradas por recorrência inteira). Update `W -= g · 2^-k` com **arredondamento estocástico** (números aleatórios inteiros). |
| `B_det` | Ablação: `B` com arredondamento determinístico (ao mais próximo) no update. |
| `B_calc` | `B_det` **sem LUT**: exp calculada (`exp(−d) = 2^(−d·log₂e)`, parte inteira do expoente vira shift, parte fracionária por polinômio inteiro de grau 3 em Q15, entrada com resolução de 1/256 nat) e divisões por elemento trocadas por uma divisão por linha mais multiplicação e shift. |

Nos modos `A*` e `B*` todo o modelo é `int64` em NumPy, simulando acumuladores
inteiros. Não há `float` nem divisão real no código dos modelos: só `+ - * // >> <<`,
`clip` e consultas a tabelas. O script confere o `dtype` de pesos e logits a cada avaliação.

## Resultados

Acurácia na avaliação (acaso = 6,25%). `max|peso|` = maior valor absoluto entre
os pesos INT8 no passo 4000 (limite 127/128).

| Modo | lr | passo 500 | passo 4000 | max\|peso\| |
|---|---|---|---|---|
| ref_adam | 0,003 | 100,0% | 100,0% | – |
| ref_sgd | 2⁻⁶ / 2⁻⁷ / 2⁻⁸ | 100,0% | 100,0% | – |
| **A** | shift 3 / 5 / 7 | 5,6% | 6,3% / 5,6% / 5,6% | 128 / 127 / 127 |
| A_sim | shift 3 / 5 / 7 | 5,8–7,7% | 6,6% / 6,6% / 7,7% | 128 / 128 / 8 |
| **B** | k = 6 | 100,0% | **6,5%** (colapsou) | 127 |
| **B** | k = 7 / 8 | 100,0% | 100,0% | 127 |
| **B_det** | k = 6 / 7 / 8 | 100,0% / 100,0% / 99,9% | 100,0% | 84 / 82 / 76 |
| **B_calc** | k = 7 / 8 | 100,0% / 99,9% | 100,0% | 77 / 74 |

### Exp calculada (`B_calc`) vs LUT (`B`)

Erro relativo máximo da exp inteira contra `exp` real, por faixa de valor:

| faixa | exp calculada | LUT de 160 pontos |
|---|---|---|
| exp > 0,1 | 0,035% | — |
| exp > 0,01 | 0,29% | — |
| exp > 0,001 | 2,9% | 1,1% nos pontos da tabela, mas a entrada anda em degraus de 1/8 nat (saltos de 11,8% entre vizinhos) |

O erro da versão calculada cresce perto de zero por causa da escala fixa de saída
(Q15, menor fração ≈ 3·10⁻⁵), não do polinômio. Melhoria possível para contexto
longo: devolver a exp como mantissa + expoente (o shift já calculado), no mesmo
esquema de expoente rastreado do modelo B.

Tempo: ~100–125 s por rodada inteira (CPU, 2 núcleos), ~51 s por rodada float (GPU).

## O que os resultados mostram

1. **Treinar em inteiros puros é possível.** `B` e `B_det` chegam a 100% com
   aritmética 100% inteira e tudo guardado em INT8, tão rápido quanto a referência
   em float (já no passo 500).
2. **O método do `SKILL.md`, como está escrito, não aprende** em nenhuma taxa testada.
3. **Causa encontrada no `A`:** em inteiro, `-1 >> k` vale −1, não 0 (o shift
   arredonda para −∞). Todo gradiente negativo vira update −1 e o peso sobe 1, e
   gradiente positivo pequeno vira 0. Medido: depois de 300 passos com `lr_shift = 7`,
   97,7% dos pesos estavam em +127 (média +126,2).
4. **Corrigir só isso não basta** (`A_sim`). Com shift 3 e 5 os pesos ainda saturam
   em ±128. Com shift 7 nunca mudam, porque o gradiente é reescalado para no máximo
   ±100 e `100 >> 7 = 0`. Normalizar o gradiente de cada tensor para ±100 faz todo
   tensor dar o passo máximo a cada iteração, por menor que seja o gradiente real.
5. **O arredondamento estocástico não é o ingrediente decisivo.** `B_det` aprende
   igual e é mais estável: os pesos ficam em ~80 em vez de colarem em 127, e não
   há o colapso de `B` com k = 6. O ruído do arredondamento estocástico parece fazer
   os pesos caminharem até o limite.
6. **A LUT não é necessária.** `B_calc` troca a tabela por cálculo inteiro
   (shift + polinômio) e as divisões por elemento por recíproco por linha, e treina
   igual (100%), sem nenhum acesso a tabela em memória.
7. Portanto o que separa `A` de `B` está na **representação dos números**: escala
   verdadeira rastreada no backward, mais resolução na RMSNorm e no softmax, e um
   passo proporcional ao gradiente real. Qual desses itens é necessário ainda não
   foi isolado (ver Próximos passos).

## Limitações (ler antes de citar qualquer número)

- **Uma semente só.** Os resultados podem variar; uma publicação precisa de várias
  sementes com média e desvio.
- **A tarefa ficou fácil demais.** Todo método que funciona chega a 100% já no
  passo 500, então o teste separa "aprende / não aprende" mas não mede eficiência.
  Faltam uma tarefa mais difícil e avaliações mais frequentes.
- **Simulação em CPU com `int64`.** Não roda na GPU nem mede velocidade. Os
  acumuladores reais na GPU seriam `int32`; os limites de overflow foram estimados,
  não verificados (ex.: o maior acúmulo de gradiente de peso é ~127·127·512 ≈ 8,3M,
  dentro do `int32`).
- **`B` guarda um expoente inteiro por tensor** (metadado de poucos bits). Os
  valores são todos INT8, mas isso é uma diferença em relação ao `A`, que descarta a escala.
- **Modelo mínimo:** 2 camadas, sem MLP, D=32. Nada aqui garante que escala para
  o modelo aluno (64 camadas, vocabulário de 248 mil).
- O colapso de `B` com k = 6 foi visto num único ponto de avaliação (passo 4000) e
  não foi investigado.

## Notas sobre o método do `SKILL.md`

- **Máscara causal com −4 vaza o futuro.** Com a LUT de 9 pontos, exp(−4) = 1
  contra até 2072 no máximo: as posições mascaradas recebem peso pequeno, mas não
  zero. Na previsão do próximo token, isso deixa a resposta entrar na atenção. Neste
  experimento a máscara do `A` usa exp = 0, o que também é 100% inteiro.
- A divisão `//` de números negativos também arredonda para −∞ (em `scale_to_safe`,
  RMSNorm etc.), o que cria um viés sistemático pequeno em todo o forward.

## Próximos passos

1. Isolar a causa: partir de `B_det` e trocar uma peça por vez pela versão do `A`
   (gradiente normalizado ±100; LUT de 9 pontos; RMSNorm em [−4, 4]; descartar a escala).
2. Tarefa mais difícil e avaliação a cada 50 passos, para comparar velocidade de
   convergência com a referência em float.
3. 5 sementes por configuração.
4. Verificar overflow com acumuladores `int32` de verdade.
5. Portar `B_det` para kernels CUDA com `__dp4a` e medir tokens/s na GTX 1650.

## Como reproduzir

Requisitos: Python 3.12, NumPy 2.5.4; PyTorch 2.6.0 só para a referência.
Hardware usado: GTX 1650 (driver 580.178.04), só na referência float.

```bash
python teste_int8_inducao.py <modo> <lr> [passos]
```

Exemplos (os logs de todas as rodadas estão em `logs/`):

```bash
python teste_int8_inducao.py ref_adam 0.003 4000
python teste_int8_inducao.py A 3 4000
python teste_int8_inducao.py A_sim 5 4000
python teste_int8_inducao.py B 7 4000
python teste_int8_inducao.py B_det 7 4000
python teste_int8_inducao.py B_calc 7 4000
```

Os coeficientes do polinômio de `B_calc` (`-1295, 7556, -22645, 32765`, em Q15) foram
ajustados uma única vez por mínimos quadrados de 2⁻ˣ em [0, 1) e ficam fixos no
código, como os valores de uma LUT. Erro relativo do ajuste: 1,9·10⁻⁴.

`lr` é o `lr_shift` nos modos `A*`, o `k` de 2⁻ᵏ nos modos `B*` e em `ref_sgd`,
e o learning rate do Adam em `ref_adam`.
