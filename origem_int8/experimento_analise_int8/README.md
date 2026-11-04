# Análise passo a passo da aritmética INT8

Experimento 3 da série "treino em inteiros puros" do projeto Jataí.
Data: 2026-11-02. Configuração alvo: D=256, 64 cabeças × head_dim 4, FF=1024,
12 camadas, T=64, lote 2. Pesos aleatórios (escala de inicialização usual), semente 0.
**Mede fidelidade numérica, não capacidade de aprender.**

## Perguntas

1. Em quais passos o valor exato passa de 8 bits, por quanto, e ele fica só no
   registrador ou vai para a memória?
2. Dá para não usar nenhum valor maior que 8 bits, nem no acumulador?
3. Para INT8: ALiBi ou RoPE? ReLU², ReGLU ou SiLU?

## Método

Uma camada transformer pre-norm (atenção com 64 cabeças + MLP), implementada só com
inteiros no esquema do modelo `B_calc` do experimento 1: mantissa INT8 + um expoente
inteiro de potência de 2 por tensor, exp calculada por shift + polinômio, recíproco
por linha em vez de divisão por elemento. ALiBi com inclinações em potência de 2
(`2⁻ˢ`, s = 0..7, 8 cabeças por inclinação), aplicado como shift exato. RoPE com
tabela cos/sin em Q14 (2 pares de frequência: 1 e 0,01 rad/token).

- **Parte 1:** bits do valor exato em cada operação e erro local (saída inteira vs
  float64 aplicado às mesmas entradas).
- **Parte 2:** erro acumulado em 12 camadas, inteiro vs float64 com os mesmos pesos.
  Régua: a mesma rede em FP16 e BF16 vs float64.
- **Parte 3:** backward local de um bloco de atenção e de um de MLP: similaridade de
  cosseno entre o gradiente inteiro e o gradiente do PyTorch em float64.
- **Parte 4:** produto escalar de K termos INT8 sem nenhum valor intermediário maior
  que 8 bits, comparado com acumulador int32.

## Resultados

### Parte 1 — onde o valor passa de 8 bits (camada 1, ALiBi + ReGLU)

| operação | bits do valor exato | onde vive | erro local |
|---|---|---|---|
| RMSNorm: soma dos quadrados | 20 | registrador | |
| RMSNorm: x·32·(1/r) | 24 | registrador | |
| RMSNorm: saída | 8 | memória INT8 | 1,8% |
| Q/K/V: acumulador (256 termos) | 18 | registrador | |
| Q/K/V: saída | 8 | memória INT8 | 1,8% |
| scores q·k (head_dim 4) | 14 | registrador | |
| score + viés ALiBi | 16 | registrador | |
| softmax: soma das exp | 22 | registrador | |
| softmax: exp·(1/soma) | **31** | registrador | |
| softmax: probabilidades | 8 (sem sinal) | memória UINT8 | 1,2% |
| P@V: acumulador | 15 | registrador | |
| O: acumulador | 17 | registrador | |
| residual: soma alinhada | 9 | registrador | |
| MLP up/gate: acumulador | 18 | registrador | |
| ReGLU: relu(gate)·up | 13 | registrador | |
| ReGLU: saída | 8 | memória INT8 | 3,4% |
| MLP down: acumulador (1024 termos) | 16 | registrador | |

Outras variantes: RoPE x·cos − y·sin → 21–22 bits, erro 1,7% (uma requantização a
mais em Q e em K); ReLU² x·x → 14 bits, erro 3,75%; SiLU: sigmoide com divisão por
elemento → **32 bits**, x·sigmoide → 23 bits, erro 3,2%.

**Tudo o que vai para a memória é INT8.** Os valores maiores existem só no registrador.

### Parte 2 — erro acumulado em 12 camadas (vs float64)

| combinação | camada 1 | 3 | 6 | 9 | 12 |
|---|---|---|---|---|---|
| ALiBi + ReLU² | 6,5% | 11,7% | 15,7% | 19,2% | 21,5% |
| ALiBi + ReGLU | 4,9% | 9,6% | 16,0% | 20,2% | 24,6% |
| ALiBi + SiLU | 4,0% | 8,8% | 13,8% | 17,1% | 19,3% |
| RoPE + ReLU² | 6,8% | 12,0% | 16,2% | 19,1% | 21,6% |
| RoPE + ReGLU | 5,1% | 10,0% | 16,9% | 21,4% | 26,3% |
| RoPE + SiLU | 4,2% | 8,4% | 13,4% | 16,4% | 18,1% |
| *régua: FP16 (ALiBi + ReLU²)* | 0,1% | 0,1% | 0,1% | 0,2% | 0,2% |
| *régua: BF16 (ALiBi + ReLU²)* | 0,6% | 0,9% | 1,1% | 1,3% | 1,5% |

### Parte 3 — backward local (cosseno inteiro vs float64; 1,0 = idêntico)

| bloco | dX | dW (entrada) | dW (saída) | zeros criados em dX |
|---|---|---|---|---|
| atenção ALiBi | 0,9971 | 0,9965 (Q) | 0,9995 (O) | 3,3% |
| atenção RoPE | 0,9950 | 0,9953 (Q) | 0,9990 (O) | 5,8% |
| MLP ReLU² | 0,9993 | 0,9993 (up) | 0,9990 (down) | 1,5% |
| MLP ReGLU | 0,9926 | 0,9993 (up) | 0,9990 (down) | 2,1% |
| MLP SiLU | 0,9995 | 0,9996 (up) | 0,9993 (down) | 1,5% |

### Parte 4 — produto escalar sem nenhum valor acima de 8 bits

| esquema | K=256: correlação com o exato | K=1024 |
|---|---|---|
| acumulador int32 (atual), só a saída vira INT8 | **1,0000** (erro 0,9%) | **1,0000** (erro 0,9%) |
| acumulador INT8 saturando (produto >>7) | 0,903 | 0,611 |
| árvore INT8, arredondamento simétrico (produto >>7) | 0,881 | 0,817 |
| tudo ≤ 8 bits, simétrico (operandos >>4) | 0,793 | 0,732 |

Arredondamento "metade para cima" na árvore dá correlação ainda menor (0,71 / 0,52)
por acumular viés a cada nível. A versão simétrica está em `resultado_referencias.txt`.

## Conclusões

1. **O acumulador largo não pode sair, e não custa nada mantê-lo.** Somar só em INT8
   leva a correlação com o resultado certo para 0,61–0,88, o equivalente a um ruído
   enorme. Na GTX 1650 os registradores e o multiplicador são de 32 bits e o `__dp4a`
   devolve int32: não existe registrador de 8 bits, então um acumulador INT8 não
   economiza nada e ainda impede o `__dp4a`. A regra que funciona é **nada maior que
   INT8 é guardado na memória**, e ela já é cumprida.
2. **Margem do int32:** os produtos escalares precisam de só 15–18 bits. Os pontos
   perigosos são o softmax (31 bits) e a sigmoide do SiLU (32 bits), no limite do
   int32. Precisam ser redesenhados com menos bits fracionários antes de ir para a GPU.
3. **Posição: ALiBi.** Erro de forward equivalente ao RoPE, backward um pouco melhor
   (dX 0,9971 × 0,9950; zeros criados 3,3% × 5,8%), sem tabela de cos/sin e sem as
   duas requantizações extras de Q e K por camada. Este teste com pesos aleatórios
   **não** mede a resolução de posição do RoPE com head_dim 4; esse argumento
   continua sendo teórico.
4. **MLP: ReLU².** É quase tão fiel quanto o SiLU (deriva 21,5% × 19,3%; backward
   0,9993 × 0,9995) com custo mínimo (uma comparação e uma multiplicação, no máximo
   14 bits). O SiLU é o mais fiel, mas precisa de exp e de divisão por elemento, no
   limite do int32. O ReGLU foi o pior aqui (deriva 24,6%, dX 0,9926) e ainda exige
   uma multiplicação de matriz a mais. Fidelidade numérica não é qualidade de modelo:
   ReGLU × ReLU² ainda precisa de um teste de treino.
5. **Achado principal: o limite de profundidade é a granularidade do expoente, não o
   acumulador.** Com um expoente por tensor, o erro chega a 19–26% na camada 12,
   contra 1,5% do BF16 e 0,2% do FP16. Cada requantização para INT8 erra ~1,8% porque
   valores pequenos dividem a escala com o maior valor do tensor inteiro.

## Próximos passos

1. Expoente por bloco de 32 valores em vez de por tensor (a mesma ideia do formato
   MXINT8 da especificação OCP Microscaling) e medir de novo a deriva em 12 camadas.
2. Escala não restrita a potência de 2 (multiplicador inteiro + shift) para usar a
   faixa INT8 inteira.
3. Redesenhar softmax e sigmoide para caberem com folga em int32.
4. Teste de treino: ReLU² × ReGLU e ALiBi × RoPE com head_dim 4.

## Como reproduzir

```bash
python analise_passo_a_passo.py     # partes 1-4 -> resultado_analise.txt
python referencias_extras.py        # soma INT8 simétrica e régua FP16/BF16 -> resultado_referencias.txt
```

Python 3.12.3, NumPy 2.5.4, PyTorch 2.6.0 (só nas referências em float).
