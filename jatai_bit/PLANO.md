# Plano: arquitetura binária (Jataí-bit)

Rascunho de planejamento, 2026-11-03. Nada aqui foi implementado ainda.

**Regras do projeto (definidas pelo autor):**
- **Projeto separado do Jataí INT8.** Biblioteca, kernels, especificação e testes próprios. INT8 pode ser usado
  como peça interna onde fizer sentido (ex.: contadores de treino), mas os dois projetos não se misturam; o Jataí
  INT8 e o float32 entram só como **réguas externas** de comparação, como o float32 foi para o INT8.
- **Genérico desde o primeiro kernel.** Todo kernel tem de funcionar para qualquer dimensão, número de camadas,
  contexto, vocabulário e (se houver) número/tamanho de cabeças. Uma biblioteca que só roda numa configuração faz o
  projeto fracassar (foi a falha registrada na biblioteca INT8, relatório técnico, seção 10, item 8). Os testes
  bit a bit rodam em várias configurações, não numa só.
- Mesma disciplina do INT8: especificação executável em inteiros, conferência bit a bit, logs que localizam o erro,
  A/B alternado nas medidas, documentação de tudo o que deu certo e errado.
- **Só binário.** Pesos e ativações do modelo são bits. Inteiros de no máximo 8 bits (int8) só onde for realmente
  necessário (ex.: gradientes, médias durante o treino); somas maiores só em registrador, dentro de um kernel,
  nunca guardadas. Auditoria em 2026-11-03: o contador-probabilidade de cada peso estava em int16 — a corrigir
  (seção 16).

## 1. O que os trabalhos anteriores em bit ensinam

Fontes lidas: `../transformer_bit/` (APB-1650 v1–v42: `Doc.md`, `1.md`, `SKILL.md`, `rules.md`, `walkthrough_v3.md`,
`Doc_vocab_semantico.md`, `SKILL_think_Heb.md`, `infer_output.txt`), `../jatai_dna_2bit/` (álgebra GF(2), blocos do
Transformer 2-bit, loss genética), `../jatai_rwkv_bit/` (telemetria v5), `../2bits/`,
`../documentacao_oficial_jatai_2bit/DOSSIE_TECNICO_OFICIAL_JATAI_V9.md`, `../Motor_Inteiro_Puro/SKILL.md`,
`../RELATORIOS_ARQUITETURAS/`.

| # | o que foi tentado | o que aconteceu (números das fontes) | lição para este plano |
|---|---|---|---|
| 1 | Treino por mutação aleatória de bits (APB v8–v9) e evolução com populações e annealing (v9–v14) | v14: Hamming 255,4–256,0 de 512 (exatamente o acaso, 512/2), 97,2% das mutações aceitas, 10 passos/s; a própria `SKILL.md` estima o espaço em 2^196608 | **Busca às cegas não escala.** Aceitar o melhor de 31 mutantes no mesmo lote seleciona ruído (por acaso, um deles sempre fica um pouco melhor). Precisamos de um sinal que diga *qual* bit virar e *em que direção*: o gradiente |
| 2 | Aritmética GF(2): soma = XOR, saída = paridade de AND (`popc(w & x) & 1`) | Teorema anotado na própria `SKILL.md`: camadas lineares em GF(2) se compõem numa só (24 camadas = 1) | A paridade é a função mais "áspera" possível: virar qualquer bit de entrada troca a saída. Não há "quase certo", então não há gradiente útil. **Usaremos ±1 com soma inteira (XNOR + popcount dá um inteiro graduado), não GF(2)** |
| 3 | Kernel com conjunção (`all_match`: o neurônio dispara se todos os bits do peso estão na entrada) | Com K = 4, ativa em 1/16 das entradas; com K = 2, 1/4, mas cada neurônio depende de só 2 bits (`1.md`) | Ativação por limiar sobre uma **soma** inteira (como num perceptron), não por conjunção |
| 4 | Hebbiano / Willshaw "one-shot" (APB v27–v31, v42 MoE com `atomicOr`/`atomicAnd`) | Saturação ("tempestade de uns"): com 512 dimensões, Hamming ~490 de 512 (v28); com 2048, ~125 (v29); explosão de empates no K-WTA (v31). Inferência da v42: tokens repetidos e de idiomas misturados (`infer_output.txt`) | Hebb é memória associativa (decorar coocorrências), não aprendizado composicional. Pode inicializar, não treinar |
| 5 | Contexto por OR de tokens passados (v25: 4 tokens; v30: 16 tokens) | v30: densidade saltou de 1,9% para 31%; Hamming de ~125 para 1992/2048 (a rede passou a ligar todos os bits) | OR satura. Atenção precisa de **pesos graduados** (somas inteiras + softmax inteira), como a do Jataí INT8 |
| 6 | Métrica de Hamming com alvos esparsos (K = 20 de 512 bits) | "Recorde" de 32,37/512 (v26) | Com 20 bits ligados no alvo, uma saída **toda zero** já erra ~20–40 bits: o número não diz se o modelo aprendeu. **Avaliar com entropia cruzada / perplexidade em validação separada, contra baselines triviais (unigrama, bigrama)** |
| 7 | Contadores saturantes por bit (SBC, `Doc.md` §4): cada bit tem um contador de 4 bits; vira quando o contador chega a 0 ou 15 | A regra pedia avaliar "se inverter o bit reduz o erro" — caro, bit a bit | **A ideia é boa e é a base deste plano:** o contador por bit é exatamente o "peso latente inteiro" do Bop (seção 3). Falta só o sinal barato: o gradiente inteiro diz, para todos os bits de uma vez, se virar ajuda |
| 8 | "RWKV-bit" v5 | Usava `nn.Linear` em float32 (relatório 02); loss 3,59–3,67 em 400 passos efetivos | Nome e implementação têm de bater: aqui, tudo inteiro e conferido bit a bit, como na biblioteca INT8 |
| 9 | Treinos longos (APB v31) | 5 horas de treino perdidas: os pesos não foram salvos; treino sem recorrência e inferência com recorrência (desvio de representação) | Salvar pesos periodicamente; **o mesmo código de forward no treino e na inferência**; conferência automática |
| 10 | Tokenizador por regex de palavras (`NSTokenizer`) | Palavras fora do vocabulário eram descartadas (`pass`) | Começar com **bytes** (vocabulário 256, sem perda), como o Jataí INT8 |
| 11 | Método de inteiros em faixa fixa ±20 com LUTs (`Motor_Inteiro_Puro/SKILL.md`) | Experimento 1 da série INT8: não aprende (5,6–7,7%, acaso 6,25%); `−1 >> k = −1` empurrou 97,7% dos pesos para +127 | Arredondamento simétrico e escala rastreada (expoente por bloco) são indispensáveis — já resolvidos no Jataí INT8 |

O que a série INT8 (`../jatai_cuda`) ensinou e vale como **método** (não como código compartilhado): gradientes,
momento e error feedback 100% inteiros funcionam; arredondamento empate-para-par é indispensável; especificação
executável conferida bit a bit; logs que localizam o erro; ocupação de SM e tamanho do código decidem a
velocidade nesta GPU mais que a contagem de operações.

## 2. O "Axioma 5" (sem STE) é compatível com este plano

O STE (Straight-Through Estimator) aparece quando a **ativação** é binarizada (a derivada de `sign` é zero em
quase todo ponto e é "fingida" como 1). Aqui:
- **Pesos** em ±1 (ou ternários {−1, 0, +1}): o gradiente da loss em relação ao valor do peso é calculado
  **exatamente** pelo backward inteiro (o peso entra no produto como o inteiro ±1). Não há derivada fingida.
- **Ativações** continuam em INT8 com expoente por bloco (o formato que já funciona). Binarizar ativações é o
  que mais destrói qualidade na literatura e é o único ponto que exigiria STE; fica fora da fase 1.
- O estado de treino é um **contador inteiro por bit** (a SBC do `Doc.md`), nunca um float.

## 3. Regra de aprendizado: virar bits guiado pelo gradiente (Bop, com inteiros)

Referência: Helwegen et al., "Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization",
NeurIPS 2019 (otimizador Bop). Adaptação inteira:

```
para cada peso binário w ∈ {−1, +1}, com contador inteiro m (a SBC do Doc.md, agora com sinal):
    m ← m − m/2^a + g          (momento inteiro do gradiente g, já calculado pelo backward INT8)
    se |m| > τ e sinal(m) == sinal(w):   w ← −w    (virar: o gradiente aponta para o outro valor)
```

- Vira **só** quando o gradiente acumulado insiste, e só na direção que reduz a loss: é o "flip" das versões
  anteriores, mas com o erro decidindo qual bit, em vez do sorteio.
- Todos os bits são avaliados de uma vez por passo (o backward dá g para todos), em vez de 1 forward por mutação.
- Hiperparâmetros: a (decaimento, potência de 2) e τ (limiar). Ternário: um segundo limiar decide o 0.

## 4. Arquitetura (a decidir com o autor; ver seção 8)

Proposta: blocos com mistura temporal estilo RWKV (seção 5b) e mistura de canais, pesos ±1 (ou ternários) com um
fator de escala por linha em potência de 2, estado de treino em contadores inteiros por bit (seção 3). Todas as
dimensões são parâmetros. Para comparar com as réguas externas, a primeira configuração medida pode ter o mesmo
número de parâmetros (ou a mesma memória de pesos) e os mesmos dados e avaliação do Jataí INT8.

## 5. Velocidade: empacotar bits no dp4a?

Medido no experimento bitwise (`../kernel_int8_b32/bitplano.cu`, 2048³):

| conta | TOPS | em relação ao dp4a |
|---|---|---|
| dp4a INT8 × INT8 | 3,33 | 1x |
| 8 planos × 8 planos (bit-serial exato) | 0,39 | 0,12x |
| 2 × 2 planos | 3,92 | 1,2x |
| 1 × 1 plano (±1 × ±1, XNOR + popcount) | 7,43 | **2,2x** |

Análise do "vários bits dentro do dp4a" (SWAR): pôr dois pesos ternários numa lane de 8 bits (`w1 + 64·w2`) faz
o dp4a devolver `S1 + 64·S2`; para separar S1 de S2 sem mistura é preciso |S1| < 32, o que só vale com ativações
de até 4 bits e **um** dp4a por extração (4 instruções extras a cada dp4a). A conta prevê perda, não ganho. Com
peso de 1 bit e ativação INT8 o ganho é de **memória** (pesos 8x menores), não de velocidade: a conta ainda é dp4a.
O ganho de velocidade real (2,2x medido) só existe com os **dois** lados em 1 bit.

Mesmo assim vale um experimento curto e barato para confirmar a análise (fase 0), porque a conta acima é teórica.

## 5b. Atenção ou recorrência linear (RWKV)?

Decisão anterior do autor (época do `jatai_rwkv_bit`): seguir com RWKV em bit por ser "mais linear", o que
ajudaria as operações em bit, enquanto o Transformer atrapalhava por não ser linear. Análise:

**Duas linearidades diferentes:**
- *Linear no tempo* (o que o RWKV tem): o estado é atualizado por uma recorrência linear,
  `S_t = decaimento ⊙ S_{t−1} + k_t ⊗ v_t`, sem a matriz T × T nem a softmax sobre todas as posições.
  **Isso ajuda muito o inteiro/bit:** custo O(T) em vez de O(T²), estado de tamanho fixo na inferência, e um
  decaimento escolhido como potência de 2 (`S − (S >> s)`) vira shift exato, o mesmo truque do ALiBi e do momento
  7/8 do Jataí INT8. No Jataí INT8 a atenção custa 36% do passo e é limitada por conta escalar (seção 7.6 do
  relatório técnico): é exatamente o custo que a recorrência linear elimina.
- *Linear na função da camada* (o que o GF(2) tinha): **isso atrapalha.** Camadas lineares se compõem numa só, e
  sem não-linearidade o modelo não aprende nada além de um mapa linear (lição 2). O RWKV não é linear nesse
  sentido: tem portas (receptance, com sigmoide), channel-mixing com ReLU² e normalização. O que ele tira é só a
  não-linearidade *entre posições* (a softmax).

**Cuidado herdado da lição 5:** o estado da recorrência tem de **somar** com magnitude (inteiros com expoente por
bloco), não acumular por OR, que satura.

**Cuidado herdado da lição 8:** o "RWKV-bit" v5 era float32 e o bloco era um `cumsum` com gate, não o RWKV real
(sem decaimento por canal, sem time-first, sem channel-mixing).

**Proposta:** a mistura temporal do projeto binário é no estilo RWKV (decaimento por canal em potência de 2,
estado inteiro que soma com magnitude, portas com sigmoide inteira por shift + polinômio). Para separar o efeito
de cada escolha, as ablações são feitas **dentro** do projeto binário (ex.: pesos ±1 × ternários; com e sem cada
porta), sempre uma mudança por vez.

## 6. Fases

| fase | o que | critério de sucesso | custo estimado |
|---|---|---|---|
| 0 | Micro-benchmark do empacotamento SWAR no dp4a (pesos ternários × ativações 4 e 8 bits) contra dp4a e XNOR-popcount; exato e medido | número medido; decide se o empacotamento entra | 1 dia |
| 1 | Tarefa de indução do experimento 1 (2 camadas): pesos ±1 com Bop inteiro × flip aleatório × INT8 | Bop chega perto do INT8; flip aleatório fica no acaso (confirma a lição 1 com número) | 1–2 dias |
| 1b | Bloco RWKV binário mínimo (1 camada, genérico) na tarefa de indução e em texto de bytes | aprende acima dos baselines; especificação conferida bit a bit em várias configurações | 3–5 dias |
| 2 | Modelo binário completo (biblioteca CUDA própria e genérica, especificação e conferência bit a bit) | loss de validação em 600 passos comparada com as réguas externas: INT8 (2,280) e float32 (2,123) nos mesmos dados | 1–2 semanas |
| 3 | Treino longo (como o noturno de 2026-11-03) | curva de loss contra INT8 e float32 | 1 noite |
| 4 | Só se a fase 2 der qualidade aceitável: ativações de poucos bits nas multiplicações grandes, kernels XNOR-popcount | velocidade e loss medidas | a definir |

## 7. Métricas (para não repetir a lição 6)

Entropia cruzada (nats por byte) em validação separada, os mesmos 8 lotes do Jataí INT8; baselines: unigrama de
bytes, bigrama, float32 e INT8 do mesmo tamanho. Nunca Hamming de vetores esparsos como métrica de aprendizado.

## 8. Decisões do autor (2026-11-03)

1. **Pesos: 4 estados {−2, −1, 0, +1}** (2 bits, 4 por byte; os do Jataí 2-bit anterior). A faixa é
   assimétrica (−2 a +1); o fator de escala por linha em potência de 2 fica. Regra de mudança: um passo de nível por
   vez (`w ← w ∓ 1`, limitado a [−2, +1]) quando o contador do peso passa do limiar.
2. **Ativações: inteiros pequenos** (4 ou 8 bits com expoente por bloco). Sem STE: o gradiente é exato.
3. **Sinal de treino: testar e comparar**, na fase 1 (tarefa de indução), lado a lado e com a mesma avaliação:
   - (A) contador por peso movido pelo gradiente inteiro (SBC do `Doc.md` + Bop);
   - (B) só operações de bit e contagem, sem backward: contador movido pela correlação local (Hebbiano com
     contagem por popcount, a versão que se aplica a pesos de 4 níveis e ativações inteiras; o backprop GF(2) e o
     solver gaussiano exigem ativações de 1 bit e por isso não se aplicam diretamente);
   - (C) controle: virar níveis ao acaso e aceitar só com melhora maior que o ruído medido em validação separada
     (a correção que o `1.md` recomendava para a v14).
4. **Mistura temporal: RWKV** (seção 5b).

Consequência para a fase 0: com pesos de 4 níveis e ativações inteiras, o que se mede é o custo de **desempacotar
2 bits por peso para o dp4a** (código c = w + 2 em {0..3}, `dp4a(c, a) − 2·Σa`, com Σa calculado uma vez por linha
de ativação) contra pesos guardados em INT8, e o empacotamento SWAR de dois pesos por lane (só separável com
ativações de até ~2 bits).

## 9. Resultado da fase 0 (2026-11-03) — `fase0_empacotamento.cu`, `resultado_fase0.txt`

Multiplicação de matriz exata (int32) com pesos de 4 níveis, mesma estrutura de kernel nas três variantes (tile
64 × 64, 256 threads, 4 × 4 saídas por thread). Conferência: 0 diferenças entre as variantes em todas as posições,
e a referência bate com a CPU (64 amostras por forma). Uma execução, GPU sem outra carga.

| M × N × K | ativações | pesos int8 | **pesos 2 bits** | SWAR (2 pesos por lane) |
|---|---|---|---|---|
| 2048×1024×256 | int8 | 3,32 TOPS | **5,68 (1,71x)** | — |
| 2048×256×1024 | int8 | 3,21 | **5,51 (1,72x)** | — |
| 2048×1024×1024 | int8 | 3,28 | **6,23 (1,90x)** | — |
| 2048×2048×2048 | int8 | 3,85 | **7,29 (1,90x)** | — |
| 2048×1024×256 | ternárias | 3,33 | 5,67 (1,70x) | 3,32 (1,00x) |
| 2048×256×1024 | ternárias | 3,24 | 5,58 (1,72x) | 3,15 (0,97x) |
| 2048×1024×1024 | ternárias | 3,28 | 6,24 (1,90x) | 3,46 (1,05x) |
| 2048×2048×2048 | ternárias | 3,85 | 7,35 (1,91x) | 3,78 (0,98x) |

**Leitura honesta:**
- **Pesos em 2 bits foram 1,7–1,9x mais rápidos** que em int8, ao contrário da previsão da seção 5 (que esperava
  perda pelo custo de desempacotar). Causa: este kernel simples, com pesos int8, é limitado pelas **leituras da
  memória compartilhada** (uma palavra de B por grupo de 4 dp4a por coluna); com 2 bits, uma palavra traz 16 pesos e
  serve 4 dp4a, e o desempacotamento (1 shift + 1 máscara por palavra) custa menos que as leituras que economiza.
  Parte do ganho vem, portanto, de a referência ser fraca nesse ponto: num kernel com 8 × 8 saídas por thread (como
  o da biblioteca INT8, que lê menos por dp4a) o ganho deve ser menor. **A medir na fase 2 com o kernel definitivo.**
  O efeito em si (menos bytes de peso por dp4a) é real e cresce com K.
- A soma das linhas de A (para a correção `− 2·Σa`) foi feita aqui com um kernel ingênuo (1 thread por linha,
  0,1–0,6 ms); ela é calculada uma vez por matriz de ativação e serve a todas as multiplicações que a usam, e pode
  ser produzida pelo kernel que gera a ativação, quase de graça. Não entra no tempo da multiplicação acima.
- **SWAR (dois pesos na mesma lane do dp4a): sem ganho (0,97–1,05x)**, como a análise previa: 8 produtos por dp4a,
  mas ~5 instruções para separar os dois resultados a cada dp4a; e só funciona com ativações de até ~2 bits.
  Descartado como caminho de velocidade.
- Consequência para o plano: **guardar os pesos em 2 bits é, ao mesmo tempo, 4x menos memória e (pelo menos neste
  kernel) mais velocidade.** O kernel definitivo do projeto binário parte do formato 2 bits.

## 10. Fase 2 (parcial): multiplicação de matriz definitiva, genérica — `fase2_gemm.cu`, `resultado_fase2_gemm.txt`

Desenho: pesos de 4 níveis em 2 bits (16 por palavra), ativações int8 com **expoente por linha** (não por bloco de
K), acumulador int32 único e exato; tile 64 × 128, 256 threads, 4 × 8 saídas por thread, K em etapas de 64 com dois
buffers. Genérico: qualquer M, N, K (K completado até múltiplo de 64 com ativação 0 e peso 0; bordas de M e N
conferidas). 85 registradores, 12 KB de memória compartilhada (2 bits) contra 87 e 24 KB (pesos int8).

Conferência: formas "tortas" (1000×300×200 e 77×1031×129) conferidas contra a CPU em **todas** as posições (0
diferenças em 300.000 e 79.387); nas demais, 256 amostras contra a CPU e todas as posições contra a variante int8:
0 diferenças.

| forma (M × N × K) | pesos int8 (mesmo kernel) | pesos 2 bits | 2 bits / int8 |
|---|---|---|---|
| 1000×300×200 (torta) | 2,31 TOPS | 3,10 | 1,34x |
| 77×1031×129 (torta) | 1,09 | 1,31 | 1,20x |
| 2048×768×256 | 4,47 | **6,73** | 1,50x |
| 2048×256×256 | 4,09 | **5,89** | 1,44x |
| 2048×1024×256 | 4,74 | **7,43** | 1,57x |
| 2048×256×1024 | 5,25 | **8,08** | 1,54x |
| 4096×4096×1024 | 6,20 | **9,23** | 1,49x |
| 2048×2048×2048 | 6,17 | **9,95** | 1,61x |

(A/B alternado no mesmo processo, 5 rodadas × 20 chamadas de cada.)

- Nas formas do modelo de 12 camadas, **5,9–8,1 TOPS, ~2x a GEMM da biblioteca INT8** nas mesmas formas (3,0–4,0).
  Duas causas: (1) sem expoente por bloco de K, um acumulador só (85 registradores em vez de ~128–209) e nenhum
  alinhamento a cada 128 de K; (2) pesos 4x menores = menos leituras.
- **Porcentagem do pico, conferida:** o pico do dp4a medido em sequência na mesma sessão deu 9,99 TOPS com o SM a
  1410 MHz e 13,50 TOPS logo depois, já aquecido, a 1875 MHz (13,50 / 9,99 = 1,35 ≈ 1875 / 1410 = 1,33): **o pico
  escala com o clock**. Na mesma condição (GPU aquecida), a GEMM de 2 bits deu 9,94 TOPS em 2048³ = **~74% do pico
  real**. O número de 99% (contra os 10,06 TOPS medidos antes, com clock baixo) estava errado.
- Custo de qualidade a medir: expoente por linha em vez de por bloco (no experimento 4 da série INT8, expoente por
  tensor derivava 21,5% em 12 camadas e por bloco de 32, 9,4%; por linha fica entre os dois). A fase 1 usa expoente
  por tensor e aprende; a fase 2 completa (modelo de texto) mede o efeito real.
- A soma das linhas de A (correção do código 2 bits) custa 0,004–0,027 ms com 1 warp por linha; na biblioteca sai
  de graça do kernel que produz a ativação.

## 11. Desenho proposto do bloco RWKV binário (fase 1b; para revisão do autor, não implementado)

Base: a forma de "estado em matriz" do RWKV-5/6 (atenção linear com decaimento por canal), porque ela é linear no
tempo **sem** exponenciais das chaves (o RWKV-4 precisa de e^k e de uma soma normalizada, o que traz de volta a
softmax). Tudo em inteiros; pesos de 4 níveis em 2 bits (fase 2); ativações int8 com expoente por linha.

Por camada, com H cabeças de tamanho S (D = H·S; ambos parâmetros, qualquer valor):

**Mistura temporal** (para cada token t, cabeça h):
- deslocamento de token (token shift) por média com o anterior, com coeficientes por canal em potência de 2:
  `x̃ = x_t + ((x_{t−1} − x_t) >> μ)` — só soma e shift;
- projeções com pesos de 4 níveis: `r = W_r x̃_r`, `k = W_k x̃_k`, `v = W_v x̃_v`;
- estado S_h (matriz S × S, int32 com expoente por linha do estado), decaimento por canal `d_c ∈ {1, 2, …}`:
  `S_t = S_{t−1} − (S_{t−1} >> d) + k_t ⊗ v_t` — o "esquecer" é **um shift exato**, como o momento 7/8 do INT8;
- saída da cabeça: `o_t = r_tᵀ S_t` (e um termo de bônus para o token atual, `+ (r_t · (u ⊙ k_t)) v_t`, como no
  RWKV);
- normalização por cabeça (RMS inteira) e porta: `y = W_o (σ(g) ⊙ norma(o))`, com σ inteira (shift + polinômio,
  como a exp do projeto INT8).

**Mistura de canais:** `r' = σ(W_r' x̃)`, `k' = ReLU(W_k' x̃)²`, `y = r' ⊙ (W_v' k')`.

**Treino:** backward pela recorrência em ordem inversa (a mesma recorrência linear, transposta), em inteiros; pesos
pela regra do contador (método A da fase 1, se ele vencer a varredura).

**Custos que caem em relação à atenção do INT8:** nada de matriz T × T, nada de exp por par de posições, contexto
sem limite fixo (o estado tem tamanho S × S por cabeça, independente de T). O custo por token é ~2·D·S operações na
mistura temporal, contra ~T·D na atenção.

**Perguntas abertas para medir, não para supor:**
1. decaimento só por shift (`d` inteiro) é fino o suficiente, ou precisa de `S − (S >> d1) − (S >> d2)` (dois
   shifts, decaimentos intermediários)?
2. estado com expoente por linha ou por cabeça?
3. a porta σ inteira é necessária ou um ReLU basta?

## 12. Treino 100% binário (pesos E ativações em bit) sem STE — ideias e análise (2026-11-03)

### 12.1 A ideia do autor: "o observador muda o resultado" / um bit representando mais de uma coisa

Proposta (resumo): aproveitar que um bit só tem dois estados para que o estado e o seu oposto signifiquem coisas
diferentes (ex.: no vocabulário, uma palavra e o seu antônimo no mesmo bit), como se a ausência de um valor já
carregasse significado; e algo como a superposição quântica, em que a observação decide o valor.

Análise:
- **O limite que não muda:** 1 bit carrega no máximo 1 bit de informação (Shannon). Dois tokens independentes não
  cabem em 1 bit. O que existe — e é real — é economizar quando há **estrutura** nos dados: se B é sempre
  "o oposto de A", basta guardar A e a regra. A economia vem da regra, não do bit.
- **Versão aproveitável do exemplo dos antônimos:** com vetores ±1, "o oposto" é de graça (XOR com uma máscara).
  Mas o complemento total faria "quente" e "frio" serem os vetores mais distantes possíveis, e na língua eles são
  parecidos em quase tudo (os dois são temperatura) e diferem em poucas dimensões. A forma que respeita isso é
  **uma máscara de negação aprendida e compartilhada**: `E[antônimo(A)] = E[A] XOR m_neg`, com m_neg ligando só as
  dimensões de "polaridade". Guarda-se metade dos vetores dos pares e uma máscara. É a operação de "binding" da
  computação hiperdimensional (que o `jatai_dna_2bit` já usava). Testável: corta o embedding dos pares pela metade e
  mede-se a perda de qualidade.
- **Versão aproveitável da "superposição / observação":** um peso binário cujo valor ainda não está decidido é
  representado pela **probabilidade** de ser 1 (o contador inteiro que já usamos é exatamente isso: quanto mais longe
  de zero, mais "certo" o bit). "Observar" = sortear o bit a partir dessa probabilidade no forward. A perda
  esperada é suave em relação à probabilidade, então ela tem gradiente **sem STE**. É a ideia de *Expectation
  Backpropagation* (Soudry, Hubara e Meir, NeurIPS 2014), que treina redes com pesos binários propagando médias, sem
  derivada fingida. Os pesos do modelo final são bits; a "superposição" só existe durante o treino.
- **"A ausência significa algo":** com ±1, os dois estados carregam informação de forma simétrica (não há
  ausência). A ausência explícita é o 0 do ternário / dos 4 níveis, que já está na decisão de pesos.

### 12.2 Outras aritméticas para o treino 100% binário sem STE

O problema central: com ativação binária, `sinal(x)` tem derivada zero em quase todo ponto; o STE "finge" que é 1.
Caminhos que não fingem nada:

1. **Crédito exato por margem (Madaline Rule II, Winter e Widrow, 1988 — "perturbação mínima").** Num neurônio ±1
   com pré-ativação inteira `z = Σ w·x`, virar uma entrada x_i muda z **exatamente** em `−2·w·x_i`, e o neurônio só
   troca de estado se |z| for menor que essa mudança. Então dá para calcular, em inteiros e sem aproximação, quais bits
   escondidos, se virados, consertam a saída; a escolha recai nos neurônios **menos certos** (menor |z| — os que
   estão "em superposição", perto do limiar). O neurônio escolhido ganha um bit-alvo e ajusta os próprios pesos pela
   regra do perceptron (contador). Tudo é popcount e soma.
2. **Média de campo inteira (Expectation Backpropagation, 12.1):** propaga a média (e a variância) de cada bit em
   ponto fixo inteiro; a não-linearidade vira uma função suave inteira (shift + polinômio, como a exp do projeto
   INT8). Gradiente exato do modelo de médias.
3. **Margem por camada (alvos locais):** cada camada recebe um padrão-alvo de bits (vindo de 1) e aprende com
   perceptron; dispensa o backward contínuo.
4. **Contadores por bit (já validado na fase 1 com ativações inteiras):** o mesmo mecanismo serve aos métodos 1–3
   como o estado que decide quando um peso vira.

### 12.3 Experimento proposto (fase 1c)

Na tarefa de indução, rede **totalmente binária** (pesos e ativações ±1, pré-ativações por popcount):
- (a) Madaline Rule II com crédito exato por margem;
- (b) média de campo inteira (pesos como probabilidades, forward com bits sorteados — a "observação");
- (c) controle: o método A da fase 1 (ativações inteiras), que é o teto;
- (d) controle: flip ao acaso.
Critério: acurácia e loss contra os controles, 3 sementes. Se (a) ou (b) chegar perto de (c), o treino binário puro
sem STE é viável neste modelo; se não, fica documentado com o motivo medido.

Depois, num modelo de texto: a máscara de negação compartilhada (12.1) no embedding, medindo memória poupada e perda.

## 13. Resultado da fase 1 (2026-11-03) — `fase1_inducao.py`, `fase1_varredura.sh`, `logs/fase1_resumo.md`

Tarefa de indução, modelo de 2 camadas (atenção), ativações inteiras, **todos** os pesos em 4 níveis {−2,−1,0,+1}
(exceto na régua int8). 3 sementes por configuração; avaliação a cada 100 passos em 512 sequências fixas.

| configuração | acurácia final | loss final (nats) | passo até 100% |
|---|---|---|---|
| régua int8, k = 7 (melhor) | 100% (3/3) | 0,025 (0,024–0,027) | 233 (200–300) |
| régua int8, k = 6 | 69% — **divergiu numa semente** (6,9%) | 1,084 | 2 de 3 |
| **A: gradiente + contador, k = 6** | **100% (3/3)** | **0,003 (0,003–0,004)** | 267 (200–300) |
| A, k = 7 | 100% (3/3) | 0,006 | 300 (200–500) |
| A, k = 8 | 100% (3/3) | 0,010 | 400 (200–500) |
| A, k = 5 | 37% — **divergiu em 2 de 3** | 3,107 | 1 de 3 |
| A com decaimento do contador (Bop), a = 6 | 100% / 100% | 0,030 / 0,058 | pior que sem decaimento |
| A com decaimento a = 4 | 99,8% / 98,5% | 0,096 / 0,244 | não chega a 100% |
| B: Hebb + regra delta só na saída | 6,0–6,4% (acaso: 6,25%) | 2,8–9,6 | nunca |
| C: flip ao acaso com validação (R = 1, 8, 64) | 6,9–7,4% | 3,4–3,9 | nunca |

Conclusões:
1. **Pesos de 4 níveis treinados pelo contador movido pelo gradiente aprendem tão rápido quanto a régua int8 e
   chegam a uma loss ~8x menor** (0,003 contra 0,025 da melhor int8). Cuidado na leitura: as duas variantes têm
   inicialização e escala de peso diferentes; o resultado garante "pelo menos tão bom", não "melhor em geral".
2. **Faixa de lr:** k = 6–8 funciona nas 3 sementes; k = 5 diverge (como o int8 em k = 6). O contador não precisa
   de decaimento: o decaimento do Bop só piorou (perde o acúmulo dos gradientes pequenos).
3. **Hebb (B) e flip ao acaso (C) ficam no acaso em todas as sementes e configurações**, mesmo com 2x mais passos
   e (C) com 4 forwards por passo. Confirma, com número, as lições 1 e 4: sem crédito vindo do erro (backward), o
   modelo não aprende a tarefa. No C, quase nenhuma mudança passa no critério de validação (0–151 aceitas em
   4000 passos): o critério corrige o "aceitar ruído" da v14, e o resultado honesto é que a busca não encontra
   melhoras.
4. Distribuição final dos níveis no método A: ~5–8% em −2, ~28% em −1, ~30% em 0, ~35% em +1: o nível −2 (só
   alcançável treinando) é usado, e o 0 é bastante usado.

## 14. O conceito do "observador" aplicado ao treino inteiro (teoria, 2026-11-03)

O exemplo dos antônimos era só uma ilustração. O conceito do autor, separado em três princípios:

- **P1 — Complementaridade:** os dois estados de um bit são duas alternativas com significado, e passar de uma
  para a outra (NOT / XOR) não custa nada.
- **P2 — Superposição até a observação:** enquanto não está decidido, o bit guarda o grau de certeza (uma
  probabilidade, que no projeto é um contador inteiro). "Observar" fixa o bit, e só se observa o que precisa.
- **P3 — A ausência informa:** o estado não escolhido (o complemento, o "não") também é informação usada pelo
  modelo.

Limite físico que vale para tudo abaixo: bits clássicos não têm interferência de amplitudes; o que se aproveita é a
probabilidade clássica (P2) e a estrutura (P1). Ganho de memória só existe onde há estrutura; ganho de velocidade,
onde há certeza.

| # | onde no treino | mecanismo | o que pode render | como testar |
|---|---|---|---|---|
| 1 | **Pesos** (P2) | peso = probabilidade num contador; observado (sinal do contador) no fim | treino de rede binária sem STE | **feito**: fase 1c (29,1% em 1000 passos contra 30,7% da régua float em 5000) |
| 2 | **Colapso progressivo dos pesos** (P2) | quando o contador de um peso fica certo (ex.: \|θ\| no máximo por N passos), o peso é "observado" de vez: vira um bit fixo, sai do treino e passa a usar o caminho rápido (XNOR + popcount / 2 bits); só os incertos continuam em superposição | **velocidade e memória do treino** crescem à medida que o modelo fica certo; o custo do treino cai ao longo do tempo | fração de pesos colapsados por passo; tempo por passo; qualidade contra a versão sem colapso |
| 3 | **Ativações** (P2) | cada neurônio carrega a média do seu bit; os certos (\|M/√V\| grande) colapsam em bit e seguem por popcount, os incertos seguem como média | a maior parte da conta vira operação de bit; só a "fronteira de decisão" paga a conta cara | fração de neurônios colapsados por camada; tempo e acurácia |
| 4 | **Observação dependente do contexto** (P1) | uma matriz de bits compartilhada e uma máscara pequena por contexto: `W_ctx = W XOR m_ctx` (o mesmo peso "lido" de outro jeito por outro observador) | várias redes (tarefas, idiomas, especialistas de um MoE) no espaço de uma + máscaras esparsas; ajuste fino barato (só a máscara treina) | E especialistas como 1 base + E máscaras esparsas, contra E matrizes completas, com o mesmo orçamento de memória |
| 5 | **Relações no embedding** (P1; generaliza o exemplo dos antônimos) | um vetor base por raiz e uma máscara por relação: `E[r(A)] = E[A] XOR m_r` (antônimo, plural, gênero, tempo verbal...) — ecoa a "genética gramatical" do `jatai_dna_2bit` | embedding de um vocabulário de palavras guardado como raízes + poucas máscaras | só com vocabulário de palavras ou subpalavras; com bytes não se aplica |
| 6 | **Atualização como bit** (P1 + P3) | o update de cada peso é um bit de direção + "nenhum" (o contador não mexe); virar = XOR | update de 1 bit por peso: barato para treino distribuído / em várias GPUs | já é o que o contador faz por dentro; medir só se houver treino distribuído |
| 7 | **Estado da recorrência (RWKV)** (P2) | componentes do estado com certeza alta guardados como bits; os incertos, como inteiros | estado menor e mais barato no contexto longo | depois do bloco RWKV binário existir (fase 1b) |
| 8 | **Evidência negativa** (P3) | em cada neurônio de saída, "não é este token" também treina (com peso proporcional à certeza errada), não só "é este" | já está na softmax/entropia cruzada; na versão binária pura, cuidado com a lição 6 (alvos esparsos enganam) | — |

Prioridade proposta (o que mais pode render e é testável já): **2** (colapso progressivo: velocidade do treino) e
**3** (ativações colapsadas: velocidade), depois **4** (capacidade por memória). Ambos medem o que o autor quer:
velocidade sem perder qualidade.

## 15. Fase 1c: rede 100% binária sem STE (2026-11-03)

**Tarefa professor-aluno** (`fase1c_binario.py`, `fase1c_kernel.py`): professor = rede binária sorteada
(256 → 512 → 512 → 16 classes, ativação sinal); aluno com a mesma forma; avaliação sempre da rede final **100%
binária** (pesos e ativações ±1) em 8192 entradas fixas. Régua de chute constante: 8,2% (classe mais comum).

**Régua de aprendibilidade (não é método do projeto):** rede float comum (pesos contínuos, tanh, Adam), 5000 passos:
30,7%, 30,6%, 30,8% (3 sementes). A tarefa é difícil por natureza: o professor binário com sinal em camadas é uma
função muito "áspera", e nem a rede float passa de ~31%.

**Resultados (3 sementes):**

| método | acurácia da rede 100% binária | observação |
|---|---|---|
| **observador (pesos como probabilidade, média de campo inteira)** | **27,6–29,4%** (12 corridas: 4 tamanhos de passo × 3 sementes) | platô já no passo ~250; estável entre sementes e tamanhos de passo |
| Madaline com crédito exato por margem | 6,6–13,9% (semente 1, 9 configurações); melhor configuração em 3 sementes: 13,9%, 14,9%, 11,2% | aprende pouco; com passo pequeno não sai do acaso |
| flip ao acaso com validação | 6,8%, 6,4%, 5,6% (3 sementes) | nada aceito em 3000 passos |
| régua float | 30,6–30,8% | — |

**O observador treina uma rede 100% binária sem STE e sem float até ~2 pontos da rede float.** A Madaline (o crédito
exato "clássico") fica muito atrás, como a literatura indicava.

**Kernels próprios (`libbit.cu`):** a 1ª versão em PyTorch (`_int_mm` + int64) levava 85 s por 1000 passos; com
kernels próprios na representação Q7 (médias int8, um dp4a por produto, contadores int16), **2,5 s por 1000 passos
(~34x)**, com a mesma acurácia (28,6%). Conferência bit a bit de um passo inteiro (forward, softmax, backward,
update) contra a especificação int64: tudo igual.

Dois ajustes de especificação encontrados pela própria conferência: (1) com a razão M/√V em só 7 bits de fração, as
médias das ativações (pesos ainda incertos) viravam 0 na 2ª camada e o gradiente da saída morria (gradiente da camada
de saída = 0 na conferência); passou a 10 bits de fração (Q10), com a saída da tanh em int8; (2) contadores começam em
±512 (peso médio ±0,5) em vez de ±64.

**Teste do "colapso progressivo" (seção 14, item 2) — negativo nesta tarefa:** fração de pesos com o contador no
limite (|θ| = 1024) depois de 1000–4000 passos: 0,0–0,9% nas camadas escondidas, 0,2–8,4% na saída. Os pesos ficam em
"superposição" o treino inteiro; colapsar os certos economizaria quase nada aqui. Ideia derivada (a testar): uma
**pressão de observação** que aumenta com o tempo (empurra os contadores incertos para o extremo mais próximo, como
um recozimento), para que o modelo termine "decidido" — o que também reduziria a diferença entre a rede de médias
(treino) e a rede observada (avaliação).

**Limitação da tarefa:** como até a rede float para em ~31%, esta tarefa não separa bem métodos bons de ótimos nem
mede escala. Próximo passo: um professor mais aprendível (ex.: camadas escondidas mais estreitas, ou pesos esparsos)
em que a rede float chegue perto de 100%, e então profundidade e largura crescentes (4, 8, 16 camadas) para medir se
o observador escala — o ponto onde os trabalhos anteriores falharam.

## 16. Fase 1d: contador int8, professor aprendível e escala em profundidade (2026-11-03) — `fase1d_escala.py`

Driver genérico (qualquer profundidade e largura no professor e no aluno), kernels da libbit; conferência bit a bit
contra a especificação int64 com 2 e 4 camadas e contador de 16 e de 8 bits: tudo igual.

**Regra do projeto aplicada:** o contador-probabilidade de cada peso passou a **int8** (θ ∈ [−127, 127], que já é a
média do peso em Q7). O que o método guarda fora de bits: só durante o treino, o contador int8 de cada peso e as
médias int8 das ativações; int32 só como soma em registrador. A rede final é 100% bits.

**Professor aprendível.** Professores binários estreitos (2 camadas de 8 neurônios) são aprendíveis: régua float
98,4% em 3000 passos, 100% em 10.000. Com professores de 16, 32, 64 e 512 neurônios, até a régua float fica em
57%, 48%, 40% e 31%.

**Primeira leitura (3000 passos, passo do contador 6): o observador ficava muito atrás da régua** (55,8% com int16,
44,0% com int8, contra 98,4%). Causa: passo pequeno demais e treino curto. Com 10.000 passos e passo maior:

| configuração (professor 2 × 8, aluno 2 × 512) | acurácia da rede 100% binária |
|---|---|
| régua float | 100% |
| int16, passo 3 / 4 / 5 / 6 / 7 / 8 | 87,0 / **89,8** / 88,5 / 77,1 / 60,3 / 51,1% |
| **int8**, passo 2 / 3 / 4 | 84,2 / 89,3 / **90,7%** |
| int8, passo 4, 30.000 passos | 90,0% (platô) |

**O contador int8 empata com o int16** (90,7% × 89,8%) — o contador fica em 8 bits.

**Escala em profundidade** (mesmo professor 2 × 8; aluno com 2, 4 e 8 camadas de 512; 10.000 passos; 1 semente):

| aluno | régua float | observador int8, passo 3 / 4 / 5 |
|---|---|---|
| 2 camadas | 100% | 89,3 / 90,7 / — % |
| 4 camadas | 99,4% | **95,5** / 91,5 / 79,6% |
| 8 camadas | 98,4% | **91,7** / 87,0 / 19,7% (não aprende) |

**O treino 100% binário pelo observador não quebra com a profundidade**: 95,5% com 4 camadas e 91,7% com 8, contra
98,4–99,4% da régua float. Os métodos clássicos sem STE (Madaline, seção 15) quebravam com 2–3 camadas. O passo do
contador precisa diminuir com a profundidade (passo 5 com 8 camadas não sai do acaso).

Pendências: 3 sementes; professores mais profundos (o aluno de 8 camadas ainda imita um professor de 2); reduzir os
bits do treino (contador e médias em 4 bits, como a SBC do `../transformer_bit/Doc.md`); a "pressão de observação"
(seção 15) para fechar a diferença entre a rede de médias e a rede observada.

### 16.1 Menos bits no treino: contador e médias em 4 bits (2026-11-03)

Kernels novos na libbit: contador de 4 bits (θ ∈ [−7, 7], média do peso = θ·18 em Q7, como a SBC do
`../transformer_bit/Doc.md`) com **arredondamento estocástico** do passo (sorteio por hash inteiro `lowbias32`,
determinístico e reproduzível, sem float: com só 15 níveis, o arredondamento determinístico zeraria quase todo
passo); e médias das ativações com 4 bits (±7 níveis). Conferência bit a bit contra a especificação int64 (3 camadas):
c8, c4, c8 + médias 4 bits e c4 + médias 4 bits — tudo igual, inclusive o sorteio.

Aluno de 4 camadas × 512, professor 2 × 8, 10.000 passos, 1 semente (referência: contador int8 + médias int8 = 95,5%):

| configuração | passo | acurácia da rede 100% binária |
|---|---|---|
| contador int8 + **médias de 4 bits** | 2 / 3 / 4 | 86,5 / **95,2** / 94,9% |
| **contador de 4 bits** (estocástico) + médias int8 | 6 / 7 / 8 / 9 | 67,3 / **71,0** / 64,4 / 59,1% |

- **Médias das ativações com 4 bits: sem perda** (95,2% × 95,5%). Passam a 4 bits.
- **Contador de 4 bits: perde ~24 pontos.** O contador precisa de 8 bits (ou de outra regra para 4 bits). Fica int8.

## 17. "Calcular metade, a outra metade é o oposto" — onde isso é verdade (2026-11-03)

Pergunta do autor: se um bit e o seu oposto representam duas coisas, a conta não cairia pela metade?

Análise:
- **Em geral, não:** o número de contas de aprendizado acompanha a informação independente que o modelo guarda; se a
  outra metade é sempre o oposto exato da primeira, ela não traz informação nova (é a mesma metade, lida ao contrário).
- **Onde já é verdade e já está sendo usado:**
  1. No produto de bits por XNOR + popcount, só se contam as concordâncias; as discordâncias são "o resto"
     (K − concordâncias): a soma ±1 sai de uma contagem só.
  2. No observador, a variância usa o quadrado da média do peso, que é igual para W e para −W.
- **Onde pode render de verdade (a testar, ideia 17a): dois bits por soma.** Cada soma calculada `z = Σ w·x` gera dois
  bits: `z > +t` e `z < −t` (o "positivo" e o "oposto"), com limiar t aprendido. Uma conta, duas ativações: a camada
  fica com o dobro de saídas pelo mesmo custo de multiplicação. É o equivalente binário do CReLU (Shang et al., 2016:
  `[ReLU(z), ReLU(−z)]`, que em redes convolucionais deu a mesma qualidade com cerca de metade dos filtros). Mede-se
  acurácia por custo contra a rede de 1 bit por soma com o dobro de neurônios.
- **E a superposição já rendeu:** é o observador (pesos como probabilidade, observados no fim), o que treinou a rede
  100% binária sem STE e escalou até 8 camadas (seção 16).

### 16.2 Três sementes e professor mais profundo (2026-11-03, `fase1d_varredura.sh`, `logs/fase1d_varredura.log`)

Professor 2 × 8, contador int8, 10.000 passos, melhor passo de cada profundidade (rede final 100% binária):

| aluno | observador (sementes 1 / 2 / 3; média) | régua float (sementes 1 / 2 / 3) |
|---|---|---|
| 2 camadas, passo 4 | 90,7 / 87,6 / 90,2 — **89,5%** | 100 / 100 / 99,6% |
| 4 camadas, passo 3 | 95,5 / 92,8 / 92,0 — **93,4%** | 99,4 / 99,2 / 99,2% |
| 8 camadas, passo 3 | 91,7 / 89,6 / 87,5 — **89,6%** | 98,4 / 99,1 / 98,3% |

Confirma em 3 sementes: com o professor de 2 camadas, o observador fica a ~6–10 pontos da régua float em todas as
profundidades testadas, sem quebrar até 8 camadas.

**Professor mais profundo (4 × 8), 1 semente — aqui a escala NÃO se confirmou com 8 camadas:**

| aluno | régua float | observador, passo 2 / 3 / 4 |
|---|---|---|
| 4 camadas | 99,1% | 82,0 / 88,6 / **89,1%** |
| 8 camadas | 99,0% | **3,3** / 19,9 / 50,8% |

Com 4 camadas o observador ainda aprende (89,1%). Com 8 camadas e o professor mais difícil, o treino fica instável:
50,8% no melhor passo, e 3,3% com passo 2 — **abaixo do chute constante (25,1%)**, ou seja, a rede convergiu para
uma resposta sistematicamente errada (sinal de divergência, não de "não aprender"). A afirmação "escala até 8
camadas" vale só para a função mais simples. Causas a investigar: a aproximação de média de campo piora a cada camada
(a variância é tratada como independente); o passo único para todas as camadas; ausência de normalização entre
camadas. Candidatos: passo por camada, a "pressão de observação" (seção 15), normalização inteira das médias entre
camadas.

## 18. As três correções para a profundidade (2026-11-03)

**Diagnóstico** (`diag_medias.py`, professor 4 × 8, aluno 8 × 512): no início, o tamanho médio das médias cai camada
a camada (23 → 5 → 1 → 0,2 → 0: da 5ª camada em diante a rede não "vê" nada). Com passo grande, todas as camadas
saturam (~120 de 127: "certas", mas erradas — daí o resultado abaixo do acaso); com passo pequeno, as camadas do meio
ficam mortas (3–10) e só as últimas crescem.

**1. Normalização inteira entre camadas — FUNCIONA.** Antes da tanh, a razão r de cada amostra é dividida pela RMS
da linha e multiplicada por um alvo T (Q10): `r' = r · T / rms(r)` (kernels `k_norm_tanh`, `k_dtanh_norm`; conferência
bit a bit com 3 camadas, com médias de 8 e de 4 bits). Dividir por um positivo **não muda o sinal**: a rede binária
observada é exatamente a mesma, a normalização só existe no treino.

Caso que falhava (professor 4 × 8, aluno 8 × 512, passo 4 salvo indicação), 1 semente:

| T (Q10) | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | **6144** | 8192 |
|---|---|---|---|---|---|---|---|---|
| passo 4 | 33,2 | 49,0 | 79,6 | 83,3 | 87,7 | 93,6 | **94,3** | 91,9 % |
| passo 3 | 25,8 | 23,6 | 79,6 | 80,8 | 77,5 | 26,2 | — | — |
| passo 5 | — | — | — | — | — | 79,4 | 79,2 | 81,1 % |

Com T = 6144 e passo 4 (antes da correção: 50,8% no melhor passo; régua float 99,0%):

| configuração | sem normalização | **com normalização** | régua float |
|---|---|---|---|
| professor 4 × 8, aluno 8 camadas (sementes 1 / 2 / 3) | 50,8% (melhor passo, semente 1) | **94,3 / 87,9 / 87,7%** | 99,0% |
| professor 4 × 8, aluno 4 camadas | 89,1% | **93,1%** | 99,1% |
| professor 2 × 8, aluno 2 camadas | 90,7% | **92,3%** | 100% |
| professor 2 × 8, aluno 4 camadas | 95,5% | **99,0%** | 99,4% |
| professor 2 × 8, aluno 8 camadas | 91,7% | **95,7%** | 98,4% |
| professor 4 × 8, aluno 8 camadas, **médias de 4 bits** | — | 91,3% | 99,0% |

- **Com 4 camadas e o professor de 2, a rede 100% binária chega a 99,0%, contra 99,4% da régua float.**
- A normalização melhorou todos os casos e é o que faz o treino profundo funcionar. Ainda há variação entre
  sementes no caso mais difícil (87,7–94,3%).
- Médias de 4 bits com normalização: 91,3% (−3 pontos contra 8 bits no caso difícil; no caso de 4 camadas, sem
  normalização, não havia perda — seção 16.1).

**2. Passo diferente por camada — NÃO ajudou.** Com T = 6144 no caso difícil: saída com passo maior (3) 91,4%; saída
com passo menor (5) 91,8%; camadas iniciais com passo maior (3) 75,5%; camadas iniciais com passo menor (5) 84,6%;
contra 94,3% com o mesmo passo em todas. A normalização por tensor do gradiente (cada tensor normalizado pelo seu
maior valor) já iguala as camadas.

**3. Pressão de observação — PIORA.** Probabilidade máxima (de 256) de empurrar o contador 1 nível para o lado mais
provável, crescendo linearmente até o fim: 16 → 90,3%; 64 → 80,1%; 128 → 68,9% (contra 94,3% sem pressão). Decidir os
pesos à força tira deles a liberdade de continuar mudando. (O kernel `k_pressao` não passou por conferência bit a
bit contra a especificação, porque o resultado foi descartado.)

## 19. Velocidade e GLU binário (2026-11-03)

**Perfil por kernel** (nvprof, aluno 8 × 512, 1000 passos): multiplicações de matriz 44%; saída de camada (raiz e
divisão por elemento) 15%; backward da normalização (2 divisões int64 por elemento) 11%; normalização (divisão int64
por elemento) 11%; máximos, transposições, quantização e update ~19%. **As divisões por elemento custavam ~37% do
passo.**

**Variância das camadas escondidas** (caso difícil, professor 4 × 8, aluno 8 × 512, 3 sementes, `JB_VAR`):

| variante | sementes 1 / 2 / 3 | média | tempo (10.000 passos) |
|---|---|---|---|
| exata (2ª multiplicação de matriz + raiz + divisão) | 94,3 / 87,9 / 87,7 | 90,0% | 65,5 s |
| aproximada (somas de linha e coluna) | 92,7 / 90,5 / 91,3 | 91,5% | 61,5 s |
| **sem variância** (a soma vai direto para a normalização) | 91,9 / 90,1 / 89,6 | 90,5% | 55,2 s |
| **sem variância + recíproco por linha** (sem divisão por elemento) | 93,0 / 90,5 / 91,3 | **91,6%** | **49,4 s** |

Com a normalização entre camadas, a variância é redundante: sem ela, **a qualidade se mantém (ou melhora) e o passo
fica 25% mais rápido**. Configuração adotada. Conferência bit a bit das três variantes contra a especificação int64.

**GLU binário (a "tabela de combinações" do autor, resumida)** — `fase1e_glu.py`, kernels `k_glu_mul`, `k_glu_back`;
conferência bit a bit (camada simples e GLU, 3 camadas): tudo igual. Camada GLU: `h = sinal(A x) · sinal(B x)` (XNOR),
média de campo `μ = μA · μB`.

| professor (2 camadas) | régua float (4 × 512) | simples 4 × 512 | GLU 4 × 384 (+12% pesos) | GLU 4 × 320 (−22% pesos) |
|---|---|---|---|---|
| 16 neurônios | 96,9% | 64,1% | 6,7% | 6,4% |
| 32 neurônios | 84,1% | 38,0% | 7,2% | 6,2% |

**O GLU não aprende** — nem a rede de médias (erro de treino ~87%). Diagnóstico em float (`diag_glu_float.py`, só
ferramenta: o mesmo modelo de médias com gradiente exato do autograd), professor 2 × 8, 3000 passos:

| alvo da normalização T | simples, 1 camada | GLU, 1 camada | simples, 2 camadas | GLU, 2 camadas |
|---|---|---|---|---|
| 6,0 | 65,9% | 36,3% | 88,3% | 18,7% |
| 0,5 / 1,0 / 2,0 | — | — | 46,1 / 54,6 / 74,3% | 25,4 / 35,0 / 44,3% |

O GLU fica bem abaixo da camada simples também em float, com qualquer T: **o problema é a formulação de média de
campo do produto, não a versão inteira** (o kernel bate bit a bit com a especificação, e a especificação, em float, já
treina mal). Hipótese não testada: a média de campo trata hA e hB como independentes, mas as duas projeções veem a
mesma entrada, e o produto de duas médias saturadas tem gradiente útil só quando as duas estão fora da saturação ao
mesmo tempo. Pela condição combinada com o autor, os "pares escolhidos" ficam para depois (o GLU não mostrou que a
combinação ajuda nesta formulação).

## 20. Checklist do treino binário (estado em 2026-11-03)

### Como o treino acontece (configuração adotada)
1. Cada peso tem um **contador int8** θ ∈ [−127, 127] = a "probabilidade" de o peso ser +1 (o observador). O peso
   do modelo é **sinal(θ)**: 1 bit.
2. **Forward do treino:** cada camada propaga a **média** de cada bit (int8): `M = Σ média_entrada × θ` (dp4a, soma
   int32), normalização por linha `M · T / rms` (recíproco por linha) e `média = tanh(0,8 · …)` em int8. A camada de
   saída usa média e variância exatas.
3. **Saída:** softmax inteira; gradiente `p − 1[classe certa]` (Q15).
4. **Backward:** gradientes quantizados em int8 por tensor, derivada da tanh `(1 − média²)`, gradiente de cada peso
   `G = gradienteᵀ · média_entrada` (dp4a, soma int32).
5. **Update:** `θ −= G normalizado pelo maior |G| do tensor >> passo` (passo 4), limitado a ±127.
6. **Rede final ("observação"):** pesos = sinal(θ), ativações = sinal da soma: **100% bits**.

### O que funciona
| item | evidência |
|---|---|
| treino de rede 100% binária sem STE e sem float (observador) | fases 1c–1d; 99,0% contra 99,4% da float (4 camadas, professor simples) |
| contador int8 (em vez de int16) | 90,7% × 89,8% (seção 16) |
| médias das ativações em 4 bits | 95,2% × 95,5% sem normalização; 91,3% × 94,3% com normalização no caso difícil |
| normalização inteira entre camadas | caso difícil de 50,8% para 87,7–94,3% (seção 18) |
| sem variância nas camadas escondidas + recíproco | mesma qualidade, −25% de tempo (seção 19) |
| kernels próprios, conferência bit a bit | ~34x a versão PyTorch; toda variante conferida contra a especificação int64 |
| pesos em 2 bits na multiplicação de matriz (fase 2) | 1,5–1,6x o int8, exato |

### O que não funciona (medido)
| item | resultado |
|---|---|
| Hebb, flip ao acaso, Madaline (crédito exato) | acaso a ~14% (seções 13, 15) |
| contador de 4 bits | −24 pontos (seção 16.1) |
| passo diferente por camada | não supera o passo único (seção 18) |
| pressão de observação | piora: 90 → 80 → 69% (seção 18) |
| colapso progressivo dos pesos | < 1% dos pesos ficam "certos" (seção 15) |
| GLU binário (média de campo do produto) | não aprende, nem em float (seção 19) |
| dois pesos por lane do dp4a (SWAR) | sem ganho (seção 9) |

### Quanto é binário (aluno 8 × 512, lote 1024: 2,0 M de pesos)
- **Rede final (inferência): 100% dos valores são bits** (pesos ±1, ativações ±1; 0,25 MB de pesos empacotados).
  Ressalva: hoje esses bits ainda são guardados em int8 e multiplicados por dp4a; empacotar em 1 bit e usar XNOR +
  popcount na inferência ainda não foi feito.
- **Treino (bytes gravados na memória num passo):** int8 34% (contadores 2,0 MB, médias das ativações 4,5 MB,
  gradientes quantizados e transpostas 1,6 MB); **int32 65%** (gradiente G de cada peso guardado até o update 8,0 MB;
  somas M e gradientes dM/dmu de cada camada 6,3 MB; saída 1,1 MB); int64 < 1% (recíproco da normalização).
- **Isso viola a regra do projeto** ("int32 só em registrador"): G, M, dM e dmu passam pela memória entre kernels.
  Correção: **fusão de kernels** — o update aplicado na saída da multiplicação que calcula G (G nunca é gravado), a
  normalização e a tanh na saída da multiplicação que calcula M, a derivada da tanh na saída da que calcula dmu. Com
  isso o treino grava só int8 (contadores, médias, gradientes quantizados), e o int32 fica em registrador. Também é a
  maior alavanca de velocidade que resta.
- **Contas:** todas as multiplicações de matriz são int8 × int8 (dp4a) com soma int32 em registrador; as funções
  (tanh por shift + polinômio, normalização por recíproco) usam int32/int64 em registrador.

## 21. Modelo de linguagem binário completo + sistema de logs por etapa (2026-11-03) — `modelo_bit.py`

### 21.1 O que foi implementado (tudo em kernels da `libbit`, conferido bit a bit)
Modelo de bytes (V = 256), todas as dimensões como parâmetro (L blocos, D, F, lote, contexto):
- **embedding binário:** contador int8 por bit, `x0 = E[token]`; gradiente = soma inteira (index_add) por token.
- **mistura temporal tipo RWKV:** `v = bits(Wv x)`; soma com decaimento por canal `S_t = S_{t−1} − S_{t−1}·2^−d_c + v_t`,
  `d_c = 1 + (c & 7)` (memórias de ~2 a ~256 tokens; kernels `k_scan_fwd/back`, 1 thread por sequência×canal);
  `s = bits(S)`; saída `Wo s`.
- **mistura de canais:** `h = bits(Wu xa)`; saída `Wd h`.
- **residual binário ("voto"):** o bit novo é `sinal(M/rms · 2^−a + bit anterior)`. No treino, média de campo (21.3).
- **saída:** 256 classes com média e variância exatas + **64 canais constantes (+1)** como viés binário.
- **rede final 100% binária** (`logits_binarios`): pesos = sinal(contador), ativações = bits.
- **especificação int64** (`ref_passo`) com os mesmos nomes de etapa; `python modelo_bit.py conferir` roda um passo
  completo (2 blocos, D = 128, F = 256) e compara **26 etapas + todos os contadores** → bate bit a bit (com e sem a
  variância no voto).

### 21.2 Sistema de logs (`Log` em `modelo_bit.py`, saída `logs/<nome>.jsonl`)
- cada etapa roda num trecho nomeado (`backward / bloco 2 / temporal`); erro do CUDA em qualquer kernel vira exceção
  com o kernel e o trecho;
- **sentinelas:** médias mortas (|média| < 3), saturadas (> 98% com |média| ≥ 120), **canais constantes entre tokens**
  (colapso), somas int32 ≥ 2^30 (kernel `k_vigia32`, com linha e coluna), gradiente zerado, contadores no limite ±127;
- modo conferir: a primeira etapa que diferir da especificação é reportada com nome, posição e os dois valores.
- Rodam nos 3 primeiros passos e a cada 500 (custo zero no resto).

### 21.3 O que os logs encontraram (cada um corrigido ou medido)
| # | onde o log apontou | causa | correção |
|---|---|---|---|
| 1 | `backward / bloco 0 / temporal`, dMv ≥ 2^30 (linha 7, col. 60) | a soma reversa dos canais lentos amplia o gradiente até 256× | dS quantizado em int8 antes da soma reversa |
| 2 | `backward / bloco 1 / canais`, soma dxa ≥ 2^30 | o caminho direto do voto cresce ~48× por bloco | o gradiente do fluxo residual volta a int8 após cada soma (a escala é comum aos dois caminhos, então é livre) |
| 3 | (antes do teste, na dedução) | o caminho direto do voto saía 2^17 vezes mais fraco que o do bloco | soma com shift com sinal `Σ shifts − 17` (−27 na temporal, que tem 3 quantizações) |
| 4 | saturação crescente até 99–100% e 100% dos contadores em ±127 | sem viés, o modelo só representa a frequência dos bytes empurrando tudo para constante | 64 canais constantes na saída (viés binário) |
| 5 | contadores em ±127 em 500 passos | o update normaliza pelo maior \|G\|: o maior peso anda 16 por passo, sempre (lr efetiva alta demais para texto) | passo 6 (máx. 4 por passo) |
| 6 | voto residual 98–100% saturado já na inicialização | sem variância, `tanh(mean)` com bit certo e bloco incerto dá derivada ~0 | peso do voto `2^−a` e **voto com variância** (`k_voto_var`: `u = (M + x·rms·2^a)/σ`, σ da variância exata como na saída) |
| 7 | canais v/h/s constantes entre tokens (52–93%) | colapso de posto 1: o componente comum a todos os tokens no gradiente arrasta as linhas dos pesos juntas | **aberto** |

### 21.4 Resultados (wiki PT+EN, bytes, lote 8 × 128, D = 256, F = 1024, 2000 passos = 2 M de bytes)
Régua: unigrama de bytes = 3,301 nats/byte.
| modelo | loss val (rede de médias) | acurácia (médias) | acurácia (rede 100% binária) |
|---|---|---|---|
| L = 0 (só embedding → saída = bigrama) | 2,846 | 22,4% | 11,3% |
| **L = 1** | **2,740** | **27,0%** | 10,3% |
| L = 2 | 2,813 | 25,5% | 9,2% |
| L = 4 (voto com variância, a = 2) | 2,935 | 22,2% | 11,6% |
| L = 4 sem variância, a = 3, passo 6 | 2,894 | 22,4% | 9,5% |
| L = 4 primeira versão (passo 4, sem viés, a = 0) | 3,48 | 14,7% | 0,8% |
Velocidade: L = 4 ≈ 58–62 mil tokens/s; L = 1 ≈ 188 mil; L = 0 ≈ 720 mil (GTX 1650, kernels sem fusão).

**Leitura honesta:**
1. O pipeline inteiro funciona e é exato; o modelo aprende abaixo do unigrama e **um bloco usa contexto** (L = 1 bate
   o bigrama: 2,74 × 2,85).
2. **A profundidade piora** (L = 2, 4 pior que L = 1): o colapso de posto 1 (achado 7) é o próximo problema.
3. **A rede 100% binária ainda é pior que o unigrama** (~10% de acurácia contra 27% da rede de médias): em 2000 passos
   os contadores não se polarizaram, e sinal(contador incerto) é ruído. No professor-aluno (10 000 passos) isso fechava.
   Fechar essa diferença é condição para o projeto: a rede de médias é, na prática, uma rede de pesos reais.
4. Ainda não comparado com INT8/float no mesmo número de passos (o treino INT8 de referência usa outro lote/passos).

### 21.5 Próximos passos (em ordem)
1. Colapso de posto 1: centrar por canal entre tokens (média móvel guardada para a inferência) ou retirar do
   gradiente o componente comum (média de dq por coluna).
2. Diferença médias × binária: treino mais longo + medir a fração de contadores "certos" por tensor (sentinela nova);
   avaliar o voto/saída binária com a mesma normalização da rede de médias.
3. Depois: fusão de kernels (int32 só em registrador), XNOR+popcount na inferência, geração, checkpoint.

## 22. Diagnóstico float + pesos fatorados (ideia da tabela linha x coluna), 2026-11-03

**Réplica float** (`diag_float_modelo.py`, só diagnóstico, Adam, 3000 passos, L4 D256 F1024):

| | inteiros (método) | float `norm` | float `var` |
|---|---|---|---|
| loss val, rede de médias | 2,650 | 2,090 | 3,35 (travou no unigrama) |
| acurácia da rede binária | 8,1% | 6,8% | 5,8% |
| pesos certos (abs(w) > 0,8) | poucos | 1% | 0% |

Conclusão: os inteiros custam ~0,56 nat, mas o problema principal é o método. A rede de médias aprende, só que os pesos não
polarizam, e a rede binária (sinal dos pesos) não é a rede que foi treinada. Variância em todas as camadas não resolveu.

**Pesos fatorados** (`JB_POSTO=R`): W[i][j] = soma_r A[i][r]·B[j][r]. Só os bits de linha A e de coluna B existem e são
treinados. A média de W = rnd(A·B^T, 7 + bitlen(R)//2). dA = G·B e dB = G^T·A (cada bit recebe a soma da linha ou coluna
de G). Rede binária: W = sinal(A)·sinal(B)^T, inteiro em [-R, R]; na inferência são 2 produtos binários de posto R.
Composição e gradientes conferidos bit a bit com int64 (posto < 64 é completado com zeros por causa do GEMM).

| modelo | bits totais | passo | loss val (médias) | acurácia binária |
|---|---|---|---|---|
| denso (atual) | 2,77 M | 2000 | 2,661 | 8,5% |
| R = 16 | 0,38 M | 2000 | 2,697 | 8,1% |
| R = 64 | 1,06 M | 1500 | 2,696 (denso no 1500: 2,684) | 11,6% |
| R = 128 | 1,98 M | 2200 | 2,674 | 8,7% |

Os treinos foram cortados pelo timeout de 1500 s, porque a avaliação a cada 100 passos é lenta. Com R = 16, as matrizes
dos blocos usam 11x menos bits (0,23 M contra 2,62 M) e perdem só ~0,04 nat. A fatoração não resolve o problema da rede
binária. Velocidade: 28-33k tokens/s contra 51k (composição e gradientes dos fatores ainda não fundidos).

### 22.1 Quantos bits por peso preservam o aprendizado (`precisao_bits.py`, L4 D256 F1024, 3000 passos)
A mesma rede treinada, avaliada com o contador θ reduzido a 2^k níveis igualmente espaçados em [-127, 127]:

| bits por peso | loss (ativações suaves) | acurácia (ativações suaves) | acurácia (ativações em bits) |
|---|---|---|---|
| 1 (só o sinal) | 10,52 | 8,1% | 8,1% |
| 2 | 2,812 | 21,9% | 15,3% |
| 3 | 2,696 | 22,3% | 16,1% |
| 4 | 2,653 | 23,5% | 16,6% |
| 8 (o contador) | 2,650 | 23,9% | 16,4% |

Contadores no fim: |θ| médio 47,8 de 127. Conclusão: a perda está quase toda na passagem de 2 para 1 bit por peso.
Com 2 bits (4 níveis) a rede guarda quase tudo, sem retreino. A segunda perda (~7 pontos) vem das ativações em bits.

### 22.2 Pesos x ativações em k bits (todas as variantes, 2026-11-03)
Acurácia de validação. Ativações "suaves" = médias int8 do treino; ativações de k bits = a média reduzida a 2^k níveis.
Linha = bits por peso; colunas = ativações suaves | ativações com 1 / 2 / 3 / 4 bits.

| modelo (pesos) | 1 bit | 2 bits | 3 bits | 4 bits | 8 bits (contador) |
|---|---|---|---|---|---|
| denso 3k passos (2,77 M) | 8,1 \| 8,1/8,2/8,1/8,1 | 21,9 \| 15,3/**18,2**/17,9/18,3 | 22,3 \| 16,1/20,0/21,3/20,4 | 23,5 \| 16,6/21,1/20,7/21,0 | 23,9 \| 16,4/20,0/20,4/20,9 |
| denso 10k passos | 5,6 \| ... | 20,0 \| 10,1/17,5/15,2/16,5 | 23,0 \| 10,3/17,0/17,5/17,6 | 23,0 \| 9,4/17,9/19,1/18,3 | 23,7 \| 9,5/17,9/18,7/18,3 |
| denso D512 F2048 (10,8 M) | 7,8 \| ... | 21,3 \| 11,4/14,2/13,7/14,7 | 22,8 \| 12,6/14,5/16,5/15,2 | 22,6 \| 12,0/15,0/16,9/18,1 | 23,1 \| 11,9/14,6/17,7/17,8 |
| fatorado R16 (0,38 M) | 8,9 \| ... | 20,5 \| 12,0/13,1/12,2/13,9 | 23,4 \| 12,3/15,0/17,4/16,1 | 23,1 \| 12,7/16,1/16,7/17,3 | 24,1 \| 12,9/15,6/18,4/17,8 |
| fatorado R64 (1,06 M) | 8,7 \| ... | 23,1 \| 14,7/18,0/16,4/17,7 | 24,4 \| 14,1/17,8/19,7/19,9 | 25,2 \| 15,8/18,7/20,8/20,5 | 24,7 \| 15,8/18,1/20,3/20,5 |

Loss da rede de médias (8 bits): denso 3k 2,650; denso 10k 2,645; D512 2,668; R16 2,671; R64 2,628.

Conclusões:
1. 1 bit por peso destrói a rede em todas as variantes; 2 bits recuperam quase tudo; a partir de 3-4 bits não há diferença.
2. Ativações de 2 bits recuperam a maior parte da perda das ativações de 1 bit (denso: 15,3% -> 18,2% com pesos de 2 bits).
3. Treinar mais (10k passos) ou alargar (D512) NÃO melhora a rede de médias (~2,65): o teto é o método de treino, não a
   capacidade nem os bits. A réplica float da mesma arquitetura chega a 2,09.
4. Treinar mais piora a versão de 1 bit/ativação de 1 bit (a rede de médias depende cada vez mais de nuances).

### 22.3 Escada float -> método (onde se perde qualidade), 2000 passos, loss val
Adam 2,205 | g (gradientes int8 por tensor, estocástico) 2,209 | u (nossa regra de update: G/max·3/127) 2,348 |
ug 2,332 | ugc (+ contador na grade 1/127) 2,355 | modelo inteiro real ~2,66.
Conclusão: os gradientes em int8 não custam nada; a regra de update custa ~0,14; o resto (~0,3) está na aritmética
inteira do forward/backward (ativações int8 Q7, tanh/normalização inteiras, soma temporal, renormalizações de 12/18
bits), que a escada ainda não isolou. Próximo degrau: ativações em int8 na réplica float.

### 22.4 Kernels novos (libbit.cu)
- `bt_empacota` + `bt_gemm_planos<PA,PB>`: pesos e ativações de k bits guardados em k planos de bits; o produto é
  soma_pq 2^(p+q) (K - 2 popc(A_p xor B_q)), só XOR e popcount. Exato para todos os pares 1..4 bits (`teste_planos.py`).
  Velocidade contra o GEMM int8 dp4a (GTX 1650, `logs/teste_planos.log`):

  | forma (tokens x saída x entrada) | int8 | 1 bit | 2 bits | 3 bits |
  |---|---|---|---|---|
  | 1024x1024x256 | 0,124 ms | 2,60x | 1,04x | 0,48x |
  | 1024x4096x4096 | 5,41 ms | 3,49x | 1,01x | 0,45x |
  | 8192x4096x4096 | 40,9 ms | 3,49x | 0,96x | 0,43x |

  Com 2 bits: mesma velocidade do int8 e 4x menos memória de pesos. Com 3 bits: 2,7x menos memória, mas metade da
  velocidade, porque são 9 pares de planos.
- `bt_compor`, `bt_grad_fatores`: tabela fatorada sem completar o posto com zeros (qualquer R). Bit a bit com int64
  (`conferir_fatores.py`, R = 16, 64 e 100). Treino: R16 28k -> 38,7k tokens/s; R64 ~30k (igual ao caminho GEMM,
  dentro do ruído). O modelo denso faz 51k.

### 22.5 Escada completa (2000 passos, loss val; cada letra somada a ugc = 2,355)
| + a (ativ. int8) | + s (soma inteira) | + d (dS 12 bits) | + r (rms e sigma constantes) | + R (só rms) | + S (só sigma) | + H (só rms escondidas) | + V (só rms do voto) |
|---|---|---|---|---|---|---|---|
| 2,333 | 2,340 | 2,343 | 2,626 | 2,648 | 2,318 | 2,366 | **2,629** |

Culpado isolado: tratar o rms do voto (o termo `ant · rms(M) · 2^A`) como constante no backward custa ~0,27.
Ativações int8, soma inteira, dS de 12 bits, sigma constante e rms das escondidas constante não custam nada.
Nota (FAT32): `escada_ugcR/S.log` sobrescreveram `escada_ugcr/s.log` (nomes que só diferem em maiúscula); os valores
de r (2,626) e S (2,340, rodada antiga) vêm da saída do terminal. Logs novos: `escada_ugc_ocultas`, `escada_ugc_voto`.

### 22.6 Correções inteiras (L4 D256 F1024, 3000 passos, shift 6, loss val da rede de médias)
- `JB_PROJ=1`: derivada completa do rms-norm nas escondidas (projeção, int128). Verificada (cos 0,237 -> 1e-5). 2,658.
  Sem ganho, como a escada previa (H inofensivo).
- `JB_VRMS=1`: derivada do rms do voto (`bt_voto_rms_back`), bit a bit com a referência. **2,627**.
- `JB_VRMS=1 JB_PROJ=1`: **2,604** (acurácia 25,9%; rede binária 7,7%).
- Base sem correções: 2,650. Ganho real ~0,05, bem menor que os ~0,27 da escada float. A lacuna até ~2,35 continua:
  a soma de degraus pequenos (a+s+d, cada um ~0) não explica; falta testar na réplica float todos juntos (ugcasdS) e
  a regra de update inteira (o degrau u, ~0,14).
- Réplica float com TODAS as peças juntas (`JF_ESCADA=ugcasdS`, `logs/escada_ugc_todos.log`): **2,312** em 2000 passos
  (acurácia 33,7%). Ou seja, nenhuma das peças modeladas, nem somadas, explica o 2,60 do modelo inteiro. O que resta
  está no que a réplica não modela: tanh/normalização em tabela inteira, sv = isqrt(K·16384 - S2) inteiro, as
  renormalizações de 12/18 bits do backward, a perda/softmax inteira e o tamanho real do passo (shift 6 ≈ 2/127 contra
  3/127 da réplica). Próximo: comparar camada a camada a réplica float e o modelo inteiro no mesmo lote e mesmos pesos.

### 22.7 Comparação camada a camada e a saturação do tanh (`comparar_camadas.py`, 2026-11-04)
- Forward inteiro = réplica float (cos >= 0,97 em todas as etapas; loss 2,627 x 2,624 no passo 1000).
- Backward: o gradiente chegava ao embedding com cos 0,57 do exato; no último bloco o voto devolvia cos 0,006.
  Causa: o tanh inteiro vira 127 já em ~2,8 e a derivada 16384 - 127^2 fica num piso de 1,6% (a verdadeira é ~0).
  O piso empurrava os contadores até ±127 (bloco 3: Wd 99%, Wo 74%), o voto saturava 100% e o gradiente virava ruído.
- Correção `JB_SAT=3584`: derivada zero onde |entrada do tanh| > 3,5 (por ora em torch inteiro, falta kernel e conferir).
  Passo 1000: cos no embedding 0,57 -> 0,84; voto do último bloco 0,006 -> 0,90.
- Treino (VRMS=1 PROJ=1 SAT=3584, 3000 passos): rede de médias **2,212** (acurácia 36,1%), contra 2,604 sem a correção
  e 2,31 da réplica float (2000 passos). A lacuna inteiro x float FECHOU.
- MAS a rede 100% binária ficou em 7,2%: a perda agora está toda na passagem média -> sinal (1 bit por peso/ativação).

### 22.8 Rede de pulsos (`pulsos.py`, logs/pulsos_3000.log): nuance no tempo em vez de bits por valor
Modelo SAT/VRMS/PROJ de 3000 passos (rede de médias 2,212 / 36,1%). Cada passada 100% binária, bits sorteados com
P(+1) = (m + 127)/254, logits somados em N passadas (acurácia val | loss calibrada):

| pesos / ativações | N=1 | N=4 | N=16 | N=64 | N=256 |
|---|---|---|---|---|---|
| sinal / sinal | 7,2% 4,15 | - | - | - | - |
| sinal / sorteio | 7,1% | 7,4% | 7,4% | 7,4% | 7,6% 4,13 |
| sorteio / sinal | 4,8% | 9,0% | 12,0% | 14,0% | 15,7% 3,27 |
| sorteio / sorteio | 5,4% | 7,8% | 11,9% | 13,6% | 15,4% 3,27 |

Leitura: a nuance perdida está nos PESOS (|θ| médio ~34 de 127: a maioria dos pesos é "incerta", ~63% de chance
de +1); sortear ativações não ajuda nada. Sortear pesos ajuda, mas devagar (15,7% com 256 passadas, contra 22% de
pesos de 2 bits numa passada só): a rede de médias não é o valor esperado da rede sorteada (as não-linearidades em
cadeia não comutam com a média). Pulsos como inferência pós-treino NÃO tornam o binário viável.

### 22.9 Ternário {-1, 0, +1} (`ternario.py`, `treino_ternario.py`, 2026-11-04)
Pós-treino (modelo SAT de 3000 passos; |θ| <= τw vira 0): melhor τw = 48 (65% de zeros): rede de médias com pesos
ternários 21,9% (loss 3,14); rede final com ativações ternárias ~15%; binária pura 7,2%; rede de médias 36,1%.
Treino com puxão explícito ao nível ternário (sem STE; θ -= sround((θ - alvo)·r/256), r crescente):
- τ48, r até 32, a partir de 50%: médias caem de 32% para 23,7%; ternária final 19,0% (médias) / 12,6% (ativ. tern.).
- τ32, r até 64, a partir de 30%: 20,3%; ternária final 18,8% / 12,2%.
O puxão briga com o gradiente (que é o da rede de médias, não o da ternária): só ~45% dos contadores polarizam e a
rede final fica PIOR que o ternário pós-treino. Conclusão: sem o treino enxergar a rede que vai rodar, nenhuma
quantização (1 bit, ternário, pulsos) chega perto da rede de médias neste tamanho (2,8 M pesos).
