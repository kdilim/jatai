# Kernel de multiplicação de matriz INT8 com expoente por bloco (dp4a), escrito do zero

Experimento 6 da série "treino em inteiros puros" do projeto Jataí. Data: 2026-11-03.
GPU: GTX 1650 (Turing TU117, 14 SMs, sem tensor cores), CUDA 12.0, `-arch=sm_75`.

**Pergunta:** o formato do Jataí (mantissa INT8 + expoente inteiro por bloco ao longo de K) multiplica
matrizes mais rápido que FP32/FP16 nesta GPU? Critério do projeto: mais de 2x, mirando o pico do `__dp4a`.

## Tetos da GPU (`pico.cu`, só registradores, sem memória)

Contagem: 1 multiplicação + 1 soma = 2 operações (um `dp4a` = 8 operações).

| unidade | teto medido |
|---|---|
| `__dp4a` (INT8) | **10,06 TOPS** |
| FMA FP32 | 2,52 TFLOPS |
| HFMA2 FP16 (`half2`) | 5,04 TFLOPS |

O dp4a vale 4x o FP32 e 2x o FP16 teórico. É o ganho máximo possível em multiplicação de matriz.
O dp4a mantém ~10 TOPS mesmo com só 8 warps por SM, desde que haja ILP (≥16 somas independentes por thread).

## O kernel (`gemm_rapido.cu`)

`C (M×N) = A (M×K) · B (N×K)ᵀ`, todos no formato: INT8 + expoente INT8 por bloco de `KBLK` valores ao longo de K;
a saída também sai no formato (expoente por grupo de 32 colunas).

- Bloco de 256 threads calcula 128×128 saídas, 8×8 por thread (64 somas independentes).
- Tile de K de 64 bytes por `__syncthreads`, dois buffers (o próximo tile vem da VRAM enquanto calcula).
- A memória compartilhada guarda `[palavra][linha]`; cada leitura de 16 bytes alimenta 32 `dp4a`.
  Colunas de cada thread intercaladas (`tx·4..+3` e `64+tx·4..+3`) para lanes vizinhas lerem bytes contíguos.
- Dentro de um bloco de expoente, os produtos somam **exatos** em int32. No fim do bloco, a soma é alinhada
  no acumulador: `acc += arred(p·2^G / 2^s)`, `s = (RA_i − ea_ib) + (RB_j − eb_jb)`, com RA/RB = maior expoente
  da linha. Arredondamento empate-para-par (sem viés). G = 6 bits de guarda (4 com bloco ≥ 128).
- Saída: maior expoente do grupo de 32 colunas por `__shfl_xor_sync`, arredondamento empate-para-par para INT8.
- Laço principal no SASS: 1024 `IDP.4A` para 64 `LDS.128`, 16 `STS` e ~10 instruções de controle.

Para a comparação ser justa, `gemm_fp32` é o **mesmo kernel** em FP32 (mesmo tile, mesmas leituras, FMA no
lugar de dp4a). O cuBLAS (FP32 `Sgemm`, FP16 `Hgemm`) aparece só como régua do treino tradicional, que é o que o
PyTorch usa. Ele não faz parte do método.

## Precisão (M = N = 128, K = 512, expoentes por bloco variando em 5 valores)

| bloco de K | erro vs conta exata em double | régua: só arredondar a saída exata para INT8 |
|---|---|---|
| 32 | 0,866% | 0,866% |
| 64 | 0,977% | 0,977% |
| 128 | 1,010% | 1,010% |
| 256 | 1,074% | 1,074% |

O kernel erra **exatamente** o arredondamento inevitável da saída: o alinhamento não acrescenta erro.

Custo do bloco maior no treino (simulação do formato completo, `../treino_int8_gpu`, modo `floatq wemag`,
600 passos, loss de validação): float 2,123 · bloco 32: 2,152 · bloco 128: 2,169 · bloco 256: 2,186.

## Velocidade (GPU sem outra carga; TOPS / TFLOPS)

| M × N × K | cuBLAS FP32 | cuBLAS FP16 | FP32 próprio | INT8 b32 | b64 | **b128** | b256 | b128 / FP32 próprio | b128 / cuBLAS FP32 |
|---|---|---|---|---|---|---|---|---|---|
| 1024×256×256 | 1,17 | 0,27 | 0,91 | 1,49 | 1,85 | **2,13** | 2,30 | 2,34x | 1,82x |
| 1024×1024×256 | 1,07 | 0,31 | 1,70 | 3,03 | 3,66 | **4,22** | 4,53 | 2,49x | 3,94x |
| 1024×256×1024 | 1,62 | 0,34 | 1,24 | 2,30 | 3,07 | **3,51** | 3,77 | 2,83x | 2,16x |
| 4096×1024×1024 | 2,56 | 0,37 | 1,93 | 3,47 | 4,55 | **5,20** | 5,59 | 2,69x | 2,03x |
| 8192×4096×1024 | 2,54 | 0,38 | 1,97 | 3,50 | 4,63 | **5,28** | 5,67 | 2,68x | 2,08x |
| 4096×4096×4096 | 1,66 | 0,38 | 1,97 | 3,92 | 5,39 | **6,44** | 7,19 | 3,26x | 3,89x |

O cuBLAS FP16 é lento nesta placa: 0,38 TFLOPS, contra 5,04 de teto do `half2`. Ele não usa o caminho
`half2` aqui.

## Histórico das otimizações (bloco 128, 4096×4096×4096)

| versão | TOPS | % do pico dp4a |
|---|---|---|
| 1ª (`bench_gemm.cu`, tile 64×64, 4×4 por thread, expoentes com conflito de banco) | ~0,45 (bloco 32) | 4% |
| expoentes transpostos na memória compartilhada | ~1,3 (bloco 32) | 13% |
| tile 128×128, 8×8 por thread, leituras de 16 bytes | 5,35 | 53% |
| dois buffers (VRAM sobreposta ao cálculo) | 5,71 | 57% |
| tile de K de 64 bytes por sincronização | 5,61 | 56% (sem ganho) |
| colunas intercaladas (sem conflito de banco na leitura de B) | **6,44** | **64%** |

## Conclusões até aqui

1. **Passa no critério de 2x:** o INT8 bloco-128 é 2,3–3,3x mais rápido que o mesmo kernel em FP32 e
   1,8–3,9x mais rápido que o cuBLAS FP32, com erro igual ao piso do formato.
2. **Ainda falta para o pico do dp4a:** 21–64% do pico, conforme o tamanho. Formas pequenas, como
   1024×256 (16 blocos para 14 SMs), deixam a GPU ociosa. Precisam de tiles menores ou de mais tokens
   por passo.
3. Bloco de expoente maior ao longo de K é mais rápido (menos alinhamentos) e custa pouco no treino
   (bloco 128: +0,017 de loss na simulação).

## Próximos passos

- Encher os 14 SMs nas formas pequenas: tile 64×128 ou mais tokens por passo.
- Reduzir o custo do alinhamento por bloco (64 saídas × ~8 instruções a cada bloco de expoente).
- Kernels do treino inteiro: RMSNorm, atenção com head_dim 4 e softmax inteiro, ReLU², backward e
  update com error feedback, todos fundidos e em INT8 bloco.
- Medir tokens/s do treino completo contra o PyTorch FP32.

## Como reproduzir

```bash
nvcc -O3 -arch=sm_75 pico.cu -o pico && ./pico
nvcc -O3 -arch=sm_75 gemm_rapido.cu -lcublas -o gemm_rapido && ./gemm_rapido
```

```bash
nvcc -O3 -arch=sm_75 bitplano.cu -o bitplano && ./bitplano   # GEMM so com operacoes de bit x dp4a
```

`bench_gemm.cu` é a 1ª versão (ingênua), mantida para registro.

## Experimento: GEMM só com operações de bit (`bitplano.cu`, `resultado_bitplano.txt`)

Cada matriz INT8 vira 8 planos de bits; o produto vira AND + popcount + shift (peso -128 no bit de sinal). Exato
(0 diferenças contra o dp4a), mas 8,6–12x mais lento: são 64 AND+popcount para cada 32 produtos, contra 8 dp4a,
e o popcount roda a 1/4 da taxa nesta GPU. Só empata com o dp4a usando 2 bits por lado e só ganha com 1 bit.
Conclusão: o dp4a já é a multiplicação bit a bit em silício; o bitwise rende nas contas de escala (expoentes,
arredondamento, ALiBi, exp), que é onde o método já o usa. Tabela completa em `../jatai_cuda/README.md`.
