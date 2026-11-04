# Custo de "tocar na VRAM" na GTX 1650

Experimento de apoio à série "treino em inteiros puros" do projeto Jataí.
Data: 2026-11-02. Medido uma vez (3 repetições por ponto em 256 MB, 200 em 512 KB).

## Pergunta

Quanto tempo se perde quando cada etapa de um cálculo grava o resultado na
memória, em vez de manter os valores intermediários dentro do processador e
gravar só o resultado final?

## Método

`bench_vram.cu` aplica uma cadeia de S etapas inteiras a cada byte de um vetor
INT8. Cada etapa é uma requantização típica do modelo:
`x = clip((x·37 + 11) >> 5, −127, 127)`.

- **Fundido:** um kernel lê o dado uma vez, faz as S etapas em registradores e
  grava só o resultado final.
- **Separado:** S kernels; cada um lê da memória, faz 1 etapa e grava.

Os dois produzem resultados idênticos (conferido: 0 diferenças). Rodado com
256 MB (não cabe no cache L2, então vai para a VRAM) e com 512 KB (cabe no L2 de 1 MB).
Também mede a banda real da VRAM e o pico de `__dp4a` só com registradores.

## Resultados (GTX 1650, 14 SMs, L2 1 MB, barramento 128 bits)

- Banda teórica da VRAM informada pelo driver: **192 GB/s** (versão GDDR6).
  Medida lendo e gravando: **121 GB/s**.
- Pico de `__dp4a` só com registradores: **9,98 TOPS INT8**.

**256 MB (VRAM):**

| etapas | fundido (ms) | separado (ms) | separado / fundido |
|---|---|---|---|
| 1 | 4,46 | 4,30 | 1,0× |
| 2 | 4,47 | 8,59 | 1,9× |
| 4 | 4,63 | 17,18 | 3,7× |
| 8 | 5,20 | 34,40 | 6,6× |
| 16 | 9,12 | 68,74 | 7,5× |
| 32 | 18,44 | 137,43 | 7,5× |
| 64 | 36,33 | 274,79 | 7,6× |
| 128 | 71,38 | 549,93 | 7,7× |

**512 KB (cache L2):**

| etapas | fundido (ms) | separado (ms) | separado / fundido |
|---|---|---|---|
| 1 | 0,006 | 0,006 | 1,1× |
| 8 | 0,012 | 0,043 | 3,7× |
| 32 | 0,035 | 0,172 | 4,9× |
| 128 | 0,130 | 0,704 | 5,4× |

## Conclusões

1. **Até ~4–8 etapas por byte, o cálculo é de graça.** O kernel fundido leva o
   mesmo tempo com 1, 2 ou 4 etapas (4,46 → 4,63 ms): a GPU fica esperando a VRAM.
   Só a partir de ~8 etapas o processador vira o limite.
2. **Gravar cada etapa custa uma viagem inteira à VRAM por etapa:** ~4,3 ms por
   etapa nesses 256 MB. Com 8 ou mais etapas o separado fica **6,6–7,7× mais lento**.
   Para operações mais baratas que esta requantização a diferença seria maior, porque
   o fundido continuaria limitado pela memória por mais etapas.
3. **Mesmo dentro do cache L2, separar custa até 5,4×**, por causa da latência do L2
   e do custo de lançar cada kernel (alguns microssegundos). Com tensores pequenos,
   lançar kernels demais também vira gargalo.

## Consequências para o projeto

- Fundir cada bloco do modelo num kernel só (RMSNorm → projeção → requantização →
  LUT → requantização), gravando apenas o que o backward precisa.
- **Recomputar no backward sai quase de graça** neste regime: se a GPU tem folga de
  ~4–8 etapas por byte lido, recalcular ativações custa menos que gravá-las e lê-las de volta.
- Uma LUT pequena (até algumas centenas de entradas) cabe em registradores ou memória
  compartilhada e não gera tráfego de VRAM; um polinômio inteiro (como no I-BERT) não
  acessa memória nenhuma.

## Como reproduzir

```bash
nvcc -O3 -arch=sm_75 bench_vram.cu -o bench_vram
./bench_vram
```

Saída completa desta medição em `resultado_bench_vram.txt`.
Ambiente: driver 580.178.04, CUDA 12.x (`nvcc` do sistema).
