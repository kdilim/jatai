# jatai_cuda: treino de Transformer 100% inteiro (INT8) em CUDA

Forward, backward e otimizador de um Transformer inteiro em kernels CUDA escritos do zero. Todo tensor guardado é
mantissa INT8 + expoente INT8 por bloco de 128 valores; as contas usam registradores int32 e todo intermediário
fica abaixo de 2^30 por construção. Nenhuma conta de ponto flutuante no treino.

GPU: GTX 1650 (Turing, sem tensor cores). Experimento 7 da série "treino em inteiros puros" do projeto Jataí.

**Documentação completa, com tudo o que foi tentado (o que deu certo, o que não deu e por quê), provas dos
limites de bits, metodologia de medição e limitações: [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md).**

## Resultados

| | float32 (PyTorch) | INT8 (este trabalho) | INT8 + CUDA Graph |
|---|---|---|---|
| tokens/s, lote 1024 tokens | 8.228 | 27.555 (3,35x) | **30.654 (3,73x)** |
| tokens/s, lote 2048 tokens | 8.926 | 32.714 (3,67x) | **34.248 (3,84x)** |
| loss de validação, passo 600 | 2,123 | 2,280 | 2,280 (idêntico) |

Mesma arquitetura (D = 256, 64 cabeças × 4, FF = 1024, 12 camadas, T = 128, vocabulário de bytes, ALiBi,
ReLU²), mesmos pesos iniciais, dados e otimizador (SGD com momento 7/8, lr 2^-4) nos dois lados.

- **Correção:** 53 casos de teste e 1044 de 1044 operações de dois passos de treino batem **bit a bit** com a
  especificação em inteiros de 64 bits (`ops.py`). O mesmo treino de 600 passos dá as mesmas losses até a 5ª casa
  decimal em cinco versões da biblioteca.
- **Logs:** qualquer anomalia numérica (estouro, expoente fora da faixa, divisão por zero, token inválido) vira uma
  mensagem com operação, trecho do modelo, posição e valor, também dentro do CUDA Graph.
- **Em aberto:** a loss fica 0,16 acima do float32 (a simulação do formato previa 0,05); ver seção 9 do relatório.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `jatai_cuda.cu` | todos os kernels e a interface C |
| `ops.py` | especificação (`OpsRef`), chamadas CUDA (`OpsCuda`), conferência bit a bit (`OpsConferido`) |
| `registro.py` | sistema de logs |
| `modelo.py` | o Transformer e o passo gravado em CUDA Graph |
| `treino_cuda.py` | conferir, treinar, medir velocidade |
| `teste_ops.py`, `teste_erro_grafo.py` | testes |
| `bench_formas.py`, `bench_grafo.py` | medições |
| `logs/` | treinos, eventos, erros reais e o código de cada rodada anterior |

## Como reproduzir

```bash
nvcc -O3 -arch=sm_75 -shared -Xcompiler -fPIC jatai_cuda.cu -o libjatai.so
python teste_ops.py                    # 53 casos, bit a bit
python teste_erro_grafo.py             # logs dentro do CUDA Graph
python treino_cuda.py conferir 2       # modelo inteiro conferido
python treino_cuda.py treinar 4 600 8  # treino de 600 passos -> logs/treino_k4_bt8.csv
python treino_cuda.py velocidade 16    # INT8 (normal e CUDA Graph) x float32
```

Python 3.12.3, PyTorch 2.6.0 (só para memória, dados e a referência em float), CUDA 12.0, driver 580.
