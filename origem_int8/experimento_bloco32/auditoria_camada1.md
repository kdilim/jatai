# Auditoria passo a passo — camada 1

Lote 0, token t=5, cabeça 0 (dimensões 0–3, bloco 0). Bloco = 32 valores. Valor real = mantissa × 2^expoente. Cada passo mostra a fórmula, os inteiros, o valor real e a referência em float64 (calculada das mesmas entradas inteiras). Toda conta abaixo foi refeita em Python escalar e conferida contra o cálculo vetorizado.

## 1. Entrada

Expoentes dos 8 blocos do token: [-5, -5, -5, -5, -5, -5, -5, -5]
Mantissas do bloco 0: [26, -2, -19, -70, 6, -28, -5, 27, 33, 29, 90, 47, -30, -33, 93, -27, -2, 70, 69, -2, 10, -65, -36, -15, 0, -13, 6, 88, 19, 12, 0, -7]

## 2. RMSNorm 1

Fórmulas: sq_b = Σ m² no bloco b;  al_b = (sq_b << 8) >> 2·(Emax − E_b);  ss = Σ al_b;  r = isqrt(ss // D)  →  rms real = r·2^(Emax − 4).

- bloco 0: E=-5, sq=55139, deslocamento 2·(-5−-5)=0, al=14115584
- bloco 1: E=-5, sq=56596, deslocamento 2·(-5−-5)=0, al=14488576
- bloco 2: E=-5, sq=65622, deslocamento 2·(-5−-5)=0, al=16799232
- bloco 3: E=-5, sq=61289, deslocamento 2·(-5−-5)=0, al=15689984
- bloco 4: E=-5, sq=46726, deslocamento 2·(-5−-5)=0, al=11961856
- bloco 5: E=-5, sq=61502, deslocamento 2·(-5−-5)=0, al=15744512
- bloco 6: E=-5, sq=30496, deslocamento 2·(-5−-5)=0, al=7806976
- bloco 7: E=-5, sq=42408, deslocamento 2·(-5−-5)=0, al=10856448

Emax=-5;  ss=107463168;  ss//D=419778;  r=isqrt=647;  rms real = 647·2^-9 = 1.26367  (float64: 1.26544)

Recíproco por linha (uma divisão só): k = bits(r) + 10 = 20;  inv = 2^k // r = 1620  (1/r ≈ inv·2^−k).  Cada elemento: produto = m·inv (≤ 18 bits);  expoente = E_b − k − Emax + 4.

- dim 0: m=26, produto = 26·1620 = 42120, expoente -16 → real 0.6427  (float64: 0.642072)
- dim 1: m=-2, produto = -2·1620 = -3240, expoente -16 → real -0.0494385  (float64: -0.0493901)
- dim 2: m=-19, produto = -19·1620 = -30780, expoente -16 → real -0.469666  (float64: -0.469206)
- dim 3: m=-70, produto = -70·1620 = -113400, expoente -16 → real -1.73035  (float64: -1.72865)

Bloco 0 vira INT8 (mantissa em ±127): maior (expoente + bits) do bloco = 2  →  E = -5;  mantissa = produto >> (-5 − (-16))
- dim 0: 21·2^-5 = 0.65625
- dim 1: -2·2^-5 = -0.0625
- dim 2: -15·2^-5 = -0.46875
- dim 3: -55·2^-5 = -1.71875

## 3. Projeção Q (produto escalar de 256 termos em 8 blocos de 32)

Fórmulas: parcial_b = Σ_(k no bloco b) xn_k·W_kn  (exato);  alinhado_b = (parcial_b << 6) >> (Emax − E_b);  acc = Σ alinhado_b;  expoente de acc = Emax + E_W − 6.

Expoentes dos blocos da entrada normalizada: [-5, -5, -5, -5, -5, -5, -6, -5];  Emax=-5;  E_W=-9

- saída 0: parciais→alinhados [-3810→-243840, 1848→118272, 2076→132864, -1903→-121792, 2649→169536, -7548→-483072, -6151→-196832, -19→-1216];  acc=-626080 → real -0.597076  (float64: -0.597076)
- saída 1: parciais→alinhados [-1649→-105536, -2482→-158848, -519→-33216, -2295→-146880, 1836→117504, -1370→-87680, 7051→225632, 6626→424064];  acc=235040 → real 0.224152  (float64: 0.224152)
- saída 2: parciais→alinhados [5908→378112, 2337→149568, -3012→-192768, -6094→-390016, -9758→-624512, 13856→886784, -1081→-34592, -9077→-580928];  acc=-408352 → real -0.389435  (float64: -0.389435)
- saída 3: parciais→alinhados [1314→84096, 5430→347520, -6015→-384960, 7541→482624, -3715→-237760, 23296→1490944, 15043→481376, -5434→-347776];  acc=1916064 → real 1.8273  (float64: 1.8273)

Bloco 0 da saída Q vira INT8 com E=-6:
- q[0] = -38·2^-6 = -0.59375
- q[1] = 14·2^-6 = 0.21875
- q[2] = -25·2^-6 = -0.390625
- q[3] = 117·2^-6 = 1.82812

K da cabeça 0 (mesmo cálculo), tokens 0..5:
- token 0: E=-5, mantissas [-15, 23, 18, 12]
- token 1: E=-5, mantissas [-12, -36, 61, -14]
- token 2: E=-6, mantissas [-51, 59, -63, -52]
- token 3: E=-5, mantissas [-56, 1, 47, -58]
- token 4: E=-6, mantissas [81, -2, 48, -1]
- token 5: E=-5, mantissas [-7, 13, 30, -32]

V da cabeça 0 (mesmo cálculo), tokens 0..5:
- token 0: E=-5, mantissas [2, 20, 14, 35]
- token 1: E=-5, mantissas [-14, -7, 3, -22]
- token 2: E=-5, mantissas [-18, 47, 55, 9]
- token 3: E=-5, mantissas [35, -8, 54, -35]
- token 4: E=-5, mantissas [-3, -3, -11, 28]
- token 5: E=-5, mantissas [24, -64, 36, 14]

## 4. Scores e softmax (cabeça 0, consulta t=5)

Fórmulas: S = Σ_(4 dims) q·k (exato);  es = E_q + E_k − 1 (o −1 é o ÷√4 = ÷2, exato);  u = S·2^(es+8) (score em unidades de 1/256 nat);  ALiBi: u' = u − (t−j)·2^(8−s) (inclinação 2^−s, s=0 nesta cabeça);  d = max(u') − u';  exp(−d/256) por shift + polinômio;  P = exp/Σexp em 0..255.

- j=0: S = -38·-15 + 14·23 + -25·18 + 117·12 = 1846;  es=-12;  u = 1846·2^-4 = 115;  viés = 1280;  u' = -1165
- j=1: S = -38·-12 + 14·-36 + -25·61 + 117·-14 = -3211;  es=-12;  u = -3211·2^-4 = -201;  viés = 1024;  u' = -1225
- j=2: S = -38·-51 + 14·59 + -25·-63 + 117·-52 = -1745;  es=-13;  u = -1745·2^-5 = -55;  viés = 768;  u' = -823
- j=3: S = -38·-56 + 14·1 + -25·47 + 117·-58 = -5819;  es=-12;  u = -5819·2^-4 = -364;  viés = 512;  u' = -876
- j=4: S = -38·81 + 14·-2 + -25·48 + 117·-1 = -4423;  es=-13;  u = -4423·2^-5 = -138;  viés = 256;  u' = -394
- j=5: S = -38·-7 + 14·13 + -25·30 + 117·-32 = -4046;  es=-12;  u = -4046·2^-4 = -253;  viés = 0;  u' = -253

max(u') = -253

- j=0: d = 912
    t = min(d,6000)·23637 = 21556944;  n = t>>22 = 5;  f = (t mod 2^22)>>7 = 4573;  polinômio(f) = 29748;  exp = p>>n = 929  (real 0.028351, float 0.028368)
- j=1: d = 972
    t = min(d,6000)·23637 = 22975164;  n = t>>22 = 5;  f = (t mod 2^22)>>7 = 15653;  polinômio(f) = 23530;  exp = p>>n = 735  (real 0.022430, float 0.022441)
- j=2: d = 570
    t = min(d,6000)·23637 = 13473090;  n = t>>22 = 3;  f = (t mod 2^22)>>7 = 6954;  polinômio(f) = 28287;  exp = p>>n = 3535  (real 0.107880, float 0.107899)
- j=3: d = 623
    t = min(d,6000)·23637 = 14725851;  n = t>>22 = 3;  f = (t mod 2^22)>>7 = 16741;  polinômio(f) = 22995;  exp = p>>n = 2874  (real 0.087708, float 0.087721)
- j=4: d = 141
    t = min(d,6000)·23637 = 3332817;  n = t>>22 = 0;  f = (t mod 2^22)>>7 = 26037;  polinômio(f) = 18892;  exp = p>>n = 18892  (real 0.576538, float 0.576499)
- j=5: d = 0
    t = min(d,6000)·23637 = 0;  n = t>>22 = 0;  f = (t mod 2^22)>>7 = 0;  polinômio(f) = 32765;  exp = p>>n = 32765  (real 0.999908, float 1.000000)

exp em Q12 (exp >> 3): [116, 91, 441, 359, 2361, 4095];  soma = 7463;  inv = 2^28 // soma = 35968;  P = (exp·inv + 2^19) >> 20

- P[0] = 4 → 0.0156  (float64: 0.0156)
- P[1] = 3 → 0.0117  (float64: 0.0123)
- P[2] = 15 → 0.0586  (float64: 0.0593)
- P[3] = 12 → 0.0469  (float64: 0.0482)
- P[4] = 81 → 0.3164  (float64: 0.3159)
- P[5] = 140 → 0.5469  (float64: 0.5487)

## 5. P@V (cabeça 0, token 5)

Fórmulas: V é alinhado ao maior expoente da cabeça entre todos os tokens (Emax_V=-5): v' = (v << 6) >> (Emax_V − E_v(j));  acc = Σ_j P_j·v'_j;  expoente = −8 + Emax_V − 6.

- dim 0: acc = 4·128 + 3·-896 + 15·-1152 + 12·2240 + 81·-192 + 140·1536 = 206912 → 206912·2^-19 = 0.394653  (float64: 0.394653)
- dim 1: acc = 4·1280 + 3·-448 + 15·3008 + 12·-512 + 81·-192 + 140·-4096 = -546240 → -546240·2^-19 = -1.04187  (float64: -1.04187)
- dim 2: acc = 4·896 + 3·192 + 15·3520 + 12·3456 + 81·-704 + 140·2304 = 363968 → 363968·2^-19 = 0.694214  (float64: 0.694214)
- dim 3: acc = 4·2240 + 3·-1408 + 15·576 + 12·-2240 + 81·1792 + 140·896 = 257088 → 257088·2^-19 = 0.490356  (float64: 0.490356)

O bloco 0 da saída (cabeças 0–7) vira INT8 com E=-5: mantissas [13, -33, 22, 16]

## 6. Projeção O e residual 1

Projeção O: mesmo cálculo da seção 3.  Residual: e = min(E_x, E_y);  s = (m_x << (E_x − e)) + (m_y << (E_y − e));  vira INT8 por bloco.

- dim 0: x = 26·2^-5 = 0.8125;  y (saída de O) = -59·2^-6 = -0.921875;  s = -7·2^-6 = -0.109375  →  -3·2^-5 = -0.09375
- dim 1: x = -2·2^-5 = -0.0625;  y (saída de O) = 2·2^-6 = 0.03125;  s = -2·2^-6 = -0.03125  →  -1·2^-5 = -0.03125
- dim 2: x = -19·2^-5 = -0.59375;  y (saída de O) = 8·2^-6 = 0.125;  s = -30·2^-6 = -0.46875  →  -15·2^-5 = -0.46875
- dim 3: x = -70·2^-5 = -2.1875;  y (saída de O) = 36·2^-6 = 0.5625;  s = -104·2^-6 = -1.625  →  -52·2^-5 = -1.625

## 7. RMSNorm 2 e MLP (ReLU²)

RMSNorm 2 (mesmo cálculo da seção 2): r = 358;  saída nas dims 0–3: -2·2^-5 = -0.0625, -1·2^-5 = -0.03125, -11·2^-5 = -0.34375, -37·2^-5 = -1.15625

MLP up (seção 3 com 1024 saídas). Unidades ocultas 0–3: 38·2^-5 = 1.1875, 0·2^-5 = 0, -5·2^-5 = -0.15625, -65·2^-5 = -2.03125

ReLU²: a = max(m, 0)² (exato, no máximo 14 bits);  expoente = 2·E;  depois vira INT8 por bloco.

- oculta 0: m=38, E=-5 → a = 1444·2^-10 → 45·2^-5 = 1.40625  (float64: 1.41016)
- oculta 1: m=0, E=-5 → a = 0·2^-10 → 0·2^-5 = 0  (float64: 0)
- oculta 2: m=-5, E=-5 → a = 0·2^-10 → 0·2^-5 = 0  (float64: 0)
- oculta 3: m=-65, E=-5 → a = 0·2^-10 → 0·2^-5 = 0  (float64: 0)

MLP down (seção 3 com 1024 entradas em 32 blocos) e residual 2 (seção 6):

- dim 0: down = -20·2^-5 = -0.625;  saída da camada = -11·2^-4 = -0.6875  (rede toda em float64: -0.724068)
- dim 1: down = -17·2^-5 = -0.53125;  saída da camada = -9·2^-4 = -0.5625  (rede toda em float64: -0.575932)
- dim 2: down = -26·2^-5 = -0.8125;  saída da camada = -20·2^-4 = -1.25  (rede toda em float64: -1.29266)
- dim 3: down = 3·2^-5 = 0.09375;  saída da camada = -24·2^-4 = -1.5  (rede toda em float64: -1.54)

Todas as contas acima conferem exatamente com o cálculo vetorizado (asserts no código).