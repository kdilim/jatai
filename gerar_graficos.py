"""Gera as figuras do README (SVG, versão clara e escura de cada uma) a partir dos logs dos experimentos.

Uso: python3 gerar_graficos.py   (só matplotlib; lê os logs de jatai_cuda/logs)
Paleta: azul / laranja / água (slots 1-3 da paleta validada para daltonismo nos dois modos); cinza para referências.
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
CUDA = os.path.join(AQUI, "jatai_cuda", "logs")
FIG = os.path.join(AQUI, "figuras")

MODOS = {
    "claro": dict(fundo="#ffffff", texto="#1f2328", texto2="#59636e", grade="#d1d9e0", ref="#8c959f",
                  s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "escuro": dict(fundo="#0d1117", texto="#f0f6fc", texto2="#9198a1", grade="#30363d", ref="#6e7681",
                   s=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}


def eixo(ax, c, titulo, xl, yl):
    ax.set_facecolor(c["fundo"])
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(c["grade"])
    ax.tick_params(colors=c["texto2"], labelsize=9, length=0)
    ax.grid(True, color=c["grade"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title(titulo, loc="left", color=c["texto"], fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xl, color=c["texto2"], fontsize=9)
    ax.set_ylabel(yl, color=c["texto2"], fontsize=9)


def salva(fig, nome, modo):
    fig.savefig(os.path.join(FIG, f"{nome}_{modo}.svg"), facecolor=fig.get_facecolor(), bbox_inches="tight")
    if os.environ.get("PREVIA"):
        fig.savefig(os.path.join(os.environ["PREVIA"], f"{nome}_{modo}.png"), facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=90)
    plt.close(fig)


def le_csv(nome, col="loss_val"):
    with open(os.path.join(CUDA, nome)) as f:
        r = list(csv.DictReader(f))
    return [int(x["passo"]) for x in r if int(x["passo"]) > 0], [float(x[col]) for x in r if int(x["passo"]) > 0]


def barras(nome, titulo, rot, val, fmt, xl, destaque=None, ref=None):
    for modo, c in MODOS.items():
        fig, ax = plt.subplots(figsize=(8, 0.55 * len(rot) + 1.2), facecolor=c["fundo"])
        eixo(ax, c, "", xl, "")
        fig.subplots_adjust(left=0.34)
        fig.suptitle(titulo, x=0.01, ha="left", color=c["texto"], fontsize=12, fontweight="bold")
        cores = [c["s"][0] if (destaque is None or i in destaque) else c["ref"] for i in range(len(rot))]
        y = list(range(len(rot)))[::-1]
        ax.barh(y, val, color=cores, height=0.62, edgecolor=c["fundo"], linewidth=2)
        ax.set_yticks(y, rot, color=c["texto"], fontsize=9.5)
        ax.grid(axis="y", visible=False)
        m = max(val)
        for yi, v in zip(y, val):
            ax.text(v + m * 0.01, yi, fmt(v), va="center", color=c["texto"], fontsize=9)
        if ref is not None:
            ax.axvline(ref[0], color=c["texto2"], linewidth=1, linestyle=(0, (3, 3)))
            ax.text(ref[0], len(rot) - 0.45, ref[1], color=c["texto2"], fontsize=8.5, ha="center")
        ax.set_xlim(0, m * 1.15)
        salva(fig, nome, modo)


def linhas(nome, titulo, series, xl, yl, logx=False, ylim=None, rotulos_fim=True, ref=None, fmt=lambda v: f"{v:.2f}".replace(".", ",")):
    estilos = ["-", (0, (5, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2))]
    for modo, c in MODOS.items():
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=c["fundo"])
        eixo(ax, c, titulo, xl, yl)
        for i, (rot, xs, ys) in enumerate(series):
            ax.plot(xs, ys, color=c["s"][i], linewidth=2, linestyle=estilos[i], label=rot, marker="o" if len(xs) < 12 else None,
                    markersize=5)
            if rotulos_fim:
                ax.annotate(fmt(ys[-1]), (xs[-1], ys[-1]), xytext=(6, 0),
                            textcoords="offset points", va="center", color=c["texto"], fontsize=9)
        if logx:
            ax.set_xscale("log", base=2)
            ax.set_xticks(series[0][1], [str(v) for v in series[0][1]])
        if ylim:
            ax.set_ylim(*ylim)
        if ref is not None:
            ax.axhline(ref[0], color=c["texto2"], linewidth=1, linestyle=(0, (3, 3)))
            ax.annotate(ref[1], (series[0][1][0], ref[0]), xytext=(0, 5), textcoords="offset points", color=c["texto2"], fontsize=8.5)
        if len(series) > 1:                             # uma série só: o título já diz o que é
            leg = ax.legend(frameon=False, fontsize=9, loc="best")
            for t in leg.get_texts():
                t.set_color(c["texto"])
        salva(fig, nome, modo)


def main():
    os.makedirs(FIG, exist_ok=True)
    # 1. velocidade do treino INT8 (12 camadas, lote 2048; RELATORIO 8.1)
    barras("int8_velocidade", "Treino INT8 x float32 na GTX 1650 (tokens/s, maior é melhor)",
           ["float32 (PyTorch)", "INT8, kernels próprios", "INT8 + CUDA Graph"], [8926, 32714, 34248],
           lambda v: f"{v:,.0f}".replace(",", ".") + ("" if v < 9000 else f"  ({v / 8926:.2f}x)".replace(".", ",")),
           "tokens por segundo", destaque=[1, 2])
    # 2. treino longo de 24 camadas
    pf, lf = le_csv("noite_float_L24_ff2048.csv")
    pi, li = le_csv("noite_int8_L24_ff2048.csv")
    pe, le = le_csv("noite_int8_L24_ff2048_resest.csv")
    p16, l16 = le_csv("noite_int8_L24_ff2048_res16.csv")
    lim = 20000
    corta = lambda p, l: ([a for a in p if a <= lim], [b for a, b in zip(p, l) if a <= lim])
    linhas("int8_treino_longo", "Treino longo, 24 camadas: loss de validação (menor é melhor)",
           [("float32", *corta(pf, lf)), ("INT8", pi, li), ("INT8 + arredondamento estocástico no resíduo", pe, le),
            ("INT8 + resíduo de 16 bits (E1)", p16, l16)],
           "passo de treino", "loss de validação (nats/byte)", ylim=(1.3, 4.1))
    # 3. absorção no resíduo
    def absor(arq):
        out = []
        for linha in open(os.path.join(CUDA, arq)):
            m = re.match(r"\s*(\d+) \|\s*[\d.]+ \|\s*[\d.]+\s+([\d.]+)%", linha)
            if m:
                out.append(float(m.group(2)))
        return out
    af, ai = absor("diag_residuo_noite_float_pesos.txt"), absor("diag_residuo_noite_int8_pesos.txt")
    linhas("int8_absorcao", "Quanto da atualização da atenção some ao somar no resíduo INT8",
           [("pesos do float32 (treino bom)", list(range(len(af))), af), ("pesos do INT8 (passo 10.000)", list(range(len(ai))), ai)],
           "camada", "% perdido no arredondamento", fmt=lambda v: f"{v:.0f}%")
    # 4. binário: o treino inteiro alcança o float
    barras("bit_treino", "Rede binária (L4, D256): loss da rede de médias após 3000 passos (menor é melhor)",
           ["inteiro, sem correções", "+ derivada do rms do voto", "+ derivada da normalização",
            "+ derivada zero na saturação do tanh", "réplica float com Adam (referência)"],
           [2.650, 2.627, 2.604, 2.212, 2.090], lambda v: f"{v:.3f}".replace(".", ","), "loss de validação (nats/byte)",
           destaque=[3])
    # 5. binário: a lacuna até a rede que roda
    barras("bit_lacuna", "Rede binária: acurácia no próximo byte da rede que realmente roda (maior é melhor)",
           ["rede de médias (treino, int8)", "pulsos: pesos sorteados, 256 passadas", "ternário {-1,0,+1} pós-treino",
            "ternário com puxão no treino", "binária pura ±1"],
           [36.1, 15.7, 15.3, 12.6, 7.2], lambda v: f"{v:.1f}%".replace(".", ","), "acurácia (%)", destaque=[0])
    # 6. pulsos
    N = [1, 4, 16, 64, 256]
    linhas("bit_pulsos", "Rede de pulsos: acurácia somando N passadas 100% binárias",
           [("pesos sorteados", N, [4.8, 9.0, 12.0, 14.0, 15.7]), ("pesos e ativações sorteados", N, [5.4, 7.8, 11.9, 13.6, 15.4]),
            ("só ativações sorteadas", N, [7.1, 7.4, 7.4, 7.4, 7.6])],
           "passadas (N)", "acurácia (%)", logx=True, ylim=(0, 40), ref=(36.1, "rede de médias: 36,1%"), rotulos_fim=False)
    # 7. precisão progressiva no INT8 de 24 camadas
    linhas("int8_bits_ativacao", "Quantos bits as ativações precisam (modelo de 24 camadas)",
           [("loss com as entradas truncadas a p bits", [1, 2, 3, 4, 5, 6, 7], [3.4364, 2.4419, 1.5441, 1.3663, 1.3408, 1.3396, 1.3399])],
           "bits de magnitude por ativação (p)", "loss de validação", ylim=(1.2, 3.6))


if __name__ == "__main__":
    main()
