"""Confere bit a bit os kernels fatorados (bt_compor, bt_grad_fatores) contra a especificação int64.
Uso: JB_POSTO=<R> python conferir_fatores.py"""
import torch
import fase1c_kernel as fk
from modelo_bit import ModeloBit, Log, POSTO, shift_posto, r_quant, I64, I32, DEV

assert POSTO > 0, "rode com JB_POSTO=R"
log = Log("fatores_conferir"); log.passo = 0
m = ModeloBit(1, 128, 256, 6, 1, log)
b = m.blocos[0]; s = shift_posto(POSTO)
for k in ("Wv", "Wo", "Wu", "Wd"):
    A, B = b[k + "A"].cpu().to(I64), b[k + "B"].cpu().to(I64)
    assert torch.equal(fk.r_rnd(A @ B.t(), s).clamp(-127, 127), b[k].cpu().to(I64)), f"compor {k}"
g = torch.Generator(device=DEV).manual_seed(3)
G = {f"bloco 0 {k}": torch.randint(-(1 << 20), 1 << 20, b[k].shape, generator=g, device=DEV, dtype=I32) for k in ("Wv", "Wo", "Wu", "Wd")}
G0 = {k: v.clone() for k, v in G.items()}
m._gradientes_fatores(G)
for k in ("Wv", "Wo", "Wu", "Wd"):
    gq, _ = r_quant(G0[f"bloco 0 {k}"].cpu().to(I64), f"b0 G{k}")
    A, B = b[k + "A"].cpu().to(I64), b[k + "B"].cpu().to(I64)
    assert torch.equal(gq @ B, G[f"bloco 0 {k}A"].cpu().to(I64)), k + "A"
    assert torch.equal(gq.t() @ A, G[f"bloco 0 {k}B"].cpu().to(I64)), k + "B"
print(f"posto {POSTO}: kernels compor e grad_fatores batem bit a bit com a especificação int64")
