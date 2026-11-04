"""Kernels de k bits (planos de bits, XOR + popcount) e kernels fatorados: exatidão e velocidade.

Exatidão: o GEMM de planos tem de dar exatamente códigos(A) . códigos(B)^T (inteiros), para todos os pares de bits.
Velocidade: GEMM de planos (k bits por peso e por ativação) contra o GEMM int8 dp4a (bt_gemm) nas mesmas formas.
Uso: python teste_planos.py
"""
import time
import torch
from fase1c_kernel import gemm, lib, P, I, I8, I32, DEV
from modelo_bit import niveis

U32 = torch.int32                      # os planos são guardados como int32 (os bits são os mesmos de um uint32)


def empacota(x, k):
    R, K = x.shape
    out = torch.empty(k, R, K // 32, dtype=U32, device=DEV)
    assert lib.bt_empacota(P(x), P(out), I(R), I(K), I(k)) == 0
    return out


def gemm_planos(Ap, Bp, M, N, K):
    C = torch.empty(M, N, dtype=I32, device=DEV)
    assert lib.bt_gemm_planos(P(Ap), P(Bp), P(C), I(M), I(N), I(K), I(Ap.shape[0]), I(Bp.shape[0])) == 0
    return C


def codigo(x, k):
    n = (1 << k) - 1
    j = torch.div((x.to(torch.int64) + 127) * n + 127, 254, rounding_mode="floor")
    return 2 * j - n


def cronometra(f, rep=50):
    f(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(rep):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / rep


def main():
    g = torch.Generator(device=DEV).manual_seed(7)
    sorteia = lambda r, c: torch.randint(-127, 128, (r, c), generator=g, device=DEV).to(I8).contiguous()
    print("== exatidão (planos x produto de códigos em int64)")
    for ka, kb in ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (4, 4)):
        M, N, K = 192, 320, 320
        a, b = niveis(sorteia(M, K), ka), niveis(sorteia(N, K), kb)
        C = gemm_planos(empacota(a, ka), empacota(b, kb), M, N, K)
        ref = codigo(a, ka).cpu() @ codigo(b, kb).cpu().t()
        print(f"  ativações {ka} bit(s) x pesos {kb} bit(s): {'EXATO' if torch.equal(C.cpu().to(torch.int64), ref) else 'DIFERENTE'}")
    print("== velocidade (ms por produto; tokens x saída x entrada)")
    for M, N, K in ((1024, 256, 256), (1024, 1024, 256), (1024, 4096, 4096), (8192, 4096, 4096)):
        a, b = sorteia(M, K), sorteia(N, K)
        t8 = cronometra(lambda: gemm(a, b))
        linha = f"  {M}x{N}x{K}: int8 dp4a {t8 * 1e3:7.3f} ms"
        for k in (1, 2, 3):
            Ap, Bp = empacota(niveis(a, k), k), empacota(niveis(b, k), k)
            tk = cronometra(lambda: gemm_planos(Ap, Bp, M, N, K))
            linha += f" | {k} bit(s) {tk * 1e3:7.3f} ms ({t8 / tk:4.2f}x)"
        print(linha, flush=True)
    print("== memória dos pesos por 1M pesos: int8 1,00 MB | 3 bits 0,375 MB | 2 bits 0,25 MB | 1 bit 0,125 MB")


if __name__ == "__main__":
    main()
