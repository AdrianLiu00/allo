# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import allo
from allo.ir.types import int8, int16, int32, int64, int128, int256, int512, bool
from allo.utils import get_np_struct_type
import allo.backend.hls as hls


from datetime import datetime
now = datetime.now()
date = now.strftime("%d%H%M")


def test_PE_simple():
    from allo.library.systolic_ws import PE_kernel_ws, PE_kernel_os

    np_type = np.int8
    allo_type = int8

    def top[Ty](A: "Ty", B_in: "Ty", C_in: "Ty", rsn:bool) -> "Ty":
        C_out: Ty
        B_out: Ty

        PE_kernel_ws[int8, int8, int8](A, B_in, C_in, B_out, C_out, rsn)
        return C_out

    s_top = allo.customize(top, instantiate=[allo_type])
    print(s_top.module)



def test_gemm_simple():
    from allo.library.systolic import systolic
    from allo.library.systolic_refine import systolic_ws, systolic_os
    from allo.library.systolic_uni import systolic_uni

    # =================================================================
    # A = 128
    # B = 768
    # C = 3072
    # B0 = 16
    # C0 = 32

    # A = 32
    # B = 768
    # C = 64
    # B0 = 16
    # C0 = 32

    # A = 6
    # B = 4
    # C = 8
    # B0 = 2
    # C0 = 4

    M0 = 128
    K0 = 768
    N0 = 3072
    Rt0 = 16
    Ct0 = 16

    # M0 = 8
    # K0 = 2
    # N0 = 4
    # Rt0 = 2
    # Ct0 = 4

    np_type = np.int8
    allo_type = int8

    # =================================================================

    X = np.random.randint(-4, 4, size=(M0, K0)).astype(np_type)
    # X = np.array([[1, 2],
    #              [3, 4]]).astype(np.int8)

    W_A_cst = np.random.randint(-4, 4, size=(K0, N0)).astype(np_type)
    # W_A_cst = np.array([[1, 1, 1, 1],
    #                    [1, 1, 1, 1]]).astype(np.int8)

    # =================================================================
    # def top[Ty](X: "Ty[A, B]", W_A: "Ty[B, C]") -> "Ty[A, C]":
    #     Z: Ty[A, C]
    #     systolic_ws[int8, int8, int8, A, B, C, B0, C0](X, W_A, Z)
    #     # systolic_os[int8, int8, int8, A, B, C, B0, C0](X, W_A, Z)
    #     return Z
    
    def top[Ty](X: "Ty[M0, K0]", W_A: "Ty[K0, N0]") -> "Ty[M0, N0]":
        Z: Ty[M0, N0]
        flowtag: bool = False
        systolic_uni[int8, M0, K0, N0, Rt0, Ct0](X, W_A, Z, flowtag)
        # systolic_ws[int8, int8, int8, M0, K0, N0, Rt0, Ct0](X, W_A, Z)
        # systolic_os[int8, int8, int8, M0, K0, N0, Rt0, Ct0](X, W_A, Z)
        return Z
    
    s_top = allo.customize(top, instantiate=[allo_type])

    # if A < 20:
    #     with open(f'systolic_{date}.mlir', 'w') as f:
    #         print(s_top.module, file=f)
    
    # =================================================================
    # CPU Testing
    
    mod = s_top.build()

    allo_C = mod(X, W_A_cst)
    np_C = X @ W_A_cst

    print(np_C)
    print(allo_C)

    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)
    print("Passed!")

    # =================================================================
    # # HLS Testing
    # s_top.compose(
    #     systolic, instantiate=[int32, int32, int32, L, D, 4 * D, M0, M1]
    # )
    # s_top.dataflow("top")  # important
    # if hls.is_available("vitis_hls"):
    #     hls_mod = s_top.build(
    #         target="vitis_hls",
    #         mode="csim",
    #         project=f"simple_{L}x{D}_tile_{M0}x{M1}_csim.prj",
    #         configs={
    #             "mappings": [
    #                 (
    #                     (L // M0, D, M0),
    #                     f"(d0 * {M0} + d2) * {D} + d1",
    #                     f"d0 * {M0} + d2, d1",
    #                 ),
    #                 (
    #                     (L // M0, 4 * D // M1, D, M1),
    #                     f"d2 * {4 * D} + d1 * {M1} + d3",
    #                     f"d2, d1 * {M1} + d3",  # does not matter a lot in FIFO
    #                 ),
    #                 (
    #                     (L // M0, 4 * D // M1, M1, M0),
    #                     f"d0 * {M0} + d3, d1 * {M1} + d2",  # does not matter a lot in FIFO
    #                     f"(d0 * {M0} + d3) * {4 * D} + d1 * {M1} + d2",
    #                 ),
    #             ]
    #         },
    #     )
    #     # Be careful about the NumPy type
    #     csim_C = np.zeros((L, 4 * D), dtype=np_type)
    #     hls_mod(packed_X, W_A_packed, csim_C)
    #     np.testing.assert_allclose(csim_C, allo_C, atol=1e-3)
    #     print("Passed!")

    return



if __name__ == '__main__':
    # test_PE_simple()
    
    test_gemm_simple()