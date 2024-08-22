# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import allo
from allo.ir.types import int8, int16, int32, int64, int128, int256, int512, bool
from allo.utils import get_np_struct_type
import allo.backend.hls as hls
from allo.ir.utils import MockBuffer


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


def test_tile():
    from allo.library.systolic_uni import systolic_tile_uni

    # =================================================================

    # M0 = 128
    # K0 = 768
    # N0 = 3072
    # Rt0 = 16
    # Ct0 = 16

    M0 = 16
    K0 = 32
    N0 = 16
    Rt0 = 4
    Ct0 = 4

    np_type = np.int8
    allo_type = int8

    # =================================================================

    s_tile = allo.customize(systolic_tile_uni, instantiate=[int8, Rt0, Ct0])

    tile_name = "systolic_tile_uni"
    pe = s_tile.unfold(f"{tile_name}:PE", [0, 1])
    s_tile.to(MockBuffer(tile_name, "R_buf"), pe, axis=0, depth=Ct0 + 1)
    s_tile.to(MockBuffer(tile_name, "C_buf"), pe, axis=1, depth=Rt0 + 1)

    code = s_tile.build(target="vhls")
    with open(f'tileHLS_{date}.cpp', 'w') as f:
        print(code, file=f)
    



def test_gemm():
    from allo.library.systolic import systolic
    from allo.library.systolic_refine import systolic_ws, systolic_os
    from allo.library.systolic_uni import systolic_uni

    # =================================================================

    # M0 = 128
    # K0 = 768
    # N0 = 3072
    # Rt0 = 16
    # Ct0 = 16

    M0 = 16
    K0 = 32
    N0 = 16
    Rt0 = 4
    Ct0 = 4

    np_type = np.int8
    allo_type = int8

    # =================================================================

    X = np.random.randint(-4, 4, size=(M0, K0)).astype(np_type)
    # X = np.array([[1, 2],
    #              [3, 4]]).astype(np.int8)

    W_A_cst = np.random.randint(-4, 4, size=(K0, N0)).astype(np_type)
    # W_A_cst = np.array([[1, 1, 1, 1],
    #                    [1, 1, 1, 1]]).astype(np.int8)

    flowtag: bool = True

    # =================================================================
    
    def top[Ty](X: "Ty[M0, K0]", W_A: "Ty[K0, N0]", flowtag: bool) -> "Ty[M0, N0]":
        Z: Ty[M0, N0]
        systolic_uni[int8, M0, K0, N0, Rt0, Ct0](X, W_A, Z, flowtag)
        # systolic_ws[int8, int8, int8, M0, K0, N0, Rt0, Ct0](X, W_A, Z)
        # systolic_os[int8, int8, int8, M0, K0, N0, Rt0, Ct0](X, W_A, Z)
        return Z
    
    def top_ori[Ty](X: "Ty[M0, K0]", W_A: "Ty[K0, N0]", flowtag: bool) -> "Ty[M0, N0]":
        Z: Ty[M0, N0]
        systolic[int8, int8, int8, M0, K0, N0, Rt0, Ct0](X, W_A, Z)
        return Z
    
    # s_uni = allo.customize(systolic_uni, instantiate=[int8, M0, K0, N0, Rt0, Ct0])
    # s_uni.unfold()
    s_top = allo.customize(top, instantiate=[allo_type])
    s_ori = allo.customize(top_ori, instantiate=[allo_type])
    # s_top.compose(s_uni)

    # if Rt0 < 20:
    #     with open(f'systolic_{date}.mlir', 'w') as f:
    #         print(s_top.module, file=f)
    
    # =================================================================
    # CPU Testing
    
    mod = s_top.build()

    allo_C = mod(X, W_A_cst, flowtag)
    allo_C_ori = mod(X, W_A_cst, flowtag)
    np_C = X @ W_A_cst

    # print(np_C)
    # print(allo_C)

    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)
    np.testing.assert_allclose(allo_C_ori, np_C, atol=1e-3)
    print("Passed Functionailty Test on CPU!")

    # =================================================================
    # # HLS Testing

    # ---------------------------------------
    # Scheduling
    tile_name = "systolic_tile_uni"
    systolic_name = "systolic_uni"
    # s_top.compose(systolic_uni, instantiate=[int8, M0, K0, N0, Rt0, Ct0])
    # s_top.partition(s_top.R_buf, dim=1)

    pe = s_top.unfold(f"{tile_name}:PE", [0, 1])
    # s_top.to(MockBuffer(systolic_name, "R_buf"), pe, axis=0, depth=Ct0 + 1)
    # s_top.to(MockBuffer(systolic_name, "C_buf"), pe, axis=1, depth=Rt0 + 1)

    # s_top.to(MockBuffer(tile_name, "R_buf"), pe, axis=0, depth=Ct0 + 1)
    # s_top.to(MockBuffer(tile_name, "C_buf"), pe, axis=1, depth=Rt0 + 1)


    # ---------------------------------------
    # code = s_top.build(target="vhls")
    # if Rt0 < 20:
    #     with open(f'systolicHLS_{date}.cpp', 'w') as f:
    #         print(code, file=f)

    # ---------------------------------------
    # mod_v = s_top.build(target="vhls", mode='csyn', project=f"gemm_{date}.prj")
    # mod_v()

    # s_ori.compose(systolic, instantiate=[int8, int8, int8, M0, K0, N0, Rt0, Ct0])
    # s_ori.dataflow("top_ori")
    # mod_v_ori = s_ori.build(target="vhls", mode='csyn', project=f"gemm_ori_{date}.prj")
    # mod_v_ori()

    # s_top.compose(
    #     systolic, instantiate=[int32, int32, int32, L, D, 4 * D, M0, M1]
    # )
    # s_top.dataflow("top")  # important
    
    # if hls.is_available("vitis_hls"):
    if hls.is_available("vitis_hls"):
        hls_mod = s_top.build(
            target="vitis_hls",
            mode="csim",
            project=f"gemm_csim_{date}.prj"
        )

        # Be careful about the NumPy type
        csim_C = np.zeros((M0, N0), dtype=np_type)
        hls_mod(X, W_A_cst, flowtag, csim_C)
        np.testing.assert_allclose(csim_C, allo_C, atol=1e-3)
        print("Passed Functionailty Test on FPGA!")

    return



if __name__ == '__main__':
    # test_PE_simple()
    # test_tile()
    test_gemm()