# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unsupported-assignment-operation

from .. import dsl, template
from ..ir.types import int4, int8, int16, int32, index, Int, UInt, bool
from ..ir.utils import MockBuffer

from ..library.systolic_refine import systolic_tile_ws, systolic_tile_os

def PE_kernel_uni[
    Ty, Rt: int32, Ct: int32
](
    S: "Ty[Rt, Ct]",
    i: index,
    j: index,

    R_in: "Ty",
    C_in: "Ty",
    R_out: "Ty[2]",
    C_out: "Ty[2]",

    flowtag: bool
):
    s: Ty = S[i, j]
    r: Ty = R_in
    c: Ty = C_in

    acti: Ty = r
    weight: Ty = c if flowtag else s
    psum: Ty = s if flowtag else c
    accu: Ty = acti * weight + psum

    R_out[0] = r
    C_out[0] = c if flowtag else accu
    S[i, j] = accu if flowtag else s


def systolic_tile_uni[
    Ty, Rt: int32, Ct: int32
](S: "Ty[Rt, Ct]", R_buf: "Ty[Rt, Ct+1, 2]", C_buf: "Ty[Rt+1, Ct, 2]", flowtag: bool):
    
    for i, j in dsl.grid(Rt, Ct, name='PE'):
        i0: index = Rt-1-i
        j0: index = Ct-1-j

        PE_kernel_uni[Ty, Rt, Ct](
            S, i0, j0, R_buf[i0, j0, 0], C_buf[i0, j0, 0], R_buf[Rt-1-i, Ct-1-j+1], C_buf[Rt-1-i+1, Ct-1-j], flowtag
        )


def systolic_uni[
    Ty, M: int32, K: int32, N: int32, Rt: int32, Ct: int32
](A: "Ty[M, K]", B: "Ty[K, N]", C: "Ty[M, N]", flowtag: bool):
    
    # ======================== Spatial ============================
    # -------------- Top Level --------------
    local_S: Ty[Rt, Ct]

    # -------------- Tile Level --------------
    R_buf: Ty[Rt, Ct+1, 2]
    C_buf: Ty[Rt+1, Ct, 2]
    C_drain: Ty


    # ======================== Temporal ============================
    R_zero: Ty = 0
    C_zero: Ty = 0
    # -------------- Top Level --------------
    Rtimes: int32 = M//Rt if flowtag else K//Rt
    Ctimes: int32 = N//Ct
    Tlength: int32 = K if flowtag else M
    Tcycles: int32 = Tlength+Rt+Ct-2

    for m, n in dsl.grid(M, N, name="initial_output"): # *
        C[m, n] = 0

    # for ri, ci in dsl.grid(Rtimes, Ctimes, name="outer_tile"):
    for ri in range(Rtimes):
        for ci in range(Ctimes):

        # -------------- Tile Level --------------
            for r, c in dsl.grid(Rt, Ct, name="initial_tile"):
                local_S[r, c] = 0 if flowtag else B[ri * Rt + r, ci * Ct + c]
                R_buf[r, c, 0] = 0 # *
                C_buf[r, c, 0] = 0 # *

            for t in range(Tcycles, name="temporal"):
                # organize the input data shape
                for rl in range(Rt, name="load_R"):
                    if t >= rl and t < rl + Tlength:
                        R_buf[rl, 0, 0] = A[ri*Rt+rl, t-rl] if flowtag else A[t-rl, ri*Rt+rl]
                    else:
                        R_buf[rl, 0, 0] = R_zero
                
                for cl in range(Ct, name="load_C"):
                    if t >= cl and t < Tlength + cl:
                        C_buf[0, cl, 0] = B[t-cl, ci*Ct+cl] if flowtag else C[t-cl, ci*Ct+cl]
                    else:
                        C_buf[0, cl, 0] = C_zero
                
                systolic_tile_uni[Ty, Rt, Ct](
                    local_S,
                    R_buf,
                    C_buf,
                    flowtag
                )

                for cd in range(Ct, name="drain_C"):
                    if t >= Rt-1+cd and t < Tlength+Rt-1+cd:
                        # if not flowtag:
                        #     C[t-(Rt-1+cd), ci*Ct+cd] = C_buf[Rt, cd, 0]
                        # else:
                        #     C_drain = C_buf[Rt, cd, 0] # *
                        if flowtag:
                            C_drain = C_buf[Rt, cd, 0] # *
                        else:
                            C[t-(Rt-1+cd), ci*Ct+cd] = C_buf[Rt, cd, 0]


            for r, c in dsl.grid(Rt, Ct, name="store_tile"):
                # if not flowtag:
                #     B[ri * Rt + r, ci * Ct + c] = local_S[r, c] # *
                # else:
                #     C[ri * Rt + r, ci * Ct + c] = local_S[r, c]
                if flowtag:
                    C[ri * Rt + r, ci * Ct + c] = local_S[r, c]
                else:
                    B[ri * Rt + r, ci * Ct + c] = local_S[r, c] # *
                    
