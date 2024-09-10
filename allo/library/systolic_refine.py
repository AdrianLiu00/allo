# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unsupported-assignment-operation

from .. import dsl, template
from ..ir.types import int4, int8, int16, int32, index, Int, UInt, bool
from ..ir.utils import MockBuffer


def PE_kernel_ws[
    TyA, TyB, TyC, Kt: int32, Nt: int32
](
    B: "TyB[Kt, Nt]",
    i: index,
    j: index,

    A_in: "TyA",
    C_in: "TyC",
    A_out: "TyA[2]",
    C_out: "TyC[2]"

):

    b: TyB = B[i, j]

    a: TyA = A_in
    c: TyC = C_in
    
    A_out[0] = a
    C_out[0] = a * b + c



def PE_kernel_os[
    TyA, TyB, TyC, Mt: int32, Nt: int32
](
    C: "TyC[Mt, Nt]",
    i: index,
    j: index,

    A_in: "TyA",
    B_in: "TyB",
    A_out: "TyA[2]",
    B_out: "TyB[2]"

):
    a: TyA = A_in
    b: TyB = B_in
    
    c: TyC = a * b + C[i, j]
    
    A_out[0] = a
    B_out[0] = b
    C[i, j] = c



def systolic_tile_ws[
    TyA, TyB, TyC, Kt: int32, Nt: int32
](B: "TyB[Kt, Nt]", A_buf: "TyA[Kt, Nt+1, 2]", C_buf: "TyC[Kt+1, Nt, 2]"):

    for i, j in dsl.grid(Kt, Nt, name='PE'):
        i0: index = Kt-1-i
        j0: index = Nt-1-j

        PE_kernel_ws[TyA, TyB, TyC, Kt, Nt](
            B, i0, j0, A_buf[i0, j0, 0], C_buf[i0, j0, 0], A_buf[Kt-1-i, Nt-1-j+1], C_buf[Kt-1-i+1, Nt-1-j]
        )

    # A_drain: TyA[1]
    # for k in range(Kt, name="A_drain"):
    #     A_drain[0] = A_buf[k, Nt, 0]


def systolic_tile_os[
    TyA, TyB, TyC, Mt: int32, Nt: int32
](C: "TyC[Mt, Nt]", A_buf: "TyA[Mt, Nt+1, 2]", B_buf: "TyB[Mt+1, Nt, 2]"):

    for i, j in dsl.grid(Mt, Nt, name='PE'):
        i0: index = Mt-1-i
        j0: index = Nt-1-j

        PE_kernel_os[TyA, TyB, TyC, Mt, Nt](
            C, i0, j0, A_buf[i0, j0, 0], B_buf[i0, j0, 0], A_buf[Mt-1-i, Nt-1-j+1], B_buf[Mt-1-i+1, Nt-1-j]
        )


def systolic_ws[
    TyA, TyB, TyC, M: int32, K: int32, N: int32, Kt: int32, Nt: int32
](A: "TyA[M, K]", B: "TyB[K, N]", C: "TyC[M, N]"):

    # ======================== Spatial ============================
    # -------------- Top Level --------------
    local_B: TyB[Kt, Nt]

    # -------------- Tile Level --------------
    A_buf: TyA[Kt, Nt+1, 2]
    C_buf: TyC[Kt+1, Nt, 2]


    # ======================== Temporal ============================
    A_zero: TyA = 0
    C_zero: TyC = 0
    # -------------- Top Level --------------
    for ki, ni in dsl.grid(K//Kt, N // Nt, name = "outer_tile"):

    # -------------- Tile Level --------------
        for bk, bn in dsl.grid(Kt, Nt, name="initial_wights"):
            local_B[bk, bn] = B[ki * Kt + bk, ni * Nt + bn]
        
        for t in range(M+Kt+Nt-2, name = "temporal"):
            # organize the input data shape
            for ak in range(Kt, name="load_A"):
                if t >= ak and t < M + ak:
                    A_buf[ak, 0, 0] = A[t-ak, ki * Kt + ak]
                else:
                    A_buf[ak, 0, 0] = A_zero

            for cn0 in range(Nt, name="load_C"):
                if ki == 0: # Initialize Partial Sum
                    C_buf[0, cn0, 0] = 0
                elif t >= cn0 and t < M + cn0:
                    C_buf[0, cn0, 0] = C[t-cn0, ni * Nt + cn0]
                else:
                    C_buf[0, cn0, 0] = C_zero

            systolic_tile_ws[TyA, TyB, TyC, Kt, Nt](
            local_B,
            A_buf,
            C_buf
            )

            for cn1 in range(Nt, name = "store_C"):
                if t >= Kt-1+cn1 and t < M+Kt-1+cn1:
                    C[t-(Kt-1+cn1), ni * Nt + cn1] = C_buf[Kt, cn1, 0]


def systolic_os[
    TyA, TyB, TyC, M: int32, K: int32, N: int32, Mt: int32, Nt: int32
](A: "TyA[M, K]", B: "TyB[K, N]", C: "TyC[M, N]"):

    # ======================== Spatial ============================
    # -------------- Top Level --------------
    local_C: TyC[Mt, Nt]

    # -------------- Tile Level --------------
    A_buf: TyA[Mt, Nt+1, 2]
    B_buf: TyB[Mt+1, Nt, 2]


    # ======================== Temporal ============================
    A_zero: TyA = 0
    B_zero: TyB = 0
    # -------------- Top Level --------------
    for mi, ni in dsl.grid(M // Mt, N // Nt, name = "outer_tile"):
        
        for cm, cn in dsl.grid(Mt, Nt, name="initial_C"):
            local_C[cm, cn] = 0
            # Initilize A and B
            A_buf[cm, cn, 0] = 0
            B_buf[cm, cn, 0] = 0

        for t in range(K+Mt+Nt-2, name = "temporal"):
            # organize the input data shape
            for am in range(Mt, name="load_A"):
                if t >= am and t < K + am:
                    A_buf[am, 0, 0] = A[mi * Mt + am, t-am]
                else:
                    A_buf[am, 0, 0] = A_zero
            
            for bn in range(Nt, name="load_B"):
                if t >= bn and t < K + bn:
                    B_buf[0, bn, 0] = B[t-bn, ni * Nt + bn]
                else:
                    B_buf[0, bn, 0] = B_zero
            
            systolic_tile_os[TyA, TyB, TyC, Mt, Nt](
                local_C,
                A_buf,
                B_buf
            )
        
        for cm, cn in dsl.grid(Mt, Nt, name="store_C"):
            C[mi * Mt + cm, ni * Nt + cn] = local_C[cm, cn]
