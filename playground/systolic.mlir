module {
  func.func @PE_kernel(%arg0: memref<16xi8, strided<[1], offset: ?>>, %arg1: memref<16xi8, strided<[1], offset: ?>>, %arg2: memref<16xi8, strided<[1], offset: ?>>, %arg3: memref<16xi8, strided<[1], offset: ?>>, %arg4: memref<4x4xi8>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.trunci %c0_i32 : i32 to i8
    %alloc = memref.alloc() {name = "v"} : memref<1xi8>
    affine.store %0, %alloc[0] {to = "v"} : memref<1xi8>
    affine.for %arg7 = 0 to 16 {
      %2 = affine.load %arg0[%arg7] {from = "A_in"} : memref<16xi8, strided<[1], offset: ?>>
      %alloc_0 = memref.alloc() {name = "a"} : memref<1xi8>
      affine.store %2, %alloc_0[0] {to = "a"} : memref<1xi8>
      %3 = affine.load %arg1[%arg7] {from = "B_in"} : memref<16xi8, strided<[1], offset: ?>>
      %alloc_1 = memref.alloc() {name = "b"} : memref<1xi8>
      affine.store %3, %alloc_1[0] {to = "b"} : memref<1xi8>
      %4 = affine.load %alloc_0[0] {from = "a"} : memref<1xi8>
      %5 = arith.extsi %4 : i8 to i16
      %6 = affine.load %alloc_1[0] {from = "b"} : memref<1xi8>
      %7 = arith.extsi %6 : i8 to i16
      %8 = arith.muli %5, %7 : i16
      %9 = arith.trunci %8 : i16 to i8
      %10 = affine.load %alloc[0] {from = "v"} : memref<1xi8>
      %11 = arith.addi %10, %9 : i8
      affine.store %11, %alloc[0] {to = "v"} : memref<1xi8>
      %12 = affine.load %alloc_0[0] {from = "a"} : memref<1xi8>
      affine.store %12, %arg2[%arg7] {to = "A_out"} : memref<16xi8, strided<[1], offset: ?>>
      %13 = affine.load %alloc_1[0] {from = "b"} : memref<1xi8>
      affine.store %13, %arg3[%arg7] {to = "B_out"} : memref<16xi8, strided<[1], offset: ?>>
    } {loop_name = "k", op_name = "S_k_0"}
    %1 = affine.load %alloc[0] {from = "v"} : memref<1xi8>
    affine.store %1, %arg4[%arg5, %arg6] {to = "C"} : memref<4x4xi8>
    return
  }
  func.func @systolic_tile(%arg0: memref<4x16xi8>, %arg1: memref<16x4xi8>, %arg2: memref<4x4xi8>) attributes {itypes = "sss", otypes = ""} {
    %alloc = memref.alloc() {name = "A_fifo"} : memref<4x5x16xi8>
    %alloc_0 = memref.alloc() {name = "B_fifo"} : memref<4x5x16xi8>
    %alloc_1 = memref.alloc() {name = "A_drain"} : memref<4xi8>
    %alloc_2 = memref.alloc() {name = "B_drain"} : memref<4xi8>
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 4 {
        %0 = affine.load %arg0[%arg4, %arg3] {from = "A"} : memref<4x16xi8>
        affine.store %0, %alloc[%arg4, 0, %arg3] {to = "A_fifo"} : memref<4x5x16xi8>
      } {loop_name = "m", op_name = "S_m_0"}
      affine.for %arg4 = 0 to 4 {
        %0 = affine.load %arg1[%arg3, %arg4] {from = "B"} : memref<16x4xi8>
        affine.store %0, %alloc_0[%arg4, 0, %arg3] {to = "B_fifo"} : memref<4x5x16xi8>
      } {loop_name = "n", op_name = "S_n_1"}
    } {loop_name = "k", op_name = "data_load"}
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        %subview = memref.subview %alloc[%arg3, %arg4, 0] [1, 1, 16] [1, 1, 1] {from = "A_fifo"} : memref<4x5x16xi8> to memref<16xi8, strided<[1], offset: ?>>
        %subview_3 = memref.subview %alloc_0[%arg4, %arg3, 0] [1, 1, 16] [1, 1, 1] {from = "B_fifo"} : memref<4x5x16xi8> to memref<16xi8, strided<[1], offset: ?>>
        %0 = arith.index_cast %arg4 : index to i34
        %c1_i32 = arith.constant 1 : i32
        %1 = arith.extsi %c1_i32 : i32 to i34
        %2 = arith.addi %0, %1 : i34
        %3 = arith.index_cast %2 : i34 to index
        %subview_4 = memref.subview %alloc[%arg3, %3, 0] [1, 1, 16] [1, 1, 1] {from = "A_fifo"} : memref<4x5x16xi8> to memref<16xi8, strided<[1], offset: ?>>
        %4 = arith.index_cast %arg3 : index to i34
        %c1_i32_5 = arith.constant 1 : i32
        %5 = arith.extsi %c1_i32_5 : i32 to i34
        %6 = arith.addi %4, %5 : i34
        %7 = arith.index_cast %6 : i34 to index
        %subview_6 = memref.subview %alloc_0[%arg4, %7, 0] [1, 1, 16] [1, 1, 1] {from = "B_fifo"} : memref<4x5x16xi8> to memref<16xi8, strided<[1], offset: ?>>
        func.call @PE_kernel(%subview, %subview_3, %subview_4, %subview_6, %arg2, %arg3, %arg4) : (memref<16xi8, strided<[1], offset: ?>>, memref<16xi8, strided<[1], offset: ?>>, memref<16xi8, strided<[1], offset: ?>>, memref<16xi8, strided<[1], offset: ?>>, memref<4x4xi8>, index, index) -> ()
      } {loop_name = "j"}
    } {loop_name = "i", op_name = "PE"}
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 4 {
        %0 = affine.load %alloc[%arg4, 4, %arg3] {from = "A_fifo"} : memref<4x5x16xi8>
        affine.store %0, %alloc_1[%arg4] {to = "A_drain"} : memref<4xi8>
      } {loop_name = "m", op_name = "S_m_4"}
      affine.for %arg4 = 0 to 4 {
        %0 = affine.load %alloc_0[%arg4, 4, %arg3] {from = "B_fifo"} : memref<4x5x16xi8>
        affine.store %0, %alloc_2[%arg4] {to = "B_drain"} : memref<4xi8>
      } {loop_name = "n", op_name = "S_n_5"}
    } {loop_name = "k", op_name = "data_drain"}
    return
  }
  func.func @systolic(%arg0: memref<16x16xi8>, %arg1: memref<16x64xi8>, %arg2: memref<16x64xi8>) attributes {itypes = "sss", otypes = ""} {
    %alloc = memref.alloc() {name = "local_A"} : memref<4x16xi8>
    %alloc_0 = memref.alloc() {name = "local_B"} : memref<16x4xi8>
    %alloc_1 = memref.alloc() {name = "local_C"} : memref<4x4xi8>
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 16 {
        affine.for %arg5 = 0 to 16 {
          affine.for %arg6 = 0 to 4 {
            %0 = arith.index_cast %arg4 : index to i33
            %c0_i32 = arith.constant 0 : i32
            %1 = arith.extsi %c0_i32 : i32 to i33
            %2 = arith.cmpi eq, %0, %1 : i33
            scf.if %2 {
              %3 = affine.load %arg0[%arg3 * 4 + %arg6, %arg5] {from = "A"} : memref<16x16xi8>
              affine.store %3, %alloc[%arg6, %arg5] {to = "local_A"} : memref<4x16xi8>
            }
          } {loop_name = "ai"}
        } {loop_name = "ak", op_name = "load_A_tile"}
        affine.for %arg5 = 0 to 16 {
          affine.for %arg6 = 0 to 4 {
            %0 = affine.load %arg1[%arg5, %arg4 * 4 + %arg6] {from = "B"} : memref<16x64xi8>
            affine.store %0, %alloc_0[%arg5, %arg6] {to = "local_B"} : memref<16x4xi8>
          } {loop_name = "bj"}
        } {loop_name = "bk", op_name = "load_B_tile"}
        func.call @systolic_tile(%alloc, %alloc_0, %alloc_1) : (memref<4x16xi8>, memref<16x4xi8>, memref<4x4xi8>) -> ()
        affine.for %arg5 = 0 to 4 {
          affine.for %arg6 = 0 to 4 {
            %0 = affine.load %alloc_1[%arg6, %arg5] {from = "local_C"} : memref<4x4xi8>
            affine.store %0, %arg2[%arg3 * 4 + %arg6, %arg4 * 4 + %arg5] {to = "C"} : memref<16x64xi8>
          } {loop_name = "si"}
        } {loop_name = "sj", op_name = "store_C_tile"}
      } {loop_name = "ni"}
    } {loop_name = "mi", op_name = "outer_tile"}
    return
  }
  func.func @top(%arg0: memref<16x16xi8>, %arg1: memref<16x64xi8>) -> memref<16x64xi8> attributes {itypes = "ss", otypes = "s"} {
    %alloc = memref.alloc() {name = "Z"} : memref<16x64xi8>
    call @systolic(%arg0, %arg1, %alloc) : (memref<16x16xi8>, memref<16x64xi8>, memref<16x64xi8>) -> ()
    return %alloc : memref<16x64xi8>
  }
}

