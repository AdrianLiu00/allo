module {
  func.func @PE_kernel_ws(%arg0: i8, %arg1: i8, %arg2: i8, %arg3: memref<2xi8, strided<[1], offset: ?>>, %arg4: memref<2xi8, strided<[1], offset: ?>>) attributes {itypes = "sssss", otypes = ""} {
    %alloc = memref.alloc() {name = "b"} : memref<1xi8>
    affine.store %arg0, %alloc[0] {to = "b"} : memref<1xi8>
    %alloc_0 = memref.alloc() {name = "a"} : memref<1xi8>
    affine.store %arg1, %alloc_0[0] {to = "a"} : memref<1xi8>
    %alloc_1 = memref.alloc() {name = "c"} : memref<1xi8>
    affine.store %arg2, %alloc_1[0] {to = "c"} : memref<1xi8>
    %0 = affine.load %alloc_0[0] {from = "a"} : memref<1xi8>
    affine.store %0, %arg3[0] {to = "A_out"} : memref<2xi8, strided<[1], offset: ?>>
    %1 = affine.load %alloc_0[0] {from = "a"} : memref<1xi8>
    %2 = arith.extsi %1 : i8 to i16
    %3 = affine.load %alloc[0] {from = "b"} : memref<1xi8>
    %4 = arith.extsi %3 : i8 to i16
    %5 = arith.muli %2, %4 : i16
    %6 = arith.extsi %5 : i16 to i17
    %7 = affine.load %alloc_1[0] {from = "c"} : memref<1xi8>
    %8 = arith.extsi %7 : i8 to i17
    %9 = arith.addi %6, %8 : i17
    %10 = arith.trunci %9 : i17 to i8
    affine.store %10, %arg4[0] {to = "C_out"} : memref<2xi8, strided<[1], offset: ?>>
    return
  }
  func.func @systolic_tile_ws(%arg0: memref<3xi8>, %arg1: memref<3x4xi8>, %arg2: memref<4xi8>, %arg3: memref<3x5x2xi8>, %arg4: memref<4x4x2xi8>) attributes {itypes = "sssss", otypes = ""} {
    %alloc = memref.alloc() {name = "A_drain"} : memref<1xi8>
    affine.for %arg5 = 0 to 3 {
      %0 = affine.load %arg0[%arg5] {from = "A"} : memref<3xi8>
      affine.store %0, %arg3[%arg5, 0, 0] {to = "A_buf"} : memref<3x5x2xi8>
    } {loop_name = "k", op_name = "A_load"}
    affine.for %arg5 = 0 to 4 {
      %c0_i32 = arith.constant 0 : i32
      %0 = arith.trunci %c0_i32 : i32 to i8
      affine.store %0, %arg4[0, %arg5, 0] {to = "C_buf"} : memref<4x4x2xi8>
    } {loop_name = "n", op_name = "C_load"}
    affine.for %arg5 = 0 to 3 {
      affine.for %arg6 = 0 to 4 {
        %0 = affine.load %arg1[-%arg5 + 2, -%arg6 + 3] {from = "B"} : memref<3x4xi8>
        %1 = affine.load %arg3[-%arg5 + 2, -%arg6 + 3, 0] {from = "A_buf"} : memref<3x5x2xi8>
        %2 = affine.load %arg4[-%arg5 + 2, -%arg6 + 3, 0] {from = "C_buf"} : memref<4x4x2xi8>
        %c3_i32 = arith.constant 3 : i32
        %3 = arith.extsi %c3_i32 : i32 to i33
        %c1_i32 = arith.constant 1 : i32
        %4 = arith.extsi %c1_i32 : i32 to i33
        %5 = arith.subi %3, %4 : i33
        %6 = arith.extsi %5 : i33 to i34
        %7 = arith.index_cast %arg5 : index to i34
        %8 = arith.subi %6, %7 : i34
        %9 = arith.index_cast %8 : i34 to index
        %c4_i32 = arith.constant 4 : i32
        %10 = arith.extsi %c4_i32 : i32 to i33
        %c1_i32_0 = arith.constant 1 : i32
        %11 = arith.extsi %c1_i32_0 : i32 to i33
        %12 = arith.subi %10, %11 : i33
        %13 = arith.extsi %12 : i33 to i34
        %14 = arith.index_cast %arg6 : index to i34
        %15 = arith.subi %13, %14 : i34
        %16 = arith.extsi %15 : i34 to i35
        %c1_i32_1 = arith.constant 1 : i32
        %17 = arith.extsi %c1_i32_1 : i32 to i35
        %18 = arith.addi %16, %17 : i35
        %19 = arith.index_cast %18 : i35 to index
        %subview = memref.subview %arg3[%9, %19, 0] [1, 1, 2] [1, 1, 1] {from = "A_buf"} : memref<3x5x2xi8> to memref<2xi8, strided<[1], offset: ?>>
        %c3_i32_2 = arith.constant 3 : i32
        %20 = arith.extsi %c3_i32_2 : i32 to i33
        %c1_i32_3 = arith.constant 1 : i32
        %21 = arith.extsi %c1_i32_3 : i32 to i33
        %22 = arith.subi %20, %21 : i33
        %23 = arith.extsi %22 : i33 to i34
        %24 = arith.index_cast %arg5 : index to i34
        %25 = arith.subi %23, %24 : i34
        %26 = arith.extsi %25 : i34 to i35
        %c1_i32_4 = arith.constant 1 : i32
        %27 = arith.extsi %c1_i32_4 : i32 to i35
        %28 = arith.addi %26, %27 : i35
        %29 = arith.index_cast %28 : i35 to index
        %c4_i32_5 = arith.constant 4 : i32
        %30 = arith.extsi %c4_i32_5 : i32 to i33
        %c1_i32_6 = arith.constant 1 : i32
        %31 = arith.extsi %c1_i32_6 : i32 to i33
        %32 = arith.subi %30, %31 : i33
        %33 = arith.extsi %32 : i33 to i34
        %34 = arith.index_cast %arg6 : index to i34
        %35 = arith.subi %33, %34 : i34
        %36 = arith.index_cast %35 : i34 to index
        %subview_7 = memref.subview %arg4[%29, %36, 0] [1, 1, 2] [1, 1, 1] {from = "C_buf"} : memref<4x4x2xi8> to memref<2xi8, strided<[1], offset: ?>>
        func.call @PE_kernel_ws(%0, %1, %2, %subview, %subview_7) : (i8, i8, i8, memref<2xi8, strided<[1], offset: ?>>, memref<2xi8, strided<[1], offset: ?>>) -> ()
      } {loop_name = "j"}
    } {loop_name = "i", op_name = "PE"}
    affine.for %arg5 = 0 to 3 {
      %0 = affine.load %arg3[%arg5, 4, 0] {from = "A_buf"} : memref<3x5x2xi8>
      affine.store %0, %alloc[0] {to = "A_drain"} : memref<1xi8>
    } {loop_name = "k", op_name = "A_drain"}
    affine.for %arg5 = 0 to 4 {
      %0 = affine.load %arg4[3, %arg5, 0] {from = "C_buf"} : memref<4x4x2xi8>
      affine.store %0, %arg2[%arg5] {to = "C"} : memref<4xi8>
    } {loop_name = "n", op_name = "C_drain"}
    return
  }
  func.func @systolic_ws(%arg0: memref<5x3xi8>, %arg1: memref<3x8xi8>, %arg2: memref<5x8xi8>) attributes {itypes = "sss", otypes = ""} {
    %alloc = memref.alloc() {name = "local_A"} : memref<3xi8>
    %alloc_0 = memref.alloc() {name = "local_B"} : memref<3x4xi8>
    %alloc_1 = memref.alloc() {name = "local_C"} : memref<4xi8>
    %alloc_2 = memref.alloc() {name = "A_buf"} : memref<3x5x2xi8>
    %alloc_3 = memref.alloc() {name = "C_buf"} : memref<4x4x2xi8>
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.trunci %c0_i32 : i32 to i8
    %alloc_4 = memref.alloc() {name = "A_null"} : memref<1xi8>
    affine.store %0, %alloc_4[0] {to = "A_null"} : memref<1xi8>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 3 {
        affine.for %arg5 = 0 to 4 {
          %1 = affine.load %arg1[%arg4, %arg3 * 4 + %arg5] {from = "B"} : memref<3x8xi8>
          affine.store %1, %alloc_0[%arg4, %arg5] {to = "local_B"} : memref<3x4xi8>
        } {loop_name = "bn"}
      } {loop_name = "bk", op_name = "initial_wights"}
      affine.for %arg4 = 0 to 10 {
        affine.for %arg5 = 0 to 3 {
          %1 = arith.cmpi sge, %arg4, %arg5 : index
          %c5_i32 = arith.constant 5 : i32
          %2 = arith.extsi %c5_i32 : i32 to i34
          %3 = arith.index_cast %arg5 : index to i34
          %4 = arith.addi %2, %3 : i34
          %5 = arith.index_cast %arg4 : index to i34
          %6 = arith.cmpi slt, %5, %4 : i34
          %7 = arith.andi %1, %6 : i1
          scf.if %7 {
            %8 = affine.load %arg0[%arg4 - %arg5, %arg5] {from = "A"} : memref<5x3xi8>
            affine.store %8, %alloc[%arg5] {to = "local_A"} : memref<3xi8>
          } else {
            %8 = affine.load %alloc_4[0] {from = "A_null"} : memref<1xi8>
            affine.store %8, %alloc[%arg5] {to = "local_A"} : memref<3xi8>
          }
        } {loop_name = "ak", op_name = "load_A"}
        func.call @systolic_tile_ws(%alloc, %alloc_0, %alloc_1, %alloc_2, %alloc_3) : (memref<3xi8>, memref<3x4xi8>, memref<4xi8>, memref<3x5x2xi8>, memref<4x4x2xi8>) -> ()
        affine.for %arg5 = 0 to 4 {
          %c3_i32 = arith.constant 3 : i32
          %1 = arith.extsi %c3_i32 : i32 to i33
          %c1_i32 = arith.constant 1 : i32
          %2 = arith.extsi %c1_i32 : i32 to i33
          %3 = arith.subi %1, %2 : i33
          %4 = arith.extsi %3 : i33 to i34
          %5 = arith.index_cast %arg5 : index to i34
          %6 = arith.addi %4, %5 : i34
          %7 = arith.index_cast %arg4 : index to i34
          %8 = arith.cmpi sge, %7, %6 : i34
          %c5_i32 = arith.constant 5 : i32
          %9 = arith.extsi %c5_i32 : i32 to i33
          %c3_i32_5 = arith.constant 3 : i32
          %10 = arith.extsi %c3_i32_5 : i32 to i33
          %11 = arith.addi %9, %10 : i33
          %12 = arith.extsi %11 : i33 to i34
          %c1_i32_6 = arith.constant 1 : i32
          %13 = arith.extsi %c1_i32_6 : i32 to i34
          %14 = arith.subi %12, %13 : i34
          %15 = arith.extsi %14 : i34 to i35
          %16 = arith.index_cast %arg5 : index to i35
          %17 = arith.addi %15, %16 : i35
          %18 = arith.index_cast %arg4 : index to i35
          %19 = arith.cmpi slt, %18, %17 : i35
          %20 = arith.andi %8, %19 : i1
          scf.if %20 {
            %21 = affine.load %alloc_1[%arg5] {from = "local_C"} : memref<4xi8>
            affine.store %21, %arg2[%arg4 - (%arg5 + 2), %arg3 * 4 + %arg5] {to = "C"} : memref<5x8xi8>
          }
        } {loop_name = "cn", op_name = "store_C"}
      } {loop_name = "t", op_name = "temporal"}
    } {loop_name = "ni", op_name = "outer_tile"}
    return
  }
  func.func @top(%arg0: memref<5x3xi8>, %arg1: memref<3x8xi8>) -> memref<5x8xi8> attributes {itypes = "ss", otypes = "s"} {
    %alloc = memref.alloc() {name = "Z"} : memref<5x8xi8>
    call @systolic_ws(%arg0, %arg1, %alloc) : (memref<5x3xi8>, memref<3x8xi8>, memref<5x8xi8>) -> ()
    return %alloc : memref<5x8xi8>
  }
}

