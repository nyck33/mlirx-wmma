module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @matmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) {
    %0 = llvm.mlir.constant(2048 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(2088 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg0, %4[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg1, %5[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg2, %6[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.insertvalue %arg3, %7[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.insertvalue %arg5, %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg4, %9[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg6, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg7, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg8, %13[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg9, %14[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %arg10, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %arg12, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %arg11, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %arg13, %18[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.insertvalue %arg14, %20[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg15, %21[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg16, %22[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %arg17, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %arg19, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %arg18, %25[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %arg20, %26[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%3 : i64)
  ^bb1(%28: i64):  // 2 preds: ^bb0, ^bb8
    %29 = llvm.icmp "slt" %28, %2 : i64
    llvm.cond_br %29, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%30: i64):  // 2 preds: ^bb2, ^bb7
    %31 = llvm.icmp "slt" %30, %0 : i64
    llvm.cond_br %31, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%3 : i64)
  ^bb5(%32: i64):  // 2 preds: ^bb4, ^bb6
    %33 = llvm.icmp "slt" %32, %0 : i64
    llvm.cond_br %33, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %34 = llvm.mlir.constant(2048 : index) : i64
    %35 = llvm.mul %28, %34  : i64
    %36 = llvm.add %35, %32  : i64
    %37 = llvm.getelementptr %arg1[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %38 = llvm.load %37 : !llvm.ptr -> f64
    %39 = llvm.mlir.constant(2048 : index) : i64
    %40 = llvm.mul %32, %39  : i64
    %41 = llvm.add %40, %30  : i64
    %42 = llvm.getelementptr %arg8[%41] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %43 = llvm.load %42 : !llvm.ptr -> f64
    %44 = llvm.mlir.constant(2048 : index) : i64
    %45 = llvm.mul %28, %44  : i64
    %46 = llvm.add %45, %30  : i64
    %47 = llvm.getelementptr %arg15[%46] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %48 = llvm.load %47 : !llvm.ptr -> f64
    %49 = llvm.fmul %38, %43  : f64
    %50 = llvm.fadd %48, %49  : f64
    %51 = llvm.mlir.constant(2048 : index) : i64
    %52 = llvm.mul %28, %51  : i64
    %53 = llvm.add %52, %30  : i64
    %54 = llvm.getelementptr %arg15[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %50, %54 : f64, !llvm.ptr
    %55 = llvm.add %32, %1  : i64
    llvm.br ^bb5(%55 : i64)
  ^bb7:  // pred: ^bb5
    %56 = llvm.add %30, %1  : i64
    llvm.br ^bb3(%56 : i64)
  ^bb8:  // pred: ^bb3
    %57 = llvm.add %28, %1  : i64
    llvm.br ^bb1(%57 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant(5 : index) : i64
    %1 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(0x4184640000000000 : f64) : f64
    %3 = llvm.mlir.constant(2048 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(2088 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(2088 : index) : i64
    %8 = llvm.mlir.constant(2048 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(4276224 : index) : i64
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.getelementptr %11[4276224] : (!llvm.ptr) -> !llvm.ptr, f64
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.call @malloc(%13) : (i64) -> !llvm.ptr
    %15 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %7, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %8, %20[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %8, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %9, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(2048 : index) : i64
    %25 = llvm.mlir.constant(2048 : index) : i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.constant(4194304 : index) : i64
    %28 = llvm.mlir.zero : !llvm.ptr
    %29 = llvm.getelementptr %28[4194304] : (!llvm.ptr) -> !llvm.ptr, f64
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.call @malloc(%30) : (i64) -> !llvm.ptr
    %32 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.insertvalue %31, %32[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %31, %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mlir.constant(0 : index) : i64
    %36 = llvm.insertvalue %35, %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %24, %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %25, %37[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %25, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %26, %39[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.mlir.constant(2088 : index) : i64
    %42 = llvm.mlir.constant(2048 : index) : i64
    %43 = llvm.mlir.constant(1 : index) : i64
    %44 = llvm.mlir.constant(4276224 : index) : i64
    %45 = llvm.mlir.zero : !llvm.ptr
    %46 = llvm.getelementptr %45[4276224] : (!llvm.ptr) -> !llvm.ptr, f64
    %47 = llvm.ptrtoint %46 : !llvm.ptr to i64
    %48 = llvm.call @malloc(%47) : (i64) -> !llvm.ptr
    %49 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %50 = llvm.insertvalue %48, %49[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.insertvalue %48, %50[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.mlir.constant(0 : index) : i64
    %53 = llvm.insertvalue %52, %51[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.insertvalue %41, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %42, %54[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %42, %55[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.insertvalue %43, %56[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%6 : i64)
  ^bb1(%58: i64):  // 2 preds: ^bb0, ^bb5
    %59 = llvm.icmp "slt" %58, %5 : i64
    llvm.cond_br %59, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%6 : i64)
  ^bb3(%60: i64):  // 2 preds: ^bb2, ^bb4
    %61 = llvm.icmp "slt" %60, %3 : i64
    llvm.cond_br %61, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %62 = llvm.mlir.constant(2048 : index) : i64
    %63 = llvm.mul %58, %62  : i64
    %64 = llvm.add %63, %60  : i64
    %65 = llvm.getelementptr %14[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %1, %65 : f64, !llvm.ptr
    %66 = llvm.add %60, %4  : i64
    llvm.br ^bb3(%66 : i64)
  ^bb5:  // pred: ^bb3
    %67 = llvm.add %58, %4  : i64
    llvm.br ^bb1(%67 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%6 : i64)
  ^bb7(%68: i64):  // 2 preds: ^bb6, ^bb11
    %69 = llvm.icmp "slt" %68, %3 : i64
    llvm.cond_br %69, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%6 : i64)
  ^bb9(%70: i64):  // 2 preds: ^bb8, ^bb10
    %71 = llvm.icmp "slt" %70, %3 : i64
    llvm.cond_br %71, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %72 = llvm.mlir.constant(2048 : index) : i64
    %73 = llvm.mul %68, %72  : i64
    %74 = llvm.add %73, %70  : i64
    %75 = llvm.getelementptr %31[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %1, %75 : f64, !llvm.ptr
    %76 = llvm.add %70, %4  : i64
    llvm.br ^bb9(%76 : i64)
  ^bb11:  // pred: ^bb9
    %77 = llvm.add %68, %4  : i64
    llvm.br ^bb7(%77 : i64)
  ^bb12:  // pred: ^bb7
    llvm.br ^bb13(%6 : i64)
  ^bb13(%78: i64):  // 2 preds: ^bb12, ^bb17
    %79 = llvm.icmp "slt" %78, %5 : i64
    llvm.cond_br %79, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%6 : i64)
  ^bb15(%80: i64):  // 2 preds: ^bb14, ^bb16
    %81 = llvm.icmp "slt" %80, %3 : i64
    llvm.cond_br %81, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %82 = llvm.mlir.constant(2048 : index) : i64
    %83 = llvm.mul %78, %82  : i64
    %84 = llvm.add %83, %80  : i64
    %85 = llvm.getelementptr %48[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %1, %85 : f64, !llvm.ptr
    %86 = llvm.add %80, %4  : i64
    llvm.br ^bb15(%86 : i64)
  ^bb17:  // pred: ^bb15
    %87 = llvm.add %78, %4  : i64
    llvm.br ^bb13(%87 : i64)
  ^bb18:  // pred: ^bb13
    %88 = llvm.call @rtclock() : () -> f64
    llvm.br ^bb19(%6 : i64)
  ^bb19(%89: i64):  // 2 preds: ^bb18, ^bb26
    %90 = llvm.icmp "slt" %89, %0 : i64
    llvm.cond_br %90, ^bb20, ^bb27
  ^bb20:  // pred: ^bb19
    llvm.br ^bb21(%6 : i64)
  ^bb21(%91: i64):  // 2 preds: ^bb20, ^bb25
    %92 = llvm.icmp "slt" %91, %5 : i64
    llvm.cond_br %92, ^bb22, ^bb26
  ^bb22:  // pred: ^bb21
    llvm.br ^bb23(%6 : i64)
  ^bb23(%93: i64):  // 2 preds: ^bb22, ^bb24
    %94 = llvm.icmp "slt" %93, %3 : i64
    llvm.cond_br %94, ^bb24, ^bb25
  ^bb24:  // pred: ^bb23
    %95 = llvm.mlir.constant(2048 : index) : i64
    %96 = llvm.mul %91, %95  : i64
    %97 = llvm.add %96, %93  : i64
    %98 = llvm.getelementptr %48[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %1, %98 : f64, !llvm.ptr
    %99 = llvm.add %93, %4  : i64
    llvm.br ^bb23(%99 : i64)
  ^bb25:  // pred: ^bb23
    %100 = llvm.add %91, %4  : i64
    llvm.br ^bb21(%100 : i64)
  ^bb26:  // pred: ^bb21
    %101 = llvm.extractvalue %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.extractvalue %40[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.extractvalue %40[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.extractvalue %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.extractvalue %40[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.extractvalue %40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.extractvalue %40[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.extractvalue %57[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.extractvalue %57[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.extractvalue %57[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.extractvalue %57[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %120 = llvm.extractvalue %57[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.extractvalue %57[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @matmul(%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> ()
    %122 = llvm.add %89, %4  : i64
    llvm.br ^bb19(%122 : i64)
  ^bb27:  // pred: ^bb19
    %123 = llvm.call @rtclock() : () -> f64
    %124 = llvm.fsub %123, %88  : f64
    %125 = llvm.fdiv %2, %124  : f64
    llvm.call @printF64(%124) : (f64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printFlops(%125) : (f64) -> ()
    llvm.call @free(%14) : (!llvm.ptr) -> ()
    llvm.call @free(%31) : (!llvm.ptr) -> ()
    llvm.call @free(%48) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @printNewline() attributes {sym_visibility = "private"}
  llvm.func @printF64(f64) attributes {sym_visibility = "private"}
  llvm.func @printFlops(f64) attributes {sym_visibility = "private"}
  llvm.func @rtclock() -> f64 attributes {sym_visibility = "private"}
}

