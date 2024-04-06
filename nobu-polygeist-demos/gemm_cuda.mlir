module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>,
    #dlti.dl_entry<f128, dense<128> : vector<2xi32>>,
    #dlti.dl_entry<f64, dense<64> : vector<2xi32>>,
    #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>,
    #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>,
    #dlti.dl_entry<i64, dense<64> : vector<2xi32>>,
    #dlti.dl_entry<f80, dense<128> : vector<2xi32>>,
    #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>,
    #dlti.dl_entry<i8, dense<8> : vector<2xi32>>,
    #dlti.dl_entry<i1, dense<8> : vector<2xi32>>,
    #dlti.dl_entry<i16, dense<16> : vector<2xi32>>,
    #dlti.dl_entry<f16, dense<16> : vector<2xi32>>,
    #dlti.dl_entry<i32, dense<32> : vector<2xi32>>,
    #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>,
    #dlti.dl_entry<"dlti.endianness", "little">>,
  llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  llvm.target_triple = "x86_64-unknown-linux-gnu",
  polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",
  polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda",
  "polygeist.target-cpu" = "x86-64",
  "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87",
  "polygeist.tune-cpu" = "generic"
} {
}
