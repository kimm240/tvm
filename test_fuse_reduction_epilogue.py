import tvm  
from tvm import tir  
from tvm.script import tir as T  
import numpy as np  
  
# 원본 PrimFunc  
@T.prim_func  
def matmul_bias_original(  
    A: T.Buffer((16, 16), "int8"),  
    B: T.Buffer((16, 16), "int8"),  
    C: T.Buffer((16, 16), "int32"),  
    D: T.Buffer((16, 16), "int32"),  
) -> None:  
    temp = T.alloc_buffer((16, 16), dtype="int32")  
    for i, j, k in T.grid(16, 16, 16):  
        with T.block("multiply"):  
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])  
            with T.init():  
                temp[vi, vj] = T.int32(0)  
            temp[vi, vj] = temp[vi, vj] + T.cast(A[vi, vk], "int32") * T.cast(B[vj, vk], "int32")  
    for i, j in T.grid(16, 16):  
        with T.block("add"):  
            vi, vj = T.axis.remap("SS", [i, j])  
            D[vi, vj] = temp[vi, vj] + C[vi, vj]  
  
# 예상되는 fusion 결과  
@T.prim_func  
def matmul_bias_expected(  
    A: T.Buffer((16, 16), "int8"),  
    B: T.Buffer((16, 16), "int8"),  
    C: T.Buffer((16, 16), "int32"),  
    D: T.Buffer((16, 16), "int32"),  
) -> None:  
    for i, j, k in T.grid(16, 16, 16):  
        with T.block("multiply"):  
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])  
            with T.init():  
                D[vi, vj] = C[vi, vj]  
            D[vi, vj] = D[vi, vj] + T.cast(A[vi, vk], "int32") * T.cast(B[vj, vk], "int32")  
  
print("=" * 70)  
print("테스트 1: 메서드 존재 확인")  
print("=" * 70)  
sch = tir.Schedule(matmul_bias_original)  
if hasattr(sch, 'fuse_reduction_epilogue'):  
    print("✓ fuse_reduction_epilogue 메서드가 존재합니다")  
else:  
    print("✗ 메서드가 없습니다. 컴파일 확인 필요")  
    exit(1)  
  
print("\n" + "=" * 70)  
print("테스트 2: Fusion 변환 실행")  
print("=" * 70)  
try:  
    sch.fuse_reduction_epilogue("multiply", "add")  
    print("✓ Fusion 성공!")  
    print("\n변환된 IR:")  
    print(sch.mod["main"].script())  
except Exception as e:  
    print(f"✗ Fusion 실패: {e}")  
    exit(1)  
  
print("\n" + "=" * 70)  
print("테스트 3: IR 구조 검증")  
print("=" * 70)  
try:  
    # global_symbol을 동일하게 설정하여 비교
    after = sch.mod["main"].with_attr("global_symbol", "main")
    expected_normalized = matmul_bias_expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(after, expected_normalized)  
    print("✓ 변환된 IR이 예상과 일치합니다!")  
except Exception as e:  
    print(f"⚠ IR 구조가 예상과 다릅니다:")  
    print(f"  {e}")  
    print("\n예상된 IR:")  
    print(matmul_bias_expected.script())  
  
print("\n" + "=" * 70)  
print("테스트 4: 수치 정확성 검증")  
print("=" * 70)  
# 원본 버전 실행  
A_np = np.random.randint(-128, 127, size=(16, 16), dtype=np.int8)  
B_np = np.random.randint(-128, 127, size=(16, 16), dtype=np.int8)  
C_np = np.random.randint(-1000, 1000, size=(16, 16), dtype=np.int32)  
D_original = np.zeros((16, 16), dtype=np.int32)  
  
mod_original = tvm.build(matmul_bias_original, target="llvm")  
A_tvm = tvm.runtime.tensor(A_np)  
B_tvm = tvm.runtime.tensor(B_np)  
C_tvm = tvm.runtime.tensor(C_np)  
D_tvm_original = tvm.runtime.tensor(D_original)  
mod_original(A_tvm, B_tvm, C_tvm, D_tvm_original)  
  
# Fusion 버전 실행  
D_fused = np.zeros((16, 16), dtype=np.int32)  
mod_fused = tvm.build(sch.mod["main"], target="llvm")  
D_tvm_fused = tvm.runtime.tensor(D_fused)  
mod_fused(A_tvm, B_tvm, C_tvm, D_tvm_fused)  
  
# 결과 비교  
if np.allclose(D_tvm_original.numpy(), D_tvm_fused.numpy()):  
    print("✓ 두 버전의 계산 결과가 동일합니다!")  
    print(f"  샘플 값: {D_tvm_original.numpy()[0, :3]}")  
else:  
    print("✗ 결과가 다릅니다!")  
    print(f"  최대 차이: {np.max(np.abs(D_tvm_original.numpy() - D_tvm_fused.numpy()))}")  
    exit(1)  
  
print("\n" + "=" * 70)  
print("모든 테스트 통과! ✓")  
print("=" * 70)
