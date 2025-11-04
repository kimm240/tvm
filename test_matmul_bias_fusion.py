import tvm  
from tvm import tir  
from tvm.script import tir as T  
  
# 원본 PrimFunc 정의  
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
  
  
# 목표: 이상적인 fusion 형태 (수동 작성)  
@T.prim_func  
def matmul_bias_fused(  
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
  
  
def test_case_1_decompose_then_inline():  
    """케이스 1: decompose_reduction 후 reverse_compute_inline 시도"""  
    print("\n=== 케이스 1: decompose_reduction + reverse_compute_inline ===")  
      
    sch = tir.Schedule(matmul_bias_original, debug_mask="all")  
      
    mult_block = sch.get_block("multiply")  
    add_block = sch.get_block("add")  
      
    # decompose_reduction 수행  
    print("1. decompose_reduction 수행...")  
    init_block = sch.decompose_reduction(mult_block, sch.get_loops(mult_block)[-1])  
    print(f"   생성된 init 블록: {sch.get(init_block).name_hint}")  
      
    # 이제 multiply_init와 multiply_update 두 블록이 temp에 씀  
    print("2. reverse_compute_inline 시도...")  
    try:  
        sch.reverse_compute_inline(add_block)  
        print("   ✓ 성공!")  
        print(sch.mod["main"].script())  
    except tvm.tir.ScheduleError as e:  
        print(f"   ✗ 실패: {e}")  
        print("   이유: multiply_init와 multiply_update 두 블록이 모두 temp에 쓰므로")  
        print("        단일 생산자 요구사항 위반")  
  
  
def test_case_2_direct_inline():  
    """케이스 2: decompose 없이 직접 reverse_compute_inline 시도"""  
    print("\n=== 케이스 2: 직접 reverse_compute_inline ===")  
      
    sch = tir.Schedule(matmul_bias_original, debug_mask="all")  
    add_block = sch.get_block("add")  
      
    print("1. reverse_compute_inline 시도...")  
    try:  
        sch.reverse_compute_inline(add_block)  
        print("   ✓ 성공!")  
        print(sch.mod["main"].script())  
    except tvm.tir.ScheduleError as e:  
        print(f"   ✗ 실패: {e}")  
        print("   이유: producer가 reduction 블록이므로 지원되지 않음")  
  
  
def test_case_3_working_example():  
    """케이스 3: 작동하는 예제 (단순 복사 epilogue)"""  
    print("\n=== 케이스 3: 작동하는 예제 (단순 복사) ===")  
      
    @T.prim_func  
    def working_case(  
        A: T.Buffer((16, 16), "float32"),  
        B: T.Buffer((16, 16), "float32"),  
        C: T.Buffer((16, 16), "float32"),  
    ) -> None:  
        temp = T.alloc_buffer((16, 16), dtype="float32")  
          
        for i, j, k in T.grid(16, 16, 16):  
            with T.block("matmul"):  
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])  
                with T.init():  
                    temp[vi, vj] = 0.0  
                temp[vi, vj] = temp[vi, vj] + A[vi, vk] * B[vk, vj]  
          
        for i, j in T.grid(16, 16):  
            with T.block("copy"):  
                vi, vj = T.axis.remap("SS", [i, j])  
                C[vi, vj] = temp[vi, vj]  # 단순 복사  
      
    sch = tir.Schedule(working_case, debug_mask="all")  
      
    print("1. reverse_compute_inline 시도...")  
    try:  
        sch.reverse_compute_inline(sch.get_block("copy"))  
        print("   ✓ 성공! (단순 복사는 inline 가능)")  
        print("\n변환 후:")  
        print(sch.mod["main"].script())  
    except tvm.tir.ScheduleError as e:  
        print(f"   ✗ 실패: {e}")  
  
  
def test_case_4_manual_fusion():  
    """케이스 4: 수동으로 fusion된 형태 검증"""  
    print("\n=== 케이스 4: 수동 fusion 형태 검증 ===")  
      
    print("수동으로 작성된 fusion 형태:")  
    print(matmul_bias_fused.script())  
      
    # 두 버전이 동일한 결과를 내는지 검증  
    import numpy as np  
      
    A_np = np.random.randint(-128, 127, size=(16, 16), dtype=np.int8)  
    B_np = np.random.randint(-128, 127, size=(16, 16), dtype=np.int8)  
    C_np = np.random.randint(-1000, 1000, size=(16, 16), dtype=np.int32)  
      
    # 원본 버전 실행  
    D_original = np.zeros((16, 16), dtype=np.int32)  
    mod_original = tvm.build(matmul_bias_original, target="llvm")  
      
    A_tvm = tvm.runtime.tensor(A_np)  
    B_tvm = tvm.runtime.tensor(B_np)  
    C_tvm = tvm.runtime.tensor(C_np)  
    D_tvm_original = tvm.runtime.tensor(D_original)  
      
    mod_original(A_tvm, B_tvm, C_tvm, D_tvm_original)  
      
    # Fusion 버전 실행  
    D_fused = np.zeros((16, 16), dtype=np.int32)  
    mod_fused = tvm.build(matmul_bias_fused, target="llvm")  
    D_tvm_fused = tvm.runtime.tensor(D_fused)  
      
    mod_fused(A_tvm, B_tvm, C_tvm, D_tvm_fused)  
      
    # 결과 비교  
    if np.allclose(D_tvm_original.numpy(), D_tvm_fused.numpy()):  
        print("\n✓ 두 버전의 결과가 동일합니다!")  
    else:  
        print("\n✗ 결과가 다릅니다!")  
        print(f"최대 차이: {np.max(np.abs(D_tvm_original.numpy() - D_tvm_fused.numpy()))}")

if __name__ == "__main__":  
    print("=" * 70)  
    print("TVM TIR Matmul + Bias Fusion 문제 재현 테스트")  
    print("=" * 70)  
      
    test_case_1_decompose_then_inline()  
    test_case_2_direct_inline()  
    test_case_3_working_example()  
    test_case_4_manual_fusion()  
      
    print("\n" + "=" * 70)  
    print("결론:")  
    print("- decompose_reduction 후 reverse_compute_inline: 실패 (단일 생산자 위반)")  
    print("- 직접 reverse_compute_inline: 실패 (reduction producer 미지원)")  
    print("- 단순 복사 epilogue: 성공")  
    print("- 수동 fusion: 가능하며 올바른 결과 생성")  
    print("=" * 70)
