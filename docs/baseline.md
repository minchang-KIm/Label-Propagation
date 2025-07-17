# Phase 2: 2-Phase Workflow 분산 라벨 전파 시스템 분석 문서

## 📊 Phase 1 완료 후 초기 상태 예제 그래프

### 예제 그래프 구조 (Phase 1 완료 후)
```
시스템 구성: 2개 MPI 프로세스, 4개 파티션 (P0, P1, P2, P3)

Process 0 (GPU 0):
정점 0-9 (총 10개 정점)
0─1─2─3─4
│ │ │ │ │
5─6─7─8─9

Process 1 (GPU 1):
정점 10-17 (총 8개 정점)  
10─11─12─13
│  │  │  │
14─15─16─17

초기 파티션 할당 (Phase 1 결과):
Process 0: [P0:0,1,5,6] [P1:2,7] [P2:3,8] [P3:4,9]
Process 1: [P0:10,14] [P1:11,15] [P2:12,16] [P3:13,17]

경계 정점 (다른 파티션과 연결):
Process 0: 1,2,3,6,7,8 (6개)
Process 1: 10,11,12,13,14,15,16 (7개)

Edge-cut 초기값: 14개
```

## 🚀 프로그램 시작 및 GPU 자원 관리

### 1. 시스템 초기화 (`main()` → `initializeCUDA()`)

#### 목적
- 다중 GPU 환경에서 각 MPI 프로세스에 GPU 할당
- GPU 리소스 확인 및 최적화된 설정 적용

#### 입력
- `argc`, `argv`: 명령행 인수
- 시스템의 GPU 하드웨어 정보

#### 출력
- GPU 할당 완료 상태
- `gpu_config.json`: GPU 정보 저장 파일
- 각 프로세스별 GPU 설정 완료

#### 내부 변수 및 용어 설명
```cpp
MultiGPUManager* gpu_manager_     // GPU 관리 시스템
device_id_                        // 현재 프로세스 할당 GPU ID
GPUInfo* gpu_info                 // GPU 상세 정보 구조체
  ├── device_id                   // GPU 디바이스 번호
  ├── name                        // GPU 모델명 (예: RTX 3080)
  ├── total_memory                // 총 GPU 메모리 (바이트)
  ├── free_memory                 // 사용 가능 메모리 (바이트)
  ├── compute_capability_major    // CUDA 컴퓨트 능력 (주버전)
  ├── multiprocessor_count        // 멀티프로세서 수
  └── max_threads_per_block      // 블록당 최대 스레드 수
```

#### 프로세스별 GPU 할당 로직
```cpp
// 라운드 로빈 방식 GPU 할당
current_gpu_ = mpi_rank_ % total_gpus_;

// 예시:
// Rank 0 → GPU 0 (RTX 3080 #1)
// Rank 1 → GPU 1 (RTX 3080 #2)
```

#### GPU 리소스 최적화 확인
```cpp
// 1. GPU 메모리 테스트 (1M 정수 배열)
testGPUMemory() {
    test_size = 1024 * 1024
    할당 → 읽기/쓰기 → 검증 → 해제
}

// 2. CUDA 컨텍스트 워밍업 (5회 반복)
warmupGPU() {
    for (int i = 0; i < 5; i++) {
        memoryCheckKernel<<<blocks, threads>>>()
    }
}

// 3. 커널 실행 파라미터 최적화
threads_per_block = min(256, gpu_info->max_threads_per_block)
blocks = (workload + threads_per_block - 1) / threads_per_block
```

## 🏗️ 시스템 아키텍처 개요

### Multi-GPU 분산 처리 설계

#### 하드웨어 구성
- **GPU**: 2x RTX 3080 (Ampere 아키텍처, 8.6 컴퓨트 능력)
- **메모리**: 각 GPU당 10GB GDDR6X
- **연결**: PCIe 4.0 x16, MPI 기반 프로세스 간 통신

#### 소프트웨어 스택
- **MPI**: OpenMPI 4.x, 프로세스 간 통신
- **CUDA**: 12.1/12.8, GPU 커널 실행
- **메모리 관리**: RAII 패턴, 자동 해제

#### 분산 알고리즘 특징
- **Phase 2**: 경계 정점 중심의 라벨 전파
- **2-Phase Workflow**: Phase 1 결과를 입력으로 받아 최적화 수행
- **7-Step Process**: 체계적인 라벨 업데이트 과정

## 📈 성능 특성 및 벤치마크

### 현재 벤치마크 결과
- **그래프 크기**: 55,476개 정점, 234,825개 엣지
- **경계 정점**: 53,413개 (전체의 96.3%)
- **실행 시간**: ~118ms 총 처리 시간
- **처리량**: ~467 정점/ms
- **하드웨어**: Dual RTX 3080 시스템

### 메모리 사용량 분석
- **정점 데이터**: ~220KB (정점 배열)
- **엣지 데이터**: ~900KB (인접성 정보)
- **GPU 메모리**: 총 ~7.5GB 할당 가능
- **확장성**: 최대 2M 정점까지 처리 가능

---

*이 문서는 Phase 2 분산 라벨 전파 시스템의 기술적 기준선을 제공합니다.*