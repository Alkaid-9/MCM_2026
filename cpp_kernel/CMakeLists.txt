cmake_minimum_required(VERSION 3.15)

# ==============================================================================
# 1. 项目元数据 (Project Metadata)
# ==============================================================================
project(MCM_2026_Core_Engine
        VERSION 4.0.0
        LANGUAGES CXX
        DESCRIPTION "High-Performance Bayesian Inverse Optimization Kernel (v4.6)"
)

# 强制使用 C++17 标准 (支持 constexpr, std::lgamma, alignas 等关键特性)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 物理直觉：
# 贝叶斯反演是计算密集型任务，我们需要强制编译器进行极致优化。
# 同时，必须保证内存对齐和多线程安全。

# ==============================================================================
# 2. 依赖查找 (Dependency Injection)
# ==============================================================================

# [核心] Pybind11: 用于生成 Python 扩展模块
find_package(pybind11 REQUIRED)

# [核心] Eigen3: 工业级线性代数库，支持 SIMD 优化
# 注意：MCM 环境需预先安装 Eigen (Header-only)
find_package(Eigen3 3.3 REQUIRED)

# [核心] OpenMP: 实现 23 核并行采样的物理引擎
find_package(OpenMP REQUIRED)

# ==============================================================================
# 3. 源文件定义 (Source Definitions)
# ==============================================================================

# 包含目录
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
)

# 核心源码清单
set(SOURCE_FILES
        src/bindings.cpp      # Python 桥接层
        src/mcmc_sampler.cpp  # Dual-Averaging 采样引擎
        src/likelihood.cpp    # 势能景观评估器
        src/math_utils.cpp    # 向量化数学算子 (Soft-Rank/LSE)
        src/diagnostics.cpp   # R-hat 与收敛审计
)

# ==============================================================================
# 4. 模块构建与性能优化 (HPC Tunnels)
# ==============================================================================

# 创建 Python 扩展模块 (mcm_core_lib)
pybind11_add_module(mcm_core_lib ${SOURCE_FILES})

# 链接 OpenMP 和 Eigen
target_link_libraries(mcm_core_lib PRIVATE
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
)

# --- [学术亮点] 开启 LTO (Link Time Optimization) ---
# 物理意义：允许编译器跨文件内联 Likelihood 函数到 MCMC 循环中，消除函数调用开销。
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    message(STATUS "LTO/IPO is enabled for cross-unit inlining.")
    set_property(TARGET mcm_core_lib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "LTO/IPO not supported: ${error}")
endif()

# --- [工业实践] 激进优化标志 (Compiler Specific) ---
if(MSVC)
    # Windows 环境优化 (Visual Studio)
    target_compile_options(mcm_core_lib PRIVATE
            /O2                 # 最大速度
            /Ob2                # 激进内联
            /Oi                 # 启用固有函数
            /Ot                 # 速度优先
            /arch:AVX2          # 强制开启 AVX2 指令集 (数院大数据标配)
            /fp:fast            # 牺牲极小精度换取浮点运算质跃
            /MP                 # 编译时多核并行
            /std:c++17
    )
else()
    # Linux/Mac 环境优化 (GCC/Clang)
    target_compile_options(mcm_core_lib PRIVATE
            -O3                 # 最高级优化
            -march=native       # [重要] 自动针对执行机 CPU 架构优化指令 (AVX-512 Ready)
            -ffast-math         # 开启快速数学模型
            -funroll-loops      # 循环展开，提升指令流水线效率
            -fvisibility=hidden # 减小二进制体积，加速符号查表
            -Wall               # 开启所有警告 (严谨性体现)
    )
endif()

# ==============================================================================
# 5. 产物部署 (Artifact Deployment)
# ==============================================================================

# 设置输出路径：直接将生成的动态库拷贝到项目的 bin 目录下
# 物理直觉：方便 Python 端的 MCMCInferenceWrapper 直接 import，无需手动搬运。
set_target_properties(mcm_core_lib PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../bin"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../lib"
        PREFIX "" # Linux 下移除 'lib' 前缀，确保 Python 识别为模块名
)

# 注入版本号和编译元数据，方便在 Python 侧审计内核版本
target_compile_definitions(mcm_core_lib PRIVATE
        VERSION_INFO="${PROJECT_VERSION}"
)

# 打印构建摘要
message(STATUS "------------------------------------------------------------")
message(STATUS " MCM 2026 Core Engine Configuration Complete")
message(STATUS " Parallelism: OpenMP 23-Core Support Enabled")
message(STATUS " Math Backend: Eigen3 with AVX2/SIMD")
message(STATUS " Output: ${CMAKE_SOURCE_DIR}/../bin/mcm_core_lib")
message(STATUS "------------------------------------------------------------")