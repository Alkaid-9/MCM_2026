cmake_minimum_required(VERSION 3.12)
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()
project(MCM_2026_Core_Engine LANGUAGES CXX)

# ==============================================================================
# 1. 编译标准与全局优化 (Industrial Performance Standards)
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 确保输出目录整洁，直接指向 Python 调用层可访问的 bin 目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../bin")

# ==============================================================================
# 2. 核心依赖寻找 (pybind11, OpenMP, Eigen)
# ==============================================================================
# pybind11: C++/Python 桥梁
find_package(pybind11 REQUIRED)

# OpenMP: 实现 23 核并行采样的核心动力
find_package(OpenMP REQUIRED)

# Eigen3: 顶级数值计算库（处理 MCMC 中的矩阵运算与线性代数）
# 建议通过 vcpkg 或 system path 安装，若未安装可先注释，但强烈建议使用
find_package(Eigen3 REQUIRED)

# ==============================================================================
# 3. 针对性编译器优化 (Aggressive Optimization Flags)
# ==============================================================================
if(MSVC)
    # Windows / Visual Studio 优化
    add_compile_options(/O2 /arch:AVX2 /openmp /Zi)
    add_definitions(-D_USE_MATH_DEFINES)
else()
    # Linux (WSL) / GCC / Clang 优化
    # -O3: 极限优化
    # -march=native: 针对你当前的 CPU (YaoGuang) 进行指令集优化
    # -ffast-math: 牺牲极小精度换取极致浮点运算速度
    add_compile_options(-O3 -march=native -ffast-math -fopenmp -fPIC)
endif()

# ==============================================================================
# 4. 包含目录定义 (Header Scoping)
# ==============================================================================
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# ==============================================================================
# 5. 定义 Python 扩展模块 (Pybind11 Binding)
# ==============================================================================
# 模块名称 mcm_core_lib 必须与 Python 中 import 的名称完全一致
pybind11_add_module(mcm_core_lib
    src/mcmc_sampler.cpp
    src/bindings.cpp
)

# ==============================================================================
# 6. 链接库与并行支持 (Linking)
# ==============================================================================
target_link_libraries(mcm_core_lib PRIVATE
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)

# ==============================================================================
# 7. 附加元数据 (Academic Rigor Metadata)
# ==============================================================================
# 在编译出的库中注入版本信息，方便在 Python 侧审计
target_compile_definitions(mcm_core_lib PRIVATE
    VERSION_INFO=${PROJECT_VERSION}
    ENGINE_TYPE="Bayesian_BIO_Kernel"
)