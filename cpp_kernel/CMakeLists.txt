cmake_minimum_required(VERSION 3.12)
project(MCM_2026_Core_Engine LANGUAGES CXX)

# ==============================================================================
# 1. 编译标准与全局优化 (Industrial Performance Standards)
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 强制输出目录，直接指向 Python 桥接层能够 import 的路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../bin")

# ==============================================================================
# 2. 核心依赖寻找 (pybind11, OpenMP, Eigen)
# ==============================================================================
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)
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
    # -O3: 极致性能优化
    # -march=native: 针对你当前的 CPU (YaoGuang) 进行指令集优化
    # -flto: 链接时优化，消除跨文件调用的开销（解决之前 symbol 问题的性能补丁）
    add_compile_options(-O3 -march=native -flto -fopenmp -fPIC -fvisibility=hidden)
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
# 【核心修正】：必须包含 likelihood.cpp，它是 MCMC 的灵魂实现
# ==============================================================================
pybind11_add_module(mcm_core_lib
        src/mcmc_sampler.cpp
        src/bindings.cpp
        src/likelihood.cpp    # <--- 这里就是之前报错的根源！
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

# 开启 LTO 支持 (如果编译器支持)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    set_property(TARGET mcm_core_lib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()