cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_OSX_ARCHITECTURES x86_64)

project(coremltools)

find_program(HAS_CCACHE ccache)
if(HAS_CCACHE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

add_subdirectory(deps)
add_subdirectory(mlmodel)

exec_program(python-config
  ARGS "--includes"
  OUTPUT_VARIABLE PYTHON_FLAGS)
exec_program(python-config
  ARGS "--libs"
  OUTPUT_VARIABLE PYTHON_LIBS)

include_directories(
  .
  deps/caffe/include
  deps/caffe/include/caffe/proto
  deps/protobuf/src
  deps/pybind11/include
)

set(CMAKE_CXX_FLAGS " \
  ${CMAKE_CXX_FLAGS} \
  ${PYTHON_FLAGS} \
  --std=c++14 \
")
set(CMAKE_EXE_LINKER_FLAGS " \
  ${CMAKE_EXE_LINKER_FLAGS} \
  ${PYTHON_LIBS} \
  --std=c++14 \
")
set(CMAKE_MODULE_LINKER_FLAGS " \
  ${CMAKE_MODULE_LINKER_FLAGS} \
  ${PYTHON_LIBS} \
  --std=c++14 \
")
set(CMAKE_SHARED_LINKER_FLAGS " \
  ${CMAKE_SHARED_LINKER_FLAGS} \
  ${PYTHON_LIBS} \
  --std=c++14 \
")

add_library(caffeconverter
  SHARED
  caffeconverter/CaffeConverterLib.cpp
  caffeconverter/CaffeConverterPython.cpp
  caffeconverter/Caffe/Pooling.cpp
  caffeconverter/Caffe/Embed.cpp
  caffeconverter/Caffe/Parameter.cpp
  caffeconverter/Caffe/Split.cpp
  caffeconverter/Caffe/Bias.cpp
  caffeconverter/Caffe/Reduction.cpp
  caffeconverter/Caffe/Batchnorm.cpp
  caffeconverter/Caffe/Slice.cpp
  caffeconverter/Caffe/Crop.cpp
  caffeconverter/Caffe/Concat.cpp
  caffeconverter/Caffe/InnerProduct.cpp
  caffeconverter/Caffe/LRN.cpp
  caffeconverter/Caffe/Power.cpp
  caffeconverter/Caffe/Scale.cpp
  caffeconverter/Caffe/MVN.cpp
  caffeconverter/Caffe/LSTM.cpp
  caffeconverter/Caffe/Abs.cpp
  caffeconverter/Caffe/Eltwise.cpp
  caffeconverter/Caffe/Exp.cpp
  caffeconverter/Caffe/InputLayers.cpp
  caffeconverter/Caffe/Softmax.cpp
  caffeconverter/Caffe/TrainingLayers.cpp
  caffeconverter/Caffe/CaffeConverter.cpp
  caffeconverter/Caffe/UpgradeProto.cpp
  caffeconverter/Caffe/Flatten.cpp
  caffeconverter/Caffe/Log.cpp
  caffeconverter/Caffe/Activation.cpp
  caffeconverter/Caffe/Convolution.cpp
  caffeconverter/Caffe/Reshape.cpp
  deps/caffe/src/caffe/proto/caffe.pb.cc
  deps/caffe/src/caffe/util/upgrade_proto.cpp
)

target_compile_definitions(caffeconverter
  PRIVATE
  CPU_ONLY=1
)

target_link_libraries(caffeconverter
  mlmodel
  libprotobuf
)

find_library(CORE_VIDEO CoreVideo)
find_library(CORE_ML CoreML)
find_library(FOUNDATION Foundation)
if (CORE_VIDEO AND CORE_ML AND FOUNDATION)
  add_library(coremlpython
    SHARED
    coremlpython/CoreMLPython.mm
    coremlpython/CoreMLPython.h
    coremlpython/CoreMLPythonArray.mm
    coremlpython/CoreMLPythonArray.h
    coremlpython/CoreMLPythonUtils.mm
    coremlpython/CoreMLPythonUtils.h
  )
  target_link_libraries(coremlpython
    ${CORE_VIDEO}
    ${CORE_ML}
    ${FOUNDATION}
  )
else()
  message(STATUS "CoreML.framework and dependent frameworks not found. Skipping libcoremlpython build.")
endif()

add_custom_target(dist
  COMMAND cp libcaffeconverter.so coremltools/
  COMMAND test ! -f libcoremlpython.so || cp libcoremlpython.so coremltools/
  COMMAND python setup.py bdist_wheel --plat-name=linux_x86_64
  DEPENDS caffeconverter coremlpython
  COMMENT "Building Python wheel for coremltools under dist/"
)