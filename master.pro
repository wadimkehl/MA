
QT += widgets core gui
TARGET = master
CONFIG -= app_bundle

QMAKE_CXXFLAGS += -std=c++11 -march=native -O3

SOURCES += main.cpp scribblearea.cpp tool.cpp icgbench.cpp
HEADERS  += scribblearea.h tool.h CImg.h
OTHER_FILES += notes.txt gpu.cu gpu.h
INCLUDEPATH += arma/include wavelet


CUDA_DIR = /usr/local/cuda
CUDA_SOURCES += gpu.cu
NVCC = $$CUDA_DIR/bin/nvcc

# OSX
macx:{
    LIBS += wavelet/libwavelet.a -lcudart
    QMAKE_LIBDIR += $$CUDA_DIR/lib
    INCLUDEPATH += /usr/local/include
}

# Linux
unix:!macx{
    QMAKE_LIBDIR += $$CUDA_DIR/lib64
    INCLUDEPATH += /opt/tum/external/include/
    LIBS += -Larma wavelet/libwavelet.a -lcudart -L/opt/tum/external/lib

}

INCLUDEPATH += $$CUDA_DIR/include
LIBS += -lcudart -lopencv_core -lopencv_imgproc
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj

cuda.commands = $$NVCC -c $$(QMAKE_CXXFLAGS) $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.input = CUDA_SOURCES

QMAKE_EXTRA_COMPILERS += cuda
