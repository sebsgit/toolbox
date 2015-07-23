TEMPLATE = app
TARGET = imgutils
INCLUDEPATH += . ../cuda_wraps/

CONFIG += c++11

DEFINES += CUWR_WITH_QT

OBJECTS_DIR = build
MOC_DIR = build
UI_DIR = build

QMAKE_CXXFLAGS += -Wall -Werror

cuda.commands = nvcc -ptx -arch=sm_20 cuwr_imgutils.cu -o cuwr_imgutils.ptx
cuda.target = cuda
QMAKE_EXTRA_TARGETS += cuda

unix{
	LIBS += -ldl
}

SOURCES += main.cpp	\
		   ../cuda_wraps/cuwrap.cpp	\
		   cuwr_img.cpp

HEADERS += \
    cuwr_img.h \
    cuwr_imgdata_priv.h

DISTFILES += \
    cuwr_imgutils.cu
