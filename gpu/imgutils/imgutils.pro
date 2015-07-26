TEMPLATE = app
TARGET = imgutils
INCLUDEPATH += . ../cuda_wraps/

CONFIG += c++11

DEFINES += CUWR_WITH_QT

OBJECTS_DIR = build
MOC_DIR = build
UI_DIR = build

QMAKE_CXXFLAGS += -Wall -Werror

cuda.commands = nvcc -ptx -arch=sm_20 -O2 cuwr_imgutils.cu -o cuwr_imgutils.ptx
cuda.target = cuda
QMAKE_EXTRA_TARGETS += cuda

cudamotion.commands = nvcc -ptx -arch=sm_20 -O2 cuwr_motion.cu -o cuwr_motion.ptx
cudamotion.target = cudamotion
QMAKE_EXTRA_TARGETS += cudamotion

unix{
	LIBS += -ldl
}

SOURCES += main.cpp	\
		   ../cuda_wraps/cuwrap.cpp	\
		   cuwr_img.cpp \
    cuwr_motion_estimator.cpp

HEADERS += \
    cuwr_img.h \
    cuwr_imgdata_priv.h \
    cuwr_motion_estimator.h

DISTFILES += \
    cuwr_imgutils.cu \
    cuwr_motion.cu
