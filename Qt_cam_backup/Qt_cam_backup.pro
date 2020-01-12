QT       += core gui multimedia positioning

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++14

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    AbstractDataSource.cpp \
    AppSettings.cpp \
    DataStorageConfig.cpp \
    SourceConfigWidget.cpp \
    dataSinks/AbstractDataSink.cpp \
    dataSinks/DataSinks.cpp \
    dataSinks/DebugSink.cpp \
    dataSinks/FtpSink.cpp \
    dataSources/DataProducer.cpp \
    dataSources/GpsDataSource.cpp \
    dataSources/PictureDataSource.cpp \
    dataSources/SoundDataSource.cpp \
    main.cpp \
    MainWindow.cpp

HEADERS += \
    AbstractDataSource.h \
    AppSettings.h \
    DataStorageConfig.h \
    MainWindow.h \
    SourceConfigWidget.h \
    dataSinks/AbstractDataSink.h \
    dataSinks/DataSinks.h \
    dataSinks/DebugSink.h \
    dataSinks/FtpSink.h \
    dataSources/DataProducer.h \
    dataSources/GpsDataSource.h \
    dataSources/PictureDataSource.h \
    dataSources/SoundDataSource.h

FORMS += \
    DataStorageConfig.ui \
    MainWindow.ui \
    SourceConfigWidget.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    android/AndroidManifest.xml \
    android/build.gradle \
    android/res/values/libs.xml

contains(ANDROID_TARGET_ARCH,armeabi-v7a) {
    ANDROID_PACKAGE_SOURCE_DIR = \
        $$PWD/android
}
