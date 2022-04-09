#ifndef TESTTEXTURES_H
#define TESTTEXTURES_H

#include "testbase.h"

class TestTextures : public QObject
{
    Q_OBJECT
public:
    explicit TestTextures(QObject *parent = nullptr);

private slots:
    void uploadQImage();
    void readWriteTextureMemory();

};

#endif // TESTTEXTURES_H
