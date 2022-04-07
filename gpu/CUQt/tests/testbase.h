#ifndef TESTBASE_H
#define TESTBASE_H

#include <QObject>
#include <QDebug>
#include <qtestcase.h>
#include <cuqt.h>

#ifndef __CUDACC__
#define blockIdx dim3{}
#define blockDim dim3{}
#define threadIdx dim3{}
#endif

//TODO move to .cpp, provide implementations for other enums and structs
inline char *toString(const cudaError &err)
{
    QString src;
    {
        QDebug(&src) << err;
    }
    const auto localAscii{src.toLocal8Bit()};
    char *dst = new char[localAscii.size() + 1];
    return qstrcpy(dst, localAscii.data());
}

#endif // TESTBASE_H
