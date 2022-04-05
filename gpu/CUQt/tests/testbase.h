#ifndef TESTBASE_H
#define TESTBASE_H

#include <QObject>
#include <QDebug>
#include <qtestcase.h>

#ifndef __CUDACC__
#define blockIdx dim3{}
#define blockDim dim3{}
#define threadIdx dim3{}
#endif

#endif // TESTBASE_H
