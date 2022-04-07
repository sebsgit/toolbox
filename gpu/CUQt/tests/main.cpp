#include "testdevice.h"
#include "teststream.h"
#include "testdevicememory.h"
#include "testtextures.h"

#include <QTest>

template <typename T>
int runTest(int argc, char **argv)
{
    T test;
    return QTest::qExec(&test, argc, argv);
}

template <typename FirstTest, typename ... Rest>
int runTests(int argc, char **argv)
{
    int ret_code{0};
    {
        ret_code = runTest<FirstTest>(argc, argv);
    }
    if constexpr (sizeof...(Rest) > 0)
    {
        if (ret_code == 0)
        {
            ret_code = runTests<Rest...>(argc, argv);
        }
    }

    return ret_code;
}

int main(int argc, char **argv)
{
    return runTests<TestDevice, TestStream, TestDeviceMemory, TestTextures>(argc, argv);
}
