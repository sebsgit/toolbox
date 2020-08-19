#ifndef DISTORTIONPARAMETERS_H
#define DISTORTIONPARAMETERS_H

#include <QtGlobal>

struct DistortionParameters
{
    qreal k1 {0.0};
    qreal k2 {0.0};
    qreal k3 {0.0};

    // camera settings
    qreal fx {1.0};
    qreal fy {1.0};
    qreal cx {0.0};
    qreal cy {0.0};
};

#endif // DISTORTIONPARAMETERS_H
