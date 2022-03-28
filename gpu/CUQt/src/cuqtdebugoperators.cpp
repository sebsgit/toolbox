#include "cuqt.h"

#include <type_traits>
#include <QDebug>

#define MAKE_CASE(x) case x: stream << #x ; break

QDebug operator<<(QDebug stream, cudaDeviceAttr value)
{
    switch (value)
    {
        MAKE_CASE(cudaDevAttrMaxThreadsPerBlock);
        MAKE_CASE(cudaDevAttrMaxBlockDimX);
        MAKE_CASE(cudaDevAttrMaxBlockDimY);
        MAKE_CASE(cudaDevAttrMaxBlockDimZ);
        MAKE_CASE(cudaDevAttrMaxGridDimX );
        MAKE_CASE(cudaDevAttrMaxGridDimY );
        MAKE_CASE(cudaDevAttrMaxGridDimZ );
        MAKE_CASE(cudaDevAttrMaxSharedMemoryPerBlock);
        MAKE_CASE(cudaDevAttrTotalConstantMemory);
        MAKE_CASE(cudaDevAttrWarpSize);
        MAKE_CASE(cudaDevAttrMaxPitch);
        MAKE_CASE(cudaDevAttrMaxRegistersPerBlock);
        MAKE_CASE(cudaDevAttrClockRate);
        MAKE_CASE(cudaDevAttrTextureAlignment);
        MAKE_CASE(cudaDevAttrGpuOverlap);
        MAKE_CASE(cudaDevAttrMultiProcessorCount);
        MAKE_CASE(cudaDevAttrKernelExecTimeout);
        MAKE_CASE(cudaDevAttrIntegrated);
        MAKE_CASE(cudaDevAttrCanMapHostMemory);
        MAKE_CASE(cudaDevAttrComputeMode);
        MAKE_CASE(cudaDevAttrMaxTexture1DWidth);
        MAKE_CASE(cudaDevAttrMaxTexture2DWidth);
        MAKE_CASE(cudaDevAttrMaxTexture2DHeight);
        MAKE_CASE(cudaDevAttrMaxTexture3DWidth);
        MAKE_CASE(cudaDevAttrMaxTexture3DHeight);
        MAKE_CASE(cudaDevAttrMaxTexture3DDepth);
        MAKE_CASE(cudaDevAttrMaxTexture2DLayeredWidth);
        MAKE_CASE(cudaDevAttrMaxTexture2DLayeredHeight);
        MAKE_CASE(cudaDevAttrMaxTexture2DLayeredLayers);
        MAKE_CASE(cudaDevAttrSurfaceAlignment);
        MAKE_CASE(cudaDevAttrConcurrentKernels);
        MAKE_CASE(cudaDevAttrEccEnabled);
        MAKE_CASE(cudaDevAttrPciBusId);
        MAKE_CASE(cudaDevAttrPciDeviceId);
        MAKE_CASE(cudaDevAttrTccDriver);
        MAKE_CASE(cudaDevAttrMemoryClockRate);
        MAKE_CASE(cudaDevAttrGlobalMemoryBusWidth);
        MAKE_CASE(cudaDevAttrL2CacheSize);
        MAKE_CASE(cudaDevAttrMaxThreadsPerMultiProcessor);
        MAKE_CASE(cudaDevAttrAsyncEngineCount);
        MAKE_CASE(cudaDevAttrUnifiedAddressing);
        MAKE_CASE(cudaDevAttrMaxTexture1DLayeredWidth);
        MAKE_CASE(cudaDevAttrMaxTexture1DLayeredLayers);
        MAKE_CASE(cudaDevAttrMaxTexture2DGatherWidth);
        MAKE_CASE(cudaDevAttrMaxTexture2DGatherHeight);
        MAKE_CASE(cudaDevAttrMaxTexture3DWidthAlt);
        MAKE_CASE(cudaDevAttrMaxTexture3DHeightAlt);
        MAKE_CASE(cudaDevAttrMaxTexture3DDepthAlt);
        MAKE_CASE(cudaDevAttrPciDomainId);
        MAKE_CASE(cudaDevAttrTexturePitchAlignment);
        MAKE_CASE(cudaDevAttrMaxTextureCubemapWidth);
        MAKE_CASE(cudaDevAttrMaxTextureCubemapLayeredWidth);
        MAKE_CASE(cudaDevAttrMaxTextureCubemapLayeredLayers);
        MAKE_CASE(cudaDevAttrMaxSurface1DWidth);
        MAKE_CASE(cudaDevAttrMaxSurface2DWidth);
        MAKE_CASE(cudaDevAttrMaxSurface2DHeight);
        MAKE_CASE(cudaDevAttrMaxSurface3DWidth);
        MAKE_CASE(cudaDevAttrMaxSurface3DHeight);
        MAKE_CASE(cudaDevAttrMaxSurface3DDepth);
        MAKE_CASE(cudaDevAttrMaxSurface1DLayeredWidth);
        MAKE_CASE(cudaDevAttrMaxSurface1DLayeredLayers);
        MAKE_CASE(cudaDevAttrMaxSurface2DLayeredWidth);
        MAKE_CASE(cudaDevAttrMaxSurface2DLayeredHeight);
        MAKE_CASE(cudaDevAttrMaxSurface2DLayeredLayers);
        MAKE_CASE(cudaDevAttrMaxSurfaceCubemapWidth);
        MAKE_CASE(cudaDevAttrMaxSurfaceCubemapLayeredWidth);
        MAKE_CASE(cudaDevAttrMaxSurfaceCubemapLayeredLayers);
        MAKE_CASE(cudaDevAttrMaxTexture1DLinearWidth);
        MAKE_CASE(cudaDevAttrMaxTexture2DLinearWidth);
        MAKE_CASE(cudaDevAttrMaxTexture2DLinearHeight);
        MAKE_CASE(cudaDevAttrMaxTexture2DLinearPitch);
        MAKE_CASE(cudaDevAttrMaxTexture2DMipmappedWidth);
        MAKE_CASE(cudaDevAttrMaxTexture2DMipmappedHeight);
        MAKE_CASE(cudaDevAttrComputeCapabilityMajor);
        MAKE_CASE(cudaDevAttrComputeCapabilityMinor);
        MAKE_CASE(cudaDevAttrMaxTexture1DMipmappedWidth);
        MAKE_CASE(cudaDevAttrStreamPrioritiesSupported);
        MAKE_CASE(cudaDevAttrGlobalL1CacheSupported);
        MAKE_CASE(cudaDevAttrLocalL1CacheSupported);
        MAKE_CASE(cudaDevAttrMaxSharedMemoryPerMultiprocessor);
        MAKE_CASE(cudaDevAttrMaxRegistersPerMultiprocessor);
        MAKE_CASE(cudaDevAttrManagedMemory);
        MAKE_CASE(cudaDevAttrIsMultiGpuBoard);
        MAKE_CASE(cudaDevAttrMultiGpuBoardGroupID);
        MAKE_CASE(cudaDevAttrHostNativeAtomicSupported);
        MAKE_CASE(cudaDevAttrSingleToDoublePrecisionPerfRatio);
        MAKE_CASE(cudaDevAttrPageableMemoryAccess);
        MAKE_CASE(cudaDevAttrConcurrentManagedAccess);
        MAKE_CASE(cudaDevAttrComputePreemptionSupported);
        MAKE_CASE(cudaDevAttrCanUseHostPointerForRegisteredMem);
        MAKE_CASE(cudaDevAttrReserved92);
        MAKE_CASE(cudaDevAttrReserved93);
        MAKE_CASE(cudaDevAttrReserved94);
        MAKE_CASE(cudaDevAttrCooperativeLaunch);
        MAKE_CASE(cudaDevAttrCooperativeMultiDeviceLaunch);
        MAKE_CASE(cudaDevAttrMaxSharedMemoryPerBlockOptin);
        MAKE_CASE(cudaDevAttrCanFlushRemoteWrites);
        MAKE_CASE(cudaDevAttrHostRegisterSupported);
        MAKE_CASE(cudaDevAttrPageableMemoryAccessUsesHostPageTables);
        MAKE_CASE(cudaDevAttrDirectManagedMemAccessFromHost);
        MAKE_CASE(cudaDevAttrMaxBlocksPerMultiprocessor);
        MAKE_CASE(cudaDevAttrReservedSharedMemoryPerBlock);
        MAKE_CASE(cudaDevAttrSparseCudaArraySupported);
        MAKE_CASE(cudaDevAttrHostRegisterReadOnlySupported);
        MAKE_CASE(cudaDevAttrMaxTimelineSemaphoreInteropSupported);
        MAKE_CASE(cudaDevAttrMemoryPoolsSupported);
    }
    return stream;
}

QDebug operator<<(QDebug stream, cudaError value)
{
    return stream << cudaGetErrorString(value);
}

QDebug operator<<(QDebug stream, const dim3 &value)
{
    stream.nospace() << "[" << value.x << ',' << value.y << ',' << value.z << ']';
    return stream.resetFormat();
}

QDebug operator<<(QDebug stream, const CUQt::Version &version)
{
    stream.nospace() << version.major << '.' << version.minor;
    return stream.resetFormat();
}

#undef MAKE_CASE
