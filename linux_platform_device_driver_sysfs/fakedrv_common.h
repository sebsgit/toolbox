#ifndef FAKEDRV_COMMON_H_
#define FAKEDRV_COMMON_H_

/*
    used to populate the .name fields of the platform device
    and device driver, used for driver <-> device name-based matching
*/
#define FAKEDRV_PLATFORM_NAME "fakedrv-plt-dev"

/*
    some extra "variant" of platform device, used to test the id-table based matching
*/
#define FAKEDRV_PLATFORM_NAME_VARIANT_A "fplt-drv-var-a"

#define FAKEDRV_RW 3
#define FAKEDRV_R  1

/*
    platform-specific data, will be passed from the platform device to the driver in the ".platform_data" field
*/
typedef struct
{
    char serial_no[20];
    unsigned size;
    int access_mode;
} fakedrv_platform_device_data_t;

#endif // FAKEDRV_COMMON_H_
