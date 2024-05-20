#include <linux/module.h>
#include <linux/platform_device.h>

#include "fakedrv_common.h"

#undef pr_fmt
#define pr_fmt(fmt) "Fakepltdev: " fmt

static void fakedrv_device_release(struct device *)
{
    pr_info("device released");
}

static fakedrv_platform_device_data_t device_data[] = {
    [0] = {
        .access_mode = FAKEDRV_RW,
        .serial_no = "DEV_SERIAL_0",
        .size = 512
    },
    [1] = {
        .access_mode = FAKEDRV_R,
        .serial_no = "DVSRIAL_11",
        .size = 128
    },
    [2] = {
        .access_mode = FAKEDRV_RW,
        .serial_no = "DVSR_3",
        .size = 32
    }
};

static struct platform_device device0 = {
    .name = FAKEDRV_PLATFORM_NAME,
    .id = 0,
    .dev = {
        .release = fakedrv_device_release,
        .platform_data = &device_data[0]
    }
};

static struct platform_device device1 = {
    .name = FAKEDRV_PLATFORM_NAME,
    .id = 1,
    .dev = {
        .release = fakedrv_device_release,
        .platform_data = &device_data[1]
    }
};

static struct platform_device device2 = {
    .name = FAKEDRV_PLATFORM_NAME_VARIANT_A,
    .id = 2,
    .dev = {
        .release = fakedrv_device_release,
        .platform_data = &device_data[2]
    }
};

static int __init fakeplatformdev_init(void)
{
    pr_info("device module init");
    platform_device_register(&device0);
    platform_device_register(&device1);
    platform_device_register(&device2);
    return 0;
}

static void __exit fakeplatformdev_exit(void)
{
    platform_device_unregister(&device0);
    platform_device_unregister(&device1);
    platform_device_unregister(&device2);
    pr_info("device module release");
}

module_init(fakeplatformdev_init);
module_exit(fakeplatformdev_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("sebb");
MODULE_DESCRIPTION("Fake Character Driver platform device module");
