#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/kdev_t.h>
#include <linux/device.h>
#include <linux/platform_device.h>
#include <linux/mod_devicetable.h>

#include "fakedrv_common.h"

#undef pr_fmt
#define pr_fmt(fmt) "Fakecdrv: " fmt

static const unsigned DEV_COUNT = 4;
static const char * DRV_NAME = "fakedrvplt";
static const char * DRV_CLASS_NAME = "fkdrvp_cls";
static const char * DRV_DEV_NAME = "fkdrvp-%d";

typedef struct
{
    fakedrv_platform_device_data_t plt_data;
    char *buffer;
    int id;
    struct cdev cdev_handle;
    struct device *device_p;
} fakedrv_dev_local_data_t;

typedef struct
{
    int num_devices;
    struct class *class_p;
    dev_t device_number_base;
} fakedrv_private_data_t;

/* Fakedrv file operations */
static int fakecdrv_open(struct inode *, struct file*);
static int fakecdrv_release(struct inode *, struct file*);
static ssize_t fakecdrv_read(struct file *, char __user *, size_t, loff_t *);
static ssize_t fakecdrv_write(struct file *, const char __user *, size_t, loff_t *);
static loff_t fakecdrv_llseek(struct file *, loff_t, int);

static struct file_operations fakedrv_fileops;
static fakedrv_private_data_t fakedrv_data;

static int init_file_ops(struct file_operations *fops)
{
    memset(fops, 0, sizeof(*fops));
    fops->owner = THIS_MODULE;
    fops->open = &fakecdrv_open;
    fops->release = &fakecdrv_release;
    fops->read = &fakecdrv_read;
    fops->write = &fakecdrv_write;
    fops->llseek = &fakecdrv_llseek;
    return 0;
}

static int fakedrv_probe(struct platform_device * platform_dev)
{
    int res = 0;
    pr_info("probe");

    const fakedrv_platform_device_data_t * dev_data_p = (const fakedrv_platform_device_data_t*)platform_dev->dev.platform_data;
    if (!dev_data_p)
    {
        pr_err("no valid platform data found");
        return -ENOENT;
    }

    fakedrv_dev_local_data_t *new_entry = (fakedrv_dev_local_data_t*)devm_kzalloc(&platform_dev->dev, sizeof(fakedrv_dev_local_data_t), GFP_KERNEL);
    if (!new_entry)
    {
        pr_err("Cannot allocate fakedrv_dev_local_data_t");
        return -ENOMEM;
    }
    
    memcpy(&new_entry->plt_data, dev_data_p, sizeof(*dev_data_p));
    new_entry->id = platform_dev->id;

    pr_info("new dev: %d, size: %d, serial: %s, access mode: %d\n",
        new_entry->id, 
        new_entry->plt_data.size, 
        new_entry->plt_data.serial_no, 
        new_entry->plt_data.access_mode);

    new_entry->buffer = (char*)devm_kzalloc(&platform_dev->dev, new_entry->plt_data.size, GFP_KERNEL);
    if (!new_entry->buffer)
    {
        pr_err("Cannot allocate device buffer");
        res = -ENOMEM;
        goto new_entry_free;
    }

    platform_dev->dev.driver_data = new_entry;

    cdev_init(&new_entry->cdev_handle, &fakedrv_fileops);
    new_entry->cdev_handle.owner = THIS_MODULE;

    const int add_res = cdev_add(&new_entry->cdev_handle, fakedrv_data.device_number_base + new_entry->id, 1);
    if (add_res < 0)
    {
        pr_err("Cannot add cdev: %d\n", add_res);
        res = add_res;
        goto new_entry_buffer_free;
    }

    new_entry->device_p = device_create(fakedrv_data.class_p, NULL, fakedrv_data.device_number_base + new_entry->id, NULL, DRV_DEV_NAME, new_entry->id);
    if (IS_ERR(new_entry->device_p))
    {
        res = PTR_ERR(new_entry->device_p);
        goto cdev_delete;
    }

    return 0;

cdev_delete:
    cdev_del(&new_entry->cdev_handle);

new_entry_buffer_free:
    devm_kfree(&platform_dev->dev, new_entry->buffer);

new_entry_free:
    devm_kfree(&platform_dev->dev, new_entry);

    return res;
}

static int fakedrv_remove(struct platform_device * pdev)
{
    pr_info("remove");
    fakedrv_dev_local_data_t* p = (fakedrv_dev_local_data_t*)(pdev->dev.driver_data);
    if (p)
    {
        pr_info("destroy device %d\n", p->id);
        device_destroy(fakedrv_data.class_p, fakedrv_data.device_number_base + p->id);
        cdev_del(&p->cdev_handle);
    }
    return 0;
}

static struct platform_device_id fakedrv_devs_id[] = {
    [0] = {.name = FAKEDRV_PLATFORM_NAME, .driver_data = 0},
    [1] = {.name = FAKEDRV_PLATFORM_NAME_VARIANT_A, .driver_data = 1}, // driver_data can be extracted in "probe" to do specific configuration
    {}
};

static struct platform_driver fakedrv_platform_driver = {
    .driver = {
        .name = FAKEDRV_PLATFORM_NAME
    },
    .probe = fakedrv_probe,
    .remove = fakedrv_remove,
    .id_table = fakedrv_devs_id
};

static int __init fakecdrv_module_init(void)
{
    pr_info("init started..\n");
    init_file_ops(&fakedrv_fileops);

    const int alloc_chrdev_res = alloc_chrdev_region(&fakedrv_data.device_number_base, 0, DEV_COUNT, DRV_NAME);
    if (alloc_chrdev_res < 0)
    {
        pr_err("Alloc chrdev region failed: %d\n", alloc_chrdev_res);
        return alloc_chrdev_res;
    }

    fakedrv_data.class_p = class_create(DRV_CLASS_NAME);
    if (IS_ERR(fakedrv_data.class_p))
    {
        pr_err("Failed to create entry in /sys/class");
        unregister_chrdev_region(fakedrv_data.device_number_base, DEV_COUNT);
        return PTR_ERR(fakedrv_data.class_p);
    }

    const int res = platform_driver_register(&fakedrv_platform_driver);
    if (res < 0)
    {
        class_destroy(fakedrv_data.class_p);
        unregister_chrdev_region(fakedrv_data.device_number_base, DEV_COUNT);
        return res;
    }
    
    return 0;
}

static void __exit fakecdrv_module_exit(void)
{
    platform_driver_unregister(&fakedrv_platform_driver);
    class_destroy(fakedrv_data.class_p);
    unregister_chrdev_region(fakedrv_data.device_number_base, DEV_COUNT);
    pr_info("exit\n");
}

static int fakecdrv_open(struct inode *, struct file*)
{
    pr_info("open\n");
    return 0;
}

static int fakecdrv_release(struct inode *, struct file*)
{
    pr_info("release\n");
    return 0;
}

static ssize_t fakecdrv_read(struct file *, char __user * user_buffer, size_t count, loff_t * f_pos)
{
    return 0;
}

static ssize_t fakecdrv_write(struct file *, const char __user * user_buffer, size_t count, loff_t * f_pos)
{
    pr_info("write %zu bytes on position %lld\n", count, *f_pos);
    return -ENOMEM;
}

static loff_t fakecdrv_llseek(struct file * filep, loff_t off, int mode)
{
    return 0;
}

module_init(fakecdrv_module_init);
module_exit(fakecdrv_module_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("sebb");
MODULE_DESCRIPTION("Fake Character Driver module for platform device");
