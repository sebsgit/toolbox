/*
Implementation of a multi-device character driver module.
*/

#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/kdev_t.h>
#include <linux/device.h>

#undef pr_fmt
#define pr_fmt(fmt) "Fakecdrv: " fmt

#define DEV_COUNT 4

typedef struct {
    char *buffer;
    unsigned size;
    const char *serial;
    int access_mode;
    struct cdev cdev_handle;
    struct device* device_p;
} fakedrv_device_data_t;

typedef struct {
    dev_t dev_number;
    struct class* class_p;
    fakedrv_device_data_t devs[DEV_COUNT];
} fakedrv_driver_data_t;

#define DEV_ACCESS_R    1
#define DEV_ACCESS_W    2
#define DEV_ACCESS_RW   3

static const char * DRV_NAME = "fakedrv";
static const char * DRV_CLASS_NAME = "fkcdr_cls";
static const char * DRV_DEV_NAME = "fkdrv-%d";

static char fakedrv_buffer_0[512];
static char fakedrv_buffer_1[256];
static char fakedrv_buffer_2[512];
static char fakedrv_buffer_3[1024];
static const unsigned FAKEDRV_DEV_SIZE_0 = sizeof(fakedrv_buffer_0) / sizeof(fakedrv_buffer_0[0]);
static const unsigned FAKEDRV_DEV_SIZE_1 = sizeof(fakedrv_buffer_1) / sizeof(fakedrv_buffer_1[0]);
static const unsigned FAKEDRV_DEV_SIZE_2 = sizeof(fakedrv_buffer_2) / sizeof(fakedrv_buffer_2[0]);
static const unsigned FAKEDRV_DEV_SIZE_3 = sizeof(fakedrv_buffer_3) / sizeof(fakedrv_buffer_3[0]);

static struct file_operations fakedrv_fileops;

/* Fakedrv file operations */
static int fakecdrv_open(struct inode *, struct file*);
static int fakecdrv_release(struct inode *, struct file*);
static ssize_t fakecdrv_read(struct file *, char __user *, size_t, loff_t *);
static ssize_t fakecdrv_write(struct file *, const char __user *, size_t, loff_t *);
static loff_t fakecdrv_llseek(struct file *, loff_t, int);

static fakedrv_driver_data_t fakedrv_driver_data;

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

/*

Character device registration steps:
 - dynamically allocate the device number dev_t
 - initialize the file operations
 - initialize the device handle (struct cdev), connecting the file operations with the device
 - register the device with the VFS (virtual file system): cdev_add
 - device file creation - export the device info to sysfs and let the udev daemon create the /dev/ file (generate uevent)

*/
static int __init fakecdrv_module_init(void)
{
    pr_info("init started..\n");
    memset(&fakedrv_driver_data, 0, sizeof(fakedrv_driver_data));

    fakedrv_driver_data.devs[0].buffer = fakedrv_buffer_0;
    fakedrv_driver_data.devs[0].size = FAKEDRV_DEV_SIZE_0;
    fakedrv_driver_data.devs[0].serial = "fake.drv.dev.0.0";
    fakedrv_driver_data.devs[0].access_mode = DEV_ACCESS_R;
    memcpy(fakedrv_buffer_0, "read only data dev0", 20);

    fakedrv_driver_data.devs[1].buffer = fakedrv_buffer_1;
    fakedrv_driver_data.devs[1].size = FAKEDRV_DEV_SIZE_1;
    fakedrv_driver_data.devs[1].serial = "fake.drv.dev.0.1";
    fakedrv_driver_data.devs[1].access_mode = DEV_ACCESS_W;

    fakedrv_driver_data.devs[2].buffer = fakedrv_buffer_2;
    fakedrv_driver_data.devs[2].size = FAKEDRV_DEV_SIZE_2;
    fakedrv_driver_data.devs[2].serial = "fake.drv.dev.0.2";
    fakedrv_driver_data.devs[2].access_mode = DEV_ACCESS_RW;

    fakedrv_driver_data.devs[3].buffer = fakedrv_buffer_3;
    fakedrv_driver_data.devs[3].size = FAKEDRV_DEV_SIZE_3;
    fakedrv_driver_data.devs[3].serial = "fake.drv.dev.0.3";
    fakedrv_driver_data.devs[3].access_mode = DEV_ACCESS_RW;

    init_file_ops(&fakedrv_fileops);

    const int alloc_res = alloc_chrdev_region(&fakedrv_driver_data.dev_number, 0, DEV_COUNT, DRV_NAME);
    if (alloc_res != 0)
    {
        pr_err("Fakedrv: can't allocate chrdev: %d\n", alloc_res);
        return alloc_res;
    }
    pr_info("Allocated char dev_t: %d\n", fakedrv_driver_data.dev_number);

    fakedrv_driver_data.class_p = class_create(DRV_CLASS_NAME);
    if (IS_ERR(fakedrv_driver_data.class_p))
    {
        pr_err("Failed to create entry in /sys/class");
        unregister_chrdev_region(fakedrv_driver_data.dev_number, DEV_COUNT);
        return PTR_ERR(fakedrv_driver_data.class_p);
    }

    for (int i = 0; i < DEV_COUNT; ++i)
    {
        cdev_init(&fakedrv_driver_data.devs[i].cdev_handle, &fakedrv_fileops);
        fakedrv_driver_data.devs[i].cdev_handle.owner = THIS_MODULE;
    
        const int add_res = cdev_add(&fakedrv_driver_data.devs[i].cdev_handle, fakedrv_driver_data.dev_number + i, 1);
        if (add_res < 0)
        {
            pr_err("Fakedrv: can't add chrdev: %d\n", add_res);
            for (int k = 0; k < i; ++k)
            {
                cdev_del(&fakedrv_driver_data.devs[k].cdev_handle);
            }
            class_destroy(fakedrv_driver_data.class_p);
            unregister_chrdev_region(fakedrv_driver_data.dev_number, DEV_COUNT);
            return add_res;
        }

        fakedrv_driver_data.devs[i].device_p = device_create(fakedrv_driver_data.class_p, NULL, fakedrv_driver_data.dev_number + i, NULL, DRV_DEV_NAME, i);
        if (IS_ERR(fakedrv_driver_data.devs[i].device_p))
        {
            for (int k = 0; k < i; ++k)
            {
                device_destroy(fakedrv_driver_data.class_p, fakedrv_driver_data.dev_number + k);
                cdev_del(&fakedrv_driver_data.devs[k].cdev_handle);
            }
            class_destroy(fakedrv_driver_data.class_p);
            unregister_chrdev_region(fakedrv_driver_data.dev_number, DEV_COUNT);
            return PTR_ERR(fakedrv_driver_data.devs[i].device_p);
        }

        pr_info("init done. Assigned dev_t: %d:%d\n", MAJOR(fakedrv_driver_data.dev_number + i), MINOR(fakedrv_driver_data.dev_number + i));
    }

    return 0;
}

static void __exit fakecdrv_module_exit(void)
{
    for (int i = 0; i < DEV_COUNT; ++i)
    {
        device_destroy(fakedrv_driver_data.class_p, fakedrv_driver_data.dev_number + i);
        cdev_del(&fakedrv_driver_data.devs[i].cdev_handle);
    }
    class_destroy(fakedrv_driver_data.class_p);
    unregister_chrdev_region(fakedrv_driver_data.dev_number, DEV_COUNT);

    pr_info("exit\n");
}

static int check_permission(int device_mode, int user_requested_mode)
{
    if (device_mode == DEV_ACCESS_R)
    {
        if ((user_requested_mode & FMODE_READ) && !(user_requested_mode & FMODE_WRITE))
        {
            return 0;
        }
    }
    else if (device_mode == DEV_ACCESS_W)
    {
        if (!(user_requested_mode & FMODE_READ) && (user_requested_mode & FMODE_WRITE))
        {
            return 0;
        }
    }
    else if (device_mode == DEV_ACCESS_RW)
    {
        if ((user_requested_mode & FMODE_READ) || (user_requested_mode & FMODE_WRITE))
        {
            return 0;
        }
    }
    return -EPERM;
}

static int fakecdrv_open(struct inode *p_inode, struct file *p_file)
{
    pr_info("open\n");
    for (int i = 0 ; i < DEV_COUNT; ++i)
    {
        if (p_inode->i_cdev == &fakedrv_driver_data.devs[i].cdev_handle)
        {
                pr_info("open device %d\n", i);
                p_file->private_data = (void*)&fakedrv_driver_data.devs[i];
                return check_permission(fakedrv_driver_data.devs[i].access_mode, p_file->f_mode);
        }
    }

    return -ENOENT;
}

static int fakecdrv_release(struct inode *, struct file*)
{
    pr_info("release\n");
    return 0;
}

static ssize_t fakecdrv_read(struct file * fp, char __user * user_buffer, size_t count, loff_t * f_pos)
{
    pr_info("read start\n");
    if (!fp->private_data)
    {
        return -ENOENT;
    }
    
    const fakedrv_device_data_t* dev_data = (const fakedrv_device_data_t*)fp->private_data;
    if (count + *f_pos > dev_data->size)
    {
        count = dev_data->size - *f_pos;
    }

    if (copy_to_user(user_buffer, dev_data->buffer + *f_pos, count))
    {
        return -EFAULT;
    }

    *f_pos += count;

    return count;
}

static ssize_t fakecdrv_write(struct file * fp, const char __user * user_buffer, size_t count, loff_t * f_pos)
{
    const fakedrv_device_data_t* dev_data = (const fakedrv_device_data_t*)fp->private_data;

    pr_info("write %zu bytes on position %lld\n", count, *f_pos);
    if (count + *f_pos > dev_data->size)
    {
        count = dev_data->size - *f_pos;
    }
    
    if (count == 0)
    {
        // no space left
        return -ENOMEM;
    }

    if (copy_from_user(dev_data->buffer + *f_pos, user_buffer, count))
    {
        return -EFAULT;
    }

    *f_pos += count;

    pr_info("updated position to %lld\n", *f_pos);

    return count;
}

static loff_t fakecdrv_llseek(struct file * filep, loff_t off, int mode)
{
    const fakedrv_device_data_t* dev_data = (const fakedrv_device_data_t*)filep->private_data;

    switch (mode)
    {
        case SEEK_SET:
            if ((off < 0) || (off > dev_data->size))
            {
                return -EINVAL;
            }
            filep->f_pos = off;
            break;
        case SEEK_CUR:
            if (((filep->f_pos + off) < 0) || ((filep->f_pos + off) > dev_data->size))
            {
                return -EINVAL;
            }
            filep->f_pos += off;
            break;
        case SEEK_END:
            if (((dev_data->size + off) < 0) || ((dev_data->size + off) > dev_data->size))
            {
                return -EINVAL;
            }
            filep->f_pos = dev_data->size + off;
            break;
        default:
            return -EINVAL;
    }
    return filep->f_pos;
}

module_init(fakecdrv_module_init);
module_exit(fakecdrv_module_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("sebb");
MODULE_DESCRIPTION("Fake Character Driver module");
MODULE_INFO(additional_info, "Some more info with custom tag");
