#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/device.h>
#include <linux/platform_device.h>
#include <linux/mod_devicetable.h>
#include <linux/of.h>
#include <linux/gpio/consumer.h>

#undef pr_fmt
#define pr_fmt(fmt) "sb-gpiosfs: " fmt

#define DRV_NAME "sb-gpio-sysfs"
static const char * DRV_CLASS_NAME = "sb-gpiosfs";

typedef struct
{
    char label[32];
    struct gpio_desc *p_desc;
} gpiosfs_local_data_t;

typedef struct
{
    int num_devices;
    struct class *class_p;
    struct device **devs;
} gpiosfs_private_data_t;

static gpiosfs_private_data_t gpiosfs_data;

static gpiosfs_local_data_t * get_gpio_data_container(struct device *dev)
{
    return (gpiosfs_local_data_t*)(dev_get_drvdata(dev));
}

static ssize_t direction_show(struct device *dev, struct device_attribute *attr,
                    char *buf)
{
    const gpiosfs_local_data_t *p = get_gpio_data_container(dev);
    if (!p)
    {
        dev_err(dev, "Failed to get gpio private data\n");
        return 0;
    }
    const int dir = gpiod_get_direction(p->p_desc);
    if ((dir != 0) && (dir != 1))
    {
        return dir;
    }
    return sprintf(buf, "%s", (dir ? "in" : "out"));
}

static ssize_t direction_store(struct device *dev, struct device_attribute *attr,
                    const char *buf, size_t count)
{
    const gpiosfs_local_data_t *p = get_gpio_data_container(dev);
    if (!p)
    {
        dev_err(dev, "Failed to get gpio private data\n");
        return 0;
    }

    if (sysfs_streq(buf, "in"))
    {  
        const int res = gpiod_direction_input(p->p_desc);
        if (res != 0)
        {
            return res;
        }
        return count;
    }
    else if (sysfs_streq(buf, "out"))
    {
        const int res = gpiod_direction_output(p->p_desc, 0);
        if (res != 0)
        {
            return res;
        }
        return count;
    }

    return -EINVAL;
}

static ssize_t value_show(struct device *dev, struct device_attribute *attr,
                    char *buf)
{
    const gpiosfs_local_data_t *p = get_gpio_data_container(dev);
    if (!p)
    {
        dev_err(dev, "Failed to get gpio private data\n");
        return 0;
    }
    const int val = gpiod_get_value(p->p_desc);
    return sprintf(buf, "%d", val);
}

static ssize_t value_store(struct device *dev, struct device_attribute *attr,
                    const char *buf, size_t count)
{
    long result;
    int res = kstrtol(buf, 10, &result);
    if (res != 0)
    {
        return res;
    }

    if ((result != 0) && (result != 1))
    {
        return -EINVAL;
    }

    const gpiosfs_local_data_t *p = get_gpio_data_container(dev);
    if (!p)
    {
        dev_err(dev, "Failed to get gpio private data\n");
        return 0;
    }

    gpiod_set_value(p->p_desc, (int)result);

    return count;
}

static ssize_t label_show(struct device *dev, struct device_attribute *attr,
                    char *buf)
{
    const gpiosfs_local_data_t *p = get_gpio_data_container(dev);
    if (!p)
    {
        dev_err(dev, "Failed to get gpio private data\n");
        return 0;
    }
    return sprintf(buf, "%s", p->label);
}

static DEVICE_ATTR_RW(direction);
static DEVICE_ATTR_RW(value);
static DEVICE_ATTR_RO(label);

static struct attribute *gpiosfs_attrs[] = {
    &dev_attr_direction.attr,
    &dev_attr_label.attr,
    &dev_attr_value.attr,
    NULL
};

static const struct attribute_group gpiosfs_group = {.attrs = gpiosfs_attrs};

static const struct attribute_group *gpiosfs_attr_groups[] = {
    &gpiosfs_group,
    NULL
};

static int gpiosfs_probe(struct platform_device * platform_dev)
{
    int res = 0;
    pr_info("probe");

    struct device_node *parent = platform_dev->dev.of_node;
    struct device_node *child = NULL;

    gpiosfs_data.num_devices = of_get_child_count(parent);
    if (gpiosfs_data.num_devices == 0)
    {
        pr_err("No devices found\n");
        return -EINVAL;
    }

    pr_info("Found %d gpio devices...\n", gpiosfs_data.num_devices);
    gpiosfs_data.devs = devm_kzalloc(&platform_dev->dev, sizeof(struct device *) * gpiosfs_data.num_devices, GFP_KERNEL);

    int i = 0;
    for_each_available_child_of_node(parent, child)
    {
        pr_info("processing child node: %s\n", child->name);

        gpiosfs_local_data_t *node_data = devm_kzalloc(&platform_dev->dev, sizeof(gpiosfs_local_data_t), GFP_KERNEL);
        if (!node_data)
        {
            return -ENOMEM;
        }

        const char *label_s_ptr = NULL;
        res = of_property_read_string(child, "label", &label_s_ptr);
        if (res != 0)
        {
            dev_err(&platform_dev->dev, "No 'label' for node\n");
            return res;
        }
        strcpy(node_data->label, label_s_ptr);
        pr_info("Node 'label': %s\n", node_data->label);

        node_data->p_desc = devm_fwnode_gpiod_get_index(&platform_dev->dev, &child->fwnode, "func", 0, GPIOD_ASIS, node_data->label);
        if (IS_ERR(node_data->p_desc))
        {
            dev_err(&platform_dev->dev, "Can't read the 'func' property\n");
            return PTR_ERR(node_data->p_desc);
        }

        res = gpiod_direction_output(node_data->p_desc, 0);
        if (res != 0)
        {
            dev_err(&platform_dev->dev, "Failed to configure the GPIO output pin\n");
            return res;
        }

        struct device *p_dev = device_create_with_groups(gpiosfs_data.class_p, &platform_dev->dev, 
            0, node_data, gpiosfs_attr_groups, node_data->label);
        if (IS_ERR(p_dev))
        {
            dev_err(&platform_dev->dev, "Failed to create sysfs attributes\n");
            return PTR_ERR(p_dev);
        }
        gpiosfs_data.devs[i] = p_dev;
        ++i;
    }

    return res;
}

static int gpiosfs_remove(struct platform_device * pdev)
{
    pr_info("remove");
    for (int i = 0; i < gpiosfs_data.num_devices; ++i)
    {
        device_unregister(gpiosfs_data.devs[i]);
    }
    return 0;
}

static struct of_device_id gpiosfs_dt_match_table[] = {
    {
        .compatible = DRV_NAME
    },
    {}
};

static struct platform_driver gpiosfs_platform_driver = {
    .driver = {
        .name = DRV_NAME,
        .of_match_table = of_match_ptr(gpiosfs_dt_match_table)
    },
    .probe = gpiosfs_probe,
    .remove = gpiosfs_remove
};

static int __init gpiosfs_module_init(void)
{
    pr_info("init started..\n");
    gpiosfs_data.class_p = class_create(DRV_CLASS_NAME);
    if (IS_ERR(gpiosfs_data.class_p))
    {
        pr_err("Failed to create entry in /sys/class");
        return PTR_ERR(gpiosfs_data.class_p);
    }

    const int res = platform_driver_register(&gpiosfs_platform_driver);
    if (res < 0)
    {
        class_destroy(gpiosfs_data.class_p);
        return res;
    }
    
    return 0;
}

static void __exit gpiosfs_module_exit(void)
{
    platform_driver_unregister(&gpiosfs_platform_driver);
    class_destroy(gpiosfs_data.class_p);
    pr_info("exit\n");
}

module_init(gpiosfs_module_init);
module_exit(gpiosfs_module_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("sebb");
MODULE_DESCRIPTION("Custom GPIO driver using sysfs backend");
