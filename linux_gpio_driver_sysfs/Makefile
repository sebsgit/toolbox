obj-m := gpio_sysfs_driver.o

CROSS_COMP=arm-none-linux-gnueabihf-
KERNEL_DIR=/home/seb/repos/linux

all:
	make ARCH=arm CROSS_COMPILE=$(CROSS_COMP) -C $(KERNEL_DIR) M=$(PWD) modules
clean:
	make ARCH=arm CROSS_COMPILE=$(CROSS_COMP) -C $(KERNEL_DIR) M=$(PWD) clean
