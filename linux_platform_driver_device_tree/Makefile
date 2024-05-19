obj-m := fakeplatformdriver.o

CROSS_COMP=arm-none-linux-gnueabihf-
KERNEL_DIR=/home/seb/repos/linux

all:
	make ARCH=arm CROSS_COMPILE=$(CROSS_COMP) -C $(KERNEL_DIR) M=$(PWD) modules
clean:
	make ARCH=arm CROSS_COMPILE=$(CROSS_COMP) -C $(KERNEL_DIR) M=$(PWD) clean

host:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
host_clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
host_insmod: host
	sudo insmod fakeplatformdevice.ko && sudo insmod fakeplatformdriver.ko && sudo dmesg | tail
host_rmmod:
	sudo rmmod fakeplatformdriver.ko && sudo rmmod fakeplatformdevice.ko && sudo dmesg | tail

