GPIO driver for exposing the pin controls through the sysfs interface.

The affected GPIO pins are defined in the device tree file. The device tree first disables the 
pins used in the default "boneblack" device tree by the "user leds" module.

The example device tree uses the "pinctrl" subsystem to configure the pin muxing, to set the selected pins as GPIOs by default.
