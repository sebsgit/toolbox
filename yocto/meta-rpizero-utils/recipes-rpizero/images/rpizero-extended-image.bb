require recipes-core/images/rpi-test-image.bb

IMAGE_INSTALL:append = " libstdc++ openssh openssl openssh-sftp-server"

