SUMMARY = "Sets up the wifi init.d script"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

#
# this recipe requires the network_settings.inc file to define the following variables
#   RPI_LOCAL_IP = "192.168.0.99"
#   ROUTER_IP = "192.168.0.1"
#
require ../network_settings.inc

inherit update-rc.d
INITSCRIPT_NAME = "setup-wifi"
INITSCRIPT_PARAMS = "defaults 90 10"

FILESEXTRAPATHS:append := ":${THISDIR}/files/"

SRC_URI += "file://setup-wifi"

S = "${WORKDIR}"

do_install() {
    install -d ${D}/etc/init.d
    sed -i 's/{RPI_LOCAL_IP}/${RPI_LOCAL_IP}/g' ${WORKDIR}/setup-wifi
    sed -i 's/{ROUTER_IP}/${ROUTER_IP}/g' ${WORKDIR}/setup-wifi
    install -m 0755 ${WORKDIR}/setup-wifi ${D}/etc/init.d
}

