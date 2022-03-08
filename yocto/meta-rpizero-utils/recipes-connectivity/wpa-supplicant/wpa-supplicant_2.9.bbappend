SUMMARY = "Replaces the /etc/wpa_supplicant.conf file with a predefined wifi config"

#
# this recipe requires the network_settings.inc file to define the following variables
#   WIFI_SSID = "YourNetworkName"
#   WIFI_PASSWD = "YourSecretPasswd"
#
require ../network_settings.inc

FILESEXTRAPATHS:append := ":${THISDIR}/files/"

SRC_URI += "file://wpa_supplicant.conf.source"

do_install:append() {
    rm -rf ${D}/etc/wpa_supplicant.conf
    sed -i 's/{RPIZEROWEXTRAS_NET_SSID}/${WIFI_SSID}/g' ${WORKDIR}/wpa_supplicant.conf.source
    sed -i 's/{RPIZEROWEXTRAS_NET_PASSWD}/${WIFI_PASSWD}/g' ${WORKDIR}/wpa_supplicant.conf.source
    install -m 644 ${WORKDIR}/wpa_supplicant.conf.source ${D}/etc/wpa_supplicant.conf
}

