SUMMARY = "Replaces the /etc/wpa_supplicant.conf file with a predefined wifi config"

FILESEXTRAPATHS:append := ":${THISDIR}/files/"

SRC_URI += "file://wpa_supplicant.conf.source"

WIFI_SSID = "MyNetwork"
WIFI_PASSWD = "secret_pass"

do_install:append() {
    rm -rf ${D}/etc/wpa_supplicant.conf
    sed -i 's/{RPIZEROWEXTRAS_NET_SSID}/${WIFI_SSID}/g' ${WORKDIR}/wpa_supplicant.conf.source
    sed -i 's/{RPIZEROWEXTRAS_NET_PASSWD}/${WIFI_PASSWD}/g' ${WORKDIR}/wpa_supplicant.conf.source
    install -m 600 ${WORKDIR}/wpa_supplicant.conf.source ${D}/etc/wpa_supplicant.conf
}
