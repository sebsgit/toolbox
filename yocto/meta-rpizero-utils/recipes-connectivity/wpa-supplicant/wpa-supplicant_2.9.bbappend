SUMMARY = "Replaces the /etc/wpa_supplicant.conf file with a predefined wifi config"

FILESEXTRAPATHS:append := ":${THISDIR}/files/"

SRC_URI += "file://wpa_supplicant.conf.source" 

do_install:append() {
    rm -rf ${D}/etc/wpa_supplicant.conf
    install -m 600 ${WORKDIR}/wpa_supplicant.conf.source ${D}/etc/wpa_supplicant.conf
}
