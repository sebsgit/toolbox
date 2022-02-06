SUMMARY = "Replaces the /etc/wpa_supplicant.conf file with a predefined wifi config"

do_install:append() {
    echo "country=us
update_config=1
ctrl_interface=/var/run/wpa_supplicant

network={
 scan_ssid=1
 ssid="-network ssid-"
 psk="-network passwd-"
}" > ${WORKDIR}/wpa_supplicant.conf
    rm -rf ${D}/etc/wpa_supplicant.conf
    install -m 600 ${WORKDIR}/wpa_supplicant.conf ${D}/etc/
}

