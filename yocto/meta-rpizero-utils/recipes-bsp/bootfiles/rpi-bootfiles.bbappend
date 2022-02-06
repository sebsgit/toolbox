SUMMARY = "Adds empty 'ssh' file to the boot partition"

do_deploy:append() {
    touch ${DEPLOYDIR}/${BOOTFILES_DIR_NAME}/ssh
}

