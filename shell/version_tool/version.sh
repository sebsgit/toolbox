#!/bin/bash

# (C) 2015
# Originally by Sebastian Baginski <s.baginski@mail.com>

# This script is Free Software, and it can be copied, distributed and
# modified as defined in the GNU General Public License.  A copy of
# its license can be downloaded from http://www.gnu.org/copyleft/gpl.html

# Script checks the version string for the command given as argument. It tries to parse
# the command's standard '--version' (etc.) output to get the M.m.p string (Major, minor and patch numbers).
#

EMAIL="s.baginski@mail.com"
AUTHOR="Sebastian Baginski ($EMAIL)"
COPYRIGHT="Copyright (C) 2015 Sebastian Baginski\nThis is free software; see the source for copying conditions. There is NO \nwarranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."
VERSION=0.0.1

print_version(){
	echo "version (ver. $VERSION)";
    echo -e $COPYRIGHT
    echo "Report bugs to $EMAIL"
	exit 0;
}
print_help(){
	echo "Usage: version [options] file..."
	exit $1;
}
print_detailed_help(){
    echo "Usage: version [options] file..."
    echo "Options:";
    echo -e "  -M\t\t\tUse only major version number"
    echo -e "  -m\t\t\tUse only minor version number"
    echo -e "  -p\t\t\tUse only patch version number"
    echo -e "  -eq <value>\t\tPrints 1 if version is equal to <value>, otherwise 0"
    echo -e "  -ne <value>\t\tPrints 1 if version is not equal to <value>"
    echo -e "  -lt <value>\t\tPrints 1 if version is less than or equal to <value>"
    echo -e "  -le <value>\t\tPrints 1 if version is less than or equal to <value>"
    echo -e "  -gt <value>\t\tPrints 1 if version is greater than <value>"
    echo -e "  -ge <value>\t\tPrints 1 if version is greater than or equal to <value>"
    echo -e "  -h, --help\t\tPrint this help page"
    echo -e "  -v, --version\t\tPrint version information"
    echo -e "\nExample usage:";
    echo -e "  version gcc\t\tThis will print the installed gcc version to stdout.\n\t\t\tIf gcc is not found, then nothing is printed and\n\t\t\tscript exits with non-zero value."
    echo -e "  version -gt 4.5 gcc\tIf gcc version installed is greater than 4.5.0 script\n\t\t\tprints 1, otherwise 0."
    echo -e "  version -M -lt 3 awk\tIf make major version installed is less than 3 script\n\t\t\tprints 1, otherwise 0"
    echo -e "\nKnown bugs:"
    echo -e "  If the command being checked does not recognize the standard 'version'";
    echo -e "  switches you may experience weird behaviour, as the script tries to call"
    echo -e "  the command with the --version type of switches. For example some programs"
    echo -e "  may want to try to open the file named '-V'. Such bugs should be reported to"
    echo -e "  and get fixed by their authors."
    echo "";
    print_version;
}

if [ $# -eq 0 ]; then
    print_help 1;
fi

while true; do
	case "$1" in
        -h) print_detailed_help 0; shift;;
        --help) print_detailed_help 0; shift;;
		-v) print_version; shift;;
		--version) print_version; shift;;
		-M) only_major=1; shift;;
		-m) only_minor=1; shift;;
		-p) only_patch=1; shift;;
		-eq) op="eq"; cmp_to=$2; shift 2;;
		-ne) op="ne"; cmp_to=$2; shift 2;;
		-lt) op="lt"; cmp_to=$2; shift 2;;
		-le) op="le"; cmp_to=$2; shift 2;;
		-ge) op="ge"; cmp_to=$2; shift 2;;
		-gt) op="gt"; cmp_to=$2; shift 2;;
		-*) echo "$0: unrecognized command line option: $1"; exit 1;;
		*) CMD=$1; break;;
	esac
done

if [[ $CMD == "" ]]; then
	echo "$0: file not specified"
	print_help 1
fi

cmd_path=`which $CMD`
if [ $? -ne 0 ]; then
    exit 127;
fi

## some tools report version info on stderr...
for opt in --version -v -V -version; do
	tmp_output=`$cmd_path $opt 2>&1`
	if [ $? -eq 0 ]; then
		ver_swtch=$opt
		break;
	fi
done
if [[ $ver_swtch == "" ]]; then
	echo "$cmd_path does not support version switch"
	exit 1
fi
rx="([0-9]+)[\.]([0-9]+)([\.]([0-9])+)?"
for tmp in $tmp_output; do
	if [[ $tmp =~ $rx ]]; then
		ver_found=${BASH_REMATCH[0]}
		ver_major=${BASH_REMATCH[1]}
		ver_minor=${BASH_REMATCH[2]}
		if [[ ${#BASH_REMATCH} -gt 3 ]]; then
			ver_patch=${BASH_REMATCH[4]}
		fi
		break
	fi
done

if [[ $cmp_to != "" ]]; then
    rx="^([0-9])+([\.]([0-9]+)([\.]([0-9]+))?)?"
    if [[ $cmp_to =~ $rx ]]; then
        cmp_to=${BASH_REMATCH[0]}
        cmp_major=${BASH_REMATCH[1]}
        if [[ ${#BASH_REMATCH} -gt 2 ]]; then
            cmp_minor=${BASH_REMATCH[3]}
        fi
        if [[ ${#BASH_REMATCH} -gt 4 ]]; then
            cmp_patch=${BASH_REMATCH[5]}
        fi
    fi
fi

# attempt to fix compare version string as user is allowed to give just single number,
# depending on the p,m or M switches it should be interpreted correctly
if [[ $cmp_to != "" ]] && [[ $cmp_minor == "" ]] && [[ $cmp_patch == "" ]]; then
    if [[ $only_minor -eq 1 ]]; then
        cmp_minor=$cmp_major
    elif [[ $only_patch -eq 1 ]]; then
        cmp_patch=$cmp_major
    fi
fi

if [[ $op != "" ]] ; then
    if [[ $cmp_minor == "" ]] ; then let cmp_minor=0; fi
    if [[ $cmp_patch == "" ]] ; then let cmp_patch=0; fi
    if [[ $ver_patch == "" ]] ; then let ver_patch=0; fi
    declare -i ver_value;
    declare -i cmp_value;
    if [[ $only_major -eq 1 ]]; then
        let ver_value=$ver_major
        let cmp_value=$cmp_major
    elif [[ $only_minor -eq 1 ]]; then
        let ver_value=$ver_minor
        let cmp_value=$cmp_minor
    elif [[ $only_patch -eq 1 ]]; then
        let ver_value=$ver_patch
        let cmp_value=$cmp_patch
    else
        let ver_value=$ver_major*100000+$ver_minor*1000+$ver_patch;
        let cmp_value=$cmp_major*100000+$cmp_minor*1000+$cmp_patch;
    fi
    if [[ $op == "eq" ]]; then
        if [[ $ver_value -eq $cmp_value ]]; then echo 1; else echo 0; fi
    elif [[ $op == "ne" ]]; then
        if [[ $ver_value -ne $cmp_value ]]; then echo 1; else echo 0; fi
    elif [[ $op == "lt" ]]; then
        if [[ $ver_value -lt $cmp_value ]]; then echo 1; else echo 0; fi
    elif [[ $op == "le" ]]; then
        if [[ $ver_value -le $cmp_value ]]; then echo 1; else echo 0; fi
    elif [[ $op == "ge" ]]; then
        if [[ $ver_value -ge $cmp_value ]]; then echo 1; else echo 0; fi
    elif [[ $op == "gt" ]]; then
        if [[ $ver_value -gt $cmp_value ]]; then echo 1; else echo 0; fi
    else
        echo "$0: unrecognized operator: $op"
        exit 1
    fi
    exit 0
fi

if [[ $ver_found != "" ]]; then
	if [[ $only_major -eq 1 ]]; then
		echo $ver_major
	elif [[ $only_minor -eq 1 ]]; then
		echo $ver_minor
	elif [[ $only_patch -eq 1 ]]; then
		echo $ver_patch
	else
		echo $ver_found
	fi
else
	exit 127
fi
