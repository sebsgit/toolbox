#!/bin/bash

set -e

for path in "$@"
do
	# workaround for issues when calling castxml on .h headers - incorrectly interpreted as C sources
	TMP_CXX=$(mktemp --suffix .cpp)
	cat $path > $TMP_CXX
	
	TMP_XML=$(mktemp --suffix .xml)
	/opt/castxml/bin/castxml --castxml-gccxml --castxml-cc-gnu g++ $TMP_CXX -o $TMP_XML
	
	OUTPUT_FILE=/var/opt/castxml/output/autogen_$(basename $path)
	python3 /var/opt/castxml/scripts/generate_enum_utils.py $TMP_XML > $OUTPUT_FILE
	echo "Code generated in: $(basename $OUTPUT_FILE)"
done
