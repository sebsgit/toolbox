#!/bin/sh
# 
# generate valgrind suppression file entry for given name
#
while read line
do
	name=$line
	echo "{\n<"$name"_cond>\nMemcheck:Cond\n...\nobj:"$line"\n...\n}"
	echo "{\n<"$name"_leak>\nMemcheck:Leak\n...\nobj:"$line"\n...\n}"
done
exit 0
