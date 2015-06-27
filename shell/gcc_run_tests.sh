#!/bin/sh

# script used to launch gcc test suite and
# generate report in txt file ready to be sent by email
#
# created as workaround for 'mail' issues
#

svn update
if [ $? -ne 0 ]; then
	echo "svn status error: $?";
	exit;
fi;
make
if [ $? -ne 0 ]; then
	echo "make error: $?";
	exit;
fi;
make -j3 -k check
if [ ! -d ~/gcc_reports/ ]; then
	mkdir ~/gcc_reports
fi;
out_file=~/gcc_reports/`date +"%m%d%Y-%H%M"`.txt
touch $out_file
sh gcc_test_gen_preamble.sh > $out_file
echo "\n\n" >> $out_file
contrib/test_summary >> $out_file
less $out_file | awk '{ if($0!~"^cat" && $0!~"^mv" && $0!~"^EOF")print;}' > tmp.txt
mv tmp.txt $out_file
svn info | grep Wersja | awk '{print "[trunk revision "$2"]";}' >> $out_file
rm -f tmp.txt
