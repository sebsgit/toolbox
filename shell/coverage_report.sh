#!/bin/sh
#
# generate html doc for source code coverage made by gcov
#
lcov --capture --directory . --output-file coverage-gcov.info --no-external
lcov --output-file coverage-gcov.info --remove coverage-gcov.info 'moc_*.cpp' 'qrc_*.cpp*' '*boost*' 'ui_*.h'
genhtml coverage-gcov.info --output-directory doc/coverage
lcov --directory . --zerocounters
