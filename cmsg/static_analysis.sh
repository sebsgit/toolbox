#!/bin/bash
clang-tidy-9 -warnings-as-errors=* -checks=*,-modernize-use-trailing-return-type,-llvm-header-guard,-fuchsia-overloaded-operator cmsg.hpp -- -I.
