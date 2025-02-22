#include "obj.hpp"
#include <iostream>

Derived::Derived(const int val) : value(val) {}

void Derived::func() {
    std::cout << "Derived::func() called with value: " << value << std::endl;
}

