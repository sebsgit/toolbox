#include "obj.hpp"
#include <memory>
#include <iostream>

int main(int argc, char **argv)
{
    auto ptr = std::make_shared<Derived>(std::atoi(argv[1]));
    ptr->func();
    auto ptr2 = std::make_shared<Other>();
    std::cout << ptr2.get() << std::endl;
    return 0;
}
