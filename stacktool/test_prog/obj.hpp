#pragma once

class Object {
public:
    virtual ~Object() {}
    virtual void func() = 0;

};

class Derived : public Object
{
public:
    Derived(const int);
    void func() override;
private:
    int value;
};

class Other
{
public:
    virtual ~Other() {}
    int x;
};
