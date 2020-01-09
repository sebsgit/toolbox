#include "cmsg.hpp"

#include <atomic>
#include <iostream>
#include <tuple>

static std::atomic_int assrtCntr { 0 };
#define ASSERT(x)                                                                              \
    if (!(x)) {                                                                                \
        std::cout << "ASSERTION failure -> " << __FILE__ << ": " << __LINE__ << " : " #x "\n"; \
        exit(1);                                                                               \
    } else {                                                                                   \
        ++assrtCntr;                                                                           \
    }

using namespace cmsg;

template <typename T>
struct Variable : Receiver<T> {
    T x_ {};

    void receive(T t) noexcept
    {
        x_ = t;
    }
};

template <typename T>
struct Slot : Receiver<T> {
    std::function<void(T)> cb_;

    template <typename F>
    explicit Slot(F f) noexcept
        : cb_ { f }
    {
    }

    void receive(T t) noexcept
    {
        cb_(t);
    }
};

static void testSignalTypes()
{
    static_assert(Signal<int, double>::isSending<int>(), "");
    static_assert(Signal<int, double>::isSending<double>(), "");
    static_assert(!Signal<int, double>::isSending<float>(), "");

    static_assert(Receiver<float, char>::isReceiving<float>(), "");
    static_assert(Receiver<float, char>::isReceiving<char>(), "");
    static_assert(!Receiver<float, char>::isReceiving<int>(), "");
}

static void testSimpleSlots()
{
    Signal<std::pair<int, float>> sign0;

    Variable<std::pair<int, float>> var0;

    sign0.connect(&var0, &Variable<std::pair<int, float>>::receive);

    sign0(std::make_pair(1, 4.13f));
    ASSERT(var0.x_.first == 1);
    ASSERT(var0.x_.second == 4.13f);

    sign0(std::make_pair(123, -0.421f));
    ASSERT(var0.x_.first == 123);
    ASSERT(var0.x_.second == -0.421f);
}

static void testDisconnects()
{
    Signal<double> sig0;
    Variable<double> var;

    ASSERT(sig0.numberOfConnections() == 0);
    ASSERT(var.numberOfSenders() == 0);

    sig0.connect(&var);
    ASSERT(sig0.numberOfConnections() == 1);
    ASSERT(var.numberOfSenders() == 1);

    sig0.disconnect(&var);

    ASSERT(sig0.numberOfConnections() == 0);
    ASSERT(var.numberOfSenders() == 0);

    Variable<int> var2;
    {
        Signal<int> sig1;
        sig1.connect(&var2);

        ASSERT(sig1.numberOfConnections() == 1);
        ASSERT(var2.numberOfSenders() == 1);
    }
    ASSERT(var2.numberOfSenders() == 0);
}

static void testMoveConstructSignals()
{
    Signal<double> sig0;
    Variable<double> v;

    sig0.connect(&v);
    ASSERT(sig0.numberOfConnections() == 1);

    sig0(3.14);
    ASSERT(v.x_ == 3.14);

    Signal<double> newSig { std::move(sig0) };
    ASSERT(sig0.numberOfConnections() == 0);

    newSig(4.12);
    ASSERT(v.x_ == 4.12);
}

static void testMoveAssignSignals()
{
    Signal<double> sig0;
    Variable<double> v;

    sig0.connect(&v);
    ASSERT(sig0.numberOfConnections() == 1);

    sig0(3.14);
    ASSERT(v.x_ == 3.14);

    Signal<double> newSig;
    newSig = std::move(sig0);
    ASSERT(sig0.numberOfConnections() == 0);

    newSig(4.12);
    ASSERT(v.x_ == 4.12);
}

static void testMoveConstructAndDeleteReceiver()
{
    int val { 0 };
    Signal<int> sig0;
    std::unique_ptr<Slot<int>> recv { std::make_unique<Slot<int>>([&val](int newVal) { val = newVal; }) };

    sig0.connect(recv.get());
    ASSERT(sig0.numberOfConnections() == 1);

    Signal<int> sig1 = std::move(sig0);
    ASSERT(sig0.numberOfConnections() == 0);
    ASSERT(sig1.numberOfConnections() == 1);

    recv.reset();
    ASSERT(sig1.numberOfConnections() == 0);
}

static void testMoveAssignAndDeleteReceiver()
{
    int val { 0 };
    Signal<int> sig0;
    std::unique_ptr<Slot<int>> recv { std::make_unique<Slot<int>>([&val](int newVal) { val = newVal; }) };

    sig0.connect(recv.get());
    ASSERT(sig0.numberOfConnections() == 1);

    Signal<int> sig1;
    sig1 = std::move(sig0);
    ASSERT(sig0.numberOfConnections() == 0);
    ASSERT(sig1.numberOfConnections() == 1);

    recv.reset();
    ASSERT(sig1.numberOfConnections() == 0);
}

static void testAutoDisconnect()
{
    Signal<int> sig0;
    int intCounter { 0 };

    {
        Slot<int> slo0 { [&](int x) { intCounter += x; } };
        ASSERT(intCounter == 0);
        sig0.connect<int>(&slo0, &Slot<int>::receive);
        sig0(1);
        ASSERT(intCounter == 1);
        sig0(-32);
        ASSERT(intCounter == -31);
        sig0(0);
        ASSERT(intCounter == -31);
    }

    ASSERT(intCounter == -31);
    sig0(10);
    ASSERT(intCounter == -31);
}

static void testManualDisconnect()
{
    Signal<std::string> sig0;
    Variable<std::string> var0;
    Variable<std::string> var1;

    sig0.connect(&var0, &Variable<std::string>::receive);
    sig0.connect(&var1);

    ASSERT(var0.x_ != "test");
    ASSERT(var1.x_ != "test");

    sig0(std::string("test"));
    ASSERT(var0.x_ == "test");
    ASSERT(var1.x_ == "test");

    sig0.disconnect(&var0);
    sig0(std::string("after disconnect"));

    ASSERT(var0.x_ == "test");
    ASSERT(var1.x_ == "after disconnect");
}

static void testSelectiveDisconnect()
{
    class Recv2 : public Receiver<int, double> {
    public:
        Recv2(int* outI, double* outD) noexcept
            : outI_ { outI }
            , outD_ { outD }
        {
        }

        void receive(int i) { *outI_ = i; }
        void receive(double d) { *outD_ = d; }

    private:
        int* outI_;
        double* outD_;
    };

    int i { 0 };
    double d { 0.0 };
    Recv2 recv2(&i, &d);

    {
        Signal<int, double, float> sig0;

        {
            Variable<float> var;

            sig0.connect<float>(&var, &Variable<float>::receive);
            sig0.connect<int>(&recv2, cmsg::overload<int>(&Recv2::receive));
            sig0.connect<double>(&recv2, cmsg::overload<double>(&Recv2::receive));

            ASSERT(var.numberOfSenders() == 1);
            ASSERT(recv2.numberOfSenders() == 1);
            ASSERT(recv2.numberOfConnections() == 2);
            ASSERT(sig0.numberOfConnections() == 3);

            sig0(10);
            ASSERT(i == 10);
            sig0(1.23f);
            ASSERT(var.x_ == 1.23f);
            sig0(0.44);
            ASSERT(d == 0.44);

            sig0.disconnect<double>(&recv2);
            ASSERT(recv2.numberOfSenders() == 1);
            ASSERT(recv2.numberOfConnections() == 1);
            ASSERT(sig0.numberOfConnections() == 2);

            sig0(15);
            ASSERT(i == 15);
            sig0(-81.23f);
            ASSERT(var.x_ == -81.23f);
            sig0(4.6632);
            ASSERT(d == 0.44);

            sig0.connect<double>(&recv2, cmsg::overload<double>(&Recv2::receive));
            ASSERT(recv2.numberOfSenders() == 1);
            ASSERT(recv2.numberOfConnections() == 2);
            sig0(5.273);
            ASSERT(d == 5.273);
            ASSERT(sig0.numberOfConnections() == 3);

            sig0.disconnect(&recv2);
            ASSERT(recv2.numberOfConnections() == 0);
            ASSERT(recv2.numberOfSenders() == 0);
            ASSERT(sig0.numberOfConnections() == 1);

            sig0(-88);
            ASSERT(i == 15);
            sig0(92.0f);
            ASSERT(var.x_ == 92.0f);
            sig0(833.22);
            ASSERT(d == 5.273);
        }
        ASSERT(sig0.numberOfConnections() == 0);
    }
    ASSERT(recv2.numberOfSenders() == 0);
}

static void testDisconnectSingleChannelAndDelete()
{
    class Recv2 : public Receiver<int, double> {
    public:
        Recv2(int* outI, double* outD) noexcept
            : outI_ { outI }
            , outD_ { outD }
        {
        }

        void receive(int i) { *outI_ = i; }
        void receive(double d) { *outD_ = d; }

    private:
        int* outI_;
        double* outD_;
    };

    int i { 0 };
    double d { 0.0 };
    Recv2 recv2(&i, &d);

    Signal<double, int> sig0;

    sig0.connect<int>(&recv2, cmsg::overload<int>(&Recv2::receive));
    sig0.connect<double>(&recv2, cmsg::overload<double>(&Recv2::receive));

    ASSERT(sig0.numberOfConnections() == 2);
    ASSERT(recv2.numberOfConnections() == 2);
    ASSERT(recv2.numberOfSenders() == 1);

    sig0.disconnect<int>(&recv2);
    ASSERT(sig0.numberOfConnections() == 1);
    ASSERT(recv2.numberOfConnections() == 1);
    ASSERT(recv2.numberOfSenders() == 1);

    sig0(0.23);
    ASSERT(d == 0.23);
}

static void testDisconnectFromInsideClass()
{
    class Recv : public Receiver<int> {
    public:
        Signal<int>* sender { nullptr };
        int signalCount { 0 };

        Recv(Signal<int>* s)
            : sender(s)
        {
        }

        void receive(int) noexcept
        {
            ++signalCount;
            if (signalCount == 10) {
                sender->disconnect(this);
            }
        }
    };

    Signal<int> sig0;
    Recv r(&sig0);

    sig0.connect(&r);

    for (int i = 0; i < 100; ++i) {
        sig0(i);
    }
    ASSERT(r.signalCount == 10);
}

static void testDeleteReceiverBeforeSender()
{
    int var { 0 };
    cmsg::Signal<int> sig0;
    Variable<int> var2;
    std::unique_ptr<Slot<int>> recv(new Slot<int>([&var](int newVal) { var = newVal; }));

    ASSERT(sig0.numberOfConnections() == 0);

    sig0.connect(&var2);

    ASSERT(sig0.numberOfConnections() == 1);
    sig0.connect(recv.get());

    ASSERT(sig0.numberOfConnections() == 2);

    ASSERT(var == 0);
    ASSERT(var2.x_ == 0);
    sig0(12);
    ASSERT(var == 12);
    ASSERT(var2.x_ == 12);

    recv.reset();
    ASSERT(sig0.numberOfConnections() == 1);

    sig0(73);
    ASSERT(var == 12);
    ASSERT(var2.x_ == 73);
}

static void testDeleteSenderBeforeReceiver()
{
    std::unique_ptr<cmsg::Signal<int>> sig0(new cmsg::Signal<int>());
    std::unique_ptr<cmsg::Signal<int>> sig1(new cmsg::Signal<int>());
    Variable<int> recv;

    sig0->connect(&recv);
    sig1->connect(&recv);

    ASSERT(recv.numberOfSenders() == 2);

    (*sig0)(9);
    ASSERT(recv.x_ == 9);
    (*sig1)(-13);
    ASSERT(recv.x_ == -13);

    sig0.reset();
    ASSERT(recv.numberOfSenders() == 1);

    (*sig1)(23);
    ASSERT(recv.x_ == 23);

    sig1.reset();
    ASSERT(recv.numberOfSenders() == 0);
}

int main()
{
    testSignalTypes();
    testSimpleSlots();
    testDisconnects();
    testMoveConstructSignals();
    testMoveAssignSignals();
    testMoveConstructAndDeleteReceiver();
    testMoveAssignAndDeleteReceiver();
    testAutoDisconnect();
    testManualDisconnect();
    testDisconnectFromInsideClass();
    testDeleteReceiverBeforeSender();
    testDeleteSenderBeforeReceiver();
    testSelectiveDisconnect();
    testDisconnectSingleChannelAndDelete();

    std::cout << "All tests done (" << assrtCntr << " assertions).\n";
    return 0;
}
