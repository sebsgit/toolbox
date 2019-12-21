#ifndef CMSG_HPP_INCLUDED_
#define CMSG_HPP_INCLUDED_

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cmsg {
namespace priv {
    template <typename T>
    static constexpr auto contains() noexcept
    {
        return false;
    }
    template <typename T, typename First, typename... List>
    static constexpr auto contains()
    {
        return std::is_same<typename std::decay<T>::type, First>::value
            ? true
            : contains<T, List...>();
    }
} // namespace priv

template <typename Param, typename Class>
static auto overload(void (Class::*ptr)(Param)) noexcept
{
    return ptr;
}

template <typename... Types>
class Meta {
    template <typename T>
    struct ConnDataMemBase {
        explicit ConnDataMemBase() noexcept = default;
        ConnDataMemBase(const ConnDataMemBase&) noexcept = default;
        ConnDataMemBase(ConnDataMemBase&&) noexcept = default;
        ConnDataMemBase& operator=(const ConnDataMemBase&) noexcept = default;
        ConnDataMemBase& operator=(ConnDataMemBase&&) noexcept = default;
        virtual ~ConnDataMemBase() = default;
        virtual void call(const T&) = 0;
        virtual void unregisterSender(const void*) = 0;
        virtual void rebindSender(const void*, const void*, const std::function<void(const void*)>&) = 0;
        virtual const void* id() const noexcept = 0;
    };

    template <typename Class, typename T, typename SigFunc>
    struct ConnDataMember : public ConnDataMemBase<T> {
        ConnDataMember(Class* obj, SigFunc cb) noexcept
            : who_(obj)
            , cb_(cb)
        {
        }

        void call(const T& d) override { (who_->*cb_)(d); }
        void unregisterSender(const void* p) override
        {
            who_->unregisterSender(p);
        }
        void rebindSender(const void* oldSender, const void* newSender, const std::function<void(const void*)>& cb) override
        {
            who_->rebindSender(oldSender, newSender, cb);
        }
        const void* id() const noexcept override { return who_; }

    private:
        Class* who_ { nullptr };
        SigFunc cb_ { nullptr };
    };

    template <typename R>
    using ReceiverData = std::vector<
        std::unique_ptr<ConnDataMemBase<typename std::decay<R>::type>>>;

public:
    explicit Meta(const void* s) noexcept
        : sender_ { s }
    {
    }

    Meta(const Meta&) = delete;
    Meta& operator=(const Meta&) = delete;

    Meta(Meta&& other) noexcept
        : data_ { std::move(other.data_) }
        , sender_ { other.sender_ }
    {
    }
    Meta& operator=(Meta&& other) noexcept
    {
        if (this != &other) {
            data_ = std::move(other.data_);
            sender_ = other.sender_;
        }
        return *this;
    }

    ~Meta() noexcept
    {
        static_cast<void>(
            std::initializer_list<int> { (callCleanups<Types>(), 0)... });
    }

    template <typename Signal, typename Class, typename SigFunc>
    void add(Class* who, SigFunc ptr) noexcept
    {
        auto& vc = std::get<ReceiverData<Signal>>(data_);
        vc.emplace_back(
            std::make_unique<ConnDataMember<Class, Signal, SigFunc>>(who, ptr));
    }

    template <typename Signal>
    void remove(const void* who) noexcept
    {
        auto& vc = std::get<ReceiverData<Signal>>(data_);
        auto it = std::remove_if(vc.begin(), vc.end(),
            [who](const auto& d) { return d->id() == who; });
        vc.erase(it, vc.end());
    }

    template <typename Signal>
    void send(Signal&& s) noexcept
    {
        auto& vc = std::get<ReceiverData<Signal>>(data_);
        if (vc.size() == 1) {
            vc.front()->call(std::forward<Signal>(s));
        } else {
            for (auto& rcv : vc) {
                rcv->call(s);
            }
        }
    }

    void remove(const void* who) noexcept
    {
        static_cast<void>(std::initializer_list<int> { (remove<Types>(who), 0)... });
    }

    auto numberOfConnections() const noexcept
    {
        size_t result { 0 };
        static_cast<void>(std::initializer_list<size_t> {
            result += countConnections<Types>()... });
        return result;
    }

    void rebindSender(const void* oldS, const void* newS, const std::function<void(const void*)>& cb) noexcept
    {
        sender_ = newS;
        static_cast<void>(std::initializer_list<int> { (rebindSenders<Types>(oldS, newS, cb), 0)... });
    }

private:
    template <typename Signal>
    void callCleanups() noexcept
    {
        auto& vc = std::get<ReceiverData<Signal>>(data_);
        for (auto& d : vc) {
            d->unregisterSender(sender_);
        }
    }

    template <typename Signal>
    auto countConnections() const noexcept
    {
        const auto& vc = std::get<ReceiverData<Signal>>(data_);
        return vc.size();
    }

    template <typename Signal>
    void rebindSenders(const void* oldS, const void* newS, const std::function<void(const void*)>& cb) noexcept
    {
        auto& vc = std::get<ReceiverData<Signal>>(data_);
        for (auto& d : vc) {
            d->rebindSender(oldS, newS, cb);
        }
    }

private:
    std::tuple<ReceiverData<Types>...> data_;
    const void* sender_ { nullptr };
};

template <typename... Types>
class Signal final {
public:
    explicit Signal() noexcept = default;

    Signal(const Signal&) = delete;
    Signal& operator=(const Signal&) = delete;

    Signal(Signal&& other) noexcept
    {
        other.meta_.rebindSender(&other, this, [this](auto* r) { meta_.remove(r); });
        meta_ = std::move(other.meta_);
    }
    Signal& operator=(Signal&& other) noexcept
    {
        if (this != &other) {
            other.meta_.rebindSender(&other, this, [this](auto* r) { meta_.remove(r); });
            meta_ = std::move(other.meta_);
        }
        return *this;
    }

    ~Signal() noexcept = default;

    template <typename T>
    static constexpr auto isSending() noexcept
    {
        return priv::contains<T, Types...>();
    }

    template <typename T>
    void operator()(T&& t) noexcept
    {
        static_assert(isSending<T>(), "Type not registered for sending");
        meta_.template send<T>(std::forward<T>(t));
    }

    template <typename T, typename IRecv, typename SigParam>
    void connect(IRecv* r, void (IRecv::*ptr)(SigParam)) noexcept
    {
        static_assert(IRecv::template isReceiving<T>(),
            "Type not registered for receiving");
        static_assert(isSending<T>(), "Type not registered for sending");

        meta_.template add<T, IRecv>(r, ptr);
        r->registerSender(this, [this](auto* r) { meta_.remove(r); });
    }

    template <typename IRecv, typename SigParam,
        typename std::enable_if<sizeof...(Types) == 1>* = nullptr>
    void connect(IRecv* r, void (IRecv::*ptr)(SigParam)) noexcept
    {
        return connect<Types...>(r, ptr);
    }

    template <typename IRecv,
        typename std::enable_if<sizeof...(Types) == 1>* = nullptr>
    void connect(IRecv* r) noexcept
    {
        return connect<Types...>(r, overload<Types...>(&IRecv::receive));
    }

    template <typename T, typename IRecv>
    void disconnect(IRecv* who)
    {
        meta_.template remove<T>(who);
        who->unregisterSender(this);
    }

    template <typename IRecv>
    void disconnect(IRecv* who) noexcept
    {
        meta_.remove(who);
        who->deleteSender(this);
    }

    auto numberOfConnections() const noexcept
    {
        return meta_.numberOfConnections();
    }

private:
    Meta<Types...> meta_ { this };
};

template <typename... Types>
class Receiver {
    template <typename...>
    friend class Signal;
    template <typename...>
    friend class Meta;

    struct SenderData {
        const void* who_ { nullptr };
        std::function<void(Receiver<Types...>*)> cb_;
        int sigCount_ { 0 };
    };

public:
    explicit Receiver() noexcept = default;

    Receiver(const Receiver&) = delete;
    Receiver& operator=(const Receiver&) = delete;

    Receiver(Receiver&& other) = delete;
    Receiver& operator=(Receiver&& other) = delete;

    virtual ~Receiver() noexcept
    {
        for (auto& d : data_) {
            d.cb_(this);
        }
    }

    template <typename T>
    static constexpr auto isReceiving() noexcept
    {
        return priv::contains<T, Types...>();
    }

    auto numberOfSenders() const noexcept { return data_.size(); }
    auto numberOfConnections() const noexcept
    {
        int result { 0 };
        for (auto& d : data_) {
            result += d.sigCount_;
        }
        return result;
    }

private:
    void unregisterSender(const void* who)
    {
        auto it = std::find_if(data_.begin(), data_.end(),
            [who](const auto& d) { return d.who_ == who; });

        if (it != data_.end()) {
            --it->sigCount_;
            if (it->sigCount_ == 0) {
                data_.erase(it);
            }
        }
    }

    void deleteSender(const void* who)
    {
        auto it = std::remove_if(data_.begin(), data_.end(),
            [who](const auto& d) { return d.who_ == who; });
        data_.erase(it, data_.end());
    }

    void registerSender(const void* who, std::function<void(Receiver<Types...>*)>&& cb)
    {
        auto it = std::find_if(data_.begin(), data_.end(),
            [who](const auto& d) { return d.who_ == who; });
        if (it == data_.end()) {
            data_.push_back(SenderData { who, std::move(cb), 1 });
        } else {
            ++it->sigCount_;
        }
    }

    void rebindSender(const void* who, const void* newS, const std::function<void(Receiver<Types...>*)>& cb) noexcept
    {
        auto it = std::find_if(data_.begin(), data_.end(),
            [who](const auto& d) { return d.who_ == who; });
        if (it != data_.end()) {
            it->who_ = newS;
            it->cb_ = cb;
        }
    }

private:
    std::vector<SenderData> data_;
};
} // namespace cmsg

#endif // CMSG_HPP_INCLUDED_
