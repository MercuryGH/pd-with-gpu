#pragma once

namespace util
{
    // from https://codereview.stackexchange.com/questions/173929/modern-c-singleton-template
    template<typename T>
    class Singleton
    {
    public:
        static T& instance();

        Singleton(const Singleton&) = delete;
        Singleton& operator= (const Singleton) = delete;

    protected:
        struct token {};
        Singleton() {}
    };

    template<typename T>
    T& Singleton<T>::instance()
    {
        static T instance{token{}};
        return instance;
    }
}
