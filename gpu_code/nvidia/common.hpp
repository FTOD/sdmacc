#ifndef COMMON_H
#define COMMON_H
inline double utils_time_us(void)
{
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};
#endif // COMMON_H