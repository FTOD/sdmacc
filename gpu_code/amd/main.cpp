#include "fp32.hpp"

int main()
{
    int repeat = 10;
    double total_time = 0;

    for (int i = 0; i < repeat; i++)
    {
        total_time += sliding_window_fp32();
    }
    total_time /= repeat;

    std::cout << "sliding window time = " << total_time << std::endl;
}