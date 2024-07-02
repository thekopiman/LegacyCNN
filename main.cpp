#include <iostream>
#include "tests/time.h"

int main()
{
    std::cout << "Running program" << std::endl;

    Time::TestEcapa(100);

    Time::TestBasic(100);

    return 0;
}
