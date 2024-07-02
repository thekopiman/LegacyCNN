#include <iostream>
#include "models/basiccnn.h"
#include "utils/helper.h"
#include "models/ecapa.h"
#include "tests/time.h"

int main()
{
    std::cout << "Running program" << std::endl;

    Time::TestEcapa(100);

    Time::TestBasic(100);

    return 0;
}
