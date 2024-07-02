#ifndef time_h
#define time_h

#include <chrono>
#include <iostream>
#include "../models/basiccnn.h"
#include "../utils/helper.h"
#include "../models/ecapa.h"

class Time
{
public:
    static void TestEcapa(int iter);
    static void TestBasic(int iter);

private:
};

#include "time.cpp"

#endif