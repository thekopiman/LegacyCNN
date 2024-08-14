#ifndef time_h
#define time_h

#include <chrono>
#include <iostream>
#include "../models/basiccnn.h"
#include "../utils/helper.h"
#include "../models/ecapa.h"
#include "../models/ecapa_classifier.h"

class Time
{
public:
    static void TestEcapa(int iter);
    static void TestBasic(int iter);
    static void TestEcapaClassifier(int iter, int classNo, float lengths);

private:
};

#include "time.cpp"

#endif