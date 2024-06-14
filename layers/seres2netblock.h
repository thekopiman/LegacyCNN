#ifndef seres2netblock_h
#define seres2netblock_h

#include "conv1d.h"
#include "../utils/helper.h"
#include "../utils/matrixfunctions.h"
#include "tdnnblock.h"
#include "res2netblock.h"
#include <string>
#include <assert.h>

template <int kernel, int channel_in, int channel_out, int dilation, int input_width, int out_dim, int input_pad, int scale, typename T>
class SERes2NetBlock
{
public:
    SERes2NetBlock() {}

private:
};

#endif
