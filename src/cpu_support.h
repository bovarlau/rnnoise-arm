
#ifndef CPU_SUPPORT_H
#define CPU_SUPPORT_H

#include "opus_types.h"
#include "common.h"

#define OPUS_ARCHMASK 0
static OPUS_INLINE int rnn_select_arch(void)
{
  return 0;
}

#endif
