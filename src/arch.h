
#ifndef ARCH_H
#define ARCH_H

#include "opus_types.h"
#include "common.h"

# if !defined(__GNUC_PREREQ)
#  if defined(__GNUC__)&&defined(__GNUC_MINOR__)
#   define __GNUC_PREREQ(_maj,_min) \
 ((__GNUC__<<16)+__GNUC_MINOR__>=((_maj)<<16)+(_min))
#  else
#   define __GNUC_PREREQ(_maj,_min) 0
#  endif
# endif

#define CELT_SIG_SCALE 32768.f

#define celt_fatal(str) _celt_fatal(str, __FILE__, __LINE__);
#ifdef ENABLE_ASSERTIONS
#include <stdio.h>
#include <stdlib.h>
#ifdef __GNUC__
__attribute__((noreturn))
#endif
static OPUS_INLINE void _celt_fatal(const char *str, const char *file, int line)
{
   fprintf (stderr, "Fatal (internal) error in %s, line %d: %s\n", file, line, str);
   abort();
}
#define celt_assert(cond) {if (!(cond)) {celt_fatal("assertion failed: " #cond);}}
#define celt_assert2(cond, message) {if (!(cond)) {celt_fatal("assertion failed: " #cond "\n" message);}}
#else
#define celt_assert(cond)
#define celt_assert2(cond, message)
#endif

#define IMUL32(a,b) ((a)*(b))

#define MIN16(a,b) ((a) < (b) ? (a) : (b))   /**< Minimum 16-bit value.   */
#define MAX16(a,b) ((a) > (b) ? (a) : (b))   /**< Maximum 16-bit value.   */
#define MIN32(a,b) ((a) < (b) ? (a) : (b))   /**< Minimum 32-bit value.   */
#define MAX32(a,b) ((a) > (b) ? (a) : (b))   /**< Maximum 32-bit value.   */
#define IMIN(a,b) ((a) < (b) ? (a) : (b))   /**< Minimum int value.   */
#define IMAX(a,b) ((a) > (b) ? (a) : (b))   /**< Maximum int value.   */
#define UADD32(a,b) ((a)+(b))
#define USUB32(a,b) ((a)-(b))

#define OPUS_FAST_INT64 0

#define PRINT_MIPS(file)

typedef float opus_val16;
typedef float opus_val32;
typedef float opus_val64;

typedef float celt_sig;
typedef float celt_norm;
typedef float celt_ener;

#ifdef FLOAT_APPROX
/* This code should reliably detect NaN/inf even when -ffast-math is used.
   Assumes IEEE 754 format. */
static OPUS_INLINE int celt_isnan(float x)
{
   union {float f; opus_uint32 i;} in;
   in.f = x;
   return ((in.i>>23)&0xFF)==0xFF && (in.i&0x007FFFFF)!=0;
}
#else
#ifdef __FAST_MATH__
#error Cannot build libopus with -ffast-math unless FLOAT_APPROX is defined. This could result in crashes on extreme (e.g. NaN) input
#endif
#define celt_isnan(x) ((x)!=(x))
#endif

#define Q15ONE 1.0f

#define NORM_SCALING 1.f

#define EPSILON 1e-15f
#define VERY_SMALL 1e-30f
#define VERY_LARGE16 1e15f
#define Q15_ONE ((opus_val16)1.f)

/* This appears to be the same speed as C99's fabsf() but it's more portable. */
#define ABS16(x) ((float)fabs(x))
#define ABS32(x) ((float)fabs(x))

#define QCONST16(x,bits) (x)
#define QCONST32(x,bits) (x)

#define NEG16(x) (-(x))
#define NEG32(x) (-(x))
#define NEG32_ovflw(x) (-(x))
#define EXTRACT16(x) (x)
#define EXTEND32(x) (x)
#define SHR16(a,shift) (a)
#define SHL16(a,shift) (a)
#define SHR32(a,shift) (a)
#define SHL32(a,shift) (a)
#define PSHR32(a,shift) (a)
#define VSHR32(a,shift) (a)

#define PSHR(a,shift)   (a)
#define SHR(a,shift)    (a)
#define SHL(a,shift)    (a)
#define SATURATE(x,a)   (x)
#define SATURATE16(x)   (x)

#define ROUND16(a,shift)  (a)
#define SROUND16(a,shift) (a)
#define HALF16(x)       (.5f*(x))
#define HALF32(x)       (.5f*(x))

#define ADD16(a,b) ((a)+(b))
#define SUB16(a,b) ((a)-(b))
#define ADD32(a,b) ((a)+(b))
#define SUB32(a,b) ((a)-(b))
#define ADD32_ovflw(a,b) ((a)+(b))
#define SUB32_ovflw(a,b) ((a)-(b))
#define MULT16_16_16(a,b)     ((a)*(b))
#define MULT16_16(a,b)     ((opus_val32)(a)*(opus_val32)(b))
#define MAC16_16(c,a,b)     ((c)+(opus_val32)(a)*(opus_val32)(b))

#define MULT16_32_Q15(a,b)     ((a)*(b))
#define MULT16_32_Q16(a,b)     ((a)*(b))

#define MULT32_32_Q31(a,b)     ((a)*(b))

#define MAC16_32_Q15(c,a,b)     ((c)+(a)*(b))
#define MAC16_32_Q16(c,a,b)     ((c)+(a)*(b))

#define MULT16_16_Q11_32(a,b)     ((a)*(b))
#define MULT16_16_Q11(a,b)     ((a)*(b))
#define MULT16_16_Q13(a,b)     ((a)*(b))
#define MULT16_16_Q14(a,b)     ((a)*(b))
#define MULT16_16_Q15(a,b)     ((a)*(b))
#define MULT16_16_P15(a,b)     ((a)*(b))
#define MULT16_16_P13(a,b)     ((a)*(b))
#define MULT16_16_P14(a,b)     ((a)*(b))
#define MULT16_32_P16(a,b)     ((a)*(b))

#define DIV32_16(a,b)     (((opus_val32)(a))/(opus_val16)(b))
#define DIV32(a,b)     (((opus_val32)(a))/(opus_val32)(b))

#define SCALEIN(a)      ((a)*CELT_SIG_SCALE)
#define SCALEOUT(a)     ((a)*(1/CELT_SIG_SCALE))

#define SIG2WORD16(x) (x)

#ifndef GLOBAL_STACK_SIZE
#define GLOBAL_STACK_SIZE 120000
#endif

#endif /* ARCH_H */
