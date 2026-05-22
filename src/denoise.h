
#include "rnnoise.h"
#include "kiss_fft.h"
#include "nnet.h"

#define FRAME_SIZE 480
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)
#define NB_BANDS 32
#define NB_FEATURES (2*NB_BANDS+1)


#define PITCH_MIN_PERIOD 60
#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

extern const WeightArray rnnoise_arrays[];

extern const int eband20ms[];


void rnn_biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N);

void rnn_pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g);

void rnn_frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in);

int rnn_compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
                                  float *Ex, float *Ep, float *Exp, float *features, const float *in);
