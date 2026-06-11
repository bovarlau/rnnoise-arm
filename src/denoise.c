/* Copyright (c) 2024 Jean-Marc Valin
 * Copyright (c) 2018 Gregor Richards
 * Copyright (c) 2017 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include "denoise.h"
#include <math.h>
#include "rnnoise.h"
#include "pitch.h"
#include "arch.h"
#include "rnn.h"
#include "cpu_support.h"
#include <time.h>

#define SQUARE(x) ((x)*(x))


#ifndef TRAINING
#define TRAINING 0
#endif


/* ERB bandwidths going in reverse from 20 kHz and then replacing the 700 and 800
   with just 750 because having 32 bands is convenient for the DNN. 
   B(1)=400;
   for k=2:35
     B(k) = B(k-1) - max(2, round(24.7*(4.37*B(k-1)/20+1)/50));
   end
   printf("%d, ", B(end:-1:1));
   printf("\n")
*/
const int eband20ms[NB_BANDS+2] = {
/*0 100 200 300 400 500 600 750 900 1.1 1.2 1.4 1.6 1.8 2.1 2.4 2.7 3.0 3.4 3.9 4.4 4.9  5.5  6.2  7.0  7.9  8.8  9.9 11.2 12.6 14.1 15.9 17.8 20.0*/
  0, 2,  4,  6,  8,  10, 12, 15, 18, 21, 24, 28, 32, 36, 41, 47, 53, 60, 68, 77, 87, 98, 110, 124, 140, 157, 176, 198, 223, 251, 282, 317, 356, 400};


struct DenoiseState {
  RNNoise model;
#if !TRAINING
  int arch;
#endif
  float analysis_mem[FRAME_SIZE];
  int memid;
  float synthesis_mem[FRAME_SIZE];
  float pitch_buf[PITCH_BUF_SIZE];
  float pitch_enh_buf[PITCH_BUF_SIZE];
  float last_gain;
  int last_period;
  float mem_hp_x[2];
  float lastg[NB_BANDS];
  float vad_smooth;
  float vad_hist[2];
  int howl_count[NB_BANDS];
  int howl_fast[NB_BANDS];
  int howl_bin[NB_BANDS];
  int howl_mode_timer;
  RNNState rnn;
  kiss_fft_cpx delayed_X[FREQ_SIZE];
  kiss_fft_cpx delayed_P[FREQ_SIZE];
  float delayed_Ex[NB_BANDS], delayed_Ep[NB_BANDS];
  float delayed_Exp[NB_BANDS];

};

static void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  float sum[NB_BANDS+2] = {0};
  for (i=0;i<NB_BANDS+1;i++)
  {
    int j;
    int band_size;
    band_size = eband20ms[i+1]-eband20ms[i];
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = SQUARE(X[eband20ms[i] + j].r);
      tmp += SQUARE(X[eband20ms[i] + j].i);
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[1] = (sum[0]+sum[1])*2/3;
  sum[NB_BANDS] = (sum[NB_BANDS]+sum[NB_BANDS+1])*2/3;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i+1];
  }
}

static void compute_band_corr(float *bandE, const kiss_fft_cpx *X, const kiss_fft_cpx *P) {
  int i;
  float sum[NB_BANDS+2] = {0};
  for (i=0;i<NB_BANDS+1;i++)
  {
    int j;
    int band_size;
    band_size = eband20ms[i+1]-eband20ms[i];
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = X[eband20ms[i] + j].r * P[eband20ms[i] + j].r;
      tmp += X[eband20ms[i] + j].i * P[eband20ms[i] + j].i;
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[1] = (sum[0]+sum[1])*2/3;
  sum[NB_BANDS] = (sum[NB_BANDS]+sum[NB_BANDS+1])*2/3;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i+1];
  }
}

static void interp_band_gain(float *g, const float *bandE) {
  int i,j;
  memset(g, 0, FREQ_SIZE);
  for (i=1;i<NB_BANDS;i++)
  {
    int band_size;
    band_size = eband20ms[i+1]-eband20ms[i];
    for (j=0;j<band_size;j++) {
      float frac = (float)j/band_size;
      g[eband20ms[i] + j] = (1-frac)*bandE[i-1] + frac*bandE[i];
    }
  }
  for (j=0;j<eband20ms[1];j++) g[j] = bandE[0];
  for (j=eband20ms[NB_BANDS];j<eband20ms[NB_BANDS+1];j++) g[j] = bandE[NB_BANDS-1];
}

extern const float rnn_dct_table[];
extern const kiss_fft_state rnn_kfft;
extern const float rnn_half_window[];

static void dct(float *out, const float *in) {
  int i;
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * rnn_dct_table[j*NB_BANDS + i];
    }
    out[i] = sum*sqrt(2./22);
  }
}

#if 0
static void idct(float *out, const float *in) {
  int i;
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * rnn_dct_table[i*NB_BANDS + j];
    }
    out[i] = sum*sqrt(2./22);
  }
}
#endif

static void forward_transform(kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  for (i=0;i<WINDOW_SIZE;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }
  rnn_fft(&rnn_kfft, x, y, 0);
  for (i=0;i<FREQ_SIZE;i++) {
    out[i] = y[i];
  }
}

static void inverse_transform(float *out, const kiss_fft_cpx *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  for (i=0;i<FREQ_SIZE;i++) {
    x[i] = in[i];
  }
  for (;i<WINDOW_SIZE;i++) {
    x[i].r = x[WINDOW_SIZE - i].r;
    x[i].i = -x[WINDOW_SIZE - i].i;
  }
  rnn_fft(&rnn_kfft, x, y, 0);
  /* output in reverse order for IFFT. */
  out[0] = WINDOW_SIZE*y[0].r;
  for (i=1;i<WINDOW_SIZE;i++) {
    out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
  }
}

static void apply_window(float *x) {
  int i;
  for (i=0;i<FRAME_SIZE;i++) {
    x[i] *= rnn_half_window[i];
    x[WINDOW_SIZE - 1 - i] *= rnn_half_window[i];
  }
}

struct RNNModel {
  /* Set either blob or const_blob. */
  const void *const_blob;
  void *blob;
  int blob_len;
  FILE *file;
};

RNNModel *rnnoise_model_from_buffer(const void *ptr, int len) {
  RNNModel *model;
  model = malloc(sizeof(*model));
  model->blob = NULL;
  model->const_blob = ptr;
  model->blob_len = len;
  return model;
}

RNNModel *rnnoise_model_from_filename(const char *filename) {
  RNNModel *model;
  FILE *f = fopen(filename, "rb");
  model = rnnoise_model_from_file(f);
  model->file = f;
  return model;
}

RNNModel *rnnoise_model_from_file(FILE *f) {
  RNNModel *model;
  model = malloc(sizeof(*model));
  model->file = NULL;

  fseek(f, 0, SEEK_END);
  model->blob_len = ftell(f);
  fseek(f, 0, SEEK_SET);

  model->const_blob = NULL;
  model->blob = malloc(model->blob_len);
  if (fread(model->blob, model->blob_len, 1, f) != 1)
  {
    rnnoise_model_free(model);
    return NULL;
  }
  return model;
}

void rnnoise_model_free(RNNModel *model) {
  if (model->file != NULL) fclose(model->file);
  if (model->blob != NULL) free(model->blob);
  free(model);
}

int rnnoise_get_size() {
  return sizeof(DenoiseState);
}

int rnnoise_get_frame_size() {
  return FRAME_SIZE;
}

int rnnoise_init(DenoiseState *st, RNNModel *model) {
  memset(st, 0, sizeof(*st));
#if !TRAINING
  if (model != NULL) {
    WeightArray *list;
    int ret = 1;
    parse_weights(&list, model->blob ? model->blob : model->const_blob, model->blob_len);
    if (list != NULL) {
      ret = init_rnnoise(&st->model, list);
      opus_free(list);
    }
    if (ret != 0) return -1;
  }
#ifndef USE_WEIGHTS_FILE
  else {
    int ret = init_rnnoise(&st->model, rnnoise_arrays);
    if (ret != 0) return -1;
  }
#endif
  st->arch = rnn_select_arch();
#else
  (void)model;
#endif
  return 0;
}

DenoiseState *rnnoise_create(RNNModel *model) {
  int ret;
  DenoiseState *st;
  st = malloc(rnnoise_get_size());
  ret = rnnoise_init(st, model);
  if (ret != 0) {
    free(st);
    return NULL;
  }
  return st;
}

void rnnoise_destroy(DenoiseState *st) {
  free(st);
}

#if TRAINING
extern int lowpass;
extern int band_lp;
#endif

void rnn_frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in) {
  int i;
  float x[WINDOW_SIZE];
  RNN_COPY(x, st->analysis_mem, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) x[FRAME_SIZE + i] = in[i];
  RNN_COPY(st->analysis_mem, in, FRAME_SIZE);
  apply_window(x);
  forward_transform(X, x);
#if TRAINING
  for (i=lowpass;i<FREQ_SIZE;i++)
    X[i].r = X[i].i = 0;
#endif
  compute_band_energy(Ex, X);
}

int rnn_compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
                                  float *Ex, float *Ep, float *Exp, float *features, const float *in) {
  int i;
  float E = 0;
  float Ly[NB_BANDS];
  float p[WINDOW_SIZE];
  float pitch_buf[PITCH_BUF_SIZE>>1];
  int pitch_index;
  float gain;
  float *(pre[1]);
  float follow, logMax;
  rnn_frame_analysis(st, X, Ex, in);
  RNN_MOVE(st->pitch_buf, &st->pitch_buf[FRAME_SIZE], PITCH_BUF_SIZE-FRAME_SIZE);
  RNN_COPY(&st->pitch_buf[PITCH_BUF_SIZE-FRAME_SIZE], in, FRAME_SIZE);
  pre[0] = &st->pitch_buf[0];
  rnn_pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);
  rnn_pitch_search(pitch_buf+(PITCH_MAX_PERIOD>>1), pitch_buf, PITCH_FRAME_SIZE,
               PITCH_MAX_PERIOD-3*PITCH_MIN_PERIOD, &pitch_index);
  pitch_index = PITCH_MAX_PERIOD-pitch_index;

  gain = rnn_remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
          PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
  st->last_period = pitch_index;
  st->last_gain = gain;
  for (i=0;i<WINDOW_SIZE;i++)
    p[i] = st->pitch_buf[PITCH_BUF_SIZE-WINDOW_SIZE-pitch_index+i];
  apply_window(p);
  forward_transform(P, p);
  compute_band_energy(Ep, P);
  compute_band_corr(Exp, X, P);
  for (i=0;i<NB_BANDS;i++) Exp[i] = Exp[i]/sqrt(.001+Ex[i]*Ep[i]);
  dct(&features[NB_BANDS], Exp);
  features[2*NB_BANDS] = .01*(pitch_index-300);
  logMax = -2;
  follow = -2;
  for (i=0;i<NB_BANDS;i++) {
    Ly[i] = log10(1e-2+Ex[i]);
    Ly[i] = MAX16(logMax-7, MAX16(follow-1.5, Ly[i]));
    logMax = MAX16(logMax, Ly[i]);
    follow = MAX16(follow-1.5, Ly[i]);
    E += Ex[i];
  }
  if (!TRAINING && E < 0.04) {
    /* If there's no audio, avoid messing up the state. */
    RNN_CLEAR(features, NB_FEATURES);
    return 1;
  }
  dct(features, Ly);
  features[0] -= 12;
  features[1] -= 4;
  return TRAINING && E < 0.1;
}

static void frame_synthesis(DenoiseState *st, float *out, const kiss_fft_cpx *y) {
  float x[WINDOW_SIZE];
  int i;
  inverse_transform(x, y);
  apply_window(x);
  for (i=0;i<FRAME_SIZE;i++) out[i] = x[i] + st->synthesis_mem[i];
  RNN_COPY(st->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
}

void rnn_biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}

void rnn_pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g) {
  int i;
  float r[NB_BANDS];
  float rf[FREQ_SIZE] = {0};
  float newE[NB_BANDS];
  float norm[NB_BANDS];
  float normf[FREQ_SIZE]={0};
  for (i=0;i<NB_BANDS;i++) {
#if 0
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = Exp[i]*(1-g[i])/(.001 + g[i]*(1-Exp[i]));
    r[i] = MIN16(1, MAX16(0, r[i]));
#else
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = SQUARE(Exp[i])*(1-SQUARE(g[i]))/(.001 + SQUARE(g[i])*(1-SQUARE(Exp[i])));
    r[i] = sqrt(MIN16(1, MAX16(0, r[i])));
#endif
    r[i] *= sqrt(Ex[i]/(1e-8+Ep[i]));
  }
  interp_band_gain(rf, r);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r += rf[i]*P[i].r;
    X[i].i += rf[i]*P[i].i;
  }
  compute_band_energy(newE, X);
  for (i=0;i<NB_BANDS;i++) {
    norm[i] = sqrt(Ex[i]/(1e-8+newE[i]));
  }
  interp_band_gain(normf, norm);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r *= normf[i];
    X[i].i *= normf[i];
  }
}

double total_biquad_time = 0.0;
double total_features_time = 0.0;
double total_rnn_time = 0.0;
double total_pitchfilter_time = 0.0;
float rnnoise_process_frame(DenoiseState *st, float *out, const float *in) {
  int i;
  kiss_fft_cpx X[FREQ_SIZE];
  kiss_fft_cpx P[FREQ_SIZE];
  float x[FRAME_SIZE];
  float Ex[NB_BANDS], Ep[NB_BANDS];
  float Exp[NB_BANDS];
  float features[NB_FEATURES];
  float g[NB_BANDS];
  float gf[FREQ_SIZE]={1};
  float vad_prob = 0;
  int silence;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};

  //clock_t start_biquad = clock();
  rnn_biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);  //input: in[480],float  output:x[480],float
  //clock_t end_biquad = clock();
  //total_biquad_time += (double)(end_biquad - start_biquad) / CLOCKS_PER_SEC;
  //printf("rnn_biquad time: %.6f seconds\n", total_biquad_time);

  //clock_t start_features = clock();
  silence = rnn_compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);  //input: x[480],float  output:features[65],float
  //clock_t end_features = clock();
  //total_features_time += (double)(end_features - start_features) / CLOCKS_PER_SEC;
  //printf("rnn_compute_frame_features time: %.6f seconds\n", total_features_time);

  if (!silence) {
#if !TRAINING
    //clock_t start_rnn = clock();
    compute_rnn(&st->model, &st->rnn, g, &vad_prob, features, st->arch);
	//clock_t end_rnn = clock();
    //total_rnn_time += (double)(end_rnn - start_rnn) / CLOCKS_PER_SEC;
    //printf("compute_rnn time: %.6f seconds\n", total_rnn_time);

    /* Howling (acoustic feedback) detector.
       A feedback squeal is a loud narrowband peak that stays on (nearly) the
       same FFT bin for hundreds of ms, unlike speech harmonics whose frequency
       keeps moving with the pitch.  Per band (above ~1 kHz), count consecutive
       frames where a single bin (+/-1) dominates the band and the band is a
       significant part of the frame energy.  Once any band confirms a squeal,
       enter "howl mode" for a few seconds: howling happens when units are
       close (high loop gain), which never coincides with far-field speech, so
       during howl mode the speech protection below is disabled for all bands
       above ~1 kHz and the model's own gains (~0 for feedback tones) apply.
       While in howl mode, re-arming the timer only needs a short stable run,
       so squeal bursts that hop a few bins keep the mode latched.
       A second, fast trigger path handles loud squeals: a stable narrow peak
       that alone holds >40% of the whole frame energy while the model gives
       that band ~0 gain cannot be speech (validated against far-field speech
       harmonics, which stay below ~30%), so 4 such frames (40 ms) are enough
       to confirm. */
#define HOWL_FIRST_BAND 8  /* only look above ~1.05 kHz; lower bands hold sustained vowels */
#define HOWL_TRIG 26       /* ~260 ms of stable narrow peak confirms howling
                              (longest such run measured on speech: 250 ms) */
#define HOWL_RETRIG 10     /* while in howl mode, ~100 ms re-arms the timer */
#define HOWL_HOLD 300      /* howl mode lasts >= 3 s past the last confirmation */
#define HOWL_COUNT_MAX 100
#define HOWL_FAST_DOM 0.4f /* fast path: peak alone exceeds 40% of frame energy */
#define HOWL_FAST_G 0.05f  /* fast path: model must agree the band is noise */
#define HOWL_FAST_TRIG 4   /* fast path: 40 ms */
    {
      int any_trig = 0;
      int th = st->howl_mode_timer > 0 ? HOWL_RETRIG : HOWL_TRIG;
      float Etot = 1e-9f;
      for (i=0;i<NB_BANDS;i++) Etot += Ex[i];
      for (i=HOWL_FIRST_BAND;i<NB_BANDS;i++) {
        int j, lo = eband20ms[i+1], hi = eband20ms[i+2];
        int pkb = lo, narrow, stable;
        float be = 1e-15f, maxbin = 0, pe;
        for (j=lo;j<hi;j++) {
          float e = SQUARE(X[j].r) + SQUARE(X[j].i);
          be += e;
          if (e > maxbin) { maxbin = e; pkb = j; }
        }
        pe = maxbin;
        if (pkb-1 >= lo) pe += SQUARE(X[pkb-1].r) + SQUARE(X[pkb-1].i);
        if (pkb+1 < hi)  pe += SQUARE(X[pkb+1].r) + SQUARE(X[pkb+1].i);
        narrow = (be > .02f*Etot) && (pe > .7f*be);
        stable = narrow && pkb - st->howl_bin[i] <= 1 && pkb - st->howl_bin[i] >= -1;
        if (stable) {
          if (st->howl_count[i] < HOWL_COUNT_MAX) st->howl_count[i]++;
        } else if (narrow) {
          /* still a narrow peak, but it jumped bins: decay softly so an
             established count survives small frequency hops */
          st->howl_count[i] -= 1;
          if (st->howl_count[i] < 0) st->howl_count[i] = 0;
        } else {
          st->howl_count[i] -= 2;
          if (st->howl_count[i] < 0) st->howl_count[i] = 0;
        }
        if (stable && pe > HOWL_FAST_DOM*Etot && g[i] < HOWL_FAST_G) {
          if (st->howl_fast[i] < HOWL_COUNT_MAX) st->howl_fast[i]++;
        } else {
          st->howl_fast[i] -= 2;
          if (st->howl_fast[i] < 0) st->howl_fast[i] = 0;
        }
        st->howl_bin[i] = narrow ? pkb : -10;
        if (st->howl_count[i] >= th || st->howl_fast[i] >= HOWL_FAST_TRIG) any_trig = 1;
      }
      if (any_trig) st->howl_mode_timer = HOWL_HOLD;
      else if (st->howl_mode_timer > 0) st->howl_mode_timer--;
    }

    /* Far-field speech protection.
       The model heavily attenuates reverberant/far-field speech (band gains drop
       to 0.05~0.3) even while its own VAD still detects voice, and the raw VAD
       fluctuates strongly from frame to frame on such speech.  So:
       1) median-filter the VAD over 3 frames: transient noises (keyboard
          clicks, taps) produce 1-2 frame VAD spikes that would otherwise open
          the protection, while real speech keeps VAD up for many frames;
       2) track a smoothed VAD with fast attack and ~50 ms release to avoid
          gain pumping within words;
       3) blend the band gains toward 1.0 in proportion to the smoothed VAD, so
          frames classified as speech are preserved while noise-only frames
          (vad ~ 0) keep the full suppression.
       During howl mode, bands above ~1 kHz get no protection at all. */
#define VAD_RELEASE 0.9f        /* per-10ms-frame release coefficient */
#define VAD_PROTECT_SCALE 1.3f  /* >1: full protection already at vad_smooth ~ 0.77 */
    {
      float a = st->vad_hist[0], b = st->vad_hist[1], c = vad_prob;
      float vad_med = a > b ? (b > c ? b : (a > c ? c : a))
                            : (a > c ? a : (b > c ? c : b));
      st->vad_hist[1] = st->vad_hist[0];
      st->vad_hist[0] = vad_prob;
      if (vad_med > st->vad_smooth) st->vad_smooth = vad_med;
      else st->vad_smooth = VAD_RELEASE*st->vad_smooth + (1.f-VAD_RELEASE)*vad_med;
    }
    {
      float protect = VAD_PROTECT_SCALE * st->vad_smooth;
      if (protect > 1.f) protect = 1.f;
      for (i=0;i<NB_BANDS;i++) {
        float p = (st->howl_mode_timer > 0 && i >= HOWL_FIRST_BAND) ? 0.f : protect;
        g[i] = g[i]*(1.f-p) + p;
      }
    }
#endif

    //clock_t start_pitchfilter = clock();
    rnn_pitch_filter(st->delayed_X, st->delayed_P, st->delayed_Ex, st->delayed_Ep, st->delayed_Exp, g);
    //clock_t end_pitchfilter = clock();
    //total_rnn_time += (double)(end_pitchfilter - start_pitchfilter) / CLOCKS_PER_SEC;
    //printf("rnn_pitch_filter time: %.6f seconds\n", total_pitchfilter_time);
	
	for (i=0;i<NB_BANDS;i++) {
      float alpha = .6f;
      /* Cap the decay at 0.6 per frame, corresponding to an RT60 of 135 ms.
         That avoids unnaturally quick attenuation. */
      g[i] = MAX16(g[i], alpha*st->lastg[i]);
      /* Compensate for energy change across frame when computing the threshold gain.
         Avoids leaking noise when energy increases (e.g. transient noise). */
      st->lastg[i] = MIN16(1.f, g[i]*(st->delayed_Ex[i]+1e-3)/(Ex[i]+1e-3));
    }
    interp_band_gain(gf, g);
#if 1
    for (i=0;i<FREQ_SIZE;i++) {
      st->delayed_X[i].r *= gf[i];
      st->delayed_X[i].i *= gf[i];
    }
#endif
  }
  frame_synthesis(st, out, st->delayed_X);

  RNN_COPY(st->delayed_X, X, FREQ_SIZE);
  RNN_COPY(st->delayed_P, P, FREQ_SIZE);
  RNN_COPY(st->delayed_Ex, Ex, NB_BANDS);
  RNN_COPY(st->delayed_Ep, Ep, NB_BANDS);
  RNN_COPY(st->delayed_Exp, Exp, NB_BANDS);
  return vad_prob;
}

