#include <stdio.h>
#include <time.h>
#include "rnnoise.h"

#define FRAME_SIZE 480   //48kHz音频源中,每毫秒48个采样点,一次处理10ms音频,也就是480个采样点, 16位宽使用short表示,short转换为float处理.
//#define TEST_FILE

//输入音频为48K,单通道,PCM格式,输出音频为48K,单通道,PCM格式
int main(int argc, char **argv) {
  int i;
  int first = 1;
  float x[FRAME_SIZE];
  FILE *f1, *fout;
  DenoiseState *st;

  clock_t start_create = clock();
  st = rnnoise_create(NULL);
  clock_t end_create = clock();
  double create_time = (double)(end_create - start_create) / CLOCKS_PER_SEC;
  printf("rnnoise_create time: %.6f seconds\n", create_time);

  if (argc!=3) {
    fprintf(stderr, "usage: %s <noisy speech> <output denoised>\n", argv[0]);
    return 1;
  }
  f1 = fopen(argv[1], "rb");
  fout = fopen(argv[2], "wb");
  double total_process_time = 0.0;
  while (1) {
    short tmp[FRAME_SIZE];
    fread(tmp, sizeof(short), FRAME_SIZE, f1);
    if (feof(f1)) break;
    for (i=0;i<FRAME_SIZE;i++) x[i] = tmp[i];
	clock_t start_process = clock();
    rnnoise_process_frame(st, x, x);
	clock_t end_process = clock();
    total_process_time += (double)(end_process - start_process) / CLOCKS_PER_SEC;
	
    for (i=0;i<FRAME_SIZE;i++) tmp[i] = x[i];
    if (!first) fwrite(tmp, sizeof(short), FRAME_SIZE, fout);
    first = 0;
	
#ifdef TEST_FILE
	break;  //for test
#endif
  }
  printf("total rnnoise_process_frame time: %.6f seconds\n", total_process_time);
  
  rnnoise_destroy(st);
  fclose(f1);
  fclose(fout);
#ifdef USE_WEIGHTS_FILE
  rnnoise_model_free(model);
#endif
  return 0;
}
