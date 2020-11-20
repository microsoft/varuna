#include "simulate-varuna.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: simulate-varuna <pipeline-depth> <num_micro_batches> <fwd-time> "
           "<bwd-time> <send-time> <allreduce-time> (all times are in microsec)\n");
    return -1;
  }
  // TODO: Can support more parameters here: e.g. stage-wise fwd/bwd times, jitter, etc.
  int pipeline_depth = atoi(argv[1]);
  int num_mini = atoi(argv[2]);
  int fwd_time = atoi(argv[3]);
  int bwd_time = atoi(argv[4]);
  int send_time = atoi(argv[5]);
  int allreduce_time = atoi(argv[6]);
  int last_fwd_time = 0,last_bwd_time = 0;
  if(argc > 7){
    last_fwd_time = atoi(argv[7]);
    last_bwd_time = atoi(argv[8]);
  }
  int send_long_time = (argc > 9) ? atoi(argv[9]) : 0;
  Simulator s(pipeline_depth, num_mini, fwd_time, bwd_time, send_time, allreduce_time, last_fwd_time, last_bwd_time, send_long_time);
  s.Simulate();
  return 0;
}