#include "simulate-varuna.h"
#include <chrono> 
#include <iostream> 

using namespace std; 
using namespace std::chrono; 

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: simulate-varuna <pipeline-depth> <num_micro_batches> <fwd-time> "
           "<bwd-time> <send-time> <allreduce-time> (all times are in microsec)\n");
    return -1;
  }
  // TODO: Can support more parameters here: e.g. stage-wise fwd/bwd times, jitter, etc.
  int arg_index = 1;
  int pipeline_depth = atoi(argv[arg_index++]);
  int num_mini = atoi(argv[arg_index++]);
  // int fwd_time = atoi(argv[arg_index++]);
  // int bwd_time = atoi(argv[arg_index++]);
  int send_time = atoi(argv[arg_index++]);
  int allreduce_time = atoi(argv[arg_index++]);
  // int last_fwd_time = 0,last_bwd_time = 0;
  // if(argc > arg_index){
  //   last_fwd_time = atoi(argv[arg_index++]);
  //   last_bwd_time = atoi(argv[arg_index++]);
  // }
  int send_long_time = (argc > arg_index) ? atoi(argv[arg_index++]) : 0;
  int dp = (argc > arg_index) ? atoi(argv[arg_index++]) : 0;
  // Simulator s(pipeline_depth, num_mini, 0, 0, send_time, allreduce_time, 0,0, send_long_time);
  Simulator s(pipeline_depth, num_mini, 0,0, send_time, allreduce_time, 0,0, send_long_time, dp);
  auto start = high_resolution_clock::now(); 
  s.Simulate();
  auto stop = high_resolution_clock::now(); 
  auto duration = duration_cast<microseconds>(stop - start); 
  cout << "Time taken by simulation: " << duration.count() << " microseconds\n";
  return 0;
}