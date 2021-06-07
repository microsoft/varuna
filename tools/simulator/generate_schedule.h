#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <deque>
#include <utility>

using namespace std;

typedef std::deque<int> TaskQueue;
typedef struct {
  int index;
  char task;
} schedule_task;


class GenSchedule {
public:
  GenSchedule(int depth, int num_mini, bool gpipe_=false, bool pd_1f1b=false): 
      pipeline_depth_(depth), num_mini_(num_mini), gpipe(gpipe_), pd_1f1b(pd_1f1b) {
    InitQueues();
    if (pd_1f1b && gpipe){
      printf("ERROR BOTH 1F1B AND GPIPE SELECTED!\n");
    }
    for(int i=0; i<pipeline_depth_; i++){
      num_fwds_done.push_back(0);
      num_bwds_done.push_back(0);
      next_is_fwd_1f1b.push_back(true);
    }
  }
  void Generate(std::vector<schedule_task> sched[]);

private:
  void InitQueues();
  TaskQueue* PickQueue(int stage, char* identifier);
  TaskQueue* PickQueue_gpipe(int stage, char* identifier);
  TaskQueue* PickQueue_varuna(int stage, char* identifier);
  TaskQueue* PickQueue_1f1b(int stage, char* identifier);
  std::vector<TaskQueue*> fwd_queues_, bi_queues_, bw_queues_, rc_queues_;
  int pipeline_depth_;
  int num_mini_;
  int rank_;
  bool gpipe, pd_1f1b;
  std::vector<bool> next_is_fwd_1f1b;
  std::vector<int> num_fwds_done;
  std::vector<int> num_bwds_done;
  // int warmup_fwds = 0;
};