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
  GenSchedule(int depth, int num_mini, bool gpipe_=false): pipeline_depth_(depth), 
            num_mini_(num_mini), gpipe(gpipe_) {
    InitQueues();
  }
  void Generate(std::vector<schedule_task> sched[]);

private:
  void InitQueues();
  TaskQueue* PickQueue(int stage, char* identifier);
  TaskQueue* PickQueue_gpipe(int stage, char* identifier);
  TaskQueue* PickQueue_varuna(int stage, char* identifier);
  std::vector<TaskQueue*> fwd_queues_, bi_queues_, bw_queues_, rc_queues_;
  int pipeline_depth_;
  int num_mini_;
  int rank_;
  bool gpipe;
};