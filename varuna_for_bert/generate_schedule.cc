#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <deque>
#include <utility>

using namespace std;

typedef std::deque<int> Queue;
class GenSchedule {
public:
  GenSchedule(int depth, int num_mini, int rank): pipeline_depth_(depth), num_mini_(num_mini), rank_(rank) {
    InitQueues();
  }
  void Generate();

private:
  void InitQueues();
  Queue* PickQueue(int stage, char* identifier);
  std::vector<Queue*> fwd_queues_, bi_queues_, bw_queues_, rc_queues_;
  int pipeline_depth_;
  int num_mini_;
  int rank_;
};

void GenSchedule::InitQueues() {
  for (int i = 0; i < pipeline_depth_; ++i) {
    fwd_queues_.push_back(new Queue);
    bi_queues_.push_back(new Queue);
    bw_queues_.push_back(new Queue);
    rc_queues_.push_back(new Queue);
  }
  // Fill the first stage's fwd queue.
  for (int i = 1; i <= num_mini_; ++i) {
    fwd_queues_[0]->push_back(i);
  }
}

Queue* GenSchedule::PickQueue(int stage, char* identifier) {
  // Defines precedence order of servicing queues
  if (!bw_queues_[stage]->empty()) {
    // *identifier = 'B';
    *identifier = '3';
    return bw_queues_[stage];
  }
  if (!bi_queues_[stage]->empty()) {
    // *identifier = 'b';
    *identifier = '2'; 
    return bi_queues_[stage];
  }
  if (!rc_queues_[stage]->empty()) {
    // *identifier = 'r'; 
    *identifier = '1';
    return rc_queues_[stage];
  }
  if (!fwd_queues_[stage]->empty()) {
    // *identifier = 'f'; 
    *identifier = '0';
    return fwd_queues_[stage];
  }
  return NULL;
}

typedef struct {
  int index;
  char task;
} schedule_task;

void GenSchedule::Generate() {
  std::vector<schedule_task> sched[pipeline_depth_];
  std::vector<int> num_bubbles;
  num_bubbles.assign(pipeline_depth_, 0);

  int time = 0;
  for (time = 0; time < 2000; ++time) {
    std::vector<int> mini_batches;
    std::vector<char> queue_ids;
    bool all_queues_empty = true;

    // First, pick events to schedule in each stage for this time quantum
    for (int i = 0; i < pipeline_depth_; ++i) {
      // Service the queue for each pipeline stage/device
      char identifier;
      Queue* queue = PickQueue(i, &identifier);
      int mini = ((queue != NULL)? queue->front(): (-1));
      mini_batches.push_back(mini);

      if (mini < 0) {
        ++num_bubbles[i];
        queue_ids.push_back('Z');        
        // printf(" -  "); 
        continue;
      }

      all_queues_empty = false;
      queue_ids.push_back(identifier);
      queue->pop_front();
      // printf("%2d%c ", mini, identifier);
      if (identifier!='3') {
        schedule_task a = {mini, identifier};
        sched[i].push_back(a);
      }
    }
    // printf("\n");
    
    if (all_queues_empty) break;

    // Now, queue events for the next time quantum, based on dependency rules
    for (int i = 0; i < pipeline_depth_; ++i) {
      int mini = mini_batches[i];
      if (mini < 0) continue;

      bool first_stage = (i == 0);
      bool last_stage = (i == pipeline_depth_ - 1);
      switch(queue_ids[i]) {
        // case 'f':
        case '0':
          if (!last_stage) {
            fwd_queues_[i+1]->push_back(mini);
          } else {
            bi_queues_[i]->push_back(mini);
          }
          break;

        // case 'b':
        case '2':
          bw_queues_[i]->push_back(mini);
          if (!first_stage) {
            rc_queues_[i-1]->push_back(mini);
          }
          break;

        // case 'r':
        case '1':
          bi_queues_[i]->push_back(mini);
          break;

        // case 'B':
        case '3':
          break;

        case 'Z':
           printf("\nShould never be here");
           break;
      }
    }
  }
  
  for (int i=0; i<pipeline_depth_; ++i) {
    // printf("[");
    if (i==rank_) {
      for (int j=0; j<sched[i].size(); ++j) {
        printf("%c,%d;", sched[i][j].task, sched[i][j].index-1);
      }
    }
    // printf("\n");
  }
  // printf("\nFraction of bubbles = %d/%d  %.2f percent\n",
  //        num_bubbles[0]-1, time-1, (float)(num_bubbles[0]-1)*100/(time-1));
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: gen-schedule <pipeline-depth> <num_micro_batches>");
    return -1;
  }
  int pipeline_depth = atoi(argv[1]);
  int num_mini = atoi(argv[2]);
  int rank = atoi(argv[3]);
  GenSchedule s(pipeline_depth, num_mini, rank);
  s.Generate();
  return 0;
}
