#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <deque>
#include <utility>

using namespace std;

typedef std::deque<int> Queue;
class GenSchedule {
public:
  GenSchedule(int depth, int num_mini): pipeline_depth_(depth), num_mini_(num_mini) {
    InitQueues();
  }
  void Generate();

private:
  void InitQueues();
  Queue* PickQueue(int stage, char* identifier);
  std::vector<Queue*> fwd_queues_, bi_queues_, bw_queues_, rc_queues_, fb_queues_;
  int pipeline_depth_;
  int num_mini_;
};

void GenSchedule::InitQueues() {
  for (int i = 0; i < pipeline_depth_; ++i) {
    fwd_queues_.push_back(new Queue);
    bi_queues_.push_back(new Queue);
    bw_queues_.push_back(new Queue);
    rc_queues_.push_back(new Queue);
    fb_queues_.push_back(new Queue);
  }
  // Fill the first stage's fwd queue.
  for (int i = 1; i <= num_mini_; ++i) {
    fwd_queues_[0]->push_back(i);
  }
}

Queue* GenSchedule::PickQueue(int stage, char* identifier) {
  // Defines precedence order of servicing queues
  if (!bw_queues_[stage]->empty()) {
    *identifier = 'B';
    return bw_queues_[stage];
  }
  if (!bi_queues_[stage]->empty()) {
    *identifier = 'b'; 
    return bi_queues_[stage];
  }
  if (!rc_queues_[stage]->empty()) {
    *identifier = 'r'; 
    return rc_queues_[stage];
  }
  if (!fwd_queues_[stage]->empty()) {
    if (stage == 0) {
      *identifier = 'f'; 
      return fwd_queues_[stage];
    } 
    else if (fb_queues_[stage]->size() < fwd_queues_[stage]->size()) { // we have one in buffer
      if (!fwd_queues_[stage]->empty()) // buffering next in parallel with forward
        fb_queues_[stage]->pop_front();
      *identifier = 'f'; 
      return fwd_queues_[stage];
    } else {
       fb_queues_[stage]->pop_front(); //delay for buffering
    }
  }
  return NULL;
}

void GenSchedule::Generate() {
  std::vector<int> num_bubbles;
  num_bubbles.assign(pipeline_depth_, 0);

  int time = 0;
  for (time = 0; time < 200000; ++time) {
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
        printf(" -  "); continue;
      }

      all_queues_empty = false;
      queue_ids.push_back(identifier);
      queue->pop_front();
      printf("%2d%c ", mini, identifier);
    }
    printf("\n");
    
    if (all_queues_empty) break;

    // Now, queue events for the next time quantum, based on dependency rules
    for (int i = 0; i < pipeline_depth_; ++i) {
      int mini = mini_batches[i];
      if (mini < 0) continue;

      bool first_stage = (i == 0);
      bool last_stage = (i == pipeline_depth_ - 1);
      switch(queue_ids[i]) {
        case 'f':
          if (!last_stage) {
            fwd_queues_[i+1]->push_back(mini);
            fb_queues_[i+1]->push_back(mini);
          } else {
            bi_queues_[i]->push_back(mini);
          }
          break;

        case 'b':
          bw_queues_[i]->push_back(mini);
          break;

        case 'r':
          bi_queues_[i]->push_back(mini);
          break;

        case 'B':
          if (!first_stage) {
            rc_queues_[i-1]->push_back(mini);
          }
          break;

        case 'Z':
           printf("\nShould never be here");
           break;
      }
    }
  }
  printf("\nFraction of bubbles = %d/%d  %.2f percent\n",
         num_bubbles[0]-1, time-1, (float)(num_bubbles[0]-1)*100/(time-1));
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: gen-schedule <pipeline-depth> <num_micro_batches>");
    return -1;
  }
  int pipeline_depth = atoi(argv[1]);
  int num_mini = atoi(argv[2]);
  GenSchedule s(pipeline_depth, num_mini);
  s.Generate();
  return 0;
}
