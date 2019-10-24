#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <deque>
#include <utility>
using namespace std;

typedef std::deque<pair<int, int> > Queue;
class GenSchedule {
public:
  GenSchedule(int depth, int num_mini): pipeline_depth_(depth), num_mini_(num_mini) {
    InitQueues();
  }
  void Generate();
private:
  void InitQueues();
  Queue* PickQueue(int stage, int time, char* identifier);
  std::vector<Queue*> fwd_queues_, bi_queues_, bw_queues_, rc_queues_;
  int pipeline_depth_;
  int num_mini_;
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
    fwd_queues_[0]->push_back(make_pair(i, i));
  }
}

Queue* GenSchedule::PickQueue(int stage, int time, char* identifier) {
  // Highest priority is input gradient, as activations will go away
  if (!bi_queues_[stage]->empty()) {
    *identifier = 'i'; 
    return bi_queues_[stage];
  }
  if (!bw_queues_[stage]->empty()) {
    *identifier = 'w';
    return bw_queues_[stage];
  }
  bool prefer_fwd = false;
  // Between fwd and bwd, schedule the oldest event first.
  if (!fwd_queues_[stage]->empty() && !rc_queues_[stage]->empty()) {
    prefer_fwd = (fwd_queues_[stage]->front().second 
                 < rc_queues_[stage]->front().second - 1);
  }
  Queue* first = fwd_queues_[stage], *second = rc_queues_[stage];
  char id_first = 'f', id_second = 'r';
  if (!prefer_fwd) {
    first = rc_queues_[stage];
    second = fwd_queues_[stage];
    id_first = 'r'; id_second = 'f';
  }
  if (!first->empty()) {
    *identifier = id_first;
    return first;
  }
  if (!second->empty()) {
    *identifier = id_second;
    return second;
  }
  return NULL;
}

void GenSchedule::Generate() {
  std::vector<int> num_bubbles;
  num_bubbles.assign(pipeline_depth_, 0);
  int time = 0;
  for (time = 0; time < 2000; ++time) {
    std::vector<int> mini_batches;
    std::vector<char> queue_ids;
    bool all_queues_empty = true;

    for (int i = 0; i < pipeline_depth_; ++i) {
      // Service the queue for each pipeline stage/device
      char identifier;
      Queue* queue = PickQueue(i, time, &identifier);
      int mini = ((queue != NULL)? queue->front().first: (-1));
      mini_batches.push_back(mini);
      queue_ids.push_back(identifier);
      if (mini < 0) {
        ++num_bubbles[i];        
        printf(" -  "); continue;
      }
      all_queues_empty = false;
      queue->pop_front();
      printf("%2d%c ", mini, identifier);
    }
    printf("\n");
    if (all_queues_empty) break;

    for (int i = 0; i < pipeline_depth_; ++i) {
      int mini = mini_batches[i];
      if (mini < 0) continue;

      bool first_stage = (i == 0);
      bool second_stage = (i == 1);
      bool last_stage = (i == pipeline_depth_ - 1);
      switch(queue_ids[i]) {
        case 'f':
          if (!last_stage) {
            fwd_queues_[i+1]->push_back(make_pair(mini, time+1));
          } else {
            bi_queues_[i]->push_back(make_pair(mini, time+1));
            rc_queues_[i-1]->push_back(make_pair(mini, time+1));
          }
          break;

        case 'i':
            bw_queues_[i]->push_back(make_pair(mini, time+1));
            break;

        case 'r':
          bi_queues_[i]->push_back(make_pair(mini, time+1));
          if (!first_stage) {
            rc_queues_[i-1]->push_back(make_pair(mini, time+1));
          }
        case 'w':
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