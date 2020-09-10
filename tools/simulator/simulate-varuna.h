#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <deque>
#include <utility>

using namespace std;

typedef long long int int64; 
typedef struct QueueEntry {
  int micro_batch;
  int64 time;

  QueueEntry(int mb, int64 t): micro_batch(mb), time(t) {}
} QueueEntry;

typedef struct Event {
  enum State {
    FWD_START = 0, 
    FWD_DONE = 1,
    SENDACT_START = 2,
    SENDACT_DONE = 3,
    BWD_START = 4,
    BWD_DONE = 5,
    SENDGRAD_START = 6,
    SENDGRAD_DONE = 7,
    RC_QUEUE = 8,
    RC_START = 9,
    RC_DONE = 10,
    RECV_ACT = 11,
    RECV_GRAD = 12,
    NUM_STATUS
  }; 

  static const char* StringState(State s) {
    switch(s) {
      case FWD_START: return "FWD_START";
      case FWD_DONE: return "FWD_DONE";
      case BWD_START: return "BWD_START";
      case BWD_DONE: return "BWD_DONE";
      case RC_START: return "RC_START";
      case RC_DONE: return "RC_DONE";
      case SENDACT_START: return "SENDACT_START";
      case SENDACT_DONE: return "SENDACT_DONE";
      case SENDGRAD_START: return "SENDGRAD_START";
      case SENDGRAD_DONE: return "SENDGRAD_DONE";
      case RC_QUEUE: return "RC_QUEUE";
      case RECV_ACT: return "RECV_ACT";
      case RECV_GRAD: return "RECV_GRAD";
      default: return "UNKNOWN";
    }
  }

  Event() { }
  
  Event(int stg, int mb, State st): stage(stg), mb_num(mb), state(st) { }
  int stage;
  int mb_num;
  State state;
} Event;

class Simulator;

typedef struct Queue {
  Queue(char identifier): id(identifier) { }
  void push_back(const QueueEntry& qe) {
    q.push_back(qe);
  }
  void pop_front() {
    q.pop_front();
  }
  QueueEntry front() { return q.front(); }
  char id;
  std::deque<QueueEntry> q;
} Queue;


class Stage {
public:
  Stage(int stage_num, int depth, int num_mb)
    : stage_num_(stage_num), depth_(depth), num_micro_(num_mb),
      gpu_busy_(false), network_busy_(false), recomputed_mb_(-1) {
    // Initialize queues
    fwd_ = new Queue('f');
    bwd_ = new Queue('b');
    rc_ = new Queue('r');
    sendact_ = new Queue('a');
    sendgrad_ = new Queue('g');
    if (stage_num_ == 0) {
      for (int i = 0; i < num_micro_; ++i) {
        fwd_->q.push_back(QueueEntry(i, 0));
      }
    }
  }
  bool ProcessEvent(Simulator* sim, Event event, int64 now);
  void ServiceQueues(Simulator* sim, int64 now);

private:
  Queue* PickNextNetworkQueue(int64 now);
  Queue* PickNextComputeQueue(int64 now);

  Queue *fwd_, *bwd_, *rc_, *sendact_, *sendgrad_;
  bool gpu_busy_;
  bool network_busy_;
  int stage_num_;
  int depth_;
  int num_micro_;
  int recomputed_mb_;
};

typedef class Stage Stage;
typedef std::map<int64, Event> EventList;


class Simulator {
public:
  Simulator(int depth, int num_mini, int fwd, int bwd, int send, int allr)
  : pipeline_depth_(depth), num_mini_(num_mini),
    fwd_time_(fwd), bwd_time_(bwd), sendact_time_(send), sendgrad_time_(send),
    allreduce_time_(allr) {

    stages_ = (Stage**)malloc(sizeof(Stage*) * pipeline_depth_);
    for (int i = 0; i < pipeline_depth_; ++i) {
      stages_[i] = new Stage(i, pipeline_depth_, num_mini);
    }
    clock_now_micros_ = 0;
    DumpState();
  }
  void Simulate();
  void DumpState();
  bool AddEvent(int64 scheduled_time, Event e);

  int64 current_time() { return clock_now_micros_;}  
  int fwd_compute_time() { return fwd_time_; }
  int bwd_compute_time() { return bwd_time_;}

  // TODO: Can incorporate jitter here.
  int sendact_time() { return sendact_time_; }
  int sendgrad_time() { return sendgrad_time_;}

private:
  bool PopNextEvent(Event* e);

  Stage** stages_;
  EventList event_list_;
  int pipeline_depth_;
  int num_mini_;
  int64 clock_now_micros_;
  int fwd_time_, bwd_time_, sendact_time_, sendgrad_time_, allreduce_time_;
};