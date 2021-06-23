#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <deque>
#include <utility>
#include <random>
#include <fstream>

#include "generate_schedule.h"

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
  void insert_rev(const QueueEntry& qe){
    std::deque<QueueEntry>::iterator it = q.begin();
    while((qe.micro_batch < (*it).micro_batch) &&
      (it != q.end())) 
        ++it;
    it = q.insert(it, qe);
  }
  void insert(const QueueEntry& qe){
    std::deque<QueueEntry>::iterator it = q.begin();
    while((qe.micro_batch > (*it).micro_batch) &&
      (it != q.end())) 
        ++it;
    it = q.insert(it, qe);
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
  Stage(int stage_num, int depth, int num_mb, vector<schedule_task> s)
    : stage_num_(stage_num), depth_(depth), num_micro_(num_mb),
      gpu_busy_(false), network_busy_(0), recomputed_mb_(-1), schedule(s) {
    // Initialize queues
    char *gpipe_env = getenv("GPIPE");
    gpipe = gpipe_env==NULL ? 0 : (bool) atoi(gpipe_env);
    char *pd_1f1b_env = getenv("PD_1F1B");
    pd_1f1b = pd_1f1b_env==NULL ? 0 : (bool) atoi(pd_1f1b_env);
    fwd_ = new Queue('f');
    bwd_ = new Queue('b');
    rc_ = new Queue('r');
    sendact_ = new Queue('a');
    sendgrad_ = new Queue('g');
    recvact_ = new Queue('A');
    recvgrad_ = new Queue('G');
    if(stage_num_ > 0) acts_left = num_mb;
    if(stage_num_ < depth-1) grads_left = num_mb;
    // acts_recvd.insert(acts_recvd.begin(), num_mb, false);
    // grads_recvd.insert(grads_recvd.begin(), num_mb, false);
    if (stage_num_ == 0) {
      for (int i = 0; i < num_micro_; ++i) {
        fwd_->q.push_back(QueueEntry(i, 0));
      }
    }
  }
  int ProcessEvent(Simulator* sim, Event event, int64 now);
  void ServiceQueues(Simulator* sim, int64 now);
  bool isDone(){
    return curr_task_ind >= schedule.size();
  }
  void MoveQueues(){
    if(recvact_->q.size()>0 || ( recvact_->q.size()>0 && acts_left<=2)){
      fwd_->insert(recvact_->front());
      recvact_->pop_front();
    }
    if(recvgrad_->q.size()>0 || ( recvgrad_->q.size()>0 && grads_left<=2)){
      rc_ -> insert(recvgrad_->front());
      recvgrad_->pop_front();
    }
  }

private:
  Queue* PickNextNetworkQueue(int64 now);
  Queue* PickNextComputeQueue(int64 now);

  Queue *fwd_, *bwd_, *rc_, *sendact_, *sendgrad_, *recvact_, *recvgrad_;
  bool gpu_busy_ = false;
  int network_busy_ = false;
  int stage_num_;
  int depth_;
  int num_micro_;
  int recomputed_mb_ = -1;
  int last_fwd_mb_ = -1;
  int last_rec_mb_ = -1;
  // bool wait_for_fwd_ = false;
  int waiting_for_stage = -1;
  std::vector<schedule_task> schedule;
  int curr_task_ind = 0;
  // std::vector<bool> acts_recvd, grads_recvd;
  int acts_left = 0;
  int grads_left = 0;
  bool gpipe, pd_1f1b;
};

typedef class Stage Stage;
typedef std::map<int64, Event> EventList;


class Simulator {
public:
  Simulator(int depth, int num_mini, int fwd, int bwd, int send, int allr,
             int last_fwd=0, int last_bwd=0, int send_long=0, int dp=1)
  : pipeline_depth_(depth), num_mini_(num_mini), sendact_time_(send), 
    sendgrad_time_(send), allreduce_time_(allr), data_depth(dp),
    distribution(0.0,10000.0), compute_distribution(0.0,2000.0) {

    char *gpipe_env = getenv("GPIPE");
    gpipe = gpipe_env==NULL ? 0 : (bool) atoi(gpipe_env);
    char *pd_1f1b_env = getenv("PD_1F1B");
    pd_1f1b = pd_1f1b_env==NULL ? 0 : (bool) atoi(pd_1f1b_env);
    char *scattered_env = getenv("SCATTERED");
    scattered = scattered_env==NULL ? 0 : (bool) atoi(scattered_env);
    char *gpus_per_vm_env = getenv("GPUS_PER_VM");
    GPUS_PER_VM = gpus_per_vm_env==NULL ? 4 : atoi(gpus_per_vm_env);

    GenSchedule s(pipeline_depth_, num_mini_, gpipe, pd_1f1b );
    std::vector<schedule_task> schedule[pipeline_depth_];
    s.Generate(schedule);
    if(fwd != 0 && bwd !=0){
      fwd_time_.insert(fwd_time_.end(), pipeline_depth_, fwd);
      bwd_time_.insert(bwd_time_.end(),pipeline_depth_, bwd);
      if(last_fwd != 0 || last_bwd != 0){
        // printf("last stages compute %d %d", last_fwd, last_bwd);
        fwd_time_[pipeline_depth_-1] = last_fwd;
        bwd_time_[pipeline_depth_-1] = last_bwd;
      }
    }else{
      read_compute_times();
    }
    sendact_long_time_ = (send_long==0) ? sendact_time_ : send_long;
    
    stages_ = (Stage**)malloc(sizeof(Stage*) * pipeline_depth_);
    for (int i = 0; i < pipeline_depth_; ++i) {
      stages_[i] = new Stage(i, pipeline_depth_, num_mini, schedule[i]);
    }
    clock_now_micros_ = 0;

    // DumpState();
  }
  void Simulate();
  void DumpState();
  bool AddEvent(int64 scheduled_time, Event e);

  void read_compute_times(){
    std::ifstream infile("compute.txt");
    int a, b;
    int i = 0;
    while (i<pipeline_depth_){
      infile >> a >> b;
      fwd_time_.push_back(a);
      bwd_time_.push_back(b);
      i++;
    }
  }

  int add_jitter(int base, int network){
    auto dist = network ? distribution : compute_distribution;
    // auto generator = network ? net_generator: comp_generator;
    int offset = (int)(dist(net_generator));
    while((base + offset) < 0 || (network && offset<0)){
      offset = (int)(dist(net_generator));
    }
    int retval = base + offset;
    return retval;
  }

  int64 current_time() { return clock_now_micros_;}  
  int fwd_compute_time(int stage) { 
    int basetime = fwd_time_[stage];
    int retval = add_jitter(basetime, 0);
    if(stage < pipeline_depth_ - 1){
      MIN_FWD = (MIN_FWD < retval) ? MIN_FWD : retval ;
      MAX_FWD = (MAX_FWD > retval) ? MAX_FWD : retval ; 
    }
    else{
      MIN_FWD_LONG = (MIN_FWD_LONG < retval) ? MIN_FWD_LONG : retval ;
      MAX_FWD_LONG = (MAX_FWD_LONG > retval) ? MAX_FWD_LONG : retval ; 
    }
    return retval; 
  }
  int bwd_compute_time(int stage) { 
    int basetime = bwd_time_[stage];
    int retval = add_jitter(basetime, 0);
    if(stage < pipeline_depth_ - 1){
      MIN_BWD = (MIN_BWD < retval) ? MIN_BWD : retval ;
      MAX_BWD = (MAX_BWD > retval) ? MAX_BWD : retval ; 
    }
    else{
      MIN_BWD_LONG = (MIN_BWD_LONG < retval) ? MIN_BWD_LONG : retval ;
      MAX_BWD_LONG = (MAX_BWD_LONG > retval) ? MAX_BWD_LONG : retval ; 
    }
    return retval; 
  }

  bool is_same_vm(int s, int d){
    if(!scattered) return (s/GPUS_PER_VM) == (d/GPUS_PER_VM);
    int vm1 = (s*data_depth) / GPUS_PER_VM;
    int vm2 = (d*data_depth) / GPUS_PER_VM;
    return vm1 == vm2;
  }

  // TODO: Can incorporate jitter here.
  // TODO: Add scattered case here
  int sendact_time(int s, int d) { 
    bool same_vm = is_same_vm(s,d);
    int basetime = same_vm ? sendact_time_ : sendact_long_time_;
    int retval = add_jitter(basetime, 1);
    if( same_vm) {
      MAX_SEND = (MAX_SEND > retval) ? MAX_SEND : retval ;
      MIN_SEND = (MIN_SEND < retval) ? MIN_SEND : retval ; 
    }
    else{
      MIN_LONG_SEND = (MIN_LONG_SEND < retval) ? MIN_LONG_SEND : retval ; 
      MAX_LONG_SEND = (MAX_LONG_SEND > retval) ? MAX_LONG_SEND : retval ; 
    }
    return retval;
  }
  int sendgrad_time(int s, int d) { 
    bool same_vm = is_same_vm(s,d);
    int basetime = same_vm ? sendgrad_time_ : sendact_long_time_;
    int retval = add_jitter(basetime, 1);
    if(is_same_vm(s,d)) {
      MAX_SEND = (MAX_SEND > retval) ? MAX_SEND : retval ;
      MIN_SEND = (MIN_SEND < retval) ? MIN_SEND : retval ; 
    }
    else{
      MIN_LONG_SEND = (MIN_LONG_SEND < retval) ? MIN_LONG_SEND : retval ; 
      MAX_LONG_SEND = (MAX_LONG_SEND > retval) ? MAX_LONG_SEND : retval ;  
    }
    return retval;
  }

private:
  bool PopNextEvent(Event* e);

  Stage** stages_;
  EventList event_list_;
  int pipeline_depth_;
  int num_mini_;
  int64 clock_now_micros_;
  int sendact_time_, sendgrad_time_, allreduce_time_;
  int sendact_long_time_;
  int GPUS_PER_VM;
  vector<int> fwd_time_, bwd_time_; 
  normal_distribution<double> distribution;
  normal_distribution<double> compute_distribution;
  default_random_engine net_generator; //, comp_generator;

  int MAX_SEND = 0, MIN_SEND = 10000000;
  int MAX_LONG_SEND = 0, MIN_LONG_SEND = 10000000;

  int MAX_FWD = 0, MIN_FWD = 10000000; 
  int MAX_BWD = 0, MIN_BWD = 10000000; 
  int MAX_FWD_LONG = 0, MIN_FWD_LONG = 10000000; 
  int MAX_BWD_LONG = 0, MIN_BWD_LONG = 10000000; 
  bool gpipe, scattered, pd_1f1b;
  int data_depth;
};