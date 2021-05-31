#include "simulate-varuna.h"

// #define DEBUG

int MAX_NETWORK = 4;

static inline Queue* QueueReady(Queue* q, int64 now) {
  if (!q->q.empty() && q->front().time <= now) return q;
  return NULL;
}

Event::State IdToState(char id) {
  switch(id) {
    case 'f': return Event::FWD_START;
    case 'r': return Event::RC_START;
    case 'b': return Event::BWD_START;
    case 'a': return Event::SENDACT_START;
    case 'g': return Event::SENDGRAD_START;
  }
}

// ===========  class Stage ===================

Queue* Stage::PickNextNetworkQueue(int64 now) {
  // Defines precedence order of servicing queues
  Queue* q = NULL;
  if (q = QueueReady(sendact_, now)) return q;
  if (q = QueueReady(sendgrad_, now)) return q;
}

// Queue* Stage::PickNextComputeQueue(int64 now) {
//   // Defines precedence order of servicing queues
//   Queue* q = NULL;
//   // To service bwd queue, we need the recomputation to be done already.
//   // Unless you are the last stage.
//   if (!wait_for_fwd_ && (q = QueueReady(bwd_, now)) &&
//       (stage_num_ == depth_-1 || q->front().micro_batch == recomputed_mb_)) return q;
//   // Only one recomputed activation can be in flight.
//   if (!wait_for_fwd_ && (recomputed_mb_ < 0) && (q = QueueReady(rc_, now))
//       && (q->front().micro_batch == last_rec_mb_ + 1)){
//         last_rec_mb_ += 1;
//         return q;
//   }
//   // Service fwd queue only if no recomputed activations are in flight.
//   if ((recomputed_mb_ < 0) && (q = QueueReady(fwd_, now))
//       && (q->front().micro_batch == last_fwd_mb_ + 1)){
//       last_fwd_mb_ += 1;   
//       return q;
//   }
//   return NULL;
// }

Queue* Stage::PickNextComputeQueue(int64 now) {
  // Defines precedence order of servicing queues
  Queue* q = NULL;

  if(curr_task_ind >= schedule.size()) return NULL;

  schedule_task t = schedule[curr_task_ind];

  // swap 
  if(!gpipe && t.task == '1' && !QueueReady(rc_,now) && (last_fwd_mb_<num_micro_-1)){
    for(int i=curr_task_ind+1; i<schedule.size(); i++){
      if(schedule[i].task=='0'){
        schedule.insert(schedule.begin() + curr_task_ind, schedule[i]);
        schedule.erase(schedule.begin() + i+1);
        break;
      }
    }
  }

  t = schedule[curr_task_ind];

  if (t.task == '2' && (q = QueueReady(bwd_, now)) && 
      q->front().micro_batch == t.index && 
      (stage_num_==depth_-1 || recomputed_mb_ == t.index)){
      curr_task_ind++;
      return q;
  }
  if (t.task == '1' && (recomputed_mb_ < 0) && (q = QueueReady(rc_, now))
      && (q->front().micro_batch == t.index)){
        last_rec_mb_++;
        curr_task_ind++;
        return q;
  }
  // Service fwd queue only if no recomputed activations are in flight.
  if (t.task == '0' && (recomputed_mb_ < 0) && 
      (q = QueueReady(fwd_, now)) && (q->front().micro_batch == t.index)){
      last_fwd_mb_++;
      curr_task_ind++;   
      return q;
  }
  return NULL;
}


void Stage::ServiceQueues(Simulator* sim, int64 now) {
  Queue* q;
  waiting_for_stage = -1;
  if (!gpu_busy_){
    if(q = PickNextComputeQueue(now)) {
      QueueEntry qe = q->front();
      q->pop_front();
      sim->AddEvent(now, Event(stage_num_, qe.micro_batch, IdToState(q->id)));
    }
    else if(curr_task_ind < schedule.size()){
      waiting_for_stage = schedule[curr_task_ind].task == '0' ?
                              stage_num_ - 1 : stage_num_ + 1 ;
      // printf("%d WAITING FOR %d\n", stage_num_, waiting_for_stage);
    }
  }
  
  // if (!network_busy_ && (q = PickNextNetworkQueue(now))) {
  //   QueueEntry qe = q->front();
  //   q->pop_front();
  //   sim->AddEvent(now, Event(stage_num_, qe.micro_batch, IdToState(q->id)));
  // }
  if ((network_busy_ < MAX_NETWORK) && (q = QueueReady(sendact_, now))){
    QueueEntry qe = q->front();
    q->pop_front();
    sim->AddEvent(now, Event(stage_num_, qe.micro_batch, IdToState(q->id)));
  }
  if ((network_busy_ < MAX_NETWORK) && (q = QueueReady(sendgrad_, now))){
    QueueEntry qe = q->front();
    q->pop_front();
    sim->AddEvent(now, Event(stage_num_, qe.micro_batch, IdToState(q->id)));
  }
}

int Stage::ProcessEvent(Simulator* sim, Event e, int64 now) {
#ifdef DEBUG
  printf("Event time: %lld  Stage: %d MB: %d state: %s\n",
         now, e.stage, e.mb_num, Event::StringState(e.state));
#endif
  int send_time;
  switch(e.state) {
    case Event::FWD_START:
      gpu_busy_ = true;
      sim->AddEvent(now + sim->fwd_compute_time(stage_num_), 
                    Event(stage_num_, e.mb_num, Event::FWD_DONE));
      break;
    case Event::FWD_DONE:
      // Figure out next compute work to schedule.
      gpu_busy_ = false;
      if (stage_num_ != depth_ - 1) {
        sendact_->push_back(QueueEntry(e.mb_num, now));
      } else if (!gpipe) {
        bwd_->insert(QueueEntry(e.mb_num, now)); 
      } else if(e.mb_num == num_micro_-1){
        rc_->push_back(QueueEntry(e.mb_num, now));
      }
      break;
    case Event::BWD_START:
      gpu_busy_ = true;
      sim->AddEvent(now + sim->bwd_compute_time(stage_num_), 
                    Event(stage_num_, e.mb_num, Event::BWD_DONE));
      // Prev. stage can potentially start recompute  halfway during my bwd exec.
      // if (stage_num_ > 0) {
      //   int offset = sim->bwd_compute_time(stage_num_) - sim->fwd_compute_time(stage_num_-1);
      //   sim->AddEvent(now + offset,Event(stage_num_ - 1, e.mb_num, Event::RC_QUEUE));
      // }
      break;
    case Event::BWD_DONE:
      // Figure out next compute work to schedule.
      gpu_busy_ = false;
      recomputed_mb_ = -1;
      if (stage_num_ > 0) {
        sendgrad_->push_back(QueueEntry(e.mb_num, now));
      }
      if (stage_num_ == depth_ - 1 && gpipe && e.mb_num > 0)
        rc_->push_back(QueueEntry(e.mb_num-1, now));
      break;
    case Event::RC_QUEUE:
      if(!gpipe) rc_->insert(QueueEntry(e.mb_num, now));
      else rc_-> insert_rev(QueueEntry(e.mb_num, now));
      break;

    case Event::RC_START:
      gpu_busy_ = true;
      sim->AddEvent(now + sim->fwd_compute_time(stage_num_), 
                    Event(stage_num_, e.mb_num, Event::RC_DONE));
      break;
    case Event::RC_DONE:
      recomputed_mb_ = e.mb_num;          
      gpu_busy_ = false;
      if(!gpipe){
        bwd_->insert(QueueEntry(e.mb_num, now));
      }else{
        if (stage_num_ == depth_-1) bwd_->insert_rev(QueueEntry(e.mb_num, now));
        if (stage_num_ > 0) sim->AddEvent(now, Event(stage_num_ - 1, e.mb_num, Event::RC_QUEUE));
      }
      break;
    case Event::SENDACT_START:
      network_busy_++;
      send_time = sim->sendact_time(stage_num_, stage_num_+1);
      sim->AddEvent(now + send_time, 
                    Event(stage_num_, e.mb_num, Event::SENDACT_DONE));
      sim->AddEvent(now + send_time, 
                    Event(stage_num_ + 1, e.mb_num, Event::RECV_ACT));

      break;
    case Event::SENDACT_DONE:
    case Event::SENDGRAD_DONE:
      network_busy_--;
      break;
    case Event::SENDGRAD_START:
      network_busy_++;
      send_time = sim->sendgrad_time(stage_num_, stage_num_-1);
      sim->AddEvent(now + send_time, 
                    Event(stage_num_, e.mb_num, Event::SENDGRAD_DONE));
      sim->AddEvent(now + send_time, 
                    Event(stage_num_ - 1, e.mb_num, Event::RECV_GRAD));
      // sim->AddEvent(now + send_time - sim->fwd_compute_time(stage_num_-1) - 1000 , 
      //               Event(stage_num_ - 1, e.mb_num, Event::RC_QUEUE));
      break;
    case Event::RECV_ACT:
      // recvact_->insert(QueueEntry(e.mb_num, now));
      // acts_left--;
      fwd_->insert(QueueEntry(e.mb_num, now));
      break;
    case Event::RECV_GRAD:
      // recvgrad_->insert(QueueEntry(e.mb_num, now));
      // grads_left--;
      if(!gpipe){
        rc_->insert(QueueEntry(
          rc_->q.empty() ? last_rec_mb_ + 1: rc_->q.back().micro_batch+1, now));
      } else{
        bwd_->insert_rev(QueueEntry(e.mb_num, now));
      }
      break;
    default:
      printf("Invalid event");
  }
  ServiceQueues(sim, now);
  return waiting_for_stage;
}

// ======== class Simulator ===================

bool Simulator::AddEvent(int64 scheduled_time, Event e) {
  int64 time_nanos = scheduled_time * 1000ll;
  EventList::iterator it = event_list_.find(time_nanos);
  if (it != event_list_.end()) {
    // Another event w/ same time exists.  Try slightly diff. time
    int nr_retries = 0;
    for (nr_retries = 0; nr_retries < 10; ++nr_retries) {
      int offset = random()%1000;
      int64 new_time = time_nanos + offset;
      EventList::iterator it1;
      if ((it1 = event_list_.find(new_time)) == event_list_.end()) {
        time_nanos = new_time;
        break;
      }
    }
    if (nr_retries == 10) {
      printf("Conflicting time even after retries.  Aborting");
      exit(0);
    }
  }
  event_list_[time_nanos] = e;
}

bool Simulator::PopNextEvent(Event* event) {
  EventList::iterator it = event_list_.begin();
  if (it == event_list_.end()) {
    return false;
  }
  clock_now_micros_ = it->first/1000;
  *event = it->second;
  event_list_.erase(it);
  return true;
}

void Simulator::DumpState() {
  printf("\nVaruna simulator.  Current time = %lld usec", clock_now_micros_);
  printf("\nNum microbatches = %d, Pipeline depth = %d", num_mini_, pipeline_depth_);
  printf("\nCompute times (fwd = %d, bwd = %d)", fwd_time_[0], bwd_time_[0]);
  printf("\nLast stage compute times (fwd = %d, bwd = %d)", fwd_time_[pipeline_depth_-1], bwd_time_[pipeline_depth_-1]);
  printf("\nCommunication times (activation = %d, grad = %d)", sendact_time_, sendgrad_time_);
  printf("\nLong comm times (activation/grad = %d)", sendact_long_time_);
  printf("\n");
}

void Simulator::Simulate() {
  // Bootstrap with first stage.

  stages_[0]->ServiceQueues(this, clock_now_micros_);

  Event event;
  int wait_stage = -1;
  while (PopNextEvent(&event)) {
    if (event.stage >= pipeline_depth_) {
      printf("Unexpected stage in event: %d vs. %d", event.stage, pipeline_depth_);
      return;
    }
    wait_stage = stages_[event.stage]->ProcessEvent(this, event, clock_now_micros_);
    // if(wait_stage >=0 && wait_stage < pipeline_depth_)
    //   stages_[wait_stage]->ServiceQueues(this, clock_now_micros_);
    // for(int i = 0; i < pipeline_depth_; i++)
    //   stages_[i]->MoveQueues();
  }
  if (!stages_[0]->isDone()){
    printf("THERE'S AN ERROR!\n");
    return;
  }
  #ifdef DEBUG
  for(int i = 0; i < pipeline_depth_; i++){
    printf("Event time: %lld  Stage: %d MB: - state: ALL_REDUCE\n",clock_now_micros_, i);
  }
  #endif
  printf("End of simulation:  Mini-batch time (usec) = %lld\n", clock_now_micros_ + allreduce_time_);
  printf("Min send: %d, max send %d\n", MIN_SEND, MAX_SEND);
  printf("Min long send: %d, max long send %d\n", MIN_LONG_SEND, MAX_LONG_SEND);
  printf("Min fwd: %d, max fwd %d; min bwd %d, max bwd %d\n", MIN_FWD, MAX_FWD, MIN_BWD, MAX_BWD);
  printf("Min long fwd: %d, max long fwd %d; min long bwd %d, max long bwd %d\n", MIN_FWD_LONG, MAX_FWD_LONG, MIN_BWD_LONG, MAX_BWD_LONG );
}