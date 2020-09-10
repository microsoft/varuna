#include "simulate-varuna.h"
#define DEBUG

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

Queue* Stage::PickNextComputeQueue(int64 now) {
  // Defines precedence order of servicing queues
  Queue* q = NULL;
  // To service bwd queue, we need the recomputation to be done already.
  // Unless you are the last stage.
  if ((q = QueueReady(bwd_, now)) &&
      (stage_num_ == depth_-1 || q->front().micro_batch == recomputed_mb_)) return q;
  // Only one recomputed activation can be in flight.
  if ((recomputed_mb_ < 0) && (q = QueueReady(rc_, now))) return q;
  // Service fwd queue only if no recomputed activations are in flight.
  if ((recomputed_mb_ < 0) && (q = QueueReady(fwd_, now))) return q;
  return NULL;
}

void Stage::ServiceQueues(Simulator* sim, int64 now) {
  Queue* q;
  if (!gpu_busy_ && (q = PickNextComputeQueue(now))) {
    QueueEntry qe = q->front();
    q->pop_front();
    sim->AddEvent(now, Event(stage_num_, qe.micro_batch, IdToState(q->id)));
  }
  if (!network_busy_ && (q = PickNextNetworkQueue(now))) {
    QueueEntry qe = q->front();
    q->pop_front();
    sim->AddEvent(now, Event(stage_num_, qe.micro_batch, IdToState(q->id)));
  }
}

bool Stage::ProcessEvent(Simulator* sim, Event e, int64 now) {
#ifdef DEBUG
  printf("Event time: %lld  Stage: %d MB: %d state: %s\n",
         now, e.stage, e.mb_num, Event::StringState(e.state));
#endif

  switch(e.state) {
    case Event::FWD_START:
      gpu_busy_ = true;
      sim->AddEvent(now + sim->fwd_compute_time(), 
                    Event(stage_num_, e.mb_num, Event::FWD_DONE));
      break;
    case Event::FWD_DONE:
      // Figure out next compute work to schedule.
      gpu_busy_ = false;
      if (stage_num_ != depth_ - 1) {
        sendact_->push_back(QueueEntry(e.mb_num, now));
      } else {
        bwd_->push_back(QueueEntry(e.mb_num, now)); 
      }
      break;
    case Event::BWD_START:
      gpu_busy_ = true;
      sim->AddEvent(now + sim->bwd_compute_time(), 
                    Event(stage_num_, e.mb_num, Event::BWD_DONE));
      // Prev. stage can potentially start recompute  halfway during my bwd exec.
      if (stage_num_ > 0) {
        sim->AddEvent(now + sim->bwd_compute_time()/2,
                      Event(stage_num_ - 1, e.mb_num, Event::RC_QUEUE));
      }
      break;
    case Event::BWD_DONE:
      // Figure out next compute work to schedule.
      gpu_busy_ = false;
      recomputed_mb_ = -1;
      if (stage_num_ > 0) {
        sendgrad_->push_back(QueueEntry(e.mb_num, now));
      }
      break;
    case Event::RC_QUEUE:
      rc_->push_back(QueueEntry(e.mb_num, now));
      break;

    case Event::RC_START:
      gpu_busy_ = true;
      sim->AddEvent(now + sim->fwd_compute_time(), 
                    Event(stage_num_, e.mb_num, Event::RC_DONE));
      break;
    case Event::RC_DONE:
      recomputed_mb_ = e.mb_num;          
      gpu_busy_ = false;
      break;
    case Event::SENDACT_START:
      network_busy_ = true;
      sim->AddEvent(now + sim->sendact_time(), 
                    Event(stage_num_, e.mb_num, Event::SENDACT_DONE));
      sim->AddEvent(now + sim->sendact_time(), 
                    Event(stage_num_ + 1, e.mb_num, Event::RECV_ACT));

      break;
    case Event::SENDACT_DONE:
    case Event::SENDGRAD_DONE:
      network_busy_ = false;
      break;
    case Event::SENDGRAD_START:
      network_busy_ = true;
      sim->AddEvent(now + sim->sendgrad_time(), 
                    Event(stage_num_, e.mb_num, Event::SENDGRAD_DONE));
      sim->AddEvent(now + sim->sendgrad_time(), 
                    Event(stage_num_ - 1, e.mb_num, Event::RECV_GRAD));
      break;
    case Event::RECV_ACT:
      fwd_->push_back(QueueEntry(e.mb_num, now));
      break;
    case Event::RECV_GRAD:
      bwd_->push_back(QueueEntry(e.mb_num, now));
      break;
    default:
      printf("Invalid event");
  }
  ServiceQueues(sim, now);
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
  printf("\nCompute times (fwd = %d, bwd = %d)", fwd_time_, bwd_time_);
  printf("\nCommunication times (activation = %d, grad = %d)", sendact_time_, sendgrad_time_);
  printf("\n");
}

void Simulator::Simulate() {
  // Bootstrap with first stage.
  stages_[0]->ServiceQueues(this, clock_now_micros_);

  Event event;
  while (PopNextEvent(&event)) {
    if (event.stage >= pipeline_depth_) {
      printf("Unexpected stage in event: %d vs. %d", event.stage, pipeline_depth_);
      return;
    }
    stages_[event.stage]->ProcessEvent(this, event, clock_now_micros_);
  }
  printf("End of simulation:  Mini-batch time = %lld usec\n", clock_now_micros_ + allreduce_time_);
}