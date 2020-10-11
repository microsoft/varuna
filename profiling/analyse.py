import os

diry = "/home/varuna/gpt2-blob/perf_analysis_1.5b/stats"

p = 8
gpus = 40
dp = gpus // p
stage = 0
mbs = 5

prefix = "varuna_logs-{}dp-{}mBS-stage{}of{}".format(dp,mbs, stage,p)

avg_fwd_time = 0
avg_bwd_time = 0
avg_comm_time = 0
avg_alr_time = 0
avg_osync_time = 0
for i in range(dp):
    logfile = os.path.join(diry, prefix + "_" + str(i))
    fwd_times = []; bwd_times = []; rec_times = []
    gsync_times = []; osync_times = []; comm_times = []
    f = open(logfile,'r')
    for line in f:
        if "BATCH END" in line:
            break
    for line in f:
        if "BATCH END" in line:
            break
        if "comm" in line or "recv" in line:
            comm_times.append(float(line.split(" ")[-1]))
        elif "fwd" in line:
            fwd_times.append(float(line.split(" ")[-1]))
        elif "bwd" in line:
            bwd_times.append(float(line.split(" ")[-1]))
        elif "rec" in line:
            rec_times.append(float(line.split(" ")[-1]))
        elif "all_reduce" in line:
            gsync_times.append(float(line.split(" ")[-1]))
        elif "overflow" in line:
            osync_times.append(float(line.split(" ")[-1]))
    print(i, sum(comm_times), sum(fwd_times), sum(bwd_times),sum(rec_times),sum(gsync_times),sum(osync_times), len(comm_times))
    avg_comm_time += sum(comm_times)
    avg_fwd_time += (sum(fwd_times) / len(fwd_times))
    avg_bwd_time += (sum(bwd_times) / len(bwd_times))
    avg_alr_time += gsync_times[0]
    avg_osync_time += osync_times[0]

avg_fwd_time /= dp
avg_bwd_time /= dp
avg_alr_time /= dp
avg_comm_time /= dp
avg_osync_time /= dp

print('Averages:')
print('fwd', avg_fwd_time, 'bwd', avg_bwd_time, 'alr', avg_alr_time, 'comm', avg_comm_time,'osync', avg_osync_time)


prefix = "test_{}p_8K_{}mbs_{}gpus-".format(p,mbs,gpus)

mb_time = 0
count = 0
for i in range(dp):
    lossfile = prefix + str((i+1)*p - 1) + ".txt"
    lossfile = open(os.path.join(diry, lossfile),"r")
    lossfile.readline()
    lossfile.readline()
    lossfile.readline()
    for line in lossfile:
        if "Loss scale" in line:
            lossfile.readline(); lossfile.readline()
            continue
        mb_time += float(line.split(",")[0])
        count += 1

print("MB time rec",mb_time/count)