import os

diry = "/home/varuna/gpt2-blob/perf_tests_1.5b/"

p = 4
gpus = 80
dp = gpus // p
stage = 3
mbs = 2

prefix = "stats_{}x{}/varuna_logs-mBS{}-stage{}of{}".format(p,dp,mbs, stage,p)

avg_fwd_time = 0; min_fwd_time = 1000; max_fwd_time = 0
avg_bwd_time = 0; min_bwd_time = 1000; max_bwd_time = 0
avg_comm_time = 0
avg_alr_time = 0
avg_osync_time = 0
for i in range(dp):
    logfile = os.path.join(diry, prefix + "_" + str(i))
    fwd_times = []; bwd_times = []; rec_times = []
    gsync_times = []; osync_times = []; comm_times = []
    f = open(logfile,'r')
    print(logfile)
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
    min_fwd_time = min(min(fwd_times), min_fwd_time)
    max_fwd_time = max(max(fwd_times), max_fwd_time)
    min_bwd_time = min(min(bwd_times), min_bwd_time)
    max_bwd_time = max(max(bwd_times), max_bwd_time)
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
print("fwd range:", min_fwd_time, max_fwd_time)
print("bwd range:", min_bwd_time, max_bwd_time)

prefix = "stats_{}x{}/test_{}p-".format(p,dp,p,mbs,gpus)

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
            lossfile.readline()
            lossfile.readline()
            continue
        mb_time += float(line.split(",")[0])
        count += 1

print("MB time rec",mb_time/count)