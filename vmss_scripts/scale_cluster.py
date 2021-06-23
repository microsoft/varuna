
import os
from datetime import datetime
import sys
import traceback

resource_group = "Varuna"
subscription = "f3ebbda2-3d0f-468d-8e23-31a0796dcac1"
cluster = "single_gpu_spots"
morph_path = "/home/varuna/t-saathl/Varuna/Megatron-LM/"

def remove_dead(ip_list, slow_machines):
    # get instance ids
    id_cmd = "az vmss nic list -g {} --subscription {} --vmss-name {} \
                --query \"[?contains({}, ipConfigurations[0].privateIpAddress)].[ipConfigurations[0].privateIpAddress, virtualMachine.id]\"\
                -o tsv".format(resource_group, subscription, cluster, str(ip_list))
    status_cmd = "az vmss get-instance-view --name {} --resource-group {} --subscription {} \
                    --instance-id {} --query {{status:statuses[1].displayStatus}} --output tsv"
    delete_cmd = "az vmss delete-instances --resource-group {} --subscription {} \
                    --name {} --instance-ids {}"

    ip_to_ids = os.popen(id_cmd).read().split("\n")
    id_map = dict()
    for pair in ip_to_ids:
        if pair == "":
            continue
        ip, rid = pair.split("\t")
        rid = int(rid.split("/")[-1] )
        id_map[ip] = rid
        # print(ip, rid)

    remove_ids = []
    for ip in ip_list:
        cmd = status_cmd.format(cluster, resource_group, subscription, id_map[ip])
        status = os.popen(cmd).read()
        if ("running" in status or "updating" in status) \
            and (ip not in slow_machines):
            continue
        remove_ids.append(id_map[ip])

    response = 0
    if len(remove_ids) > 0:
        print("removing: ", remove_ids)
        concat_ids = " ".join([ str(i) for i in remove_ids])
        response = os.system( delete_cmd.format(resource_group, subscription, cluster, concat_ids) )
    return response == 0

def get_available_machines():
    # gets reachable machines
    ping_script = os.path.join(morph_path, "get_available_machines.sh")
    bash_out = os.popen("bash {} 1 {} 1 1".format(ping_script, cluster)).read()
    machines = bash_out.split("\n")
    if machines[-1] == "":
        machines = machines[:-1]
    split = -1
    for i,m in enumerate(machines):
        if "unreachable" in m:
            split = i
            break
    assert split >= 0, "unable to get reachable machines!"
    up = machines[:split]
    down = machines[split+1:]
    return up, down

def get_slow_machines():
    filename = os.path.join(morph_path, "slow_machines.out")
    with open(filename,"r") as f:
        slow_out = f.read()
    machines = slow_out.split("\n")
    if machines[-1] == "":
        machines = machines[:-1]
    return machines

def scale_out(max_size=87):
    global resource_group, subscription, cluster
    command = "az vmss scale --new-capacity {} --name {} --resource-group {} \
                --subscription {} 1>&2".format(max_size, cluster, resource_group, subscription)
    response = os.system(command)
    return response == 0


def setup_machines():
    os.chdir(morph_path)
    get_available_cmd = f"bash get_available_machines.sh 0 {cluster} 1 1"
    os.system(f"{get_available_cmd} > all_machines")
    # twice for safety - idempotent
    os.system(f"bash copy_and_run_init.sh all_machines")
    os.system(f"bash copy_and_run_init.sh all_machines")
    # all_machines = os.popen(get_available_cmd).read().split("\n")
    # if all_machines[-1] == "":
    #     all_machines = all_machines[:-1]
    # old_machines = open("available_machines.out","r").read().split()
    # new_out = open("new_machines.out","w")
    # for m in all_machines:
    #     if m not in old_machines:
    #         new_out.write(m+"\n")
    # new_out.close()
    # os.system("bash warmup.sh new_machines.out")



if __name__ == "__main__":

    max_size = int(sys.argv[1])
    print(datetime.now())
    current_machines, dead_machines = get_available_machines()
    current_size = len(current_machines)
    # grow_to = max(int(current_size * 1.1), current_size + 10)
    # max_size = min(max_size, grow_to)
    slow_machines = get_slow_machines()
    if len(dead_machines) > 0:
        print("Dead:",dead_machines)
        print("Slow:",slow_machines)
        success = remove_dead(dead_machines, slow_machines)
        print("remove success", success)
        if success:
            # empty slow machines log
            open(os.path.join(morph_path, "slow_machines.out"),"w").close()

    if len(current_machines) < max_size:
        try:
            success = False
            scale_size = max_size
            while scale_size > current_size and not success:
                print("try scaling to {}".format(scale_size))
                success = scale_out(scale_size)
                new_up, new_down  = get_available_machines()
                success = success or len(new_up) > current_size
                if success:
                    print("Scaled out to {}!".format(len(new_up)))
                    setup_machines()
                    break
                remove_dead(new_down,[])
                scale_size -= 10
            else:
                print("Didn't scale, at {}".format(current_size))            
        except Exception as e:
            print("couldn't scale:", e)
            traceback.print_exc()
    else:
        print("already at max!")