
import os
from datetime import datetime
import sys

cluster = sys.argv[1]
resource_group = "Varuna"
subscription = "f3ebbda2-3d0f-468d-8e23-31a0796dcac1"
# cluster = "megatron"
morph_path = "/home/varuna/t-saathl/Varuna/Megatron-LM/"

def remove_dead(ip_list, cluster):
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
        if ip not in id_map:
            continue
        cmd = status_cmd.format(cluster, resource_group, subscription, id_map[ip])
        status = os.popen(cmd).read()
        if "running" in status or "updating" in status:
            continue
        remove_ids.append(id_map[ip])

    if len(remove_ids) > 0:
        print("removing: ", remove_ids)
        concat_ids = " ".join([ str(i) for i in remove_ids])
        response = os.system( delete_cmd.format(resource_group, subscription, cluster, concat_ids) )
        return response == 0

def get_available_machines():
    # gets reachable machines
    ping_script = os.path.join(morph_path, "get_available_machines.sh")
    bash_out = os.popen("bash {} 1 {} 1".format(ping_script, cluster)).read()
    machines = bash_out.split("\n")
    if machines[-1] == "":
        machines = machines[:-1]
    split = -1
    for i,m in enumerate(machines):
        if "unreachable" in m:
            split = i
            break
    assert split > 0, "unable to get reachable machines!"
    up = machines[:split]
    down = machines[split+1:]
    return up, down


def scale_out(max_size=87):
    global resource_group, subscription, cluster
    command = "az vmss scale --new-capacity {} --name {} --resource-group {} \
                --subscription {}".format(max_size, cluster, resource_group, subscription)
    response = os.system(command)
    return response == 0


if __name__ == "__main__":

    max_size = 25
    dt = datetime.now()
    print(dt, cluster)
    print(dt, cluster, file=sys.stderr)
    current_machines, dead_machines = get_available_machines()
    if len(dead_machines) > 0:
        print("Dead:",dead_machines)
        success = remove_dead(dead_machines, cluster)
        print("remove success", success)
    if len(current_machines) < max_size:
        try:
            success = False
            current_size = len(current_machines)
            scale_size = max_size
            while scale_size > current_size and not success:
                print("try scaling to {}".format(scale_size))
                success = scale_out(scale_size)
                if success:
                    print("Scaled out to {}!".format(scale_size))
                    break
                scale_size -= 5
            else:
                print("Didn't scale, at {}".format(current_size))            
        except Exception as e:
            print("couldn't scale:", e)
    else:
        print("already at max!")