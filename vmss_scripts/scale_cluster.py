
import os

def remove_dead(ip_list):
    # get instance ids
    id_cmd = "az vmss nic list -g Varuna --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a --vmss-name megatron \
                --query \"[?contains({}, ipConfigurations[0].privateIpAddress)].[ipConfigurations[0].privateIpAddress, virtualMachine.id]\"\
                -o tsv".format(str(ip_list))
    status_cmd = "az vmss get-instance-view --name megatron --resource-group Varuna --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a \
                    --instance-id {} --query {{status:statuses[1].displayStatus}} --output tsv"
    delete_cmd = "az vmss delete-instances --resource-group Varuna --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a \
                    --name megatron --instance-ids {}"

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
        cmd = status_cmd.format(id_map[ip])
        status = os.popen(cmd).read()
        if "running" in status or "updating" in status:
            continue
        remove_ids.append(id_map[ip])

    print("removing: ", remove_ids)
    concat_ids = " ".join([ str(i) for i in remove_ids])
    response = os.system( delete_cmd.format(concat_ids) )
    return response == 0

def get_available_machines():
    # gets reachable machines
    bash_out = os.popen("bash /home/varuna/t-nisar/Varuna/Megatron-LM/get_available_machines.sh megatron 1").read()
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
    command = "az vmss scale --new-capacity {} --name megatron --resource-group Varuna \
                --subscription a947bb9f-f570-40a9-82cc-2fdd09f1553a".format(max_size)
    response = os.system(command)
    return response == 0


if __name__ == "__main__":

    max_size = 74

    current_machines, dead_machines = get_available_machines()
    if len(dead_machines) > 0:
        remove_dead(dead_machines)
    if True:#len(current_machines) < max_size:
        try:
            success = scale_out(max_size)
            if success:
                print("Scaled out to max!")
        except Exception as e:
            print("couldn't scale:", e)