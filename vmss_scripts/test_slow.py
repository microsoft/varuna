from torch.nn import Linear
import torch
import time
import socket

def client(ip, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(bytes(message, 'ascii'))

m = Linear(16384, 16384)
m.cuda()

total_time = 0.0
for i in range(200):
    time_i = time.time()
    input = torch.randn(1024, 16384)
    input = input.cuda()
    output = m(input)
    torch.cuda.synchronize()
    time_i = time.time() - time_i
    total_time += time_i

client("10.0.3.4",4100, str(total_time))