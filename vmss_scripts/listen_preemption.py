#!/usr/bin/python
# to be run as a background thread in worker nodes
import json
import socket
import urllib.request
import time
from datetime import datetime

metadata_url = "http://169.254.169.254/metadata/scheduledevents?api-version=2019-01-01"
this_host = socket.gethostname()

def client(ip, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(bytes(message, 'ascii'))

def get_scheduled_events():
    req = urllib.request.Request(metadata_url)
    req.add_header('Metadata', 'true')
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    return data


def handle_scheduled_events(data, ip, port):
    for evt in data['Events']:
        eventid = evt['EventId']
        status = evt['EventStatus']
        resources = evt['Resources']
        eventtype = evt['EventType']
        resourcetype = evt['ResourceType']
        notbefore = evt['NotBefore'].replace(" ", "_")
        print(datetime.now())
        print("+ Scheduled Event. This host " + this_host + " is scheduled for " + eventtype + " not before " + notbefore)
        # Logic for handling event, send morph signal to master
        print(this_host in resources)
        client(ip, port, "preempt {}".format(notbefore))

def main():
    ip, port = "10.0.3.4", 4200       # manager IP
    while True:
        data = get_scheduled_events()
        handle_scheduled_events(data, ip, port)
        time.sleep(5)

if __name__ == '__main__':
    main()