from __future__ import print_function
import socket
import select
import sys
import Pyro4
import Pyro4.core
import Pyro4.naming
import Pyro4.socketutil
from pyspark import SparkContext
from threading import Thread
from ArtificialNeuralNetwork import *
from neural_network_api import *

if sys.version_info < (3, 0):
    input = raw_input


class ThreadedServer(Thread):
    def __init__(self):
        Thread.__init__(self)
    
    def run(self):
       pass


class EmbeddedServer(object):

    def initialise_object(self, user_id):
        network_api = NeuralNetworkAPI(sc)
        network_uri = pyrodaemon.register(network_api)
        print("network_uri: " + str(network_uri))
        nameserverDaemon.nameserver.register("example.embedded.server.network." + str(user_id), network_uri)
        print("complete API URI: ")


if __name__ == "__main__":
    ThreadedServer().start()
    print("Make sure that you don't have a name server running already.")     
    servertype = 't'
    if servertype == 't':
         Pyro4.config.SERVERTYPE = "thread"
    else:
        Pyro4.config.SERVERTYPE = "multiplex"
    hostname = socket.gethostname()
    my_ip = Pyro4.socketutil.getIpAddress(None, workaround127=True)
    print("initializing services... servertype=%s" % Pyro4.config.SERVERTYPE)
    # start a name server with broadcast server as well
    nameserverUri, nameserverDaemon, broadcastServer = Pyro4.naming.startNS(host=my_ip)
    assert broadcastServer is not None, "expect a broadcast server to be created"

    print("got a Nameserver, uri=%s" % nameserverUri)
    print("ns daemon location string=%s" % nameserverDaemon.locationStr)
    print("ns daemon sockets=%s" % nameserverDaemon.sockets)
    print("bc server socket=%s (fileno %d)" % (broadcastServer.sock, broadcastServer.fileno()))

    # create a Pyro daemon
    pyrodaemon = Pyro4.core.Daemon(host=hostname)
    print("daemon location string=%s" % pyrodaemon.locationStr)
    print("daemon sockets=%s" % pyrodaemon.sockets)

    # register a server object with the daemon
    serveruri = pyrodaemon.register(EmbeddedServer())
    print("server uri=%s" % serveruri)

    # register it with the embedded nameserver directly
    nameserverDaemon.nameserver.register("example.embedded.server", serveruri)
    sc = SparkContext(appName = "DistributedServer.py")
    # below is our custom event loop.
    while True:
            print("Waiting for events...")
            # create sets of the socket objects we will be waiting on
            # (a set provides fast lookup compared to a list)
            nameserverSockets = set(nameserverDaemon.sockets)
            pyroSockets = set(pyrodaemon.sockets)
            rs = [broadcastServer]  # only the broadcast server is directly usable as a select() object
            rs.extend(nameserverSockets)
            rs.extend(pyroSockets)
            rs, _, _ = select.select(rs, [], [], 3)
            eventsForNameserver = []
            eventsForDaemon = []
            for s in rs:
                if s is broadcastServer:
                    print("Broadcast server received a request")
                    broadcastServer.processRequest()
                elif s in nameserverSockets:
                    eventsForNameserver.append(s)
                elif s in pyroSockets:
                    eventsForDaemon.append(s)
            if eventsForNameserver:
                print("Nameserver received a request")
                nameserverDaemon.events(eventsForNameserver)
            if eventsForDaemon:
                print("Daemon received a request")
                pyrodaemon.events(eventsForDaemon)

    sc.stop()
    nameserverDaemon.close()
    broadcastServer.close()
    pyrodaemon.close()
    print("done")
