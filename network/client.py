from __future__ import print_function
import Pyro4
import sys
import socket
import Pyro4.core
import Pyro4.socketutil


if sys.version_info < (3, 0):
    input = raw_input

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print >> sys.stderr, "Usage: distribution <user id>"

    with Pyro4.core.Proxy("PYRONAME:example.embedded.server@10.6.31.75:9090") as proxy:
        print("argument: " + str(sys.argv[1]))
        proxy.initialise_object(str(sys.argv[1])) # API client may not have to enter this.
        network_instance = Pyro4.Proxy("PYRONAME:example.embedded.server.network." + str(sys.argv[1]) + "@10.6.31.75:9090")
        print("network_instance: " + str(network_instance))
        network_instance.build([{'num_nodes' : 2, 'learning_rate' : 3.0,
        	'weight_penalty' : 0.001, 'weight_init_range' : [-1.0,+1.0]},
        	{'num_nodes' : 10, 'learning_rate' : 3.0,
        	'weight_penalty' : 0.001, 'weight_init_range' : [-1.0,+1.0]},
        	{'num_nodes' : 1}])
        # @param parameters: tuple of one or more equal-length lists
        # @param targets: tuple of one of more equal-length lists, which
        # must contain the same number of lists as 'parameters'.
        # @param num_iterations: An integer specifiying the num_iterations
        # that should occur.
        # @param batch_size: Temporarily, this defaults to defining the number of
        # partitions.
        print(network_instance.train(([0.4,0.3,1.0],[1.0,0.9,1.0],[0.3,0.2,1.0],
            [0.0,0.3,1.0],[0.5,0.6,1.0],[0.7,0.3,1.0],[0.4,0.23,1.0],[1.0,0.5,1.0]),
            ([0.0],[1.0],[0.0],[0.0],[1.0],[1.0],[0.0],[1.0]), 100, 3))
        print(network_instance.predict(([0.0,0.2,1.0],[1.0,0.8,1.0],[0.4,0.5,1.0])))
        print(network_instance.add_layer({'num_nodes' : 4, 'learning_rate' : 4.0,
            'weight_penalty' : 0.0001, 'weight_init_range' : [-1.0,+1.0]}))
        print(network_instance.train(([0.4,0.3,1.0],[1.0,0.9,1.0],[0.3,0.2,1.0],
                                          [0.0,0.3,1.0],[0.5,0.6,1.0],[0.7,0.3,1.0],[0.4,0.23,1.0],[1.0,0.5,1.0]),
                                         ([0.0],[1.0],[0.0],[0.0],[1.0],[1.0],[0.0],[1.0]), 100, 3))
        print(network_instance.predict(([0.0,0.2,1.0],[1.0,0.8,1.0],[0.4,0.5,1.0])))
