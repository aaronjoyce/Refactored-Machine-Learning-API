Dependencies:
1. NumPy;
2. Pyro4;
3. Apache Spark. 

To start the server, having downloaded the latest distribution of Apache Spark, within the 'Spark' directory, input the following to the terminal:
'./bin/pyspark ~/<path to network directory>/Refactored-Machine-Learning-API/network/server.py', which will start up the server. 

To run the application, set the client.py's 'ADDRESS' to the host's address, followed by the port number, and run the client.py file as per normal, with one command line argument - the client's id (this can be arbitrary). 
