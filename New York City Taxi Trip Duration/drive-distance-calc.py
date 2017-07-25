# Calculate drive distance using OpenStreetMap and NetworkX
# data processing
# Credit to:
# https://www.kaggle.com/ankasor/driving-distance-using-open-street-maps-data/notebook
# https://www.toptal.com/python/beginners-guide-to-concurrency-and-parallelism-in-python
import pandas as pd
import numpy as np

# System
import datetime as dtime
from datetime import datetime
import sys
from inspect import getsourcefile
import os.path
import re
import time
import logging
from queue import Queue
from threading import Thread

# Other
from geographiclib.geodesic import Geodesic
import osmnx as ox
import networkx as nx

DATA_DIR = "data-temp"
STREETGRAPH_FILENAME = 'streetnetwork.graphml'
LOCATION = 'New York, USA'
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
NUM_THREAD = 3
BLOCK_SIZE = 10


class DriveDistance(Thread):
    def __init__(self, thread_count=0, in_queue=None, out_queue=None, area_graph=None, combine_data=None):
        Thread.__init__(self)
        self.thread_count = thread_count
        self.graph_filename = DATA_DIR + "/" + STREETGRAPH_FILENAME
        self.train_distance_filename = DATA_DIR + "/train_distance.csv"
        self.eval_distance_filename = DATA_DIR + "/test_distance.csv"
        self.distance_filename = DATA_DIR + "/distance.csv"
        self.geod = Geodesic.WGS84  # define the WGS84 ellipsoid
        if in_queue is not None:
            self.in_queue = in_queue
        if out_queue is not None:
            self.out_queue = out_queue
        if area_graph is not None:
            self.area_graph = area_graph
        if combine_data is not None:
            self.combine_data = combine_data

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            n_block = self.in_queue.get()
            logger.info(str(self.thread_count) +
                        " got block number:" + str(n_block))
            self.calc_drive_distance(n_block)
            self.in_queue.task_done()

    def get_area_graph(self):
        return self.area_graph

    def get_combine_data(self):
        return self.combine_data

    def init_osm_graph(self):
        if not os.path.isfile(self.graph_filename):
            # There are many different ways to create the Network Graph. See
            # the osmnx documentation for details
            logger.info("Downloading graph for " + LOCATION)
            self.area_graph = ox.graph_from_place(
                LOCATION, network_type='drive_service')
            ox.save_graphml(
                self.area_graph, filename=STREETGRAPH_FILENAME, folder=DATA_DIR)
            logger.info("Graph saved to " + self.graph_filename)
        else:
            logger.info("Loading graph from " + self.graph_filename)
            self.area_graph = ox.load_graphml(
                STREETGRAPH_FILENAME, folder=DATA_DIR)

    def load_data(self):
        logger.info("Loading data from " + DATA_DIR)
        train_data = pd.read_csv(DATA_DIR + "/train.csv")
        eval_data = pd.read_csv(DATA_DIR + "/test.csv")
        features = eval_data.columns.values
        train_data = train_data[features]
        self.combine_data = pd.concat(
            [train_data[features], eval_data])
        # self.combine_data = self.combine_data[:11]

    def point_distance(self, startpoint, endpoint):
        distance = geod.Inverse(
            startpoint[0], startpoint[1], endpoint[0], endpoint[1])
        return distance['s12']

    def driving_distance(self, area_graph, startpoint, endpoint):
        """
        Calculates the driving distance along an osmnx street network between two coordinate-points.
        The Driving distance is calculated from the closest nodes to the coordinate points.
        This can lead to problems if the coordinates fall outside the area encompassed by the network.

        Arguments:
        area_graph -- An osmnx street network
        startpoint -- The Starting point as coordinate Tuple
        endpoint -- The Ending point as coordinate Tuple
        """

        # Find nodes closest to the specified Coordinates
        node_start = ox.utils.get_nearest_node(area_graph, startpoint)
        node_stop = ox.utils.get_nearest_node(area_graph, endpoint)
        # Calculate the shortest network distance between the nodes via the edges
        # "length" attribute
        try:
            distance = nx.shortest_path_length(
                self.area_graph, node_start, node_stop, weight="length")
        except:
            logger.error(str(self.thread_count) + " Can not calculate path from (" + str(startpoint[0]) +
                         "," + str(startpoint[0]) + ")" + " to (" +
                         str(endpoint[0]) + "," +
                         str(endpoint[1]) + "). Using fallback function")
            distance = self.point_distance(startpoint, endpoint)
        return distance

    def calc_drive_distance(self, n_block=0):
        n_start = n_block * BLOCK_SIZE
        n_end = n_start + BLOCK_SIZE
        logger.info(str(self.thread_count) + " processing from " +
                    str(n_start) + " to " + str(n_end))
        data = self.combine_data[n_start:n_end]
        # data_index = data.index.values
        # print(self.thread_count, len(data), data_index)
        start = time.time()
        distance = data.apply(lambda row: self.driving_distance(
            self.area_graph, (row['pickup_latitude'],
                              row['pickup_longitude']),
            (row['dropoff_latitude'], row['dropoff_longitude'])),
            axis=1)
        # print("data size:", len(data), " distance size:", len(distance))
        # data.loc[n_start:n_end, 'drive_distance'] = distance.values
        # self.combine_data.loc[n_start:n_end, 'drive_distance'] = distance.values
        column = 'drive_distance'
        distance_pd = pd.DataFrame(
            data={'id': data['id'], column: distance.values}, columns=['id', column])
        self.out_queue.put(distance_pd)
        end = time.time() - start
        logger.info(str(self.thread_count) + " processed time:" + str(end))

    def save_distance(self):
        logger.info(str(self.thread_count) + " Saving to" + DATA_DIR)
        column = 'drive_distance'
        distance = pd.DataFrame(columns=['id', column])
        while True:
            try:
                data = self.out_queue.get_nowait()
                distance = distance.append(data)
            except:
                logger.info(str(self.thread_count) +
                            " Total data:" + str(len(distance)))
                break

        distance.to_csv(
            self.distance_filename, index=False)
        logger.info(str(self.thread_count) + " data saved ")


# ---------------- Main -------------------------
if __name__ == "__main__":
    # create logger
    logger = logging.getLogger('kaggle')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(LOG_LEVEL)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(DATA_DIR + '/drive_distance.log', mode='a')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Create global queue
    in_queue = Queue()
    out_queue = Queue()

    # create base class
    base_class = DriveDistance(0, in_queue=in_queue, out_queue=out_queue)
    base_class.init_osm_graph()
    base_class.load_data()
    area_graph = base_class.get_area_graph()
    combine_data = base_class.get_combine_data()
    # base_class.calc_drive_distance_thread(0)
    # quit()

    # create worker
    for x in range(NUM_THREAD):
        worker = DriveDistance(x + 1, in_queue=in_queue, out_queue=out_queue,
                               area_graph=area_graph, combine_data=combine_data)
        # Setting daemon to True will let the main thread exit even though
        # the workers are blocking
        worker.daemon = True
        worker.start()

    ts = time.time()
    # Put the tasks into the queue as a tuple
    # data_len = len(combine_data)
    data_len = 55
    remain = data_len % BLOCK_SIZE
    total_blocks = data_len // BLOCK_SIZE
    if remain > 0:
        total_blocks = total_blocks + 1
    logger.info("Data size: " + str(data_len) +
                " total blocks:" + str(total_blocks))
    for i in range(total_blocks):
        in_queue.put(i)
    # Causes the main thread to wait for the queue to finish processing all
    # the tasks
    in_queue.join()
    logger.info('Took ' + str(time.time() - ts))
    base_class.save_distance()
