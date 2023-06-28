#! /usr/bin/env python3


import argparse
import logging
import time

import psutil


class LoggerConstructor:
    def __init__(self, logger_name, file_name, log_level=logging.INFO, mode="w"):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        self.__fh = logging.FileHandler(filename=file_name, mode=mode)
        self.__formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        self.__fh.setFormatter(self.__formatter)
        self.__logger.addHandler(self.__fh)

    def get_logger(self):
        return self.__logger


def main(args):
    id = args.id
    nic = args.nic
    logger = LoggerConstructor("logger", f"./netif_{id}.log").get_logger()

    recv0 = psutil.net_io_counters(pernic=True)[nic].bytes_recv
    sent0 = psutil.net_io_counters(pernic=True)[nic].bytes_sent
    time0 = time.time()
    while True:
        time.sleep(0.01)
        recv1 = psutil.net_io_counters(pernic=True)[nic].bytes_recv
        sent1 = psutil.net_io_counters(pernic=True)[nic].bytes_sent
        time1 = time.time()
        time_diff = time1 - time0
        bw_in = (recv1 - recv0) / time_diff / 1048576
        bw_out = (sent1 - sent0) / time_diff / 1048576
        logger.info(f"{bw_in} {bw_out}")
        recv0 = recv1
        sent0 = sent1
        time0 = time1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A NET Interface Monitor")
    parser.add_argument("-i", "--id", type=int, default=0)
    parser.add_argument("-n", "--nic", type=str, default="ens3")

    args = parser.parse_args()

    main(args)
