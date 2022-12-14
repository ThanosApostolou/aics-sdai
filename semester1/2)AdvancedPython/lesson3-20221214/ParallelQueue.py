# Import Required Python Modules
from time import sleep
import random
from multiprocessing import Process
from multiprocessing import Queue
from typing import TypedDict, Dict

StatusType = Dict[int, str]


# Define the function which will facilitate the execution code for
# both generator and consumer processes
# p [p1, p2, ..., pn], id
def process(queue: Queue, idx: int, max_value: int, sentinel_value: int, STATUS: StatusType, status_idx: int):
    # Initialize an internal list that willl save the individual values generated or
    # consumed by each process
    F: list[int] = []
    print("Process {0} has just been created as a {1} process".format(idx, STATUS[status_idx]), flush=True)
    while True:
        if status_idx == 1:
            # Sample the new random value and sleep for the specified amount of time
            rand_value = random.randint(0, max_value)
            print("{0} process {1} has just generated {2} \n".format(STATUS[status_idx], idx, rand_value), flush=True)
            sleep(1)
            queue.put(rand_value)
            if rand_value == sentinel_value:
                print("{0} process {1} has just terminated".format(STATUS[status_idx], idx), flush=True)
                break
        else:
            rand_value = queue.get()
            print("{0} process {1} has just consumed {2}".format(STATUS[status_idx], idx, rand_value), flush=True)
            F.append(rand_value)
            if rand_value == sentinel_value:
                # print("{0} process {1} has just terminated with F={2}".format(STATUS[status_idx], idx, F), flush=True)
                break


def main():
    # Set the status dictionary
    STATUS: dict[int, str] = {
        0: "Consumer",
        1: "Generator"
    }
    # Set the number and the various types of the processs to be created
    STATUS_IDS: list[int] = [0, 0, 0, 1, 1, 1]
    # Set the max_value parameters
    max_value = 5
    # Set the sentinel value
    sentinel_value = 0
    # Set the number of process
    process_num = len(STATUS_IDS)
    # Create the shared queue object
    queue = Queue()
    # Create the list of processes
    processes = []
    # Create all process
    for proc_idx in range(process_num):
        rand_process = Process(target=process,
                               args=(queue, proc_idx, max_value, sentinel_value, STATUS, STATUS_IDS[proc_idx]))
        processes.append(rand_process)
    # Start all process
    for proc in processes:
        proc.start()
    # Ensure that the main process will terminate after the completion
    # of the other processess
    for proc in processes:
        proc.join()


# Main Program
if __name__ == '__main__':
    main()
