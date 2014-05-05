import multiprocessing
import time


def run_pool(workers, n_jobs=12, sleep=0.1):
    # defensive copy
    workers = workers[:]
    if n_jobs < 1:
        n_jobs = multiprocessing.cpu_count()
    processes = []
    p = None
    try:
        while True:
            active = multiprocessing.active_children()
            while len(active) < n_jobs and len(workers) > 0:
                p = workers.pop(0)
                p.start()
                processes.append(p)
                active = multiprocessing.active_children()
            if len(workers) == 0 and len(active) == 0:
                break
            time.sleep(sleep)
    except KeyboardInterrupt, SystemExit:
        if p is not None:
            p.terminate()
        for p in processes:
            p.terminate()
        raise
