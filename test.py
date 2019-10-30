#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import time
from multiprocessing import Pool, Manager


def work(queue):

	while True:
		el = queue.get()

		if el is None:
			break

		print("processing {}".format(el))
		time.sleep(1)
	print("Worker ready")

def produce(queue, count):
	for i in range(count):
		print("putting {}".format(i))
		queue.put(i)

	queue.put(None)
	print("Producer ready")

def main():

	with Pool(2) as pool, Manager() as m:

		queue = m.Queue(maxsize=3)

		worker_result = pool.apply_async(work, args=(queue,))
		producer_result = pool.apply_async(produce, args=(queue, 10))

		producer_result.wait()
		worker_result.wait()


	print("Jobs ready!")


main()
