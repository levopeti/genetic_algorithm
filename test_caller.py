from test_base import increment_counter, logger
from multiprocessing.pool import Pool

p = Pool(18)
p.map(increment_counter, [_ for _ in range(18)])
