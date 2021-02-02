import numpy as np
from utils import groupby, car, cadr, cdr, info, load, parse_input, get_input_size

records = load("records")

for record in records:
    groups = record["op_groups"]
    numbers = [len(group) for group in groups]
    info(sorted(numbers))

