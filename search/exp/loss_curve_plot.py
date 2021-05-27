import numpy as np
x = []
for line in open('/tmp/ts-out.VYz8ee'):
    try:
        x.append(float(line))
    except:
        continue

x = np.convolve(x, np.ones(39)/39, mode='valid')

for i, x in enumerate(x):
    if i % 5 == 0:
        print(f"({i*50/1000},{x/50:.4g})", end=' ')
