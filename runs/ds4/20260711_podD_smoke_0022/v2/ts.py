import sys, time
t0 = time.monotonic()
for line in sys.stdin:
    sys.stdout.write("%10.3f %s" % (time.monotonic()-t0, line))
    sys.stdout.flush()
