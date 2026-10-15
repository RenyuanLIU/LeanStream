# Dynamic I/O example

Build the CPU targets, then run from the project root:

```sh
python3 tests/run_io_example.py ./build/dynamic_io_example dynamic
```

The harness creates and removes a synthetic aligned file. The example caches group 0, begins the layer, submits priorities `[0,1,2]` followed immediately by `[0,1,3]`, freezes final groups `[1,3]`, waits diagnostically, and retires. Generation count 2 confirms both priority updates reached the manager. Reads already started remain non-preemptive; exact read counts depend on timing. Controlled runtime tests verify replacement of unstarted requests.

For an existing aligned file, use `./build/dynamic_io_example /path/to/weights.bin 8 4096`. This CPU-only example has no GPU consumer retaining slots. Normal GPU submission must not wait for the diagnostic I/O wait used here.
