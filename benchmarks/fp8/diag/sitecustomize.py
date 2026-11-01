"""The FP8 investigation's servers only: SIGUSR1 makes a process dump every thread's stack to stderr, so a stalled
engine can say where it waits without ptrace."""

import faulthandler
import signal

faulthandler.register(signal.SIGUSR1, all_threads=True)
