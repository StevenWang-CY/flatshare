"""E2 (``flatshare-bench e2``): voice-to-voice latency beside agent load, one arm file per point.

- :mod:`.arm`: the arm file, which says everything about a point;
- :mod:`.run`: measuring a point on cyan, into its run directory (``flatshare-bench e2 run``);
- :mod:`.engine`: the engine a point runs on, checked against its arm file and the GPU node's profile;
- :mod:`.cyan`: cyan's settings and units, configured for a point (``flatshare-bench e2 reset`` undoes it);
- :mod:`.tools`: the tools a point runs, each in a process group of its own, and the run's waits;
- :mod:`.record`: a run directory's summary and its compact record (``flatshare-bench e2 export``);
- :mod:`.report`: the frontier across points (``flatshare-bench e2 report``);
- :mod:`.figures`: the report's figures.
"""
