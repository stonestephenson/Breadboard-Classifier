import os

# onnxruntime (1.30, macOS) queues usage telemetry for Microsoft and uploads it
# from a background thread. At exit that thread can race the shutdown and abort
# the process ("recursive_mutex lock failed"). The switch only takes effect if
# set before onnxruntime is first imported, which happens in breadboard.rectify;
# disable_telemetry_events() after import does not stop it.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
