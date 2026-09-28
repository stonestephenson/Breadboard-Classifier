"""breadboard_normalizer, vendored from
https://github.com/hartleyblakey/breadboard-normalizer at 7cc71d0 (MIT, see LICENSE).

vendor/icp/ is ClayFlannigan/icp (Apache 2.0, see vendor/icp/LICENSE).

Local changes to normalizer.py, kept few so syncing with upstream stays a small
diff:

1. `import tensorflow` is optional. It only loads a corner classifier that
   normalize_image() never uses, and it has no wheels for this project's Python.
2. `from docaligner import DocAligner` is optional, and Normalizer.__init__ takes a
   `corner_model=` callable that replaces it. breadboard/rectify.py passes an
   equivalent that runs the same ONNX model on onnxruntime directly. DocAligner's
   support library pins an onnxruntime with no Python 3.14 build.
3. `from pycpd import ...` is optional. It is only used by the *_cpd registration
   methods; we use the default, icp_ransac.

Not vendored: notebooks, demo images, weights/corner_orientation.keras (unused),
vendor/icp/test.py and README.md. This directory is excluded from ruff and pyright
so it keeps upstream's style.
"""
