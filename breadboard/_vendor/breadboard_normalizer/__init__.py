"""breadboard_normalizer, vendored.

Author: Hartley Blakey, a member of this project, who wrote it as Stage 2 of a
CS470 project on the same problem.
Upstream: https://github.com/hartleyblakey/breadboard-normalizer
Commit: 34ea9b774cb189fcac876b70cdb05d7cc796e588 (2026-08-14)
License: the upstream repository does not yet have a license file. It is included
here with the author's permission, and a license file is to be added upstream.
Update this note when it is.

vendor/icp/ is Clay Flannigan's ICP (https://github.com/ClayFlannigan/icp), Apache
2.0, license in vendor/icp/LICENSE. Hartley's edit lets it take point sets of
different lengths.

What it does: it finds the board's four corners with a pretrained document-corner
model (DocAligner) and warps roughly to 1024 x 340. It then detects pinholes and
snaps them onto a hole template, using ICP, RANSAC homography and an off-by-one
column search. Finally, if the red and blue rail stripes are the wrong way round,
it rotates the result 180 degrees. breadboard/rectify.py wraps it with our hole
names and extra self-checks.

Local changes to normalizer.py, kept few so syncing with upstream stays a small
diff:

1. `import tensorflow` is optional. TF only loads a corner classifier upstream
   calls non-functional, which normalize_image() never uses, and it has no wheels
   for this project's Python.
2. `from docaligner import DocAligner` is optional, and Normalizer.__init__ takes a
   `corner_model=` callable that replaces it. We pass an equivalent that runs the
   same ONNX model on onnxruntime directly. DocAligner's support library pins an
   onnxruntime with no Python 3.14 build. The corners match exactly (max
   difference 0.0000 px on 13 photos, 2026-09-27).
3. `from pycpd import ...` is optional. It is only used by the *_cpd registration
   methods, and we use the default icp_ransac.

Not vendored: notebooks, demo images, weights/corner_orientation.keras (unused),
vendor/icp/test.py and README.md.

This directory is excluded from ruff and pyright on purpose, to keep upstream's
style.
"""
