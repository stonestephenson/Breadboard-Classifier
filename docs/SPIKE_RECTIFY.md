# Rectification: the rectifier in use

**Question:** can we fit the WB-102 hole lattice on real photos accurately enough to place
component leads in the right electrical node? This was the go/no-go for the architecture
(ARCHITECTURE.md §3[2] and risk table row 1). The answer is yes.

**The rectifier** is breadboard-normalizer (github.com/hartleyblakey/breadboard-normalizer),
vendored in `breadboard/_vendor/breadboard_normalizer/` and wrapped by `breadboard/rectify.py`.
Try it: `./venv/bin/python -m breadboard rectify PHOTO -o out/`.

**Test photos:** 13. Twelve were sampled evenly across the 248 in
`../breadboard_generator/data/real/`, covering empty boards through full Metro Mini builds, three
background surfaces, varied lighting and mild perspective. The thirteenth is the assembled
BasicBoard on a cluttered desk (keyboard, carpet, laptop in frame, bright LED glow).

This file began on 2026-08-06 as the write-up of our own spike (`spikes/hole_detect.py`,
`spikes/lattice_fit.py`), which the normaliser replaced on 2026-09-27. The spike's full text was
removed on 2026-10-05 and is in git history. What it taught is at the end. The spike's code stays
as a record; nothing depends on it.

## How it works

It goes coarse to fine instead of finding the grid from nothing.

1. A pretrained document-corner model (DocAligner, used off the shelf) finds the board's four
   corners. That alone fixes scale and position, and it removes the half-pitch and diagonal
   grids the spike kept choosing.
2. Pinholes detected in the rough warp are snapped onto a hole template with ICP and RANSAC.
   An explicit off-by-one column search then corrects the corner model's error.
3. The red and blue rail stripes decide which end is column 1.

## Results

| | our spike | breadboard-normalizer |
|---|---|---|
| 12 sample photos | 8 fitted, some on the wrong grid or upside down | **12**, all the right way round |
| BasicBoard desk photo | failed (20 columns for 63) | **fitted** |
| Time per photo | seconds | ~0.2 s |

Judged by eye, because a low residual does not prove the grid is right (see "What the spike
taught"). The printed column numbers line up with the fitted grid on every photo, with column 1
at the same end. The BasicBoard photo's four corner holes are pinned in `tests/test_rectify.py`
at positions checked against the printed letters and numbers.

Live webcam frames work too (`tools/live.py`). **Not yet tested:** the full 248-photo set.

## Two checks on top of the normaliser's own grade

A cold review found that the grade alone does not guarantee a usable fit:

- **Orientation.** When the rail stripes cannot be read, the normaliser assumes the board is
  the right way round. Greyscale copies of all 13 photos still graded perfect, and 7 of them
  came out upside down. `Rectification.oriented` now re-reads the stripes on the final view.
- **Column registration.** The hole grid repeats, so a fit shifted by one column still lands
  most holes on template holes, and still grades perfect. Only the rails' gaps and the grid's
  ends give it away. `Rectification.column_margin` compares the fit against the same fit
  shifted by one or two columns. On the 13 photos the true fit won by 0.038–0.062 of detected
  holes. `ok` requires at least 0.02 (`MIN_COLUMN_MARGIN`).

`Rectification.ok` requires all three: the grade, the orientation and the column. When it is
False, the result must not be read.

## Found along the way

`board.hole_position` had the power rails in the wrong place. They started over column 1 instead
of column 3, and the top rail sat 1.8 pitches too close to row a. The template's terminal holes
agree with our geometry to 0.02 pitch, which vouches for its rail positions too. The fix is
tested to keep the two in agreement. The generator's `board_spec.json` gives the rail offset as
4.0 mm; the template gives 7.2 mm.

## What the spike taught

- **Geometry is not the limit.** Where the spike's fit locked onto the right lattice it was
  accurate to 0.02–0.06 of a hole pitch, against a requirement of half a pitch. So occlusion
  and how many holes are detected are what limit a rectifier, not geometric precision.
- **A low residual does not mean the lattice is right.** The diagonals of a square lattice form
  another square lattice at sqrt(2) the spacing. It fits a homography perfectly while indexing
  every other hole, so residuals look excellent and the pitch is silently wrong by 41%. A fit
  must be checked against the board's known size and layout, never by residual alone. The two
  checks above are the same lesson met again.
- **A bare grid cannot tell column 1 from column 63.** The spike recovered the lattice only up
  to an unknown origin and a 180° turn, and three of four rectified samples came out upside
  down. The features that settle it are the ones molded and printed into the board: the 3-pitch
  centre channel (7.62 mm = 3 × 2.54 mm) and the order of the red and blue rail stripes.
- **Finding the grid from nothing fails on clutter.** The four sample photos the spike could
  not fit were the ones with the heaviest background speckle (2400–3050 detections against
  about 830 real holes). The normaliser finds the board's outline first, and fits the holes
  only inside it.
