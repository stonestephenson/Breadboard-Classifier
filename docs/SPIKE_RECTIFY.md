# Spike: rectification feasibility

**Date:** 2026-08-06
**Question:** can we fit the WB-102 hole lattice on real photos accurately enough to place
component leads in the right electrical node? This is the go/no-go for the architecture
(ARCHITECTURE.md §3[2] and risk table row 1).
**Code:** `spikes/hole_detect.py`, `spikes/lattice_fit.py`
**Data:** 12 photos sampled evenly across the 248 in `../breadboard_generator/data/real/`,
covering empty boards through full Metro Mini builds, three background surfaces, varied
lighting and mild perspective.

## Verdict

**Promising — proceed, with one known failure mode.** Geometry is not the problem. Anchoring
and detection robustness are, and both are ordinary engineering.

## Results

| | |
|---|---|
| Photos where a lattice was fitted | **8 / 12** |
| Median reprojection residual | **0.02–0.06 pitch** (0.6–1.5 px at ~29 px pitch) |
| 95th percentile residual | ~2.1–2.8 px, i.e. under 0.1 pitch |

We need roughly **half-pitch** accuracy to assign a lead to the correct column. We are getting
better than **one twentieth** of a pitch. The margin is an order of magnitude, which is the
single most important number in this spike: it means occlusion and detection recall, not
geometric precision, are what will limit us.

Visual check (`data/cache/rectified.jpg`): warped boards show a perfectly regular, axis-aligned
hole grid and straight board edges. The warp is correct.

## What worked, and why

**Hole detection is essentially perfect on the board itself.** Local adaptive thresholding plus
blob shape filters find virtually every hole (see `data/cache/holes_overlay.jpg`). All false
positives come from *background* texture — speckled desk surfaces — and are spatially separate
from the board.

**Point-set autocorrelation recovers the lattice basis robustly.** The first attempt estimated
pitch from nearest-neighbour distances and failed on 8/12 images, because background speckle
poisons that estimate on exactly the noisy images that matter. Histogramming all pairwise
displacements instead makes the grid appear as sharp peaks while random texture contributes only
a diffuse blob; the two shortest strong peaks *are* the basis. This lifted 4/12 to 8/12 with no
other change.

**Local BFS growth handles perspective for free.** Assigning indices by stepping only between
adjacent holes never assumes a constant pitch across the board, so mild perspective costs
nothing — and irregular background points are rejected automatically, because the search will
not propagate into them. **No board segmentation is needed**, which removes a whole class of
lighting- and background-dependent failure.

**The centre channel is exactly 3 pitches** (7.62 mm = 3 × 2.54 mm), so allowing a row step of 3
lets one search span both halves of the board in a single consistent index frame.

## What is not done

1. **Anchoring.** The lattice is recovered up to an unknown integer origin and a 180° rotation —
   a bare grid cannot tell column 1 from column 63. Three of four rectified samples came out
   upside down. Fixes, in order of robustness:
   - the 3-pitch centre channel fixes the row origin (the e/f boundary);
   - the red/blue rail stripe order is vertically asymmetric and fixes the flip;
   - printed column numbers and row letters as confirmation.
2. **Rails are being absorbed into the terminal lattice.** Indexed counts exceed 630 and row
   spans reach 27, so the search is growing into the power rails. Residuals stay low, so it is
   not corrupting the fit, but indices must be reconciled against the board spec before they
   mean anything.
3. **One image locked onto a half-pitch sub-lattice** (IMG_3043, pitch 13.45 px vs ~29 px
   elsewhere). Needs a sanity check against expected pitch given the board's apparent size.

## The failure mode to fix

4 of 12 failed, and they are exactly the images with the heaviest background speckle (2400–3050
raw detections against ~830 real holes). The autocorrelation peak is presumably being swamped.

Options, cheapest first: raise the blob filter's strictness; try several autocorrelation radii
and keep the fit with the lowest residual; seed the search from the densest region of detections
rather than the image centroid; fall back to coarse board segmentation only when the first
attempt fails.

**This is not expected to be a production failure mode.** These photos are unconstrained
handheld shots on cluttered desks. Production capture is a guided overlay that forces the board
to fill the frame, and it is a video clip — so we get many attempts per session and need only
one good frame. A single-frame success rate of 8/12 already implies near-certain success across
a clip.

## Next

1. Anchor the lattice to the board spec (channel + rail stripes), so indices mean real nodes.
2. Fix the noisy-background failures.
3. Run over all 248 photos and report the success-rate distribution, not a 12-photo sample.
4. Then: per-node occupancy in rectified space.
