# Project Scope

## Objective

Evaluate whether ESD (Embedded Seamless Data) can improve landfill prediction when combined with the current annotated remote-sensing dataset.

## What We Already Have

- annotated landfill segmentation masks
- matching remote-sensing image tiles
- a SAM + LoRA segmentation codebase adapted for landfill experiments
- an ESD download helper and tile list

## What We Need To Answer

1. Can the current image dataset produce a trustworthy landfill segmentation baseline?
2. Does ESD contain enough discriminative semantic information for landfill detection?
3. If yes, is ESD better used as:
   - a classifier input
   - a coarse localization prior
   - a fused feature source for the segmentation model

## Non-Goals

- claiming ESD alone is sufficient for high-resolution, precise landfill boundaries before validation
- uploading raw data or checkpoints into the repository
- deeply refactoring all historical experiment code before a baseline is established

## Recommended Experimental Order

### Phase 1: Image-Only Baseline

- use the current labeled image dataset
- fix data loading and path issues
- report `mIoU`, landfill IoU, precision, recall, F1

### Phase 2: ESD-Only Feasibility

- align each sample with ESD tile and year
- extract ESD vectors
- start with binary classification or coarse grid prediction

### Phase 3: Image + ESD Fusion

- use ESD as a semantic prior
- compare against the image-only baseline

## Working Hypothesis

- `image-only` should be the strongest baseline for precise segmentation
- `ESD-only` is more likely to help coarse detection than boundary-level segmentation
- `image + ESD` is the most realistic path to a meaningful gain

## Immediate Deliverables

1. clean project repository
2. reproducible data sanity check
3. documented dataset inventory
4. baseline training entrypoint ready for local configuration
