NEW MODEL PLACEHOLDER

This folder is a starting point for adding a newly published flood emulator.
It is not an active benchmark model and must not be added to a publication
experiment without review.

Recommended steps:

1. Copy this folder and give the copy a short, descriptive model ID.
2. Put the exact published source in the copied folder (for example under
   upstream/) or pin the package dependency, then connect it in connector.py.
3. Record the paper, code URL, release or commit, license, pretrained
   checkpoint checksum, and implementation files returned by source_files().
4. Keep the required input/output contract: 13 input maps and three one-hour
   state-change maps, unless you intentionally define a new training recipe.
5. Run the connector shape/gradient check and an actual-grid GPU smoke test.
6. Add the model to a new versioned experiment; never edit a completed one.
