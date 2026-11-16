# Vendored baseline model definitions

These files are exact copies of the model definitions used by the completed
U-Net and FNO campaigns. They are vendored so reproduction does not depend on
the sibling `AI_foundation/Perlmutter_transfer` source tree.

| Local file | Original path | Original SHA-256 |
|---|---|---|
| `unet_model.py` | `AI_foundation/Perlmutter_transfer/code/AI_foundation/UNet/model.py` | `b6761003de6a3efe447f4e9d60440ab49a85d9aec11d1d2421fb2118cdce8919` |
| `fno_model.py` | `AI_foundation/Perlmutter_transfer/code/AI_foundation/FNO/model.py` | `139a964f43023c5f4b356b9978f50e4d6f0755ee6e7133ad2eafe5a4c108a686` |

`fno_model.py` intentionally imports `neuralop`; that package remains an
environment dependency rather than vendored code.
