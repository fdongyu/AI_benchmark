# U-Net (`unet_v2`)

Three-level U-Net used as an architecture baseline. It maps the common 13
input maps to three normalized one-hour state-change maps. The retained setup
uses 16 base channels and no shallow-water-equation loss.

The exact network source is `src/vendor/baselines/unet_model.py`.
