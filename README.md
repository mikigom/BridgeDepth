# BridgeDepth

Official implementation of paper:

[**BridgeDepth: Bridging Monocular and Stereo Reasoning with Latent Alignment**](https://www.arxiv.org/abs/2508.04611), **ICCV 2025**<br/>
Tongfan Guan, Jiaxin Guo, Chen Wang, Yun-Hui Liu<br/>

# Abstract
Monocular and stereo depth estimation offer complementary strengths: monocular methods capture rich contextual priors but lack geometric precision, while stereo approaches leverage epipolar geometry yet struggle with ambiguities such as reflective or textureless surfaces. Despite post-hoc synergies, these paradigms remain largely disjoint in practice. We introduce a unified framework that bridges both through iterative bidirectional alignment of their latent representations. At its core, a novel cross-attentive alignment mechanism dynamically synchronizes monocular contextual cues with stereo hypothesis representations during stereo reasoning. This mutual alignment resolves stereo ambiguities (e.g., specular surfaces) by injecting monocular structure priors while refining monocular depth with stereo geometry within a single network. Extensive experiments demonstrate state-of-the-art results: **it reduces zero-shot generalization error by `>40%` on Middlebury and ETH3D**, while addressing longstanding failures on transparent and reflective surfaces. By harmonizing multi-view geometry with monocular context, our approach enables robust 3D perception that transcends modality-specific limitations.

![Example of reconstructions](assets/overview.png)

**TLDR**: Why choose between blurry monocular guesses and stereo confusion when you have both? This AI bridges mono and stereo depth to finally take it out --- like therapy, but for pixels. Now solves reflections like a champ.

## Get Started

### Installation

1. Clone BridgeDepth
```bash
git clone https://github.com/aeolusguan/BridgeDepth
cd BridgeDepth
```

2. Create the environment, here we recommend using conda.
```bash
conda create -n bridgedepth python=3.10
conda activate bridgedepth
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126  # use the correct version of cuda for your system
pip install -r requirement.txt
# Optional, but recommend (~30% faster)
pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu126  # use the correct version of cuda for your system
```

### Checkpoints

We provide several pre-trained models:


| Model name | Benchmark | Training resolutions | Stereo encoder | Training Config |
|------------|-----------|----------------------|----------------|-----------------|
| [`sf.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_sf.pth) | Scene Flow | 368x784 | BasicEncoder | [`default.py`](bridgedepth/config/default.py) |
| [`l_sf.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_l_sf.pth) | Scene Flow | 368x784 | ConvNext-Tiny | [`l_train.yaml`](configs/L_train.yaml) |
| [`kitti.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_kitti.pth) | KITTI 2012/2015 | 304x1152 |ConvNext-Tiny | [`kitti_mix_train.yaml`](configs/kitti_mix_train.yaml) |
|[`eth3d_pretrain.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_eth3d_pretrain.pth),  [`eth3d.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_eth3d.pth)  | ETH3D | 384x512 | ConvNext-Tiny | [`eth3d_pretrain.yaml`](configs/eth3d_pretrain.yaml), [`eth3d.yaml`](configs/eth3d.yaml) |
| [`middlebury_pretrain.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_middlebury_pretrain.pth), [`middlebury.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_middlebury.pth) | Middlebury | 384x512, 512x768 | ConvNext-Tiny | [`middlebury_pretrain.yaml`](configs/middlebury_pretrain.yaml), [`middlebury.yaml`](configs/middlebury.yaml) |
| [`rvc_pretrain.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_rvc_pretrain.pth), [`rvc.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_rvc.pth) | Robust Vision Challenge | 384x768, 384x768 | ConvNext-Tiny | [`rvc_pretrain.yaml`](configs/rvc_pretrain.yaml), [`rvc.yaml`](configs/rvc.yaml) |


### Run demo
```bash
python demo.py --model_name rvc  # also try with [rvc_pretrain | eth3d_pretrain | middlebury_pretrain]
```

Tips:
- For in the wild deployment, we generally recommend the [`bridge_rvc_pretrain.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_rvc_pretrain.pth) checkpoint. You are encouraged to also try other models for your best fit ([`bridge_middlebury_pretrain.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_middlebury_pretrain.pth), [`bridge_eth3d_pretrain.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_eth3d_pretrain.pth), or [`bridge_rvc.pth`](https://huggingface.co/aeolusguan/BridgeDepth/resolve/main/bridge_rvc.pth) maybe your favorite).
- For high-resolution image (>720p), you are highly suggested to run with smaller scale, e.g., **downsampled to 720p**, not only for faster inference but also better performance.