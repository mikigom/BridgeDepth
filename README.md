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


| Modelname | Benchmark | Training resolutions | Stereo encoder | Training Config |
|-----------|-----------|----------------------|----------------|---------------|
| [`bridge_sf.pth`](https://drive.google.com/file/d/17SdccWLVvCBJQAWcsmFfrwjn6HAV4Z6F/view?usp=sharing) | Scene Flow | 368x784 | BasicEncoder | [`default.py`](bridgedepth/config/default.py) |
| [`bridge_l_sf.pth`](https://drive.google.com/file/d/1TLiDY6PPq-WSK1fXYUHQofkM_UyhsyzO/view?usp=sharing) | Scene Flow | 368x784 | ConvNext-Tiny | [`l_train.yaml`](configs/L_train.yaml) |
| [`bridge_kitti.pth`](https://drive.google.com/file/d/1DaKQc-6jOwKr_vgSeXytCJnaWyrAidn7/view?usp=sharing) | KITTI 2012/2015 | 304x1152 |ConvNext-Tiny | [`kitti_mix_train.yaml`](configs/kitti_mix_train.yaml) |
|[`bridge_eth3d_pretrain.pth`](https://drive.google.com/file/d/1MImpyN27Q19zRizy7QlcgXV_ji5CKu2n/view?usp=sharing),  [`bridge_eth3d.pth`](https://drive.google.com/file/d/1wr3hBRvD_iEp1U83vP_yxEqdWUMSP1Om/view?usp=sharing)  | ETH3D | 384x512 | ConvNext-Tiny | [`eth3d_stage1.yaml`](configs/eth3d_stage1.yaml), [`eth3d_stage2.yaml`](configs/eth3d_stage2.yaml) |
| [`bridge_middlebury_pretrain.pth`](https://drive.google.com/file/d/1Ay2G-RG6b48iO3X5os4I7Kv2Mr8pqgkn/view?usp=sharing), [`bridge_middlebury.pth`](https://drive.google.com/file/d/1NExaTOSR7nKy47FOyGrMS7-iw_33K0_z/view?usp=sharing) | Middlebury | 384x512, 512x768 | ConvNext-Tiny | [`middlebury_stage1.yaml`](configs/middlebury_stage1.yaml), [`middlebury_stage2.yaml`](configs/middlebury_stage2.yaml) |
| [`bridge_rvc_pretrain.pth`](https://drive.google.com/file/d/1aSeVqq0YzpPwo4A2lRdTf2ly3mC1trYz/view?usp=sharing), [`bridge_rvc.pth`](https://drive.google.com/file/d/1bg6aw9nSyDikragSRaERmbYhgHUPszEN/view?usp=sharing) | Robust Vision Challenge | 384x768, 384x768 | ConvNext-Tiny | [`rvc_stage1.yaml`](configs/rvc_stage1.yaml), [`rvc_stage2.yaml`](configs/rvc_stage2.yaml) |

Tips:
- For in the wild deployment, we generally recommend the [`bridge_rvc_pretrain.pth`](https://drive.google.com/file/d/1aSeVqq0YzpPwo4A2lRdTf2ly3mC1trYz/view?usp=sharing) checkpoint. You are encouraged to also try other models for your best fit ([`bridge_middlebury_pretrain.pth`](https://drive.google.com/file/d/1Ay2G-RG6b48iO3X5os4I7Kv2Mr8pqgkn/view?usp=sharing), [`bridge_eth3d_pretrain.pth`](https://drive.google.com/file/d/1MImpyN27Q19zRizy7QlcgXV_ji5CKu2n/view?usp=sharing), or [`bridge_rvc.pth`](https://drive.google.com/file/d/1bg6aw9nSyDikragSRaERmbYhgHUPszEN/view?usp=sharing) maybe your favorite).
- For high-resolution image (>720p), you are highly suggested to run with smaller scale, e.g., **downsampled to 720p**, not only for faster inference but also better performance.

### Run demo
Coming Soon !