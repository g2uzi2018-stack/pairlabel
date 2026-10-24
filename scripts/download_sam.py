#!/usr/bin/env python3
import download_models as dl
# Official Meta checkpoint, pinned by Hugging Face commit and SHA256.
dl.BASE='https://hf-mirror.com/facebook/sam2.1-hiera-large/resolve/665f8e2ad61cf5f53d65644ff27c8ee525124610/'
dl.download('sam2.1_hiera_large.pt',898083611,'2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318')
