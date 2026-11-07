"""Change the training background while preserving the full labelled nodule and its rim."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import preference_gradients as training

PROTOCOL = Path(__file__).resolve().parents[1]/"docs/lidc_91/context_protocol.md"


def background_layers(image, masks, alpha):
    union = masks.astype(bool).any(axis=0)
    matte = alpha.astype(np.float32)/255
    support = union | (matte>0.05)
    distance = cv2.distanceTransform((~union).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    position = np.clip((distance-3)/9, 0, 1)
    editable = np.minimum(position*position*(3-2*position), 1-matte)
    editable[union] = 0
    removal = cv2.dilate(support.astype(np.uint8), np.ones((5,5),np.uint8))
    background = cv2.inpaint(image, removal, 3, cv2.INPAINT_TELEA)
    annulus = (distance>5) & (distance<22) & ~support
    if not annulus.any():
        annulus = ~support
    tissue = image[annulus] if annulus.any() else image.ravel()
    profile = np.quantile(tissue, (0.1,0.25,0.5,0.75,0.9)).astype(np.float32)
    return background, editable, profile


def matched_partners(profiles, patients):
    patients = np.asarray(patients)
    different = patients[:,None] != patients[None,:]
    count = min(16, int(different.sum(axis=1).min()))
    if count == 0:
        raise ValueError("Context augmentation needs two fitting patients")
    scale = np.maximum(np.quantile(profiles,0.75,axis=0)-np.quantile(profiles,0.25,axis=0),8)
    distances = np.square((profiles[:,None]-profiles[None,:])/scale).mean(axis=2)
    distances[~different] = np.inf
    return np.argsort(distances, axis=1, kind="stable")[:, :count]


class ContextMix:
    def __init__(self, image, masks, alpha, patients, seed):
        self.image = image
        layers = [background_layers(im, y, a) for im,y,a in zip(image[:,0].cpu().numpy(), masks, alpha)]
        backgrounds, editable, profiles = (np.stack(items) for items in zip(*layers))
        device = image.device
        self.backgrounds = torch.from_numpy(backgrounds[:,None]).to(device)
        self.editable = torch.from_numpy(editable[:,None]).to(device)
        self.profiles = torch.from_numpy(profiles).to(device)
        self.partners = torch.from_numpy(matched_partners(profiles,patients)).to(device)
        self.generator = torch.Generator(device=device).manual_seed(seed+82171)

    def __call__(self, indices, step):
        if step == 1:
            return self.image[indices]
        device = indices.device
        picked = torch.randint(self.partners.shape[1], (len(indices),), device=device, generator=self.generator)
        donors = self.partners[indices,picked]
        active = torch.rand(len(indices),device=device,generator=self.generator)<0.5
        strength = (0.35+0.45*torch.rand(len(indices),device=device,generator=self.generator))*active
        ref, other = self.profiles[indices], self.profiles[donors]
        scale = ((ref[:,3]-ref[:,1]).clamp_min(8)/(other[:,3]-other[:,1]).clamp_min(8)).clamp(0.7,1.4)
        matched = ((self.backgrounds[donors].float()-other[:,2,None,None,None])*scale[:,None,None,None]
                   + ref[:,2,None,None,None]).clamp(0,255)
        change = (matched-self.backgrounds[indices].float())*self.editable[indices]
        return (self.image[indices].float()+strength[:,None,None,None]*change).clamp(0,255)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("fit","check"))
    for name in ("run","data","align"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--texture",type=Path)
    parser.add_argument("--fold",type=int,choices=range(4),required=True)
    parser.add_argument("--variant",choices=("control","context"),required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    if args.phase=="fit":
        training.fit(args, ContextMix if args.variant=="context" else None, PROTOCOL)
    else:
        training.check(args)
