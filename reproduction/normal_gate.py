"""Engineering smoke and held-inner-patient acceptance of a fitted boundary refiner."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from datasets.lidc import BaseDataset
from reproduction import boundary_eval as v1
from reproduction import normal_refiner as normal
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote


@torch.inference_mode()
def gate(data_path, root, v3_root, fold, smoke=False):
    mode = "smoke" if smoke else "inner"
    folder = root/(f"smoke-{fold}" if smoke else f"fold-{fold}-inner")
    fit = v1.read(folder/"fit.json")
    if fit["checkpoint"]!=rc.hash_file(folder/"final.pt") or fit["mode"]!=mode or fit["source"]!=rc.hash_file(normal.__file__):
        raise ValueError("Fitted refiner provenance differs")
    if rc.hash_file(data_path)!=rc.DATA_HASH:
        raise ValueError("LIDC identity")
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    data = BaseDataset(str(data_path))
    _,train,valid = rc.split_for(data,fold)
    if smoke:
        valid = valid[:4]
    models = []
    for view in vote.VIEWS:
        d = v3_root/f"fold-{fold}-{view}-inner"
        plan,result = v1.read(d/"plan.json"),v1.read(d/"result.json")
        if plan["train_indices"]!=train or rc.hash_file(d/"final.pt")!=result["checkpoint"]:
            raise ValueError("Inner vote guidance provenance")
        models.append(vote.load_net(d/"final.pt"))
    images = np.stack([data.images[i] for i in valid])[:,None]
    guides = []
    for image in torch.from_numpy(images).cuda().split(12):
        x = image.float()/255
        votes = vote.combine_views(vote.single_view(models[0],x,"fullfield"),vote.single_view(models[1],x,"zoom"))[:,8:]
        guides.append(normal.image_guide(votes.cpu().numpy()))
    logits = normal.predict(normal.load_model(folder/"final.pt"),images,np.concatenate(guides))
    truth = np.array([data.mask_labels[i] for i in valid],bool)
    pred = logits[:,:4]>0
    den = truth.sum((2,3))+pred.sum((2,3))
    dice = np.divide(2*(truth&pred).sum((2,3)),den,out=np.ones(den.shape),where=den>0)
    mean = float(dice.mean())
    record = {**fit,"inner_dice":mean,"cases":len(valid),"passed":bool(np.isfinite(mean) and (smoke or mean>=0.78))}
    rc.write_json(folder/"result.json",record)
    print(record,flush=True)
    if not record["passed"]:
        raise ValueError("Refiner failed inner gate")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("data","run","v3"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    gate(a.data,a.run,a.v3,a.fold,a.smoke)
