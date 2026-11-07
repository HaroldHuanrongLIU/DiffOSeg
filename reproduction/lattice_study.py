import argparse
from pathlib import Path
import cv2
import torch
from reproduction import lattice_preference,semantic_study

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("run","data","texture"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    cv2.setNumThreads(1);torch.set_num_threads(3)
    semantic_study.run(p.parse_args(),backend=lattice_preference,load=lattice_preference.load_net)
