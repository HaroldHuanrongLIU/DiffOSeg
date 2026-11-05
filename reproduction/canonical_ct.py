"""Object-relative CT views with an explicit native-coordinate inverse."""
import numpy as np
import torch
from torch.nn import functional as F

from reproduction import rank_contours as rc
from reproduction import vote_contours as vote


def pixel_grid(h,w,device):
    y,x=torch.meshgrid(torch.arange(h,device=device,dtype=torch.float32),
                       torch.arange(w,device=device,dtype=torch.float32),indexing="ij")
    return torch.stack((x,y),-1)


def geometry(weight,diameter):
    b,h,w=weight.shape
    xy=pixel_grid(h,w,weight.device).reshape(-1,2)
    values=torch.where(weight>=0.05,weight,0).reshape(b,-1)
    mass=values.sum(1,keepdim=True)
    center=(values@xy)/mass.clamp_min(1e-6)
    centered=xy[None]-center[:,None]
    covariance=(centered.transpose(1,2)@(centered*values[:,:,None]))/mass.clamp_min(1e-6)[:,:,None]
    eig,axes=torch.linalg.eigh(covariance)
    axis=eig.clamp_min(1).sqrt()
    geometric=axis.prod(1,keepdim=True).sqrt()
    scale=(geometric/(diameter/4)*(axis/geometric).sqrt()).clamp(0.45,1.8)
    matrix=(axes*scale[:,None,:])@axes.transpose(1,2)
    enough=mass>=9
    center=torch.where(enough,center,torch.full_like(center,64.5))
    matrix=torch.where(enough[:,:,None],matrix,torch.eye(2,device=weight.device)[None])
    return center,matrix


def image_grid(center,matrix,h,w,inverse):
    xy=pixel_grid(h,w,center.device).reshape(1,-1,2)
    if inverse:
        coords=(xy-center[:,None])@torch.linalg.inv(matrix).transpose(1,2)+64.5
    else:
        coords=(xy-64.5)@matrix.transpose(1,2)+center[:,None]
    return coords.reshape(-1,h,w,2)


def unit_grid(xy,h,w):
    return xy*xy.new_tensor([2/(w-1),2/(h-1)])-1


def warp_image(image,center,matrix):
    h,w=image.shape[-2:]
    grid=image_grid(center,matrix,h,w,False)
    return F.grid_sample(image,unit_grid(grid,h,w),mode="bicubic",padding_mode="border",align_corners=True).clamp(0,1)


def pull_back(logits,native,center,matrix):
    h,w=native.shape[-2:]
    xy=image_grid(center,matrix,h,w,True)
    warped=F.grid_sample(logits,unit_grid(xy,h,w),align_corners=True,padding_mode="border")
    x,y=xy.unbind(-1)
    edge=torch.minimum(torch.minimum(x,w-1-x),torch.minimum(y,h-1-y))
    blend=(edge/6).clamp(0,1)[:,None]
    return native+(warped-native)*blend


def image_logits(models,x,alpha):
    a,b,old=models
    rank=vote.d4(old,x)[:,:4]
    logits=vote.combine_views(vote.single_view(a,x,"fullfield"),vote.single_view(b,x,"zoom"))
    return torch.cat(((1-alpha)*rank+alpha*logits[:,:4],logits[:,8:]),1),rank>0


@torch.inference_mode()
def features(images,old,weights,alpha):
    base=rc.RankContours().cuda().to(memory_format=torch.channels_last).eval()
    base.load_state_dict(torch.load(old/"final.pt",map_location="cpu",weights_only=True))
    models=(*map(vote.load_net,weights),base)
    output,aligned,cart=[],[],[]
    for i in range(0,len(images),12):
        x=torch.from_numpy(images[i:i+12]).cuda().float()/255
        raw,det=image_logits(models,x,alpha)
        support=(2*raw[:,5:7]).sigmoid().mean(1)
        views=[]
        for diameter in (22.,30.):
            center,matrix=geometry(support,diameter)
            warped=warp_image(x,center,matrix)
            new,_=image_logits(models,warped,alpha)
            views.append(pull_back(new,raw,center,matrix))
        output.append(raw.cpu().numpy())
        aligned.append(((views[0]+views[1])/2).cpu().numpy())
        cart.append(det.cpu().numpy())
    result=tuple(np.concatenate(parts) for parts in (output,aligned,cart))
    if not all(np.isfinite(a).all() for a in result):
        raise ValueError("Canonical view produced nonfinite values")
    return result
