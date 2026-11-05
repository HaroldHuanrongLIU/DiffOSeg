"""CT-dependent reliability gates for preference masks and population frequencies."""
import numpy as np
import torch
from torch import nn

from reproduction import joint_energy as energy
from reproduction import normal_refiner as normal


@torch.inference_mode()
def contrast(images,guides):
    values=[]
    for offset in range(0,len(images),24):
        x=torch.from_numpy(images[offset:offset+24]).cuda().float()/255
        g=torch.from_numpy(guides[offset:offset+24]).cuda()
        p=normal.profile_features(x,g,True)
        # Columns are four times the difference from the central CT value.
        c=(p[:,7+4:7+5]-p[:,7+1:7+2]).tanh()
        values.append(c.cpu().numpy())
    return np.concatenate(values)


class PreferenceGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.intercept=nn.Parameter(torch.full((4,),-0.8))
        self.context=nn.Parameter(torch.zeros(3))
        self.temp=nn.Parameter(torch.zeros(4))
        self.offset=nn.Parameter(torch.zeros(4))

    def forward(self,old,new,spread,ct):
        gap=(old-new.sigmoid()).abs()
        z=(self.intercept[None,:,None,None]+self.context[0]*ct+
           self.context[1]*gap+self.context[2]*spread)
        weight=z.sigmoid()
        baseline=torch.logit(old.clamp(1e-6,1-1e-6))
        calibrated=new*(0.9*self.temp.tanh()).exp()[None,:,None,None]
        return ((1-weight)*baseline+weight*calibrated+
                1.5*self.offset.tanh()[None,:,None,None]).sigmoid()


class PopulationGate(nn.Module):
    def __init__(self,readout):
        super().__init__()
        self.readout=readout.requires_grad_(False)
        self.intercept=nn.Parameter(torch.tensor([-0.5,0.5,0.0]))
        self.context=nn.Parameter(torch.zeros(3,4))
        self.temp=nn.Parameter(torch.zeros(4))
        self.offset=nn.Parameter(torch.zeros(4))
        self.support_bias=nn.Parameter(torch.zeros(4))
        self.edge_bias=nn.Parameter(torch.zeros(4))

    def forward(self,raw,joint,new,ct,prior):
        refined=(new*(0.9*self.temp.tanh()).exp()[None,:,None,None]+
                 1.5*self.offset.tanh()[None,:,None,None]).sigmoid()
        q0=self.readout(raw,prior)
        q1=self.readout(joint,prior)
        q2=self.readout(refined,prior)
        features=torch.cat((ct,(q2-q1).abs()[:,None],raw[:,:1]-raw[:,-1:],2*(q1[:,None]-0.5).abs()),1)
        logits=self.intercept[None,:,None,None]+torch.einsum("jc,bchw->bjhw",self.context,features)
        w=logits.softmax(1)
        blended=w[:,0]*q0+w[:,1]*q1+w[:,2]*q2
        support=w[:,:1]*raw+w[:,1:2]*joint+w[:,2:]*refined
        bias=(self.support_bias.tanh()[None,:,None,None]*support+
              self.edge_bias.tanh()[None,:,None,None]*(4*support*(1-support))).sum(1)
        return (torch.logit(blended.clamp(1e-6,1-1e-6))+bias).sigmoid()


def export(model):
    return {name:p.detach().cpu().tolist() for name,p in model.named_parameters() if p.requires_grad}


def restore(model,record,device):
    model=model.to(device)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:
                p.copy_(torch.as_tensor(record[name],device=device))
    return model.eval()


def fit(parent,ancestor,new,contrast_map,priors,ranks,truth,names,seed):
    raw,decoded,proposal,ct,base,order,y=(torch.as_tensor(x,device="cuda") for x in
                                        (ancestor[0],ancestor[1],new,contrast_map,priors,ranks,truth))
    y=y.float()
    raw_s=raw[:,4:].sigmoid()
    spread=raw_s[:,:1]-raw_s[:,-1:]
    preference=PreferenceGate().cuda()
    population=PopulationGate(energy.FrequencyReadout.restore(parent["readout"],"cuda")).cuda()
    trainable=[p for model in (preference,population) for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(trainable,lr=0.025)
    _,inv,n=np.unique(names,return_inverse=True,return_counts=True)
    probability=torch.from_numpy(1/n[inv]).cuda().float()
    generator=torch.Generator(device="cuda").manual_seed(seed)
    for step in range(400):
        idx=torch.multinomial(probability,5,replacement=True,generator=generator)
        which=torch.randint(2,(len(idx),),device="cuda",generator=generator)
        pref=preference(decoded[idx,:4],proposal[idx,:4],spread[idx],ct[idx]).clamp(1e-6,1-1e-6)
        sharp=pref.square()/(pref.square()+(1-pref).square())
        label=y[idx]
        dice=(2*(sharp*label).sum((-2,-1))+1)/((sharp+label).sum((-2,-1))+1)
        freq=population(raw_s[idx],decoded[idx,4:],proposal[idx,4:],ct[idx],base[which,idx])
        loss=(1-dice.mean())+energy.distribution_loss(freq,order[which,idx],label)
        loss=loss+0.0005*(preference.offset.square().mean()+preference.temp.square().mean()+
                         preference.context.square().mean()+population.context.square().mean()+
                         population.temp.square().mean()+population.offset.square().mean()+
                         population.edge_bias.square().mean()+population.support_bias.square().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,3,error_if_nonfinite=True)
        opt.param_groups[0]["lr"]=0.025*(1-0.9*step/399)
        opt.step()
    return preference.eval(),population.eval()
