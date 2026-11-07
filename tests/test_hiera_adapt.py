import unittest
import torch
from torch import nn
from reproduction.hiera_adapt import parameter_groups


def toy():
    model=nn.Module()
    model.encoder=nn.Module()
    model.encoder.trunk=nn.Module()
    model.encoder.trunk.blocks=nn.ModuleList([nn.Linear(3,3) for _ in range(7)])
    model.encoder.neck=nn.Module()
    model.encoder.neck.convs=nn.ModuleList([nn.Conv2d(3,3,1)])
    model.encoder.requires_grad_(False)
    model.head=nn.Module()
    model.head.decoder=nn.Module()
    model.head.decoder.conv_s0=nn.Conv2d(3,3,1)
    model.head.decoder.conv_s1=nn.Conv2d(3,3,1)
    model.head.output=nn.Linear(3,3)
    return model


class ImageAdapterGroupsTests(unittest.TestCase):
    def test_exact_scope_for_tuned_and_frozen(self):
        for flag in (False,True):
            model=toy()
            groups=parameter_groups(model,flag)
            trained=[p for g in groups for p in g["params"]]
            self.assertEqual(len({id(p) for p in trained}),len(trained))
            self.assertEqual(set(map(id,trained)),{id(p) for p in model.parameters() if p.requires_grad})
            self.assertTrue(all(not p.requires_grad for p in model.encoder.trunk.blocks[:3].parameters()))
            self.assertTrue(all(p.requires_grad==flag for p in model.encoder.trunk.blocks[-4:].parameters()))
            self.assertTrue(all(p.requires_grad==flag for p in model.encoder.neck.parameters()))
            self.assertTrue(all(p.requires_grad==flag for p in model.head.decoder.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.head.output.parameters()))
            self.assertEqual([g["lr"] for g in groups],[.00005,.00001,.00004] if flag else [.00005])
