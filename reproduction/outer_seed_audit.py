"""Recompute every formal B rank score directly from the original area-sorted masks."""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from reproduction.outer_preference_audit import digest, mask_dice


def run(args):
    root=args.run
    if digest(args.data)!="6c401c96b0e1737d43fdccfc446cb0c3601473d3b7668ad4a68d22ec0c1ee81b":
        raise ValueError("Unknown original LIDC data")
    assert digest(args.base/"fourfold_summary.json")=="d27211bd93bc8eb2eccd728bd9a04840dfb0e7864ec0d15cd7826c782edeb2a0"
    assert digest(args.previous/"assessment/outer_report.json")=="799311c1c176f2d8239c619e4c6f35a67fa0ec9300c60c3194cb87d97ade58a3"
    reportpath=root/"assessment/outer_report.json";report=json.loads(reportpath.read_text())
    assert report["selection"]==digest(root/"selection.json")
    with args.data.open("rb") as stream:cases=list(pickle.load(stream).values())
    means=[];patients=set();coverage=set();comparisons=0;maximum_error=0.
    names=("e11","reference","candidate")
    for f in range(4):
        folder=root/f"assessment/fold-{f}"
        stage=args.base/f"fold_{f}/lidc/fold_{f}/stage_2"
        manifest=json.loads((stage/"split.json").read_text())
        seal=json.loads((folder/"sealed.json").read_text());table=json.loads((folder/"paired.json").read_text())
        assert table["seal"]==digest(folder/"sealed.json") and seal["payload"]==digest(folder/"predictions.npz")
        guide_file=seal.get("guide_file",f"guide-{f}.npz")
        assert guide_file in (f"guide-{f}.npz",f"field-full-{f}.npz")
        assert seal["selection"]==report["selection"] and seal["guide"]==digest(root/guide_file)
        if guide_file==f"field-full-{f}.npz":
            field=json.loads((root/f"field-full-{f}.json").read_text())
            replay=json.loads((root/f"replay-full-{f}.json").read_text())
            assert replay=={**field,"byte_equal":True} and field["payload"]==seal["guide"]
        with np.load(folder/"predictions.npz",allow_pickle=False) as z:
            ids=z["ids"].tolist();pred=np.unpackbits(z["masks"],axis=-1)
        prior=args.previous/f"assessment/fold-{f}"
        frozen=json.loads((prior/"sealed.json").read_text())
        assert frozen["payload"]==digest(prior/"predictions.npz")
        with np.load(prior/"predictions.npz",allow_pickle=False) as z:
            assert ids==z["ids"].tolist();accepted=np.unpackbits(z["masks"],axis=-1)
        actual_patients={str(cases[i]["series_uid"]).split("_")[0] for i in ids}
        assert ids==manifest["test_indices"] and actual_patients==set(manifest["test_patients"])
        assert not (actual_patients&patients or set(ids)&coverage)
        coverage.update(ids);patients.update(actual_patients)
        values=[]
        for j,(idx,row) in enumerate(zip(ids,table["rows"],strict=True)):
            assert row["index"]==idx and row["patient"]==str(cases[idx]["series_uid"]).split("_")[0]
            masks=np.asarray(cases[idx]["masks"],dtype=bool)
            y=masks[np.argsort(np.count_nonzero(masks,axis=(1,2)),kind="stable")]
            with np.load(stage/f"cases/{j:05d}_masks.npz",allow_pickle=False) as z:
                baseline=np.unpackbits(z["predictions"])[:4*128*128].reshape(4,128,128)
            scores=np.array([[mask_dice(p,t) for p,t in zip(method,y,strict=True)]
                             for method in (baseline,accepted[j],pred[j])])
            maximum_error=max(maximum_error,float(np.abs(scores-np.array([row[k] for k in names])).max()))
            comparisons+=scores.size;values.append(scores)
        means.append(np.mean(values,axis=(0,2)))
        np.testing.assert_allclose(means[-1],[report["folds"][f]["means"][k] for k in names],rtol=0,atol=1e-12)
        np.testing.assert_allclose(np.mean(values,axis=0),[report["folds"][f]["rank_means"][k] for k in names],
                                   rtol=0,atol=1e-12)
    aggregate=np.mean(means,axis=0)
    assert len(patients)==214 and coverage==set(range(1609)) and comparisons==19308
    assert maximum_error<1e-12 and report["target"]==.91 and report["target_pass"]==bool(aggregate[2]>=.91)
    np.testing.assert_allclose(aggregate,[report["means"][k] for k in names],rtol=0,atol=1e-12)
    audit={"scope":"independent original four-fold set-intersection audit","method":report.get("method","formal B"),"cases":len(coverage),
           "patients":len(patients),"rank_scores_recalculated":comparisons,"maximum_error":maximum_error,
           "means":dict(zip(names,aggregate.tolist())),"target_pass":bool(aggregate[2]>=.91),
           "report":digest(reportpath),"source":digest(Path(__file__))}
    (root/"assessment/independent_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for field in ("run","data","base","previous"):parser.add_argument("--"+field,type=Path,required=True)
    run(parser.parse_args())
