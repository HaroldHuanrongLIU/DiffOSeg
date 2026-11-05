"""Check serialization through the same aggregate entry point as the four-fold jobs."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from reproduction import vote_eval as study


class VoteReportTest(unittest.TestCase):
    def test_writes_native_boolean_endpoints(self):
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch)
            prior=root/"prior"
            def save(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value))
            keys=study.v1.PRIMARY
            old=dict.fromkeys(keys,0.4)
            new=dict(zip(keys,[0.3,0.5,0.5]))
            start=0
            for f,(n,p) in enumerate(zip((450,375,412,372),(54,54,53,53))):
                ids=list(range(start,start+n))
                start+=n
                names=[f"p{f}-{i}" for i in range(p)]
                lock=root/f"lock-{f}.json"
                save(lock,{"expert_alpha":0.5,"hardness":1.,"strength":0.9})
                folder=root/f"assessment/fold-{f}"
                seal=folder/"sealed.json"
                save(seal,{"fold":f,"source":study.rc.hash_file(study.__file__),
                           "protocol":study.rc.hash_file(study.vote.PROTOCOL),
                           "lock_sha256":study.rc.hash_file(lock)})
                rows=[{"index":idx,"patient":names[i%p],"e11":old,"v1":old,"v3":new,"det":new}
                      for i,idx in enumerate(ids)]
                save(folder/"paired.json",{"rows":rows,"seal_sha256":study.rc.hash_file(seal),"reference_max_abs_error":0.})
                save(prior/f"baseline/fold_{f}/lidc/fold_{f}/stage_1/split.json",
                     {"test_indices":ids,"test_patients":names})
            save(prior/"assessment/fourfold_comparison.json",{"metrics":{k:{"baseline":0.4,"corrected":0.4} for k in keys}})
            study.report(SimpleNamespace(run=root,prior=prior))
            result=json.loads((root/"assessment/fourfold_comparison.json").read_text())
            self.assertEqual(set(result["primary_pass"]),set(keys))
            for flag in result["primary_pass"].values():
                self.assertIs(flag,True)
