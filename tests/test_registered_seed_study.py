import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from reproduction import registered_seed_study as study


class RegisteredFieldTests(unittest.TestCase):
    def test_verified_field_requires_an_independent_byte_equal_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "protocol.md"
            doc.write_text("test protocol")
            path = root / "field-inner-0.npz"
            np.savez(path, ids=[1])
            record = {"scope":"inner","fold":0,"protocol":study.rc.hash_file(doc),
                      "payload":study.rc.hash_file(path),"adapted":"known"}
            path.with_suffix(".json").write_text(json.dumps(record))
            with self.assertRaises(FileNotFoundError):
                study.verified_field(root,"inner",0,doc)
            replay = root / "replay-inner-0.json"
            replay.write_text(json.dumps({**record,"adapted":"changed","byte_equal":True}))
            with self.assertRaises(ValueError):
                study.verified_field(root,"inner",0,doc)
            replay.write_text(json.dumps({**record,"byte_equal":True}))
            self.assertEqual(study.verified_field(root,"inner",0,doc), (path,record))
