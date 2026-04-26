import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_qa_dataset.py"
SPEC = importlib.util.spec_from_file_location("generate_qa_dataset_script", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class FakeUpdateResult:
    def __init__(self, modified_count: int) -> None:
        self.modified_count = modified_count


class FakeCollection:
    def __init__(self, docs):
        self.docs = [dict(doc) for doc in docs]

    def find(self, query):
        expected_state = query.get("state")
        return [dict(doc) for doc in self.docs if doc.get("state") == expected_state]

    def update_many(self, query, update):
        expected_state = query.get("state")
        target_state = ((update or {}).get("$set") or {}).get("state")
        modified_count = 0
        for doc in self.docs:
            if doc.get("state") != expected_state:
                continue
            doc["state"] = target_state
            modified_count += 1
        return FakeUpdateResult(modified_count)


class GenerateQADatasetScriptTests(unittest.TestCase):
    def test_rollback_all_dry_run_only_previews(self):
        collection = FakeCollection(
            [
                {"node_id": "node-1", "state": 1},
                {"node_id": "node-2", "state": 1},
                {"node_id": "node-3", "state": 2},
            ]
        )

        summary = MODULE.rollback_all_qa_source_states(collection, from_state=1, to_state=2, dry_run=True)

        self.assertEqual(summary["matched_count"], 2)
        self.assertEqual(summary["rolled_back_count"], 0)
        self.assertEqual(summary["preview_node_ids"], ["node-1", "node-2"])
        self.assertEqual(collection.docs[0]["state"], 1)
        self.assertEqual(collection.docs[1]["state"], 1)

    def test_rollback_all_updates_matching_docs(self):
        collection = FakeCollection(
            [
                {"node_id": "node-1", "state": 1},
                {"node_id": "node-2", "state": 1},
                {"node_id": "node-3", "state": 2},
            ]
        )

        summary = MODULE.rollback_all_qa_source_states(collection, from_state=1, to_state=2, dry_run=False)

        self.assertEqual(summary["matched_count"], 2)
        self.assertEqual(summary["rolled_back_count"], 2)
        self.assertTrue(all(doc["state"] == 2 for doc in collection.docs))


if __name__ == "__main__":
    unittest.main()
