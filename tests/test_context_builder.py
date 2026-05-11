import unittest

from core.custom_types import DocumentMetadata
from src.rag.context.builder import ContextBuilder
from src.types.rag_type import DocumentInfo


class ContextBuilderTest(unittest.TestCase):
    def test_builds_traditional_blocks_without_metadata_headers(self):
        docs = [
            {
                "node_id": "node-1",
                "content": "The net profit was 2,042 million yuan.",
                "metadata": {
                    "file_name": "annual_report.pdf",
                    "section_title": "Financial summary",
                    "page": 12,
                    "chunk_index": 3,
                },
                "rerank_score": 0.91234,
            }
        ]

        context = ContextBuilder().run(docs)

        self.assertEqual(context, "[node_id:node-1]\nThe net profit was 2,042 million yuan.")
        self.assertNotIn("metadata:", context)
        self.assertNotIn("content:", context)
        self.assertNotIn("rerank_score", context)

    def test_deduplicates_by_node_id_and_content_while_preserving_order(self):
        docs = [
            {"node_id": "node-1", "content": "alpha"},
            {"node_id": "node-1", "content": "alpha updated"},
            {"node_id": "node-2", "content": " alpha "},
            {"node_id": "node-3", "content": "beta"},
        ]

        context = ContextBuilder().run(docs)

        self.assertIn("[node_id:node-1]", context)
        self.assertNotIn("alpha updated", context)
        self.assertNotIn("[node_id:node-2]", context)
        self.assertIn("[node_id:node-3]", context)
        self.assertLess(context.index("[node_id:node-1]"), context.index("[node_id:node-3]"))

    def test_accepts_document_info_instances(self):
        doc = DocumentInfo(
            node_id="node-1",
            content="The board held 8 meetings.",
            metadata=DocumentMetadata(file_name="governance_report.pdf", section_title="Board", page=6),
            dense_score=0.5,
        )

        context = ContextBuilder().run([doc])

        self.assertIn("[node_id:node-1]", context)
        self.assertNotIn("file_name=governance_report.pdf", context)
        self.assertNotIn("dense_score=0.5000", context)
        self.assertIn("The board held 8 meetings.", context)

    def test_truncates_using_full_block_length(self):
        docs = [
            {
                "node_id": "node-1",
                "content": "A" * 300,
                "metadata": {"file_name": "large.txt", "page": 1},
            }
        ]

        context = ContextBuilder(max_length=120).run(docs)

        self.assertLessEqual(len(context), 120)
        self.assertIn("[node_id:node-1]", context)
        self.assertNotIn("content:\n", context)


if __name__ == "__main__":
    unittest.main()
