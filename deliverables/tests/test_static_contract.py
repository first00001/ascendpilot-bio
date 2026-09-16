from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticContractTests(unittest.TestCase):
    def test_official_training_skill_is_present(self):
        skill = (ROOT / "qwen35-mindspeed-training" / "SKILL.md").read_text(encoding="utf-8")
        runner = (ROOT / "qwen35-mindspeed-training" / "scripts" / "run_training.sh").read_text(encoding="utf-8")
        self.assertIn("Qwen3.5-0.8B", skill)
        self.assertIn("MindSpeed-MM/FSDP", skill)
        self.assertIn("torchrun", runner)
        self.assertIn("NPUS_PER_NODE", runner)

    def test_training_log_comparison_has_official_metrics(self):
        result = json.loads((ROOT / "results" / "migration" / "training-log-comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(result["window"]["common_iterations"], 100)
        self.assertIn("mean_absolute_difference", result["loss_alignment"])
        self.assertIn("throughput_change_percent", result["ascendc_vs_triton"])

    def test_official_08b_config_and_preparation(self):
        config = (ROOT / "qwen35-mindspeed-training" / "configs" / "qwen3_5_0.8B_config.yaml").read_text(encoding="utf-8")
        prepare = (ROOT / "qwen35-mindspeed-training" / "scripts" / "prepare_target.sh").read_text(encoding="utf-8")
        data = (ROOT / "qwen35-mindspeed-training" / "scripts" / "prepare_data.sh").read_text(encoding="utf-8")
        self.assertIn("Qwen3.5-0.8B", config)
        self.assertIn("train_iters: 100", config)
        self.assertIn("Qwen35Converter hf_to_dcp", prepare)
        self.assertIn("llava_instruct_150k.json", data)

    def test_health_has_a_return_value(self):
        tree = ast.parse(
            (ROOT / "deployment/qwen35_server.py").read_text(encoding="utf-8")
        )
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "health"
        )
        self.assertTrue(any(isinstance(node, ast.Return) for node in function.body))

    def test_demo_and_server_contract_match(self):
        server = (ROOT / "deployment/qwen35_server.py").read_text(encoding="utf-8")
        page = (ROOT / "demo/index.html").read_text(encoding="utf-8")
        for path in ("/migration/report", "/ricefm/umap", "/ricefm/demo-samples", "/ricefm/annotate", "/agent/run"):
            self.assertIn(path, server)
            self.assertIn(path, page)
        for element_id in ("migrationStatus", "precision", "memorySave", "qwenRate", "umap", "token", "annotate", "answer"):
            self.assertIn(f'id="{element_id}"', page)

    def test_offline_demo_sample_matrix_widths(self):
        import importlib.util

        path = ROOT / "offline_demo.py"
        spec = importlib.util.spec_from_file_location("offline_demo", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        data = module.DEMO_SAMPLES
        self.assertEqual("public-evidence", data["mode"])
        self.assertGreaterEqual(len(data["samples"]), 4)
        for sample in data["samples"]:
            self.assertEqual(len(sample["genes"]), len(sample["counts"][0]))
            self.assertLessEqual(len(sample["genes"]), 512)
            self.assertTrue(all(value >= 0 for value in sample["counts"][0]))

    def test_offline_demo_is_explicitly_non_inference(self):
        source = (ROOT / "offline_demo.py").read_text(encoding="utf-8")
        page = (ROOT / "demo/index.html").read_text(encoding="utf-8")
        self.assertIn('MODE = "public-evidence"', source)
        self.assertIn("no Qwen, riceFM, or NPU inference", source)
        self.assertIn("公开数据证据模式", page)
        self.assertIn("未执行 Qwen、riceFM 或 Ascend NPU 实时推理", page)

    def test_public_dataset_has_provenance_and_real_points(self):
        public = json.loads(
            (ROOT / "public_data/e_enad_52_review_subset.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("E-ENAD-52", public["dataset"]["atlas_accession"])
        self.assertEqual("GSE146035", public["dataset"]["geo_accession"])
        self.assertEqual(28, public["dataset"]["cluster_count"])
        self.assertEqual(28857, public["atlas_reported_cells"])
        self.assertEqual(28856, public["total_cells"])
        self.assertGreaterEqual(len(public["points"]), 1000)
        self.assertIn("not curated", public["dataset"]["notice"])

    def test_public_skill_is_independently_runnable(self):
        skill = (ROOT / "public-rice-root-skill/SKILL.md").read_text(
            encoding="utf-8"
        )
        query = (ROOT / "public-rice-root-skill/scripts/query_cluster.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("E-ENAD-52", skill)
        self.assertIn("GSE146035", skill)
        self.assertIn('sub.add_parser("summary")', query)
        self.assertIn('sub.add_parser("markers")', query)

    def test_benchmark_hits_model_endpoints(self):
        source = (ROOT / "benchmark.py").read_text(encoding="utf-8")
        self.assertIn('choices=("chat", "ricefm")', source)
        self.assertNotIn('"/analyze/plan"', source)

    def test_chat_has_evidence_boundary(self):
        source = (ROOT / "deployment/qwen35_server.py").read_text(encoding="utf-8")
        self.assertIn("Never invent", source)
        self.assertIn("禁止补写任何基因名或通路名", source)
        self.assertIn("不是独立外部验证", source)
        self.assertIn("generation_budget", source)
        self.assertIn('Literal["auto", "fixed"]', source)

    def test_grounding_ab_has_both_profiles(self):
        source = (ROOT / "scripts/evaluate_qwen_grounding.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"baseline"', source)
        self.assertIn('"evidence_guarded"', source)
        self.assertIn("unsupported_gene_response_rate", source)

    def test_migration_skill_has_automatic_policy(self):
        skill = (ROOT / "qwen35-ascend-migration/SKILL.md").read_text(encoding="utf-8")
        selector = (ROOT / "qwen35-ascend-migration/scripts/select_policy.py").read_text(encoding="utf-8")
        self.assertIn("Automatically inspect, migrate, optimize, deploy, and verify", skill)
        self.assertIn("selection_score", selector)
        report = json.loads((ROOT / "results/migration/migration-report.json").read_text(encoding="utf-8"))
        self.assertEqual("float16", report["policy"]["selected_precision"])

    def test_agent_contract_is_present(self):
        server = (ROOT / "deployment/qwen35_server.py").read_text(encoding="utf-8")
        evaluator = (ROOT / "scripts/evaluate_agent_tasks.py").read_text(encoding="utf-8")
        for term in ("AgentRunReq", "planner_fallback", "validate_plan", "verify_report", "input_required", "verification_failed", "rewrite_attempted", "agent_contract", "/agent/run"):
            self.assertIn(term, server + evaluator)

    def test_tsv_summary_uses_tab_delimiter(self):
        import sys

        sys.path.insert(0, str(ROOT))
        from skill.plantcell_skill import summarize_table

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.tsv"
            path.write_text("gene_a\tgene_b\n1\t2\n3\t4\n", encoding="utf-8")
            result = summarize_table(path)
        self.assertEqual(2, result["columns"])
        self.assertEqual(2, result["numeric_features"])

    def test_explanation_routes_to_report(self):
        import sys

        sys.path.insert(0, str(ROOT))
        from skill.plantcell_skill import plan

        result = plan("解释标签迁移的可信度和限制，不得举未验证基因")
        self.assertEqual(["report"], result["steps"])
        self.assertFalse(result["requires_riceFM"])


if __name__ == "__main__":
    unittest.main()
