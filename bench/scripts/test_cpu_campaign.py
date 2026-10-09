import csv
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


AWS = Path(__file__).resolve().parent.parent / "aws"
spec = importlib.util.spec_from_file_location("cpu_campaign", AWS / "cpu-campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CpuCampaignTests(unittest.TestCase):
    def bundle(self, root):
        runtime = root / "runtime"
        (runtime / "bin").mkdir(parents=True)
        cert = {"binaries": {}}
        for name in campaign.BINARIES:
            path = runtime / "bin" / name
            path.write_bytes(b"\x7fELF\x02\x01" + b"\x00" * 12 + b"\x3e\x00" + name.encode())
            cert["binaries"][name] = dict(
                sha256=campaign.digest(path),
                compiler="test compiler",
                source="a" * 40,
                verification="synthetic fixture only",
                flags="",
            )
        cert["binaries"]["snarkrs-adx"]["flags"] = campaign.FLAGS["adx"]
        campaign.save_json(runtime / "certification.json", cert)
        artifacts = root / "artifacts"
        for variant in campaign.VARIANTS:
            directory = artifacts / variant
            directory.mkdir(parents=True)
            for name in campaign.ARTIFACTS:
                (directory / name).write_text("synthetic fixture\n")
        bundle = root / "bundle"
        campaign.pack(runtime, artifacts, bundle)
        return runtime, bundle

    def test_runtime_whitelist_and_mutation_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bundle = self.bundle(Path(directory))
            (runtime / "unrelated-source.rs").write_text("not shipped")
            hashes = campaign.verify_bundle(bundle)
            self.assertFalse(any("target/" in name or "unrelated" in name for name in hashes))
            self.assertIn("bench/artifacts/js_16x16_d32/input.json", hashes)
            (bundle / "bench/bin/snarkrs-adx").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                campaign.verify_bundle(bundle)

    def test_build_flags_and_certification_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = self.bundle(Path(directory))
            cert = campaign.read_json(runtime / "certification.json")
            cert["binaries"]["snarkrs-adx"]["flags"] = "-C target-cpu=native"
            campaign.save_json(runtime / "certification.json", cert)
            with self.assertRaisesRegex(ValueError, "build flags"):
                campaign.certify(runtime)
            cert["binaries"]["snarkrs-adx"]["flags"] = campaign.FLAGS["adx"]
            cert["binaries"]["snarkrs-generic"]["verification"] = ""
            campaign.save_json(runtime / "certification.json", cert)
            with self.assertRaisesRegex(ValueError, "certification"):
                campaign.certify(runtime)

    def test_deploy_and_retrieve_only_transfer_runtime_and_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, bundle = self.bundle(root)
            with mock.patch.object(campaign.subprocess, "run") as run:
                campaign.transfer("ubuntu@host", root / "key", "/home/ubuntu/cpu-run", bundle)
                campaign.transfer(
                    "ubuntu@host", root / "key", "/home/ubuntu/cpu-run", root / "retrieved", True
                )
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual([cmd[0] for cmd in commands], ["ssh", "scp", "scp"])
            self.assertEqual(commands[0][-1], "mkdir /home/ubuntu/cpu-run")
            self.assertIn("StrictHostKeyChecking=yes", commands[0])
            self.assertIn(str(bundle) + "/.", commands[1])
            self.assertIn("ubuntu@host:/home/ubuntu/cpu-run/results", commands[2])

    def test_unsafe_remote_address_rejected_without_ssh(self):
        with mock.patch.object(campaign.subprocess, "run") as run:
            for host, remote in (
                ("-option", "/safe"),
                ("host;evil", "/safe"),
                ("host", "/tmp/../safe"),
                ("host", "/safe;evil"),
            ):
                with self.subTest(host=host, remote=remote), self.assertRaises(ValueError):
                    campaign.transfer(host, Path("/key"), remote, Path("/unused"))
            run.assert_not_called()

    def test_count_sequence_verified_load_and_finite_timing(self):
        row = dict(
            rep="1",
            variant="v",
            prover="ours",
            mode="warm",
            backend="cpu",
            verified="yes",
            ms="1.25",
            loaded="no",
        )
        campaign.validate_rows([row], "v", "ours", "warm", 1)
        for key, value in (
            ("rep", "2"),
            ("variant", "other"),
            ("mode", "cold"),
            ("verified", "no"),
            ("loaded", "yes"),
            ("ms", "nan"),
            ("ms", "inf"),
            ("ms", "0"),
            ("ms", "-1"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                campaign.validate_rows([dict(row, **{key: value})], "v", "ours", "warm", 1)
        with self.assertRaises(ValueError):
            campaign.validate_rows([row], "v", "ours", "warm", 2)

    def test_warm_sample_count_and_sequence(self):
        self.assertEqual(campaign.warm_samples("rep 1 1.25\nrep 2 2.5\n", 2), [1.25, 2.5])
        for text in (
            "rep 1 1.25\n",
            "rep 2 1.25\nrep 1 2\n",
            "rep 1 nan\nrep 2 2\n",
            "rep 1 0\nrep 2 2\n",
            "rep 1 1.25 verified\nrep 2 2\n",
            "rep 1 1\nrep 2 2\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                campaign.warm_samples(text, 2)

    def competitor_case(self, root, failure=""):
        results = root / "results"
        results.mkdir()
        commands = []

        def logged(command, directory, env, deadline):
            commands.append(command)
            if command[0].endswith("/rapidsnark-warm"):
                self.assertEqual(command[-3], str(root / "bench/artifacts/v/vkey.json"))
                self.assertEqual(command[-2], "--proof-dir")
                if failure == "status":
                    (directory / "stdout.log").write_text("rep 1 1.234567\n")
                    campaign.save_json(directory / "exit.json", {"exit": 7})
                    raise ValueError("configuration exited 7")
                export = Path(command[-1])
                for rep in (1, 2):
                    if failure == "missing" and rep == 2:
                        continue
                    (export / f"proof-{rep}.json").write_text("synthetic proof")
                    (export / f"public-{rep}.json").write_text("synthetic public")
                (directory / "stderr-rss.log").write_text("ffiasm_pool_threads=16\n")
                return "rep 1 1.234567\nrep 2 2.345678\n"
            if command[0].endswith("/rapidsnark"):
                Path(command[-2]).write_text("synthetic proof")
                Path(command[-1]).write_text("synthetic public")
            if command[0] == "python3" and failure == "reference":
                raise ValueError("public signals differ")
            if command[0].endswith("/snarkrs-generic"):
                if failure == "oracle":
                    raise ValueError("configuration exited 1")
                if failure == "marker":
                    return "no success marker\n"
            return "OK!\n"

        with mock.patch.object(campaign.os, "cpu_count", return_value=16), mock.patch.object(
            campaign, "idle", return_value=0
        ), mock.patch.object(campaign, "run_logged", side_effect=logged), mock.patch.object(
            campaign.time, "monotonic", side_effect=[0, 0.00125]
        ):
            campaign.competitor(root, results, "v", 1, 2, {"OMP_NUM_THREADS": "16"}, 100, 1)
        return commands

    def test_warm_exact_interface_independent_oracle_and_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = self.competitor_case(root)
            warm = commands[0]
            self.assertEqual(
                warm[1:7],
                [
                    str(root / "bench/artifacts/v/circuit.zkey"),
                    str(root / "bench/artifacts/v/circuit.wtns"),
                    str(root / "results/rapidsnark-native16-v-warm/last-proof.json"),
                    str(root / "results/rapidsnark-native16-v-warm/last-public.json"),
                    "2",
                    str(root / "bench/artifacts/v/vkey.json"),
                ],
            )
            verifiers = [cmd for cmd in commands if cmd[0].endswith("/snarkrs-generic")]
            references = [cmd for cmd in commands if cmd[0] == "python3"]
            self.assertEqual(len(verifiers), 3)
            self.assertEqual(len(references), 3)
            self.assertTrue(all(cmd[1:3] == ["groth16", "verify"] for cmd in verifiers))
            self.assertFalse(any("rapidsnark-oracle" in cmd[0] for cmd in commands))
            accepted = campaign.read_json(root / "results/rapidsnark-native16-v-warm/accepted.json")
            self.assertEqual(accepted["ms"], [1.234567, 2.345678])
            self.assertEqual(accepted["actual_pool_threads"], 16)
            self.assertEqual(accepted["independent_oracle"], "snarkrs-generic groth16 verify")
            self.assertEqual(accepted["snarkjs_compatible"], "unknown")

    def test_missing_rep_failed_wrapper_or_oracle_never_accepts_warm(self):
        for failure in ("missing", "status", "reference", "oracle", "marker"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(ValueError):
                    self.competitor_case(root, failure)
                warm = root / "results/rapidsnark-native16-v-warm"
                self.assertFalse((warm / "accepted.json").exists())
                if failure == "status":
                    self.assertTrue((warm / "stdout.log").is_file())
                    self.assertEqual(campaign.read_json(warm / "exit.json")["exit"], 7)

    def test_openmp_environment_does_not_prove_pool_size(self):
        for diagnostic in (
            "",
            "OMP_NUM_THREADS=16",
            "ffiasm_pool_threads=8",
            "ffiasm_pool_threads=16\nffiasm_pool_threads=4",
        ):
            with self.subTest(diagnostic=diagnostic), self.assertRaises(ValueError):
                campaign.native_pool(diagnostic)
        self.assertEqual(campaign.native_pool("ffiasm_pool_threads=16\n"), 16)
        with mock.patch.object(campaign.os, "cpu_count", return_value=4), mock.patch.object(
            campaign, "run_logged"
        ) as run:
            with self.assertRaisesRegex(ValueError, "native-16"):
                campaign.competitor(
                    Path("/unused"), Path("/unused"), "v", 3, 15, {"OMP_NUM_THREADS": "16"}, 100, 1
                )
            run.assert_not_called()

    def test_timeout_kills_process_group_and_preserves_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = mock.Mock(pid=123, returncode=-9)
            process.wait.side_effect = [subprocess.TimeoutExpired("prover", 1), -9]
            with mock.patch.object(
                campaign.subprocess, "Popen", return_value=process
            ), mock.patch.object(campaign.os, "killpg") as kill:
                with self.assertRaisesRegex(ValueError, "budget exhausted"):
                    campaign.run_logged(
                        ["prover"], root, {"RAYON_NUM_THREADS": "4"}, campaign.time.monotonic() + 1
                    )
            kill.assert_called_once_with(123, campaign.signal.SIGKILL)
            self.assertEqual(campaign.read_json(root / "exit.json"), {"exit": -9, "timeout": True})

    def test_serial_guest_plan_build_labels_and_one_competitor_per_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, bundle = self.bundle(root)
            watchdog = root / "watchdog"
            watchdog.mkdir()
            (watchdog / "config").write_text("3600 7200 300\n")
            (watchdog / "lease").touch()
            commands = []

            def logged(command, directory, env, deadline):
                commands.append((command, dict(env)))
                if "--csv" in command:
                    variant = command[command.index("--variants") + 1]
                    stages = (
                        "prepare_ms",
                        "gather_us",
                        "ntt_us",
                        "pointwise_us",
                        "msm_us",
                        "assemble_us",
                    )
                    fields = [
                        "rep",
                        "variant",
                        "prover",
                        "mode",
                        "backend",
                        "verified",
                        "ms",
                        "loaded",
                        "snarkjs_compatible",
                        "independent_verification_scope",
                        *stages,
                    ]
                    with Path(command[command.index("--csv") + 1]).open("w") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                        for mode, count in (("cold", 3), ("warm", 15)):
                            for rep in range(1, count + 1):
                                writer.writerow(
                                    dict(
                                        rep=rep,
                                        variant=variant,
                                        prover="ours",
                                        mode=mode,
                                        backend="cpu",
                                        verified="yes",
                                        ms=1.25,
                                        loaded="no",
                                        snarkjs_compatible="unknown",
                                        independent_verification_scope="each-proof"
                                        if mode == "cold"
                                        else "first-proof",
                                        **{name: 1 for name in stages},
                                    )
                                )
                return ""

            cpuinfo = "flags: adx bmi2\n"
            original_read = Path.read_text
            original_copy = campaign.shutil.copyfile

            def read(path, *args, **kwargs):
                return (
                    cpuinfo
                    if str(path) == "/proc/cpuinfo"
                    else original_read(path, *args, **kwargs)
                )

            def copy(source, destination, **kwargs):
                if str(source) == "/etc/os-release":
                    Path(destination).write_text("synthetic Linux guest")
                    return destination
                return original_copy(source, destination, **kwargs)

            with mock.patch.object(
                campaign, "platform_machine", return_value="x86_64"
            ), mock.patch.object(campaign.os, "cpu_count", return_value=16), mock.patch.object(
                campaign.shutil, "copyfile", side_effect=copy
            ), mock.patch.object(
                Path, "read_text", read
            ), mock.patch.object(
                campaign, "idle", return_value=0
            ), mock.patch.object(
                campaign, "run_logged", side_effect=logged
            ), mock.patch.object(
                campaign, "competitor"
            ) as competitor:
                campaign.guest(bundle, 3, 15, 3600, 1, watchdog)
            comparisons = [(cmd, env) for cmd, env in commands if "--csv" in cmd]
            self.assertEqual(len(comparisons), 16)
            self.assertEqual(
                [int(env["RAYON_NUM_THREADS"]) for _, env in comparisons],
                [t for t in campaign.THREADS for _ in range(4)],
            )
            self.assertEqual(competitor.call_count, 2)
            for call in competitor.call_args_list:
                self.assertEqual(call.args[5]["OMP_NUM_THREADS"], "16")
            for cmd, env in comparisons:
                self.assertIn("--warm-oracle", cmd)
                self.assertIn("--skip-rapidsnark", cmd)
                self.assertEqual(env["OMP_NUM_THREADS"], env["RAYON_NUM_THREADS"])
            self.assertTrue((watchdog / "done").exists())
            self.assertEqual(
                campaign.read_json(bundle / "results/complete.json")["configurations"], 36
            )

    def test_loaded_or_expired_budget_rejects_without_reducing_reps(self):
        with mock.patch.object(
            campaign.os, "getloadavg", return_value=(100, 100, 100)
        ), mock.patch.object(campaign.time, "monotonic", return_value=100):
            with self.assertRaisesRegex(ValueError, "loaded host"):
                campaign.idle(101, 1)
        with mock.patch.object(campaign.time, "monotonic", return_value=100), mock.patch.object(
            campaign.subprocess, "Popen"
        ) as process:
            with self.assertRaisesRegex(ValueError, "not reduced"):
                campaign.run_logged(["unused"], Path("/unused"), {}, 99)
            process.assert_not_called()

    def test_failed_process_retains_raw_logs_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = mock.Mock()
            process.wait.return_value = 7
            with mock.patch.object(campaign.subprocess, "Popen", return_value=process):
                with self.assertRaisesRegex(ValueError, "exited 7"):
                    campaign.run_logged(
                        ["prover"], root, {"RAYON_NUM_THREADS": "4"}, campaign.time.monotonic() + 10
                    )
            self.assertEqual(campaign.read_json(root / "exit.json")["exit"], 7)
            self.assertTrue((root / "stdout.log").exists())
            self.assertTrue((root / "stderr-rss.log").exists())


if __name__ == "__main__":
    unittest.main()
