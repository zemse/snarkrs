//! Ceremony MSM errors must reach the caller, never the legacy panic methods.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use snarkrs_ceremony::write::BinFileWriter;
use snarkrs_ceremony::{
    contribute, phase1, prepare, setup, transcript, CeremonyError, ContributionParams, CpuGroupFft,
    CpuKeyScale,
};
use snarkrs_field::{Fr, G1Affine, G1Projective, G2Affine, G2Projective, PrimeField, Zero};
use snarkrs_msm::{AccelError, CpuMsm, MsmBackend};

struct FailMsm {
    op: &'static str,
    fail_at: usize,
    calls: AtomicUsize,
    largest_slot: AtomicUsize,
}

impl FailMsm {
    fn new(op: &'static str, fail_at: usize) -> Self {
        Self {
            op,
            fail_at,
            calls: AtomicUsize::new(0),
            largest_slot: AtomicUsize::new(0),
        }
    }

    fn check(&self, op: &'static str, n: usize) -> Result<(), AccelError> {
        if op == self.op {
            self.largest_slot.fetch_max(n, Ordering::Relaxed);
            if self.calls.fetch_add(1, Ordering::Relaxed) + 1 == self.fail_at {
                return Err(AccelError::device("fake", op, "injected device loss"));
            }
        }
        Ok(())
    }
}

impl MsmBackend for FailMsm {
    fn name(&self) -> &'static str {
        "fake"
    }
    fn msm_g1(&self, _: &[G1Affine], _: &[Fr]) -> G1Projective {
        panic!("ceremony must not call legacy G1 MSM")
    }
    fn msm_g2(&self, _: &[G2Affine], _: &[Fr]) -> G2Projective {
        panic!("ceremony must not call legacy G2 MSM")
    }
    fn try_msm_g1(&self, bases: &[G1Affine], scalars: &[Fr]) -> Result<G1Projective, AccelError> {
        self.check("msm_g1", bases.len())?;
        CpuMsm::new().try_msm_g1(bases, scalars)
    }
    fn try_msm_g2(&self, bases: &[G2Affine], scalars: &[Fr]) -> Result<G2Projective, AccelError> {
        self.check("msm_g2", bases.len())?;
        CpuMsm::new().try_msm_g2(bases, scalars)
    }
}

/// A legacy-only implementation remains source compatible and supplies an independent sum.
struct ReferenceMsm;

impl MsmBackend for ReferenceMsm {
    fn name(&self) -> &'static str {
        "reference"
    }
    fn msm_g1(&self, bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
        assert_eq!(bases.len(), scalars.len());
        bases
            .iter()
            .zip(scalars)
            .fold(G1Projective::zero(), |sum, (b, s)| sum + *b * s)
    }
    fn msm_g2(&self, bases: &[G2Affine], scalars: &[Fr]) -> G2Projective {
        assert_eq!(bases.len(), scalars.len());
        bases
            .iter()
            .zip(scalars)
            .fold(G2Projective::zero(), |sum, (b, s)| sum + *b * s)
    }
}

struct Fixture {
    dir: PathBuf,
    r1cs: PathBuf,
    ptau: PathBuf,
}

impl Fixture {
    fn new(test: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "snarkrs-msm-recovery-{test}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let r1cs = dir.join("circuit.r1cs");
        let ptau = dir.join("prepared.ptau");
        write_r1cs(&r1cs);
        let raw = dir.join("new.ptau");
        let contributed = dir.join("contributed.ptau");
        phase1::ptau_new(7, &raw).unwrap();
        phase1::contribute_with(
            &raw,
            &contributed,
            ContributionParams::default(),
            transcript::rng_from_entropy_with(&[7; 64], "recovery phase1"),
            &CpuKeyScale,
        )
        .unwrap();
        prepare::prepare_phase2(&contributed, &ptau, &CpuGroupFft).unwrap();
        Self { dir, r1cs, ptau }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Repeating x * 1 = y gives G1 and G2 slots above setup's 32-term crossover.
fn write_r1cs(path: &Path) {
    let mut w = BinFileWriter::create(path, b"r1cs", 1, 3).unwrap();
    w.start_section(1).unwrap();
    w.write_u32(32).unwrap();
    for limb in Fr::MODULUS.as_ref() {
        w.write_u64(*limb).unwrap();
    }
    for value in [3, 1, 0, 1] {
        w.write_u32(value).unwrap();
    }
    w.write_u64(3).unwrap();
    w.write_u32(33).unwrap();
    w.end_section().unwrap();
    w.start_section(2).unwrap();
    for _ in 0..33 {
        for signal in [2, 0, 1] {
            w.write_u32(1).unwrap();
            w.write_u32(signal).unwrap();
            w.write_fr_plain(&Fr::from(1u64)).unwrap();
        }
    }
    w.end_section().unwrap();
    w.start_section(3).unwrap();
    for label in 0..3 {
        w.write_u64(label).unwrap();
    }
    w.end_section().unwrap();
    w.finish().unwrap();
}

fn assert_device(error: CeremonyError, op: &'static str) {
    assert!(matches!(error, CeremonyError::Accel(AccelError::Device {
        backend: "fake", op: actual, ..
    }) if actual == op));
}

#[test]
fn setup_propagates_g1_and_g2_errors_above_crossover() {
    let fixture = Fixture::new("setup-errors");
    for op in ["msm_g1", "msm_g2"] {
        let msm = FailMsm::new(op, 1);
        let error = setup::setup(
            &fixture.r1cs,
            &fixture.ptau,
            &fixture.dir.join(format!("{op}.zkey")),
            &msm,
        )
        .unwrap_err();
        assert_device(error, op);
        assert!(msm.calls.load(Ordering::Relaxed) >= 1);
        assert!(msm.largest_slot.load(Ordering::Relaxed) >= 33);
    }
}

#[test]
fn cpu_setup_matches_reference_and_verifies() {
    let fixture = Fixture::new("reference");
    let cpu = fixture.dir.join("cpu.zkey");
    let reference = fixture.dir.join("reference.zkey");
    setup::setup(&fixture.r1cs, &fixture.ptau, &cpu, &CpuMsm::new()).unwrap();
    setup::setup(&fixture.r1cs, &fixture.ptau, &reference, &ReferenceMsm).unwrap();
    assert_eq!(
        std::fs::read(&cpu).unwrap(),
        std::fs::read(&reference).unwrap()
    );
    let final_key = fixture.dir.join("final.zkey");
    contribute::contribute_with(
        &cpu,
        &final_key,
        None,
        transcript::rng_from_entropy_with(&[9; 64], "recovery phase2"),
        &CpuKeyScale,
    )
    .unwrap();
    contribute::verify_from_init(&cpu, &fixture.ptau, &final_key, &CpuMsm::new()).unwrap();
    contribute::verify_from_init(&reference, &fixture.ptau, &final_key, &ReferenceMsm).unwrap();
}

#[test]
fn zkey_verification_propagates_each_of_four_msm_errors() {
    let fixture = Fixture::new("verify-errors");
    let init = fixture.dir.join("init.zkey");
    let final_key = fixture.dir.join("final.zkey");
    setup::setup(&fixture.r1cs, &fixture.ptau, &init, &CpuMsm::new()).unwrap();
    contribute::contribute_with(
        &init,
        &final_key,
        None,
        transcript::rng_from_entropy_with(&[9; 64], "recovery phase2"),
        &CpuKeyScale,
    )
    .unwrap();
    for fail_at in 1..=4 {
        let msm = FailMsm::new("msm_g1", fail_at);
        let error =
            contribute::verify_from_init(&init, &fixture.ptau, &final_key, &msm).unwrap_err();
        assert_device(error, "msm_g1");
        assert_eq!(msm.calls.load(Ordering::Relaxed), fail_at);
    }
    let msm = FailMsm::new("msm_g1", usize::MAX);
    contribute::verify_from_init(&init, &fixture.ptau, &final_key, &msm).unwrap();
    assert_eq!(msm.calls.load(Ordering::Relaxed), 4);
}

fn entries(dir: &Path) -> Vec<std::ffi::OsString> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    entries.sort();
    entries
}

fn assert_preserved(out: &Path, mut operation: impl FnMut(&Path) -> Result<(), CeremonyError>) {
    for existing in [false, true] {
        if existing {
            std::fs::write(out, b"existing destination").unwrap();
        }
        let before = entries(out.parent().unwrap());
        let error = operation(out).unwrap_err();
        assert!(matches!(
            error,
            CeremonyError::Accel(AccelError::Device {
                backend: "fake",
                ..
            })
        ));
        if existing {
            assert_eq!(std::fs::read(out).unwrap(), b"existing destination");
        } else {
            assert!(!out.exists());
        }
        assert_eq!(entries(out.parent().unwrap()), before, "orphan stage");
    }
}

#[test]
fn setup_late_errors_preserve_absent_and_existing_destinations() {
    let fixture = Fixture::new("publication-setup");
    for (op, fail_at) in [("msm_g1", 2), ("msm_g2", 1)] {
        assert_preserved(&fixture.dir.join(op), |out| {
            let msm = FailMsm::new(op, fail_at);
            let result = setup::setup(&fixture.r1cs, &fixture.ptau, out, &msm).map(|_| ());
            assert!(msm.calls.load(Ordering::Relaxed) >= fail_at);
            result
        });
    }
}

struct LateScale(AtomicUsize);

impl LateScale {
    fn check(&self) -> Result<(), AccelError> {
        if self.0.fetch_add(1, Ordering::Relaxed) == 1 {
            return Err(AccelError::device(
                "fake",
                "apply_key",
                "second section failed",
            ));
        }
        Ok(())
    }
}

impl snarkrs_msm::KeyScale for LateScale {
    fn name(&self) -> &'static str {
        "fake"
    }
    fn apply_key_g1(&self, points: &mut [G1Affine], first: Fr, inc: Fr) -> Result<(), AccelError> {
        self.check()?;
        snarkrs_msm::KeyScale::apply_key_g1(&CpuKeyScale, points, first, inc)
    }
    fn apply_key_g2(&self, points: &mut [G2Affine], first: Fr, inc: Fr) -> Result<(), AccelError> {
        self.check()?;
        snarkrs_msm::KeyScale::apply_key_g2(&CpuKeyScale, points, first, inc)
    }
}

struct LateFft(AtomicUsize);

impl snarkrs_msm::GroupFft for LateFft {
    fn name(&self) -> &'static str {
        "fake"
    }
    fn ifft_g1(
        &self,
        a: &mut [snarkrs_msm::xyzz::Xyzz<snarkrs_field::raw::RawFq>],
    ) -> Result<(), AccelError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        snarkrs_msm::GroupFft::ifft_g1(&CpuGroupFft, a)
    }
    fn ifft_g2(
        &self,
        _: &mut [snarkrs_msm::xyzz::Xyzz<snarkrs_field::raw::RawFq2>],
    ) -> Result<(), AccelError> {
        assert!(self.0.load(Ordering::Relaxed) > 0);
        Err(AccelError::device(
            "fake",
            "ifft_g2",
            "second section failed",
        ))
    }
}

#[test]
fn prepare_late_error_preserves_absent_and_existing_destinations() {
    let fixture = Fixture::new("publication-prepare");
    assert_preserved(&fixture.dir.join("output.ptau"), |out| {
        prepare::prepare_phase2(
            &fixture.dir.join("contributed.ptau"),
            out,
            &LateFft(AtomicUsize::new(0)),
        )
    });
}

fn scale_operation(
    kind: usize,
    input: &Path,
    output: &Path,
    scale: &dyn snarkrs_msm::KeyScale,
) -> Result<(), CeremonyError> {
    let rng = transcript::rng_from_entropy_with(&[11; 64], "publication");
    let beacon = [1; 32];
    match kind {
        0 => phase1::contribute_with(input, output, ContributionParams::default(), rng, scale)
            .map(|_| ()),
        1 => phase1::beacon(input, output, None, &beacon, 10, scale).map(|_| ()),
        2 => contribute::contribute_with(input, output, None, rng, scale).map(|_| ()),
        3 => contribute::beacon(input, output, None, &beacon, 10, scale).map(|_| ()),
        4 => snarkrs_ceremony::challenge::challenge_contribute_with(input, output, rng, scale)
            .map(|_| ()),
        5 => snarkrs_ceremony::bellman::bellman_contribute_with(input, output, rng, scale)
            .map(|_| ()),
        _ => unreachable!(),
    }
}

fn scale_inputs(fixture: &Fixture) -> [PathBuf; 6] {
    let raw = fixture.dir.join("contributed.ptau");
    let init = fixture.dir.join("init.zkey");
    let challenge = fixture.dir.join("challenge");
    let bellman = fixture.dir.join("bellman");
    setup::setup(&fixture.r1cs, &fixture.ptau, &init, &CpuMsm::new()).unwrap();
    snarkrs_ceremony::challenge::export_challenge(&raw, &challenge).unwrap();
    snarkrs_ceremony::bellman::export_bellman(&init, &bellman).unwrap();
    [raw.clone(), raw, init.clone(), init, challenge, bellman]
}

#[test]
fn late_scale_errors_preserve_all_six_public_operations() {
    let fixture = Fixture::new("publication-scale");
    for (kind, input) in scale_inputs(&fixture).iter().enumerate() {
        assert_preserved(&fixture.dir.join(format!("output-{kind}")), |out| {
            let scale = LateScale(AtomicUsize::new(0));
            let result = scale_operation(kind, input, out, &scale);
            assert_eq!(scale.0.load(Ordering::Relaxed), 2);
            result
        });
        let alias = fixture.dir.join(format!("failed-alias-{kind}"));
        std::fs::copy(input, &alias).unwrap();
        let original = std::fs::read(&alias).unwrap();
        let before = entries(&fixture.dir);
        let error =
            scale_operation(kind, &alias, &alias, &LateScale(AtomicUsize::new(0))).unwrap_err();
        assert!(matches!(error, CeremonyError::Accel(_)));
        assert_eq!(std::fs::read(&alias).unwrap(), original);
        assert_eq!(entries(&fixture.dir), before);
    }
}

#[test]
fn successful_in_place_operations_match_separate_outputs() {
    let fixture = Fixture::new("publication-alias");
    for (kind, input) in scale_inputs(&fixture).iter().enumerate() {
        let expected = fixture.dir.join(format!("expected-{kind}"));
        let alias = fixture.dir.join(format!("alias-{kind}"));
        std::fs::copy(input, &alias).unwrap();
        scale_operation(kind, input, &expected, &CpuKeyScale).unwrap();
        let before = entries(&fixture.dir);
        scale_operation(kind, &alias, &alias, &CpuKeyScale).unwrap();
        assert_eq!(
            std::fs::read(&alias).unwrap(),
            std::fs::read(&expected).unwrap()
        );
        assert_eq!(entries(&fixture.dir), before);
    }
    let alias = fixture.dir.join("prepare-alias");
    std::fs::copy(fixture.dir.join("contributed.ptau"), &alias).unwrap();
    prepare::prepare_phase2(&alias, &alias, &CpuGroupFft).unwrap();
    assert_eq!(
        std::fs::read(&alias).unwrap(),
        std::fs::read(&fixture.ptau).unwrap()
    );
    let expected = fixture.dir.join("setup-expected");
    setup::setup(&fixture.r1cs, &fixture.ptau, &expected, &CpuMsm::new()).unwrap();
    for source in [&fixture.r1cs, &fixture.ptau] {
        let alias = fixture.dir.join("setup-alias");
        std::fs::copy(source, &alias).unwrap();
        let (r1cs, ptau) = if source == &fixture.r1cs {
            (&alias, &fixture.ptau)
        } else {
            (&fixture.r1cs, &alias)
        };
        setup::setup(r1cs, ptau, &alias, &CpuMsm::new()).unwrap();
        assert_eq!(
            std::fs::read(&alias).unwrap(),
            std::fs::read(&expected).unwrap()
        );
    }
}
