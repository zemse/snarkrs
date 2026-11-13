//! Host metadata only for the canonical coefficient route; no CUDA objects or clocks.

use super::*;
use snarkrs_formats::json_out::JsonWriter;
use std::io::{self, Write};

#[derive(Clone, Copy)]
pub(crate) struct Profile {
    l2_bytes: usize,
}

impl Profile {
    pub(crate) fn parse(l2: Option<&str>, branch: Option<&str>) -> Result<Self, ProveError> {
        if branch != Some("assumed-single-batch") {
            return Err(bad("set G16_PAIR_MSM_BRANCH=assumed-single-batch; split/observed-budget modes are unsupported"));
        }
        let l2_bytes = l2
            .ok_or_else(|| bad("set G16_PAIR_L2_BYTES explicitly"))?
            .parse::<usize>()
            .map_err(|e| bad(format!("invalid model L2: {e}")))?;
        if l2_bytes == 0 {
            return Err(bad("model L2 must be positive"));
        }
        Ok(Self { l2_bytes })
    }
}

fn supported_controls(env: impl IntoIterator<Item = (String, String)>) -> Result<(), ProveError> {
    for (key, value) in env {
        if key.starts_with("G16_CUDA_MSM_")
            && !matches!(
                (key.as_str(), value.as_str()),
                (
                    "G16_CUDA_MSM_FULL_CLEAR" | "G16_CUDA_MSM_LEGACY_ACC" | "G16_CUDA_MSM_POISON",
                    "0"
                )
            )
        {
            return Err(bad(format!(
                "count model supports shipped variable shapes only, unset {key}"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct View {
    identity: (usize, usize, usize, usize, usize),
    general: usize,
    index_span: Option<usize>,
}

fn view(
    prefix: &[u32],
    positions: Option<&[u32]>,
    index_identity: usize,
    range: Range<usize>,
) -> View {
    if let Some(positions) = positions {
        let (lo, hi, general) = selected_metadata(positions, Some(prefix), &range);
        View {
            identity: plan_identity(1, 0, hi - lo, index_identity, lo),
            general,
            index_span: Some(range.len()),
        }
    } else {
        View {
            identity: plan_identity(1, range.start, range.len(), 0, 0),
            general: (prefix[range.end] - prefix[range.start]) as usize,
            index_span: None,
        }
    }
}

fn shape(view: View, mix: Mix, shapes: Shapes) -> Result<Geometry, ProveError> {
    geometry_for_metadata(
        view.identity.2,
        view.identity.1,
        view.general,
        view.index_span,
        pick_window(None, view.general, mix, Work::Variable),
        Work::Variable,
        shapes,
    )
}

fn num<W: Write>(json: &mut JsonWriter<W>, key: &str, value: usize) -> io::Result<()> {
    json.field_raw(key, &value.to_string())
}

fn words<W: Write>(json: &mut JsonWriter<W>, key: &str, values: &[usize]) -> io::Result<()> {
    json.key(key)?;
    json.begin_array()?;
    for value in values {
        json.raw(&value.to_string())?;
    }
    json.end()
}

pub(crate) fn canonical_json(
    pk: &ProvingKey,
    witness: &[Fr],
    profile: Profile,
    coefficient_launches: usize,
) -> Result<String, ProveError> {
    supported_controls(std::env::vars())?;
    if witness.len() != pk.n_vars || pk.n_public >= pk.n_vars {
        return Err(bad("canonical count witness/public dimensions disagree"));
    }
    let prefix =
        scalar_prefix(witness, Work::Variable).ok_or_else(|| bad("variable prefix required"))?;
    let (a, b) = key_live(pk);
    // Logical borrowed-handle identities, not fake device pointers: witness=1, H=2,
    // separately uploaded A/B live handles=3/4, even when their contents match.
    let av = view(&prefix, a.as_deref(), 3, 0..pk.n_vars);
    let bv = view(&prefix, b.as_deref(), 4, 0..pk.n_vars);
    let lv = view(&prefix, None, 0, pk.n_public + 1..pk.n_vars);
    let hv = View {
        identity: plan_identity(2, 0, pk.domain_size, 0, 0),
        general: pk.domain_size,
        index_span: None,
    };
    let shapes = Shapes {
        scatter_rows: scatter_rows_for(profile.l2_bytes),
        ..Shapes::default()
    };
    let jobs = [
        ("A", Group::G1, av),
        ("B2", Group::G2, bv),
        ("B1", Group::G1, bv),
        ("L", Group::G1, lv),
        ("H", Group::G1, hv),
    ];
    let mut plans = Vec::new();
    let mut by_key = HashMap::new();
    let mut job_plan = Vec::new();
    let mut allocation_words = Vec::new();
    let mut digit_launches = 0;
    let mut mixes: HashMap<_, Mix> = HashMap::new();
    for (_, group, view) in jobs {
        mixes.entry(view.identity).or_default().add(group);
    }
    for (_, _, view) in jobs {
        let next = plans.len();
        let index = match by_key.get(&view.identity) {
            Some(&index) => index,
            None => {
                let geometry = shape(view, mixes[&view.identity], shapes)?;
                let scatter = geometry.n_windows.div_ceil(scatter_windows(
                    shapes.scatter_rows,
                    geometry.n_buckets + geometry.dummy_rows,
                    geometry.n_windows,
                ));
                digit_launches += 3 + scatter;
                allocation_words.extend(plan_words(&geometry)?);
                plans.push((view, geometry, scatter));
                by_key.insert(view.identity, next);
                next
            }
        };
        job_plan.push(index);
    }
    let mut point_launches = 0;
    let mut downloads = 0;
    let mut download_bytes = 0;
    for (index, (_, group, _)) in jobs.iter().enumerate() {
        let g = &plans[job_plan[index]].1;
        allocation_words.extend(output_words(g, group.words())?);
        point_launches += 4 + usize::from(g.merge_wide_min != 0) + usize::from(g.ones_groups != 0);
        downloads += 1 + usize::from(g.ones_groups != 0);
        download_bytes += (g.window_sums() + g.ones_groups) * group.words() * 4;
    }
    let scratch_bytes = vram_bytes(&allocation_words);
    let mut bytes = Vec::new();
    let mut json = JsonWriter::new(&mut bytes);
    let render = |json: &mut JsonWriter<&mut Vec<u8>>| -> io::Result<()> {
        json.begin_object()?;
        json.field_string(
            "route",
            "canonical-coefficient-five-jobs-variable-reused-witness-no-arena",
        )?;
        json.field_string(
            "budget_branch",
            "assumed-single-batch; fit UNVERIFIED until observed budget covers requested scratch",
        )?;
        json.field_raw("fit_verified", "false")?;
        num(json, "caller_supplied_l2_bytes", profile.l2_bytes)?;
        json.field_raw("hardware_profile_observed", "false")?;
        json.field_raw("full_clear", "false")?;
        json.field_raw("legacy_accumulator", "false")?;
        json.field_raw("poison", "false")?;
        json.field_raw("arena", "false")?;
        num(
            json,
            "prepared_MSM_no_index_buffer_bytes_once",
            core::mem::size_of::<u32>(),
        )?;
        num(json, "geometry_evaluations_per_proof", plans.len() * 3)?;
        num(json, "scatter_rows", shapes.scatter_rows)?;
        num(json, "wide_span", shapes.wide_span)?;
        num(json, "witness_general", *prefix.last().unwrap() as usize)?;
        json.field_string("H_classification", "device_std unclassified: all domain entries general for safe geometry, not observed nonzero digits")?;
        json.key("plans")?;
        json.begin_array()?;
        for (view, g, scatter) in &plans {
            json.begin_object()?;
            words(
                json,
                "logical_handle_range_index_identity",
                &[
                    view.identity.0,
                    view.identity.1,
                    view.identity.2,
                    view.identity.3,
                    view.identity.4,
                ],
            )?;
            for (key, value) in [
                ("n", g.n),
                ("scalar_off", g.scalar_off),
                ("general", view.general),
                ("c", g.c as usize),
                ("n_windows", g.n_windows),
                ("n_buckets", g.n_buckets),
                ("cap", g.cap),
                ("slice_len", g.slice_len),
                ("slices", g.slices),
                ("rows", g.rows()),
                ("spill_slots", g.spill_slots()),
                ("window_sums", g.window_sums()),
                ("reduce_groups", g.reduce_groups),
                ("ones_groups", g.ones_groups),
                ("merge_wide_min", g.merge_wide_min),
            ] {
                num(json, key, value)?;
            }
            json.key("indexed_original_span")?;
            if let Some(span) = view.index_span {
                json.raw(&span.to_string())?;
            } else {
                json.raw("null")?;
            }
            words(
                json,
                "requested_plan_buffer_words",
                &plan_words(g).map_err(io::Error::other)?,
            )?;
            num(json, "histogram_zero_launches", 1)?;
            num(json, "count_launches", 1)?;
            num(json, "scan_launches", 1)?;
            num(json, "scatter_launches", *scatter)?;
            num(json, "histogram_zero_word_stores", g.rows())?;
            num(json, "scan_bucket_rows", g.rows())?;
            num(json, "count_scalar_visits", g.n)?;
            num(json, "scatter_scalar_visits", g.n * scatter)?;
            num(
                json,
                "count_scalar_window_visits_upper_bound",
                g.n * g.n_windows,
            )?;
            json.end()?;
        }
        json.end()?;
        json.key("jobs_in_submission_order")?;
        json.begin_array()?;
        for (index, (name, group, _)) in jobs.iter().enumerate() {
            let g = &plans[job_plan[index]].1;
            json.begin_object()?;
            json.field_string("query", name)?;
            num(json, "plan", job_plan[index])?;
            num(json, "base_off", 0)?;
            num(json, "point_words", group.words())?;
            words(
                json,
                "requested_output_buffer_words",
                &output_words(g, group.words()).map_err(io::Error::other)?,
            )?;
            json.key("kernel_order")?;
            json.begin_array()?;
            for kernel in ["typed-clear", "segmented-accumulate", "merge"] {
                json.string(kernel)?;
            }
            if g.merge_wide_min != 0 {
                json.string("merge-wide")?;
            }
            json.string("reduce")?;
            if g.ones_groups != 0 {
                json.string("ones")?;
            }
            json.end()?;
            for (key, value) in [
                ("clear_bucket_rows", g.rows()),
                ("accumulate_window_slices", g.n_windows * g.slices),
                ("merge_bucket_rows", g.rows()),
                ("reduce_partial_points", g.window_sums()),
                (
                    "ones_original_scalar_span",
                    plans[job_plan[index]].0.index_span.unwrap_or(g.n),
                ),
                ("D2H_downloads", 1 + usize::from(g.ones_groups != 0)),
                (
                    "D2H_bytes",
                    (g.window_sums() + g.ones_groups) * group.words() * 4,
                ),
                (
                    "host_point_conversion_calls",
                    g.window_sums() + g.ones_groups,
                ),
                ("host_window_partial_add_calls", g.window_sums()),
                ("host_horner_double_calls", (g.n_windows - 1) * g.c as usize),
                ("host_horner_window_add_calls", g.n_windows - 1),
                ("host_ones_partial_add_calls", g.ones_groups),
            ] {
                num(json, key, value)?;
            }
            json.field_string(
                "variable_group_arithmetic",
                "data-dependent; logical calls above are not exact Fq operations",
            )?;
            json.end()?;
        }
        json.end()?;
        words(
            json,
            "all_requested_scratch_buffer_words",
            &allocation_words,
        )?;
        num(
            json,
            "scratch_allocation_requests_per_proof",
            allocation_words.len(),
        )?;
        num(
            json,
            "scratch_allocation_requests_per_serial_pair",
            allocation_words.len() * 2,
        )?;
        json.field_raw(
            "requested_single_batch_scratch_bytes",
            &scratch_bytes.to_string(),
        )?;
        json.field_string("allocation_initialization", "MSM unsafe stream.alloc uninitialized; no per-allocation memset in this model; explicit histogram/typed bucket clears counted")?;
        num(
            json,
            "MSM_launches_per_proof",
            digit_launches + point_launches,
        )?;
        num(
            json,
            "total_proof_kernel_launches",
            coefficient_launches + digit_launches + point_launches,
        )?;
        num(
            json,
            "total_serial_pair_kernel_launches",
            2 * (coefficient_launches + digit_launches + point_launches),
        )?;
        num(json, "MSM_event_creations_per_proof", 2)?;
        num(json, "MSM_event_records_per_proof", 2)?;
        num(json, "MSM_budget_queries_per_proof", 2)?;
        num(json, "planning_mutex_acquisitions_per_proof", 1)?;
        num(json, "D2H_downloads_per_proof", downloads)?;
        num(json, "D2H_bytes_per_proof", download_bytes)?;
        json.key("explicit_stream_sync_sites_per_proof")?;
        json.begin_object()?;
        for (key, count) in [
            ("stage_group", 1),
            ("budget_queries", 2),
            ("MSM_batch", 1),
            ("readback", downloads),
        ] {
            num(json, key, count)?;
        }
        json.end()?;
        num(json, "explicit_stream_syncs_per_proof", 4 + downloads)?;
        num(
            json,
            "explicit_stream_syncs_per_serial_pair",
            2 * (4 + downloads),
        )?;
        json.field_string("sync_count_scope", "conditional source sites only: warm pooled stage scratch, witness reuse, single batch, no memory trace/debug; excludes driver-internal waits, cold setup, fallback and split branches")?;
        json.field_string("lifetime_order", "prepared queries/live maps survive all proofs; HHandle owns stage scratch through five MSM consumers; plan/output scratch and two events live through submission, completion and host combine then drop (no arena); HHandle returns frame to pool; second proof repeats allocations, not two simultaneous frames")?;
        json.key("ordered_proof_segments")?;
        json.begin_array()?;
        for stage in [
            "host RNG draws and blinder-term work dispatched alongside stages 0-9; host work not counted as zero",
            "host pack/classification/fingerprint",
            "witness H2D and coefficient kernels",
            "H-stage completion",
            "MSM planning mutex and outer budget query",
            "batch budget query and event start",
            "unique plan allocations, then five output allocations",
            "all plan digit submissions, then five point jobs",
            "event end and batch synchronize",
            "five window/ones D2H reads and host conversion/Horner",
            "scratch/events drop and H frame return",
            "host blinder-term join and assembly; verify and serialize outside pair clock",
        ] {
            json.string(stage)?;
        }
        json.end()?;
        json.key("remaining_unknown")?;
        json.begin_object()?;
        for key in [
            "observed_available_budget",
            "physical_allocation_rounding_and_pool_retention",
            "host_allocations_pack_rng_blinding_assembly_cost",
            "exact_field_operations",
            "data_dependent_entry_and_bucket_traffic",
            "actual_DRAM_bytes_and_cache_reuse",
            "pinned_staging_and_module_margin",
            "primitive_floors",
            "two_lane_capacity",
        ] {
            json.field_raw(key, "null")?;
        }
        json.end()?;
        json.field_raw("full_workload_trace", "false")?;
        json.end()
    };
    render(&mut json).map_err(|e| bad(e.to_string()))?;
    json.finish().map_err(|e| bad(e.to_string()))?;
    String::from_utf8(bytes).map_err(|e| bad(e.to_string()))
}

#[test]
fn metadata_geometry_matches_prior_plan_expression_including_index_span() {
    for work in [Work::Variable, Work::Constant] {
        for n in [0, 1, 2, 256, 59170] {
            for general in [0, n / 2, n] {
                for span in [None, Some(n + 10000)] {
                    let c = 3;
                    let mut previous = Geometry::with_shapes(
                        n,
                        7,
                        general,
                        c,
                        slice_len_for(n, c, work),
                        work,
                        Shapes::default(),
                    )
                    .unwrap();
                    if let (Some(span), true) = (span, previous.ones_groups != 0) {
                        previous.ones_groups = ones_groups_for(span);
                    }
                    assert_eq!(
                        geometry_for_metadata(n, 7, general, span, c, work, Shapes::default())
                            .unwrap(),
                        previous
                    );
                }
            }
        }
    }
    assert!(
        geometry_for_metadata(usize::MAX, 0, 0, None, 3, Work::Variable, Shapes::default())
            .is_err()
    );
    assert!(
        geometry_for_metadata(1, usize::MAX, 0, None, 3, Work::Variable, Shapes::default())
            .is_err()
    );
    assert!(geometry_for_metadata(1, 0, 2, None, 3, Work::Variable, Shapes::default()).is_err());
}

#[test]
fn metadata_selection_and_identity_preserve_empty_equal_and_different_maps() {
    let w = [
        Fr::from(0u64),
        Fr::from(1u64),
        Fr::from(7u64),
        Fr::from(8u64),
    ];
    let p = scalar_prefix(&w, Work::Variable).unwrap();
    assert_eq!(p, [0, 0, 0, 1, 2]);
    assert_eq!(selected_metadata(&[0, 2, 3], Some(&p), &(1..4)), (1, 3, 2));
    assert_eq!(selected_metadata(&[], Some(&p), &(0..4)), (0, 0, 0));
    assert_eq!(live_positions([false, false].into_iter()), None);
    assert_eq!(live_positions([true, true].into_iter()), Some(vec![]));
    let a = view(&p, Some(&[0, 2]), 3, 0..4);
    let b = view(&p, Some(&[0, 2]), 4, 0..4);
    assert_ne!(a.identity, b.identity);
    assert_eq!(b.identity, view(&p, Some(&[0, 2]), 4, 0..4).identity);
    assert_eq!(a.general, 1);
    assert_eq!(view(&p, None, 0, 2..4).general, 2);
    assert_eq!(
        view(&p, None, 3, 0..4).identity,
        view(&p, None, 4, 0..4).identity
    );
    assert_eq!(view(&p, Some(&[]), 3, 0..4).index_span, Some(4));
    assert_ne!(a.identity, view(&p, Some(&[1, 3]), 4, 0..4).identity);
    assert_eq!(Group::G1.words() * 4, core::mem::size_of::<PackedXyzzG1>());
    assert_eq!(Group::G2.words() * 4, core::mem::size_of::<PackedXyzzG2>());
}

#[test]
fn profile_rejects_unsupported_budget_modes_and_hidden_controls() {
    for branch in [
        None,
        Some(""),
        Some("rows"),
        Some("observed"),
        Some("split"),
    ] {
        assert!(Profile::parse(Some("4194304"), branch).is_err());
    }
    for l2 in [None, Some("0"), Some("bad"), Some("18446744073709551616")] {
        assert!(Profile::parse(l2, Some("assumed-single-batch")).is_err());
    }
    assert!(Profile::parse(Some("1"), Some("assumed-single-batch")).is_ok());
    assert!(supported_controls([("G16_CUDA_MSM_FULL_CLEAR".into(), "0".into())]).is_ok());
    for name in ["G16_CUDA_MSM_C", "G16_CUDA_MSM_WIDE", "G16_CUDA_MSM_POISON"] {
        assert!(supported_controls([(name.into(), "1".into())]).is_err());
    }
}
