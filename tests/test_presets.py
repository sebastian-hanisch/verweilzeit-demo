"""Regler-Spezifikation: Permalink-Auswertung (begrenzen, einrasten, Müll ignorieren), Presets innerhalb der Reglergrenzen, Konsistenz mit den Konstanten."""

import vwz_constants as C
import vwz_presets as P

S = P.SETTING_SPECS


def test_parse_clamps_to_range():
    assert P.parse_setting(S["n_stacks_slider"], "99") == 8 and P.parse_setting(S["n_stacks_slider"], "-5") == 3 and P.parse_setting(S["n_stacks_slider"], "6") == 6
    assert P.parse_setting(S["max_height_slider"], "1") == 3 and P.parse_setting(S["max_height_slider"], "9") == 6
    assert P.parse_setting(S["seed_input"], "12345") == 9999 and P.parse_setting(S["train_seed_input"], "500") == 99 and P.parse_setting(S["train_seed_input"], "-3") == 0
    assert P.parse_setting(S["announce_slider"], "9") == 4 and P.parse_setting(S["train_slider"], "9") == 5 and P.parse_setting(S["train_slider"], "-1") == 0


def test_parse_snaps_to_step_from_lower_bound():
    fill, cont = S["fill_slider"], S["n_containers_slider"]
    assert P.parse_setting(fill, "82") == 80 and P.parse_setting(fill, "83") == 85 and P.parse_setting(fill, "10") == 40 and P.parse_setting(fill, "200") == 100
    assert P.parse_setting(cont, "320") == 300 and P.parse_setting(cont, "340") == 350 and P.parse_setting(cont, "50") == 100 and P.parse_setting(cont, "900") == 500
    assert P.parse_setting(fill, "42") == 40 and P.parse_setting(fill, "43") == 45


def test_step_grid_starts_at_the_lower_bound_not_at_zero():
    spec = P.SettingSpec("x", int, 1, 1, 21, 5)
    assert [P.parse_setting(spec, str(v)) for v in (1, 3, 4, 7, 9, 14, 19, 21)] == [1, 1, 6, 6, 11, 16, 21, 21]


def test_parse_ignores_garbage():
    for key in ("n_stacks_slider", "seed_input", "fill_slider", "announce_slider", "train_seed_input"):
        assert P.parse_setting(S[key], "abc") is None and P.parse_setting(S[key], None) is None and P.parse_setting(S[key], "") is None
    assert P.parse_setting(S["n_stacks_slider"], "4.5") is None                     # nur ganze Zahlen
    assert P.parse_setting(S["learner_select"], "junk") is None and P.parse_setting(S["learner_select"], "gruppe") is None      # das Gruppenmittel ist nicht wählbar
    assert P.parse_setting(S["learner_select"], "linear") == "linear" and P.parse_setting(S["learner_select"], "boosting") == "boosting"
    assert P.parse_setting(S["view_radio"], "junk") is None and P.parse_setting(S["view_radio"], "niedrigster_stapel") is None       # der Bezug steht immer links, ist keine Wahl rechts
    assert P.parse_setting(S["view_radio"], "bestfit_median") == "bestfit_median" and P.parse_setting(S["view_radio"], "unsicherheitsbewusst") == "unsicherheitsbewusst"
    assert P.parse_setting(P.SettingSpec("f", float, 1.0, 0.0, 2.0), "nan") is None


def test_specs_match_constants_and_defaults_inside_bounds():
    assert S["seed_input"].default == C.SEED_DEFAULT == 268 and S["train_seed_input"].default == C.TRAIN_SEED_DEFAULT == 5 and S["n_containers_slider"].default == 300
    assert S["announce_slider"].default == 2 and C.NU_LEVELS[2] == 0.6 and S["train_slider"].default == 3 and C.TRAIN_SIZES[3] == 1000 and S["learner_select"].default == "boosting"
    for key, spec in S.items():
        assert P.bounds(key) == (spec.lo, spec.hi)
        if spec.lo is not None:
            assert spec.lo <= spec.default <= spec.hi
            if spec.step and spec.step > 1:
                assert (spec.default - spec.lo) % spec.step == 0
    assert len({spec.url_param for spec in S.values()}) == len(S) and {spec.url_param for spec in S.values()} == {"ns", "mh", "fp", "nc", "an", "tn", "lr", "seed", "ts", "vw"}
    assert S["view_radio"].default in C.VIEW_KEYS and C.BASELINE not in C.VIEW_KEYS
    assert P.bounds("announce_slider") == (0, len(C.NU_LEVELS) - 1) and P.bounds("train_slider") == (0, len(C.TRAIN_SIZES) - 1)


def test_every_preset_is_inside_bounds_on_the_step():
    assert list(C.PRESETS) == ["Nur Standard", "Mit Ankündigung", "Zu wenig Daten", "Sehr verlässlich", "Voller Block"] and all(len(n) <= 16 for n in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_STATE_KEYS)
        for field, state_key in P.PRESET_STATE_KEYS.items():
            spec = S[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[field] <= spec.hi, (name, field)
                if spec.step and spec.step > 1:
                    assert (p[field] - spec.lo) % spec.step == 0, (name, field)
            else:
                assert spec.caster(p[field]) == p[field]                                # Lerner: wählbar
    assert len({p["seed"] for p in C.PRESETS.values()}) == 1 and len({p["train_seed"] for p in C.PRESETS.values()}) == 1      # ein gemeinsamer Block-Seed und Trainings-Seed


def test_presets_differ_from_the_default_in_exactly_the_intended_setting():
    base = C.PRESETS["Mit Ankündigung"]
    diffs = {name: {k for k in base if base[k] != p[k]} for name, p in C.PRESETS.items() if name != "Mit Ankündigung"}
    assert diffs == {"Nur Standard": {"announce"}, "Zu wenig Daten": {"train_idx"}, "Sehr verlässlich": {"announce"}, "Voller Block": {"fill_pct"}}
    assert base == dict(n_stacks=6, max_height=5, fill_pct=80, n_containers=300, announce=2, train_idx=3, learner="boosting", seed=268, train_seed=5)
    assert (C.PRESETS["Nur Standard"]["announce"], C.PRESETS["Sehr verlässlich"]["announce"], C.PRESETS["Zu wenig Daten"]["train_idx"]) == (0, 3, 0)
    assert C.NU_LEVELS[3] == 0.3 and C.TRAIN_SIZES[0] == 30 and C.PRESETS["Voller Block"]["fill_pct"] == 100


def test_the_preset_seed_lies_outside_the_kernabschnitt_sample():
    assert all(p["seed"] >= C.SAMPLE_INSTANCES for p in C.PRESETS.values())


def test_encoders_roundtrip_through_parse():
    for key, spec in S.items():
        assert P.parse_setting(spec, spec.encoder(spec.default)) == spec.default


def test_preset_state_keys_map_the_right_widgets():
    assert P.PRESET_STATE_KEYS == {"n_stacks": "n_stacks_slider", "max_height": "max_height_slider", "fill_pct": "fill_slider", "n_containers": "n_containers_slider", "announce": "announce_slider",
                                   "train_idx": "train_slider", "learner": "learner_select", "seed": "seed_input", "train_seed": "train_seed_input"}


def test_labels_cover_every_option():
    assert len(C.NU_LEVELS) == len(C.NU_LABELS) == len(C.NU_SHORT) == 5 and C.NU_LEVELS[0] is None and len(C.TRAIN_SIZES) == 6
    assert set(C.RULE_LABELS) == set(C.RULE_KEYS) == set(C.RULE_SHORT) == set(C.RULE_COLORS) == set(C.RULE_DESCRIPTIONS) and set(C.LEARNER_LABELS) == {"linear", "boosting", "gruppe"}
    assert set(C.TAB_RULES) <= set(C.RULE_KEYS) and set(C.VIEW_KEYS) <= set(C.TAB_RULES) and len(set(C.RULE_COLORS.values())) == len(C.RULE_COLORS)
