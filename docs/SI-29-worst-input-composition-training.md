# SI-29 — Worst-input composition-aware training for the miniature (as-built)

> **Scope:** RFC SI-28 §14 (the SI-29 outline) and §15 (the ten
> acceptance criteria — each addressed below with its
> measurements), plus §6.2–§6.7 and §17. Composition-awareness
> is now a recorded per-cycle configuration with exactly three
> named values, and signal 2 was trained at the landed head
> `d0001`, base-anchored, with exactly one added worst-input
> composition term (λ_w = 1.0, top-k mean with k = 4, both
> fixed in advance as the configuration's definition) computed
> only over signal 2's own train partition — exactly one
> variable changed against SI-27's decisive run, whose
> treatment is re-run inside this slice as a control.
> **Cycle-2 ADMIT was explicitly NOT required (RFC §15
> criterion 7).** The gate's verdict on the decisive cycle 2
> is reported as measured in the decisive-run section below.
> **This document is ordered as RFC §14 item 1 requires:
> Phase 0 — the per-input composition measurements, the
> §6.2 form choice, and the pre-registered F3 reading rule —
> comes first, before any decisive number.**

## What was built

- **`si/tiny_training.py`** — the SI-29 additions. (a) The
  third composition configuration: `COMPOSITION_CONFIGS`
  (exactly `"none"`, `"movement-composition"`,
  `"worst-input-composition"`), `WORST_INPUT_COMPOSITION`,
  and `validate_composition_config(...)` serving all three
  with the existing `ValueError` discipline. λ_w and k live
  in exactly one place: `WORST_INPUT_LAMBDA = 1.0` and
  `WORST_INPUT_K = 4`, part of the
  `"worst-input-composition"` configuration's definition —
  never read from an intent or any other caller input, so
  no per-cycle λ_w or k dial exists. (b) The worst-input
  machinery: `worst_input_selection(...)` (per-input `p`
  values, the deterministic top-k set, the single worst
  input by population index), `worst_input_composition(...)`
  (the term `W` — the mean of the k largest per-input `p(x)`
  values), `analytic_gradients_worst_input(...)` (signal
  gradient plus the mean of the selected inputs' SI-27
  fixed-matrix gradients through the same readout fold),
  `worst_input_factor_loss(...)` (the same loss through the
  float32 shared fold and the forwards — the
  finite-difference check's target), and
  `worst_input_composition_record(...)` (the allowed-input
  worst-input record: SI-27's movement-composition record
  plus `max_p`, the worst allowed input's index/sequence,
  and the top-k indices/mean). (c) `train_candidate`
  dispatches on the intent's composition configuration:
  under `"worst-input-composition"` it prepares `m1` on the
  signal's train partition only (base assembly exactly as
  SI-27's), trains against `signal_MSE + λ_w · W`, records
  the worst term's values on both partitions (train:
  optimised; held-out: reported only) and the worst train
  input by train-partition identity; under `"none"` and
  under `"movement-composition"` it runs the SI-25 and
  SI-27 paths byte-for-byte (proven by exact reproduction
  of four published blobs inside this slice — see the
  reproduction criterion). The mean-penalty hyperparams
  stay null under the worst-input configuration and the
  worst-input hyperparams stay null under both prior
  configurations: the two terms are never blended
  (§17 item 17). The non-improvement refusal still judges
  the *signal* held-out loss alone under all three
  configurations.
- **`si/tiny_transformer.py`** — the recorded composition
  configuration on `TinyTransformerMind` now serves the
  third value through the same constructor-level
  `composition_config` (default `"none"`) and the same
  explicit driver call `set_composition_config(...)`;
  `propose` carries the configured value on the intent
  beside the configured signal and anchoring. The setter
  takes no λ_w or k argument (asserted).
- **`si/gate.py`** — `Intent.composition_config`'s comment
  now names the three served values. No gate check,
  verdict mapping, tolerance, cap, or policy default
  changed; the field itself and its default (`"none"`)
  are unchanged from SI-27.
- **`tests/test_worst_input_composition.py`** (new,
  13 tests), the two `COMPOSITION_CONFIGS` assertions in
  `tests/test_movement_composition.py` updated to the new
  three-value tuple (the SI-27 slice's exactness assertion,
  carried forward to the configuration's new definition —
  every other SI-27 assertion is untouched), and this doc.
  No probe-suite change, no serve change (501 for every
  transformer bundle), no CLI wiring, the Glimmer seed
  untouched, the demo path unchanged, both signals' ids,
  streams, corrections, and partitions unchanged, both
  anchorings' definitions unchanged, both prior composition
  configurations' definitions unchanged.

There is no autonomous, learned, depth-driven,
verdict-driven, or probe-driven composition selection
anywhere in the slice — the mind chooses neither its
signal, its anchoring, nor its composition configuration;
the driver configures all three, and the journal shows all
three. Exactly one worst-input decisive run was performed,
at exactly λ_w = 1.0 and k = 4; no sweep of λ_w, k, or τ
of any kind was run.

## Phase 0 — the outstanding per-input measurements (RFC §§5.5, 6.7, 14 item 1)

Computed from existing machinery (`movement_composition_record`,
the SI-27 diagnostic) on the two existing decisive
candidates, re-run inside this slice before any new
training semantics were finalised — the SI-25 control
candidate (signal 2, base-anchored, composition `"none"`,
at `d0001`) and the SI-27 treatment candidate (same labels,
composition `"movement-composition"`, λ = 1.0). Both
candidates reproduced exactly (blobs and gate figures in
the reproduction criterion below), so these records describe
the published candidates, not approximations of them.
Allowed inputs only: signal 2's train partition (32
sequences) and held-out partition (16 sequences). Sign
agreements are published beside the magnitude-weighted
figures (the inner products / `p` values), never alone
(§17 item 19). A worst-case figure here names its
population (§17 item 20): "worst" below always means the
largest `p(x)` — the least opposing input — *in the named
allowed partition*.

### SI-25 control (composition "none") — blob `sha256:646d1e8ba3978d858d4173d561f8799bce37350f88ff2de65f40c223d9e4d675`, candidate norm 0.2515524481203623

**train partition** (allowed inputs; 32 train sequences): aggregate inner product -1.6873837019991191, mean p -0.0016478356464835148, sign agreement 0.267578125, ‖m1‖ 1.4131161397867473, ‖m2‖ 1.6765305214306858, **maximum p (worst allowed input in this partition) -0.0006878804284308964 at partition index 19, sequence [4, 5, 1, 28, 31, 9, 22]**, top-4 mean p -0.0006952966321423392 (top-4 indices [19, 31, 27, 11]).

| Index in partition | Sequence (allowed input) | Inner product m1·m2 | p(x) = inner/32 | Sign agreement | ‖m1‖ | ‖m2‖ |
|---|---|---|---|---|---|---|
| 0 | [4, 19, 17, 18, 12, 30] | -0.07444421595462357 | -0.0023263817485819865 | 0.25 | 0.2966011006025265 | 0.3254209919006127 |
| 1 | [17, 10, 18, 12] | -0.08102865659140707 | -0.002532145518481471 | 0.21875 | 0.2836062593174113 | 0.3562194241225748 |
| 2 | [25, 10, 10, 29, 5, 2] | -0.05333325852787642 | -0.001666664328996138 | 0.25 | 0.21873929349260565 | 0.3223985156710135 |
| 3 | [18, 6, 28, 20, 13, 22, 9, 22] | -0.023218651436043358 | -0.0007255828573763549 | 0.5 | 0.21357287049391846 | 0.21248756181359763 |
| 4 | [24, 23, 22, 9, 0, 14, 19] | -0.061759646731947726 | -0.0019299889603733664 | 0.1875 | 0.2495054965743626 | 0.32820574270164765 |
| 5 | [22, 21, 18, 12, 14, 0] | -0.05038851763996473 | -0.0015746411762488977 | 0.1875 | 0.27319369440438357 | 0.26426669585349377 |
| 6 | [18, 18] | -0.05670506519468709 | -0.0017720332873339716 | 0.28125 | 0.2388840561039818 | 0.31668145028608496 |
| 7 | [30, 2, 6, 9] | -0.022736103962532627 | -0.0007105032488291447 | 0.5625 | 0.21596106614984828 | 0.21309366313541422 |
| 8 | [19, 17, 26, 24, 20, 30] | -0.07420307386763172 | -0.0023188460583634914 | 0.21875 | 0.2919717297536117 | 0.3299230710564805 |
| 9 | [20, 13, 13, 20, 31, 12] | -0.08210916178417184 | -0.00256591130575537 | 0.1875 | 0.29018667792943115 | 0.3548497268742245 |
| 10 | [22, 29, 31, 5, 25, 5, 2] | -0.05506053023019075 | -0.001720641569693461 | 0.28125 | 0.22340326621088447 | 0.3228169844093542 |
| 11 | [26, 22] | -0.022545821798948398 | -0.0007045569312171375 | 0.28125 | 0.20970481457909182 | 0.21149657827905394 |
| 12 | [3, 19] | -0.06057143034179842 | -0.0018928571981812004 | 0.1875 | 0.24860966707704105 | 0.3229515653206534 |
| 13 | [0, 14, 0, 0] | -0.0517764970771236 | -0.0016180155336601124 | 0.1875 | 0.27699865400183726 | 0.2666208836796902 |
| 14 | [6, 23, 18, 18] | -0.05553894447001326 | -0.0017355920146879145 | 0.15625 | 0.2337032524305807 | 0.32008732664022344 |
| 15 | [31, 24, 4, 26, 13, 27, 10, 9] | -0.022747754592148152 | -0.0007108673310046296 | 0.40625 | 0.21045552281347335 | 0.2123865324243905 |
| 16 | [8, 18, 3, 30] | -0.0755432607000035 | -0.0023607268968751094 | 0.3125 | 0.29847131182567754 | 0.32789373131384003 |
| 17 | [3, 24, 12] | -0.07839560954681915 | -0.0024498627983380985 | 0.15625 | 0.27706906126671604 | 0.35288182534478924 |
| 18 | [16, 2, 21, 29, 2] | -0.0539947017153955 | -0.0016873344286061096 | 0.25 | 0.2210387373570659 | 0.31911487451813186 |
| 19 | [4, 5, 1, 28, 31, 9, 22] | -0.022012173709788686 | -0.0006878804284308964 | 0.3125 | 0.21327071180271867 | 0.21043367684089026 |
| 20 | [29, 5, 19] | -0.0607409280419937 | -0.0018981540013123031 | 0.1875 | 0.2480749304725319 | 0.32353210452362674 |
| 21 | [3, 31, 18, 31, 31, 0] | -0.0495493894681232 | -0.0015484184208788498 | 0.15625 | 0.2702844243567621 | 0.2629329471638519 |
| 22 | [26, 19, 28, 11, 15, 0, 18] | -0.05605323497821871 | -0.0017516635930693346 | 0.1875 | 0.23607611618488403 | 0.31761825105523267 |
| 23 | [28, 9] | -0.022746451448972537 | -0.0007108266077803917 | 0.375 | 0.21175210217611146 | 0.21380598237719964 |
| 24 | [8, 26, 8, 5, 30, 15, 10, 30] | -0.07468646417655339 | -0.002333952005517293 | 0.25 | 0.2927373388435464 | 0.32959908787179737 |
| 25 | [20, 15, 23, 12] | -0.08080395961333615 | -0.0025251237379167544 | 0.25 | 0.28562845193290143 | 0.3532190975183195 |
| 26 | [14, 2] | -0.05116446638305944 | -0.0015988895744706076 | 0.28125 | 0.21807797165824344 | 0.31621420611653295 |
| 27 | [5, 3, 20, 0, 8, 17, 27, 22] | -0.022302459451143276 | -0.0006969518578482274 | 0.40625 | 0.21158347622062923 | 0.21107282382925585 |
| 28 | [0, 15, 30, 19, 3, 19] | -0.061768893981112835 | -0.001930277936909776 | 0.21875 | 0.24735272397838892 | 0.3299845249772436 |
| 29 | [29, 7, 3, 4, 7, 0] | -0.05157943393965147 | -0.0016118573106141086 | 0.15625 | 0.27296650742775436 | 0.264604567543908 |
| 30 | [23, 24, 20, 5, 6, 25, 9, 18] | -0.055737430689499885 | -0.0017417947090468714 | 0.21875 | 0.23822149240072643 | 0.31448952995321106 |
| 31 | [6, 29, 3, 9] | -0.022137513954339057 | -0.0006917973110730954 | 0.5 | 0.21504993382191423 | 0.21131293494478304 |

**held_out partition** (allowed inputs; 16 held-out sequences, reported only): aggregate inner product -0.845596374259425, mean p -0.0016515554184754395, sign agreement 0.26953125, ‖m1‖ 0.9997655382584564, ‖m2‖ 1.1850839719703719, **maximum p (worst allowed input in this partition) -0.0006654083712165222 at partition index 11, sequence [17, 22]**, top-4 mean p -0.0006984553203186485 (top-4 indices [11, 3, 15, 7]).

| Index in partition | Sequence (allowed input) | Inner product m1·m2 | p(x) = inner/32 | Sign agreement | ‖m1‖ | ‖m2‖ |
|---|---|---|---|---|---|---|
| 0 | [25, 30] | -0.0740626760799191 | -0.0023144586274974717 | 0.1875 | 0.29230395465794035 | 0.32925469381367295 |
| 1 | [17, 2, 4, 12] | -0.0800165151429649 | -0.0025005160982176536 | 0.21875 | 0.281730472467715 | 0.35467455268322323 |
| 2 | [29, 2] | -0.053001904917095666 | -0.0016563095286592396 | 0.25 | 0.22245871127522943 | 0.3178105198929116 |
| 3 | [30, 28, 22] | -0.02178895928306586 | -0.0006809049775958081 | 0.4375 | 0.20667754516337455 | 0.20940126442148457 |
| 4 | [23, 17, 21, 23, 17, 0, 24, 19] | -0.06008933108150172 | -0.0018777915962969287 | 0.25 | 0.2478540679470391 | 0.32269507710351686 |
| 5 | [4, 0, 0] | -0.052829315987794806 | -0.001650916124618588 | 0.15625 | 0.2781382798176755 | 0.26977801386582984 |
| 6 | [19, 23, 21, 24, 18] | -0.05597737086421588 | -0.0017492928395067464 | 0.25 | 0.23705973326532234 | 0.3160330141911612 |
| 7 | [11, 30, 19, 24, 9] | -0.023834845993129048 | -0.0007448389372852829 | 0.34375 | 0.21718941757144988 | 0.21412981715421142 |
| 8 | [14, 12, 9, 0, 30] | -0.07476731304340808 | -0.0023364785326065026 | 0.1875 | 0.29520140726575567 | 0.32737998345851277 |
| 9 | [24, 24, 15, 7, 2, 12] | -0.08063787241176049 | -0.002519933512867515 | 0.21875 | 0.28246929136975096 | 0.35604953991400484 |
| 10 | [17, 4, 31, 6, 2] | -0.05438069199493413 | -0.0016993966248416913 | 0.25 | 0.2199096201496736 | 0.3220215922448087 |
| 11 | [17, 22] | -0.02129306787892871 | -0.0006654083712165222 | 0.4375 | 0.20750419805982967 | 0.20795189901394973 |
| 12 | [3, 18, 19] | -0.06128325725707098 | -0.0019151017892834682 | 0.1875 | 0.2511501402275015 | 0.32235346854420727 |
| 13 | [23, 26, 0] | -0.05193964255019715 | -0.0016231138296936608 | 0.25 | 0.27739807708644076 | 0.267650202289912 |
| 14 | [26, 15, 19, 20, 13, 18] | -0.057208201927775114 | -0.0017877563102429725 | 0.1875 | 0.24130331301083258 | 0.3179598870755009 |
| 15 | [31, 9] | -0.022485407845663396 | -0.0007026689951769811 | 0.5 | 0.20916142805259239 | 0.21258050880843418 |

### SI-27 treatment (composition "movement-composition", λ = 1.0) — blob `sha256:1bb277ab40b9aa53a6218d971e46f29b261ba2bb5f7932ffb98e3951f372b637`, candidate norm 0.2969693161862222

**train partition** (allowed inputs; 32 train sequences): aggregate inner product -2.687755446349435, mean p -0.00262476117807562, sign agreement 0.2763671875, ‖m1‖ 1.4131161397867473, ‖m2‖ 2.2303160062528544, **maximum p (worst allowed input in this partition) -0.001378012707059647 at partition index 31, sequence [6, 29, 3, 9]**, top-4 mean p -0.001389653406875007 (top-4 indices [31, 15, 23, 7]).

| Index in partition | Sequence (allowed input) | Inner product m1·m2 | p(x) = inner/32 | Sign agreement | ‖m1‖ | ‖m2‖ |
|---|---|---|---|---|---|---|
| 0 | [4, 19, 17, 18, 12, 30] | -0.11769121418966504 | -0.0036778504434270324 | 0.3125 | 0.2966011006025265 | 0.4490991048832279 |
| 1 | [17, 10, 18, 12] | -0.12152582138866455 | -0.003797681918395767 | 0.125 | 0.2836062593174113 | 0.47851469305495103 |
| 2 | [25, 10, 10, 29, 5, 2] | -0.07803842172392703 | -0.0024387006788727196 | 0.09375 | 0.21873929349260565 | 0.4114613357526399 |
| 3 | [18, 6, 28, 20, 13, 22, 9, 22] | -0.04798330313566527 | -0.00149947822298954 | 0.40625 | 0.21357287049391846 | 0.27729206413057844 |
| 4 | [24, 23, 22, 9, 0, 14, 19] | -0.09380044931312322 | -0.002931264041035101 | 0.25 | 0.2495054965743626 | 0.42956283623378944 |
| 5 | [22, 21, 18, 12, 14, 0] | -0.08640346004039706 | -0.002700108126262408 | 0.40625 | 0.27319369440438357 | 0.3698749296531412 |
| 6 | [18, 18] | -0.08476302587105179 | -0.0026488445584703685 | 0.1875 | 0.2388840561039818 | 0.4130479494410985 |
| 7 | [30, 2, 6, 9] | -0.045096553639344936 | -0.0014092673012295295 | 0.46875 | 0.21596106614984828 | 0.2816915789169196 |
| 8 | [19, 17, 26, 24, 20, 30] | -0.11678974267475034 | -0.0036496794585859473 | 0.25 | 0.2919717297536117 | 0.45257659849260284 |
| 9 | [20, 13, 13, 20, 31, 12] | -0.12356363120909064 | -0.0038613634752840825 | 0.125 | 0.29018667792943115 | 0.4769799229045445 |
| 10 | [22, 29, 31, 5, 25, 5, 2] | -0.08038365420919538 | -0.0025119891940373557 | 0.03125 | 0.22340326621088447 | 0.4122972666062193 |
| 11 | [26, 22] | -0.04646251626591989 | -0.0014519536333099969 | 0.375 | 0.20970481457909182 | 0.27671914223256033 |
| 12 | [3, 19] | -0.09226222499747397 | -0.0028831945311710615 | 0.34375 | 0.24860966707704105 | 0.42409334286653133 |
| 13 | [0, 14, 0, 0] | -0.0886461567945153 | -0.002770192399828603 | 0.375 | 0.27699865400183726 | 0.3738329295022617 |
| 14 | [6, 23, 18, 18] | -0.08278805883338153 | -0.0025871268385431728 | 0.15625 | 0.2337032524305807 | 0.415224057722739 |
| 15 | [31, 24, 4, 26, 13, 27, 10, 9] | -0.04417734375679427 | -0.0013805419923998206 | 0.4375 | 0.21045552281347335 | 0.2803883595071242 |
| 16 | [8, 18, 3, 30] | -0.11935496934325947 | -0.003729842791976858 | 0.375 | 0.29847131182567754 | 0.4513743805448149 |
| 17 | [3, 24, 12] | -0.11739738621542209 | -0.00366866831923194 | 0.15625 | 0.27706906126671604 | 0.47265628442222046 |
| 18 | [16, 2, 21, 29, 2] | -0.07842746832864379 | -0.0024508583852701185 | 0.03125 | 0.2210387373570659 | 0.4083585056638185 |
| 19 | [4, 5, 1, 28, 31, 9, 22] | -0.046342330799895685 | -0.00144819783749674 | 0.34375 | 0.21327071180271867 | 0.2716635705946684 |
| 20 | [29, 5, 19] | -0.09259019054502363 | -0.0028934434545319883 | 0.28125 | 0.2480749304725319 | 0.42623202393313553 |
| 21 | [3, 31, 18, 31, 31, 0] | -0.0852139481294016 | -0.0026629358790438 | 0.40625 | 0.2702844243567621 | 0.3688075298987033 |
| 22 | [26, 19, 28, 11, 15, 0, 18] | -0.08380569617392045 | -0.0026189280054350144 | 0.15625 | 0.23607611618488403 | 0.4140644702879363 |
| 23 | [28, 9] | -0.044505332057952984 | -0.0013907916268110303 | 0.53125 | 0.21175210217611146 | 0.2832114906260487 |
| 24 | [8, 26, 8, 5, 30, 15, 10, 30] | -0.11775121221711712 | -0.00367972538178491 | 0.28125 | 0.2927373388435464 | 0.4541426804679301 |
| 25 | [20, 15, 23, 12] | -0.12117024122226948 | -0.0037865700381959216 | 0.15625 | 0.28562845193290143 | 0.4738422474019884 |
| 26 | [14, 2] | -0.0751772208329424 | -0.00234928815102945 | 0.125 | 0.21807797165824344 | 0.4042537606166153 |
| 27 | [5, 3, 20, 0, 8, 17, 27, 22] | -0.04670710768710539 | -0.0014595971152220437 | 0.375 | 0.21158347622062923 | 0.2745279490517166 |
| 28 | [0, 15, 30, 19, 3, 19] | -0.09364251143134267 | -0.0029263284822294584 | 0.34375 | 0.24735272397838892 | 0.4317981033187759 |
| 29 | [29, 7, 3, 4, 7, 0] | -0.08753654625408754 | -0.002735517070440236 | 0.375 | 0.27296650742775436 | 0.3705661728366745 |
| 30 | [23, 24, 20, 5, 6, 25, 9, 18] | -0.08366130044218167 | -0.0026144156388181773 | 0.1875 | 0.23822149240072643 | 0.4107473578692595 |
| 31 | [6, 29, 3, 9] | -0.044096406625908705 | -0.001378012707059647 | 0.375 | 0.21504993382191423 | 0.27845310203383894 |

**held_out partition** (allowed inputs; 16 held-out sequences, reported only): aggregate inner product -1.3460015097625617, mean p -0.0026289091987550034, sign agreement 0.28125, ‖m1‖ 0.9997655382584564, ‖m2‖ 1.57723836245857, **maximum p (worst allowed input in this partition) -0.0013635384980049186 at partition index 15, sequence [31, 9]**, top-4 mean p -0.0014092771637171442 (top-4 indices [15, 11, 3, 7]).

| Index in partition | Sequence (allowed input) | Inner product m1·m2 | p(x) = inner/32 | Sign agreement | ‖m1‖ | ‖m2‖ |
|---|---|---|---|---|---|---|
| 0 | [25, 30] | -0.11675867894081648 | -0.003648708716900515 | 0.28125 | 0.29230395465794035 | 0.4524389387729271 |
| 1 | [17, 2, 4, 12] | -0.12003326971246675 | -0.003751039678514587 | 0.125 | 0.281730472467715 | 0.47599795108366577 |
| 2 | [29, 2] | -0.07768255758857624 | -0.002427579924643008 | 0.0625 | 0.22245871127522943 | 0.4067051389704285 |
| 3 | [30, 28, 22] | -0.04543557287814813 | -0.0014198616524421289 | 0.28125 | 0.20667754516337455 | 0.2732668776120363 |
| 4 | [23, 17, 21, 23, 17, 0, 24, 19] | -0.09164886491328106 | -0.002864027028540033 | 0.3125 | 0.2478540679470391 | 0.42305339394366953 |
| 5 | [4, 0, 0] | -0.08994790720214829 | -0.002810872100067134 | 0.46875 | 0.2781382798176755 | 0.3766159490560681 |
| 6 | [19, 23, 21, 24, 18] | -0.08380930499248952 | -0.002619040781015297 | 0.1875 | 0.23705973326532234 | 0.41197784934036275 |
| 7 | [11, 30, 19, 24, 9] | -0.04627207195773735 | -0.0014460022486792923 | 0.4375 | 0.21718941757144988 | 0.2842610784915616 |
| 8 | [14, 12, 9, 0, 30] | -0.11760490976843546 | -0.003675153430263608 | 0.25 | 0.29520140726575567 | 0.44926683462966016 |
| 9 | [24, 24, 15, 7, 2, 12] | -0.12081795751973726 | -0.0037755611724917893 | 0.21875 | 0.28246929136975096 | 0.4776587788070195 |
| 10 | [17, 4, 31, 6, 2] | -0.07906236113532952 | -0.0024706987854790476 | 0.03125 | 0.2199096201496736 | 0.4115974191857468 |
| 11 | [17, 22] | -0.04504660018375159 | -0.0014077062557422372 | 0.375 | 0.20750419805982967 | 0.27195700398822764 |
| 12 | [3, 18, 19] | -0.09344939048426337 | -0.0029202934526332304 | 0.34375 | 0.2511501402275015 | 0.42486933424184553 |
| 13 | [23, 26, 0] | -0.08907596669751486 | -0.002783623959297339 | 0.4375 | 0.27739807708644076 | 0.3749842576586635 |
| 14 | [26, 15, 19, 20, 13, 18] | -0.08572286385170838 | -0.002678839495365887 | 0.15625 | 0.24130331301083258 | 0.4147955642563202 |
| 15 | [31, 9] | -0.0436332319361574 | -0.0013635384980049186 | 0.53125 | 0.20916142805259239 | 0.28004275838448633 |

### Phase 0 readings

- **Every allowed input already opposes, under both
  candidates, on both partitions.** The maximum `p` is
  negative everywhere (control train −0.000688, treatment
  train −0.001378; control held-out −0.000665, treatment
  held-out −0.001364). There is no allowed input on which
  the two movements align in `p`; the tail question is one
  of degree, not of sign.
- **The worst allowed input opposes only about half as
  strongly as the mean.** Control train: worst `p`
  −0.000688 against a mean of −0.001648 (42%). Treatment
  train: worst `p` −0.001378 against a mean of −0.002625
  (53%); its top-4 mean (−0.001390) is likewise about half
  its mean. Held-out shows the same shape under both
  candidates. The mean penalty *did* improve the worst
  allowed input (it roughly doubled, −0.000688 →
  −0.001378, as the mean deepened) — but it left the tail
  proportionally weaker than the bulk, so a worst-case
  reduction still has a measured, distinct target on
  allowed inputs: the gap between the tail and the mean
  that the mean objective does not price.
- **The per-input distribution is clustered, not smooth.**
  On the train partition the sorted `p` values fall in
  visible clusters (treatment: eight inputs near −0.0014,
  eight near −0.0025, eight near −0.0027, eight near
  −0.0037), and the top-4 values are extremely close
  (treatment: −0.001378, −0.001381, −0.001391, −0.001409).
  Membership within the weakest cluster can be expected to
  shuffle under small factor changes while the priced
  quantity (the top-k mean) barely moves — a stability
  property the gradient check reports on directly
  (§6.3 criterion, below).

### The §6.2 form choice: top-k mean, k = 4 (made on Phase 0's evidence)

**Chosen: the top-k mean, `W` = mean of the k = 4 largest
per-input `p(x)` values over the 32 training sequences** —
RFC §6.2's recommended form, restated here with the
Phase-0-grounded argument:

- Phase 0 shows `p`'s scale is ~1e-3 (values from −0.00069
  to −0.0039 across both candidates and partitions). A
  log-sum-exp term at any temperature τ of natural scale
  (e.g. τ = 1.0, or even τ = 0.1) would be, at this scale,
  numerically indistinguishable from the *mean* plus a
  constant — the very statistic F2 falsified. A τ small
  enough to price the tail (τ ~ 1e-3) would be a tuned
  scale fitted to `p`'s observed magnitude — exactly the
  kind of fitted dial §16 forbids, and its "worst case"
  would be a soft claim the labelling rules would have to
  carry carefully (§6.2).
- The top-k mean prices the tail directly, with no scale
  parameter: k = 4 of 32 is large enough that the gradient
  is a combination of several inputs' cached hidden states
  rather than a single input's (the weakest cluster has
  eight members, so k = 4 sits inside it), and small
  enough that the term prices the tail, not the bulk.
  Its piecewise-smooth gradient is the mean of the
  selected inputs' existing SI-27 fixed-matrix gradients
  — no new backward machinery (§6.3).
- The log-sum-exp form is therefore **not** implemented;
  τ appears nowhere in this slice and no τ sweep exists
  to report.

### Which slice this turned out to be (§6.7)

Phase 0 does **not** undercut the slice's premise in the
strong form §18 item 4 of the RFC worried about: the
SI-27 treatment's worst allowed input does *not* oppose
as strongly as its mean (it opposes about half as
strongly), so the premise does not narrow to "already
worst-case-healthy on allowed inputs". The honest
narrowing Phase 0 *does* impose is recorded here: the
mean penalty already improved the worst allowed input
along with the mean, so SI-29 is a measurement of whether
pricing the tail *directly* — rather than incidentally —
changes anything where it matters, against a tail that is
weaker than the bulk but already opposing everywhere.
That is the slice that was built; the decisive readings
below are read against exactly that premise.

### The pre-registered F3 reading rule (§§5.5, 6.6 — stated before the decisive numbers)

RFC §6.6's F3′ turns on a qualitative boundary ("the
destination no longer approached") that §5.5 requires
this slice to state in advance, in the held-out record's
own terms. **The rule, registered here ahead of every
decisive figure in this document and applied as written
in the falsification readings:**

> The decisive candidate's destination counts as
> **approached** (F3′ does **not** fire) iff ALL THREE of
> the following hold for its signal held-out record
> (signal `tiny-predecessor-lean-001`, head `d0001`,
> target source `base`, anchoring `base-anchored`,
> composition `"worst-input-composition"`):
> (a) the held-out signal loss **falls** —
> `held_out_loss_after < base_held_out_loss`;
> (b) at least **12 of 16** held-out examples are closer
> after training (`held_out_examples_closer >= 12`, a
> stated fraction of 3/4); and
> (c) the held-out residual is at most **2× the re-run
> SI-27 control's residual** — completeness relative to
> the `"movement-composition"` control re-run inside
> this slice (its published residual is
> 0.0005227812955486338, so the boundary is
> 0.0010455625910972676).
> If any one of (a)–(c) fails, the destination counts as
> **traded away** and F3′ **is measured**.

## The worst-input term's two terms, separately (§6.2/§17 item 1)

Decisive cycle 2 (signal `tiny-predecessor-lean-001`, head
`d0001`, target source `base`, anchoring `base-anchored`,
composition `worst-input-composition`, λ_w = 1.0, k = 4).
Training numbers — labelled with signal, head, target
source, anchoring, and composition configuration wherever
they appear; the term's train values are optimised, its
held-out values are reported only and never entered the
loss:

| Term (train partition, optimised) | At initialisation | End of training |
|---|---|---|
| Signal MSE | 0.0042998498036725735 | 0.0006509292305944755 |
| Worst-input term `W` (top-4 mean `p`) | +0.0004697766937776053 | **−0.0023225615446231264** |
| Total (signal + λ_w·W) | 0.004769626497450179 | **−0.001671632314028651** |

| Term (held-out partition, reported only) | At initialisation | End of training |
|---|---|---|
| Worst-input term `W` (top-4 mean `p`) | +0.00037773328307693754 | −0.0023250720244677755 |

Both properties the signed term carries are measured, not
hidden: the total loss ends **negative** (driven by the
worst-input term, as SI-27's total was driven by its
penalty), and the term *fights* the signal loss — the
worst-input candidate's signal train residual
(0.0006509292305944755) is ~69× the SI-25 control's
(9.412649442227854e-06) and ~1.25× SI-27's treatment's
(0.0005191352673009495). The destination record is read
under the pre-registered rule in the falsification
readings.

**The worst train inputs, by train-partition identity
(allowed inputs, named as such — §17 items 18/20).** At
initialisation the term's top-4 set on the train partition
is indices **[12, 28, 4, 20]**, its single worst train
input index **12**, sequence **[3, 19]**. At the end of
training the top-4 set is **[27, 11, 19, 26]**, its single
worst train input index **27**, sequence
**[5, 3, 20, 0, 8, 17, 27, 22]** — the input whose
record appears in the treatment's allowed-input record
below (`max_p` −0.0023129050472012187). The top-k set
turned over completely between initialisation and the end
of training: the selection-oscillation dynamics RFC §6.3
/ §18 item 2 named as an open question are **measured
present** in membership (no input is in both sets), while
the priced quantity itself moved monotonically in the
published endpoints' terms (+0.000470 → −0.002323) and the
run is deterministic (criterion 6). Membership turnover
is reported as a dynamics observation; nothing in this
slice smooths it away or claims it away.

**Coincidence configuration (§15 criterion 5).**
Configured with signal 2, base-anchoring, and the
worst-input configuration, a first cycle at base records
the new term at initialisation and at the end of training,
on both partitions: **identically zero** (measured —
`m1` is the zero vector at base, so every `p` is 0.0 and
the top-k mean of zeros is 0.0; the recorded top-k set at
initialisation is the deterministic tie-break [0, 1, 2, 3]),
the totals equal the signal losses, and the run reproduces
SI-25's/SI-27's at-base run **exactly** — candidate blob
`sha256:131cc242728024b4f833bc78ea627825f1e8616972fc2670603673e1814645f4`,
identity drift 0.16672966815531254, capability regression
0.2418281203135848, ADMIT → `d0001`, §6.6 VERIFY pass. The
worst-input machinery is therefore proven correct where it
must vanish, before the decisive cycle is trusted. These
are at-base facts; they are not the slice's decisive claim.

## Gradient check for the worst-input loss (§6.3, §15 criterion 3)

Central finite differences of signal 2's **base-anchored,
worst-input** loss — base-derived targets, head hidden
states and readout, landed `m1` on the train partition —
through `worst_input_factor_loss` (the float32 shared fold
plus the existing forwards), at signal 2's pinned initial
factors over an 8-example subset, `eps = 0.01`, EVERY one
of the 512 factor entries:

| Configuration | Worst absolute disagreement | Tolerance |
|---|---|---|
| Signal 2 base-anchored + worst-input at base (new code path) | **5.5249277028738815e-08** | 1e-5 (SI-19's pin, unchanged) |
| Signal 2 base-anchored + worst-input at head `d0001` (the decisive configuration) | **5.13570189768905e-08** | 1e-5 |

Both pass with ~180× headroom. The at-base figure is
identical to SI-25's/SI-27's at-base figure, as it must
be: at base the worst-input loss coincides with the
unpenalised loss — and the coincidence is measured under
the new code path (worst term exactly 0.0 at the pinned
factors; worst-input gradients array-equal to the
unpenalised gradients), not asserted from the derivation.
No new backward machinery was needed (§6.3: on a fixed
top-k set the term's gradient is the mean of the selected
inputs' fixed-matrix gradients), and no tolerance was
loosened.

**Top-k membership stability at the check points
(§6.3, reported with the figures).** On the 8-example
check subset the top-k set at the pinned factors is
[0, 1, 2, 3] at base (every `p` is 0.0 there; the set is
the deterministic index tie-break) and [4, 5, 2, 3] at
head `d0001` (subset `W` = 0.0003049183564004984).
Recomputing the set under **every** single-entry ±eps
perturbation of the factors (1,024 perturbations per
check point) changes membership in **0 of 1,024** cases
at base and **0 of 1,024** at `d0001`: both check points
sit inside a locally fixed-top-k region, so the finite
differences above price the worst-input machinery
itself, not a membership jump. On the full 32-sequence
train partition at `d0001` the pinned top-k set is
[12, 28, 4, 20] — the decisive run's recorded initial
set — with a margin of 0.00017773181992969926 between
the 4th- and 5th-largest `p`, and it is likewise
unchanged under all 1,024 single-entry ±eps
perturbations. Membership *does* turn over across the
full training run (the initial and final top-k sets are
disjoint — see the term-values section); stability here
is a statement about the check points only, and is not
claimed for the trajectory.

## Probe-blindness of training (§15 criterion 4)

Three assertions, all in `tests/test_worst_input_composition.py`:

- A spy on `analytic_gradients_worst_input` during a real
  worst-input `train_candidate` run captures every call's
  example set and `m1` key set: each is **exactly signal
  2's train partition** (as a set) — the term's input set
  is the train partition, asserted as a set, on the code
  path that trained the decisive candidate (which the spy
  run reproduces: blob `sha256:11cab02e…b8db5`).
- The train partition's disjointness from both probe
  suites (identity and capability) and from the signal's
  held-out partition is re-asserted under the new code
  path.
- A copy of the bundle whose `probe_suite.json` was
  perturbed on disk (probe inputs and expected logits
  replaced) produces a **byte-identical candidate** through
  `train_candidate` called directly with the
  pre-perturbation verified manifest — the training path
  never opens the probe artifact, so no probe input, probe
  logit, or probe-derived movement can reach the loss, the
  worst-input term, factor initialisation, or selection.
  (`load_bundle` itself refuses the perturbed copy —
  fail-closed tamper evidence, a separate, pre-existing
  guarantee.)

## The allowed-input worst-input records (§15 criteria 2 and 8)

For the re-run **SI-25 control** candidate (blob
`sha256:646d1e8b…d675`), the re-run **SI-27 control**
candidate (blob `sha256:1bb277ab…b637`), and the
**treatment** candidate (same decisive labels, composition
`"worst-input-composition"` — blob `sha256:11cab02e…b8db5`):
the worst-input record under SI-27's movement definitions
on signal 2's train and held-out partitions. Per-input
rows for the two existing candidates are published in
Phase 0 above (the same records, computed before the
treatment existed); the treatment's aggregates and
worst-case figures are:

| Record (partition labelled) | Aggregate inner product | Sign agreement | Mean `p` | Max `p` (worst allowed input) | Top-4 mean `p` (= `W`) | ‖m1‖ | ‖m2‖ |
|---|---|---|---|---|---|---|---|
| SI-25 control, train | −1.6873837019991191 | 0.267578125 (274/1024) | −0.0016478356464835148 | −0.0006878804284308964 (index 19, seq [4, 5, 1, 28, 31, 9, 22]) | −0.0006952966321423392 | 1.4131161397867473 | 1.6765305214306858 |
| SI-25 control, held-out | −0.845596374259425 | 0.26953125 (138/512) | −0.0016515554184754395 | −0.0006654083712165222 (index 11) | −0.0006984553203186485 | 0.9997655382584564 | 1.1850839719703719 |
| SI-27 control, train | −2.687755446349435 | 0.2763671875 (283/1024) | −0.00262476117807562 | −0.001378012707059647 (index 31, seq [6, 29, 3, 9]) | −0.001389653406875007 | 1.4131161397867473 | 2.2303160062528544 |
| SI-27 control, held-out | −1.3460015097625617 | 0.28125 (144/512) | −0.0026289091987550034 | −0.0013635384980049186 (index 15) | −0.0014092771637171442 | 0.9997655382584564 | 1.57723836245857 |
| **SI-29 treatment, train** | **−2.4928287954527795** | 0.2314453125 (237/1024) | **−0.002434403120559355** | **−0.0023129050472012187 (index 27, seq [5, 3, 20, 0, 8, 17, 27, 22])** | **−0.002322561516640352** | 1.4131161397867473 | 2.1423496133427675 |
| **SI-29 treatment, held-out** | **−1.2470211966092322** | 0.255859375 (131/512) | **−0.002435588274627406** | **−0.0022598436988714773 (index 3, seq [30, 28, 22])** | **−0.002325071873910509** | 0.9997655382584564 | 1.5154484262995547 |

Readings, stated plainly:

- **On the worst allowed inputs, the term worked —
  decisively.** The treatment's worst train input
  (−0.002313) opposes ~3.4× as strongly as the SI-25
  control's (−0.000688) and ~1.7× as strongly as the
  SI-27 control's (−0.001378); its top-4 mean `W`
  (−0.002323) is likewise far below both controls'. The
  same holds on the held-out partition (reported only).
  The treatment's tail is now nearly as opposing as its
  mean (max `p` −0.002313 against mean −0.002434, 95% —
  against 53% for SI-27's treatment in Phase 0): pricing
  the tail directly closed the tail–bulk gap Phase 0
  measured, on the population training may read.
- **The price shows in the bulk.** The treatment's mean
  and aggregate on allowed inputs are slightly *less*
  opposing than SI-27's treatment's (train mean
  −0.002434 vs −0.002625; aggregate −2.493 vs −2.688) —
  worst-case pressure concentrated on the tail, exactly
  the trade the form makes, published rather than
  smoothed.
- **Cross-check:** the record's forward-path top-4 mean
  for the treatment equals the training run's analytic
  worst-input values (train −0.0023225615446231264,
  held-out −0.0023250720244677755) to ~3e-11 — two
  independent paths to the same numbers. `m1`'s norms are
  identical under all three candidates on both
  partitions, as they must be: `m1` is a property of the
  landed head, not of the candidate.

No figure above is a probe figure, and no probe figure
below is an allowed-input figure (§17 item 18). A worst
*allowed input* and the worst *probe* are different
quantities over disjoint populations (§17 item 20) — the
gap between this section and the diagnostics below is
exactly that difference, stated in both places.

## The decisive two-cycle run (measured on this VM, this slice's runs)

Driver shape: **one reused `TinyTransformerMind`** driven
through `si.run.run_cycles` — one `run_cycles` call for
cycle 1 (signal 1, default anchoring, default composition
configuration), then the driver explicitly reconfigures
the same mind to signal 2 (`set_training_signal`), to
base-anchored targets (`set_target_anchoring`), **and**
to the worst-input composition configuration
(`set_composition_config`) — three recorded selections —
then a second `run_cycles` call for cycle 2. Exactly two
authored cycles were attempted, and exactly one
worst-input decisive run was performed. This is SI-27's
decisive run with exactly one configuration changed:
cycle 2's composition configuration.

| | Cycle 1 | Cycle 2 (treatment) |
|---|---|---|
| Signal (recorded selection) | `tiny-successor-lean-001` (default) | `tiny-predecessor-lean-001` (explicit) |
| Anchoring (recorded selection) | `weights-under-training` (default) | **`base-anchored` (explicit)** |
| Composition (recorded selection) | `none` (default), λ 0.0 / λ_w 0.0, composition fields null | **`worst-input-composition` (explicit), λ_w 1.0, k 4** |
| Head trained against | `base` | `d0001` |
| Target-source head | `base` | **`base`** |
| Stack depth at training | 0 | 1 |
| Candidate blob | `sha256:2761e8ff…6ca9e` (2,360 B) | `sha256:11cab02e…b8db5` (2,360 B) |
| Candidate norm | 0.23555380990005192 | 0.28252038537434354 |
| Signal train loss, initial → final | 0.0028961365437885785 → 1.824004756500067e-06 | 0.0042998498036725735 → 0.0006509292305944755 |
| Worst-input term (train), initial → final | not measured (configuration `"none"`) | +0.0004697766937776053 → −0.0023225615446231264 |
| Pre-training held-out signal loss | 0.00140625 | **0.002757805049077469** (genuine measurement, as in SI-25) |
| Held-out signal loss after training | 3.2118398935406443e-06 | 0.0006602814776996296 (~4.2× lower) |
| Held-out examples improved | 16/16 | **12/16** |
| Norm-guard activations | 0 | 0 |
| Gate verdict | **ADMIT** → `d0001`, §6.6 VERIFY pass | **REJECT** (identity), nothing landed |
| Blob-store growth | +2,360 B | +0 B (candidate never written) |

Full candidate blob hashes: cycle 1
`sha256:2761e8ffc139ce3e107c11da035e8a73c20792945a494f4e493dccf418b6ca9e`;
cycle 2
`sha256:11cab02e560e3661ea419fa0c8b6a95de25d3c7fb1ee23ec52ec8462a36b8db5`.
Cycle 1 reproduces SI-19/SI-21/SI-23/SI-25/SI-27
**exactly** (criterion 9). Both controls re-run inside
this slice reproduce their decisive candidates **exactly**
(SI-25: blob `sha256:646d1e8b…d675`, drift
0.3329036422073841, regression 0.10096919978968799, under
`"none"`; SI-27: blob `sha256:1bb277ab…b637`, drift
0.32509003952145576, regression 0.14536726917140186, under
`"movement-composition"` with λ = 1.0), so the treatment's
difference is attributable to the reduction over inputs
and not to the refactor that introduced it.

## Cumulative vs incremental accounting (§15 criterion 8) — side by side

| Measurement | Cycle 1 | Cycle 2 (treatment) | Accounting |
|---|---|---|---|
| Identity drift (gate, candidate overlay) | 0.17189218290150166 (pass, tol 0.25) | **0.31215621903538704 (FAIL, tol 0.25)** | **seed-absolute / cumulative** |
| Capability regression (gate) | 0.1657181391492486 (pass, eps 0.25) | 0.11724743689410388 (pass, eps 0.25) | **head-incremental** |
| Echo accuracy | 1.0 (16/16, floor 0.9375) | 1.0 (16/16, floor 0.9375) | absolute, vs the recorded floor |
| Landed head's cumulative identity drift after the cycle | 0.17189218290150166 (§6.6 VERIFY, measured-pass) | 0.17189218290150166 (unchanged — re-measured after the REJECT) | seed-absolute, of the head that actually landed |

No incremental figure above is presented as cumulative or
vice versa, no base-anchored figure as a head-relative
one, and no figure under one composition configuration as
a figure under another (§17 items 9, 15, 17).

## The diagnostics (published whatever the verdict)

All diagnostics below are computed against head `d0001`
with the decisive cycle-2 candidate overlaid. They are
published to *attribute* the gate's verdict after the gate
has spoken; they are not gate checks and influenced no
verdict (§17 item 12). Per-probe drift values are
measurements (max absolute logit coordinate per probe);
the incremental movement column is a **derivation** from
two measured forwards (composed forward minus the `d0001`
forward), labelled as such. Candidate norm and ‖m2‖ stand
beside every drift figure (§§5.3, 10 of the RFC): the
treatment's candidate norm is 0.28252038537434354 and its
probe-space ‖m2‖ is 0.9424814001349618.

**(i) Per-probe drift contributions (treatment):**

| Probe | `d0001` drift (measured) | Signal-2 worst-input incremental movement (derivation) | Composed drift (measured) |
|---|---|---|---|
| tiny-id-0 | 0.17189218290150166 | 0.1409909427165985 | **0.31215621903538704** |
| tiny-id-1 | 0.15088725090026855 | 0.07746927067637444 | 0.1174006462097168 |
| tiny-id-2 | 0.11858986946754158 | 0.14978677220642567 | 0.14997844584286213 |
| tiny-id-3 | 0.14782869815826416 | 0.3379737436771393 | 0.19014504551887512 |
| tiny-id-4 | 0.11720322235487401 | 0.16256514564156532 | 0.11752748489379883 |
| tiny-id-5 | 0.09891846403479576 | 0.10969258844852448 | 0.13741302862763405 |
| tiny-id-6 | 0.12363102938979864 | 0.1797015145421028 | 0.15289807319641113 |
| tiny-id-7 | 0.08795571327209473 | 0.16283829044550657 | 0.19354498013854027 |

Worst probes, named: `d0001`'s own drift — **tiny-id-0**
(0.17189218290150166); the composed drift — **tiny-id-0**
(0.31215621903538704, equal to the gate's measured cycle-2
drift). The incremental movement's largest probe is
tiny-id-3 (0.3379737436771393). At tiny-id-0 the composed
drift is again, to the printed digits, the sum of the two
contributions (0.17189218290150166 +
0.1409909427165985 ≈ 0.3122) — the aligned extreme, for
the fourth decisive run in a row at the probe where cycle
1 moved most. The worst *probe* is tiny-id-0; the worst
*allowed input* is train index 27 — different populations,
different quantities, never shortened to the same words
(§17 item 20).

**(ii) Alignment diagnostic (derivation):** `m1` =
(`d0001` forward − seed expected) and `m2` = (composed
forward − `d0001` forward), per identity probe:

- **In aggregate the treatment opposes, but less than
  SI-27's treatment did.** Over all 8 identity probes
  (256 coordinates): inner product
  **−0.28751472160662706** (opposing — more than SI-25's
  control's −0.21256230517288188, less than SI-27's
  treatment's −0.3271477846470207; SI-23's was
  +0.07637087369869666, aligned); sign agreement
  **0.38671875** (99/256) — published beside the
  magnitude-weighted figure, never alone (§17 item 19);
  movement-vector norms ‖m1‖ = 0.7119678353441855
  (unchanged — cycle 1's movement), ‖m2‖ =
  0.9424814001349618 (SI-25: 0.7816722548913912; SI-27:
  0.9688074699782941).
- **At the binding probe it fails, worse than both
  composition controls.** At `tiny-id-0`: inner product
  **+0.04944786620153699** (positive — aligned; SI-25
  control: +0.0338873784299374; SI-27 treatment:
  +0.03801222744105431). The worst-input term improved
  composition at the worst *allowed* input decisively —
  and at the one *probe* where the gate binds, the
  alignment increased again, past both composition
  predecessors.

**(iii) Folded updates (measured):** Frobenius inner
product of the two folded `lm_head.weight` updates
**−0.0173962867164366** (opposing in weight space; SI-25:
−0.011457914936582613; SI-27: −0.02013451284871203); norms
‖Δ1‖_F = 0.23555380990005192, ‖Δ2‖_F = 0.28252038537434354
(Δ2's norm equals its candidate norm, as it must for a
single-target LoRA). The treatment's movement is smaller
than SI-27's penalised movement (candidate norm 0.28252
vs 0.29697; probe ‖m2‖ 0.94248 vs 0.96881) and larger than
SI-25's control's (0.25155 / 0.78167): the composed
maximum's improvement on both composition predecessors
(0.31216 vs 0.33290 / 0.32509) is therefore **not** an
improvement by turning at the binding probe — the
binding probe's alignment worsened — and it is only
partly an improvement by shrinking — and, at the
binding probe, an improvement by a smaller step rather
than by turning: the treatment's incremental movement
at tiny-id-0 is smaller than SI-27's (0.14099 vs
0.15392), so the composed sum there is smaller despite
the movements aligning more (overall the movement also
shrank modestly against SI-27: candidate norm −4.9%,
probe ‖m2‖ −2.7%, while the maximum fell −4.0%).
§5.3's distinguishability obligation is discharged by
those figures standing together.

**(iv) Composed capability (measured):** head-incremental
regression **0.11724743689410388** ≤ eps 0.25 (pass;
SI-25: 0.10096919978968799; SI-27: 0.14536726917140186);
echo accuracy **1.0 (16/16)** vs the recorded floor
0.9375.

**(v) The four-run control/treatment table (§14 item 8,
§17 item 16).** Same signal, same head, same target, same
hyperparameters — differing only in anchoring (SI-23 vs
SI-25) and then only in composition configuration (SI-25
vs SI-27 vs SI-29). Each figure under its own run's
labels:

| Per-probe composed drift | tiny-id-0 | tiny-id-1 | tiny-id-2 | tiny-id-3 | tiny-id-4 | tiny-id-5 | tiny-id-6 | tiny-id-7 | Max (worst probe) |
|---|---|---|---|---|---|---|---|---|---|
| SI-23 — signal 2, weights-under-training, composition n/a (pre-SI-27) | 0.2092923242598772 | 0.15134692192077637 | 0.14906740188598633 | **0.2963416576385498** | 0.23394920490682125 | 0.14004023000597954 | 0.1332004750147462 | 0.2018739078193903 | 0.2963416576385498 (tiny-id-3) |
| SI-25 — signal 2, base-anchored, composition `"none"` | **0.3329036422073841** | 0.09286642074584961 | 0.14936351776123047 | 0.1489148810505867 | 0.14728415571153164 | 0.11245626583695412 | 0.14491976890712976 | 0.14189285784959793 | 0.3329036422073841 (tiny-id-0) |
| SI-27 — signal 2, base-anchored, composition `"movement-composition"`, λ 1.0 | **0.32509003952145576** | 0.09391975402832031 | 0.1488508302718401 | 0.14755446463823318 | 0.10617947578430176 | 0.1450248844921589 | 0.15063166618347168 | 0.18760260567069054 | 0.32509003952145576 (tiny-id-0) |
| SI-29 — signal 2, base-anchored, composition `"worst-input-composition"`, λ_w 1.0, k 4 | **0.31215621903538704** | 0.1174006462097168 | 0.14997844584286213 | 0.19014504551887512 | 0.11752748489379883 | 0.13741302862763405 | 0.15289807319641113 | 0.19354498013854027 | 0.31215621903538704 (tiny-id-0) |

| Gate / candidate figure | SI-23 (head-relative) | SI-25 (base-anchored, `"none"`) | SI-27 (base-anchored, mean-penalised) | SI-29 (base-anchored, worst-input) |
|---|---|---|---|---|
| Candidate blob | `sha256:184cc7f0…6e6cf7` | `sha256:646d1e8b…d675` | `sha256:1bb277ab…b637` | `sha256:11cab02e…b8db5` |
| Candidate norm | 0.21317186599953494 | 0.2515524481203623 | 0.2969693161862222 | 0.28252038537434354 |
| Identity drift (cumulative) | 0.2963416576385498 — FAIL | 0.3329036422073841 — FAIL | 0.32509003952145576 — FAIL | 0.31215621903538704 — FAIL |
| Capability regression (head-incremental) | 0.15023326850496233 — pass | 0.10096919978968799 — pass | 0.14536726917140186 — pass | 0.11724743689410388 — pass |
| Echo accuracy | 1.0 — pass | 1.0 — pass | 1.0 — pass | 1.0 — pass |
| Aggregate probe inner product | +0.07637087369869666 | −0.21256230517288188 | −0.3271477846470207 | −0.28751472160662706 |
| Inner product at own worst probe | (SI-23 doc: aligned at tiny-id-3) | +0.0338873784299374 (tiny-id-0) | +0.03801222744105431 (tiny-id-0) | +0.04944786620153699 (tiny-id-0) |
| Verdict | REJECT | REJECT | REJECT | REJECT |

Per probe, the treatment's composed drift improves on
SI-27's treatment at 2 of 8 probes (tiny-id-0,
tiny-id-5) and worsens at 6 (tiny-id-1, tiny-id-2,
tiny-id-3, tiny-id-4, tiny-id-6, tiny-id-7); against
SI-25's control it likewise improves at 2 (tiny-id-0,
tiny-id-4) and worsens at 6.
The maximum improves on both composition predecessors
(0.33290 → 0.32509 → 0.31216) and stays far above tol
0.25, and worsens on SI-23's maximum (0.29634). The
treatment's incremental movement exceeds cycle 1's
remaining headroom (0.07810781709849834) at **7 of 8**
probes (all but tiny-id-1, at 0.07746927067637444 —
just under). No diagnostic was uncomputable; nothing in
this set is reported "not evaluated".

**(vi) The falsification readings (RFC §6.6), each stated
as measured, under the rules registered in Phase 0:**

- **F1′ — the worst-case penalty does not transfer:
  MEASURED.** The treatment's worst-input composition on
  allowed inputs improves decisively against both
  controls (train max `p`: −0.0006878804284308964 →
  −0.001378012707059647 → **−0.0023129050472012187**;
  top-4 mean `W` likewise) while its probe composition
  at the worst probe is **worse** than SI-27's
  treatment's (+0.03801222744105431 →
  **+0.04944786620153699** at `tiny-id-0`). The
  train-to-probe gap swallows the worst-case term as it
  swallowed the destination argument in SI-25 and the
  mean term in SI-27 — now at the worst-case statistic
  itself. Falsified premise, in the RFC's words: that
  worst-case composition measured on allowed inputs
  shapes worst-case composition on judged inputs,
  through this readout, at this scale. RFC §18 item 1
  recorded this as the *expected* outcome; it is
  reported as a measurement, not as a surprise.
- **F2′ — the worst case transfers on allowed inputs and
  the judged maximum still does not move: NOT measured
  as defined; its defining aggregate condition fails,
  and the maximum's failure is measured beside it.**
  F2′ as written requires the treatment's aggregate
  probe composition to be *no worse than SI-27's
  treatment's*; measured, it is worse in aggregate
  (−0.28751472160662706 against −0.3271477846470207 —
  still opposing, still better than SI-25's control's
  −0.21256230517288188, but not no-worse than SI-27's).
  The reading therefore does not fire in its registered
  form, and this document does not round it into one.
  What *is* measured, separately: the worst allowed
  input opposes decisively (F1′'s first clause), and
  the composed maximum stays above tol 0.25
  (0.31215621903538704) — so the maximum's immobility
  under a worst-case objective is a fact of this run
  even though F2′'s conjunction is not.
- **F3′ — the destination is traded away: NOT measured
  under the pre-registered rule; the destination is
  approached, at the rule's boundary.** Applying the
  Phase-0 rule as written to the treatment's signal
  held-out record (signal
  `tiny-predecessor-lean-001`, head `d0001`, target
  source `base`, anchoring `base-anchored`, composition
  `"worst-input-composition"`): (a) the loss falls —
  0.002757805049077469 → 0.0006602814776996296 ✓;
  (b) closer fraction — **12/16**, exactly the rule's
  stated 3/4 boundary ✓ (SI-27's treatment: 16/16);
  (c) residual completeness — 0.0006602814776996296 ≤
  0.0010455625910972676 (2× the re-run SI-27 control's
  residual) ✓. All three hold, so F3′ does not fire:
  opposition was NOT achieved by abandoning the lean
  the cycle exists to author, under the boundary stated
  before the run. The approach is nonetheless the
  least complete of any composition treatment on
  record (~4.2× fall against the control's ~264× and
  SI-27's ~5.3×; residual ~63× the SI-25 control's and
  ~1.26× SI-27's; four held-out examples not closer,
  where SI-27's treatment had none) — a partial
  destination trade measured beside the reading, and
  any future reading of this run that quotes the
  worst-input improvement without this cost beside it
  is blending the record.

**Attribution:** all five cycle-2 REJECTs in the project's
record are now measured, and the composition line's
attribution has sharpened at each statistic. SI-27
proved the mean of `p(x)` on allowed inputs can be
shaped and does not bind the judged maximum. SI-29
proves the same — more strongly, because the experiment
was designed against the escape route SI-27 left open —
for the **worst case** of `p(x)` on allowed inputs: the
tail was priced directly, the tail moved decisively
(worst train `p` ~1.7× past SI-27's, top-4 mean likewise,
tail–bulk gap closed on allowed inputs), and the judged
maximum moved only from 0.32509 to 0.31216 while the
alignment at the binding probe *increased* to +0.04945,
the worst on record. By F2′'s registered logic's
neighbourhood — and by F1′ measured outright — the
remaining honest reading is the one RFC §6.6 wrote in
advance for this outcome: worst-case composition is not
a population-independent property of the movement pair
that any training-side statistic of `p(x)` on allowed
inputs has yet priced. That is a successful slice with
a negative experimental result, in the
SI-21/SI-23/SI-25/SI-27 pattern.

## VERIFY and unwind (§15 criterion 10)

- **Cycle 1** ADMITted and passed the §6.6 probe VERIFY
  exactly as in SI-19/SI-21/SI-23/SI-25/SI-27 (journaled
  `transformer-probe-verify`, measured-pass on both arms,
  echo 1.0).
- **Cycle 2** was REJECTed at the gate and never
  materialised, so there is no cycle-2 VERIFY and no
  VERIFY rollback — nothing landed to verify or roll
  back.
- **Unwind, honestly conditioned:** because cycle 2
  REJECTed, **no authored two-delta stack exists** —
  nothing is unwound, and no unwind is claimed. The
  bundle head after a decisive SI-29 run is `d0001`.
  SI-21's fixture-proven unwind remains the only unwind
  evidence. The transformer-squash precondition (an
  authored two-delta stack) therefore remains **unmet**
  after SI-29 as well.

## Acceptance criteria (RFC SI-28 §15) — status

1. ✅ **Recorded composition configuration, three named
   values** — exactly `"none"`, `"movement-composition"`,
   `"worst-input-composition"`; every other value refuses
   with the existing `ValueError` discipline at the
   constructor, the setter, the intent, and the
   `train_candidate` dispatch (asserted). Recorded
   selection asserted per cycle on both verdict paths:
   cycle 1's journaled note records `"none"`; cycle 2's
   records the full triple — signal 2,
   `target_anchoring = "base-anchored"`,
   `composition_config = "worst-input-composition"`,
   `composition_lambda_w = 1.0`,
   `composition_worst_k = 4` — with the worst term's
   values beside the signal loss's. λ_w and k are not
   independently settable (asserted: neither the
   constructor nor the setter takes such an argument);
   exactly one worst-input decisive run was performed
   (no sweep).
2. ✅ **Phase 0 published before the decisive run is
   trusted** — the per-input composition record for both
   existing candidates on both allowed partitions
   (per-input inner products, maximum per partition,
   sign agreements beside the magnitude-weighted
   figures, movement norms, beside the published
   aggregates), the §6.2 form choice (top-k mean, k = 4)
   stated with its Phase-0 argument, and the
   pre-registered F3 reading rule — all in the Phase 0
   section above, ahead of the decisive numbers.
3. ✅ **Gradient check for the worst-input loss, at the
   decisive head** — re-run at `d0001` AND at base under
   the new code path, over EVERY factor entry, tolerance
   1e-5 unchanged, both figures published above; top-k
   membership stability at the check points reported
   with them.
4. ✅ **Probe-blindness of training, asserted** — the
   worst-input term's input set asserted as a set
   (exactly the train partition, via a spy on the new
   gradient path during a real training run); the train
   partition's disjointness from both probe suites
   re-asserted under the new code path; a
   perturbed-probe-artifact bundle copy produces a
   byte-identical candidate.
5. ✅ **Coincidence at base** — signal 2 base-anchored +
   worst-input at base reproduces the established
   at-base run exactly (blob `sha256:131cc242…45f4`,
   drift 0.16672966815531254, regression
   0.2418281203135848), with the new term measured at
   initialisation and at the end of training on both
   partitions (identically zero).
6. ✅ **Determinism per cycle** — a full decisive run
   repeated from the seed reproduces both candidates'
   blobs/records, the landed delta record, and the
   journaled gate measurements exactly; hyperparams
   equal except the labelled `wall_time_seconds`. The
   worst-input `train_delta` alone is likewise
   byte-identical across repeats (asserted).
7. ✅ **Gate honesty per cycle** — cycle 2's journaled
   checks 3/4 equal an independent `ProbeContext`
   evaluation of the same candidate against `d0001`
   (abs 1e-12). Cycle 2's verdict is **REJECT, as
   measured**. No ADMIT was required: SI-29 succeeds by
   §6's claim — a worst-input candidate was genuinely
   authored against the landed head under the new
   configuration and genuinely judged, with its
   premise's falsification readings published either
   way.
8. ✅ **Cumulative accounting, diagnostics, the four-run
   table, and the falsification readings published** —
   the side-by-side accounting table, the full
   diagnostic set with candidate norm and ‖m2‖ beside
   every drift figure, the Phase-0 and decisive
   allowed-input worst-input records (worst allowed
   inputs named by train-partition identity), the
   treatment's signal held-out record read under the
   pre-registered F3 rule, the four-run table with each
   figure under its own run's labels, and the
   F1′/F2′/F3′ readings (F1′ measured; F2′ not measured
   as defined, its aggregate condition failing, with
   the maximum's immobility measured beside it; F3′
   not measured under the pre-registered rule, the
   destination approached at the rule's closer-fraction
   boundary) — all above, measurements and derivations
   labelled, worst probes named, nothing in the
   diagnostic set reported "not evaluated" that was
   computable.
9. ✅ **First-cycle and both-control reproduction** —
   cycle 1 reproduces SI-19/SI-21/SI-23/SI-25/SI-27
   exactly (blob `sha256:2761e8ff…6ca9e`, drift
   0.17189218290150166, regression 0.1657181391492486);
   the re-run `"none"` control reproduces SI-25's
   decisive candidate exactly (blob
   `sha256:646d1e8b…d675`, drift 0.3329036422073841);
   the re-run `"movement-composition"` control
   reproduces SI-27's decisive candidate exactly (blob
   `sha256:1bb277ab…b637`, drift 0.32509003952145576) —
   the configuration machinery proven
   behaviour-preserving for both existing semantics
   before the new configuration's results are trusted.
10. ✅ **VERIFY, unwind, and untouched scope** — cycle-1
    VERIFY as in SI-19; unwind honestly conditioned (no
    authored stack exists, none unwound, none claimed —
    see VERIFY and unwind). Exactly two authored cycles
    attempted in the decisive run and exactly one
    worst-input decisive run performed;
    `max_stack_depth` 8, `identity_tol` 0.25,
    `capability_eps` 0.25, `max_targets_per_delta` 1,
    `allow_additive` false and every other policy
    default unchanged (asserted); both signals' ids,
    streams, corrections, and partitions unchanged
    (asserted); both anchorings' and both prior
    composition configurations' definitions unchanged
    (asserted); `si squash` still fails closed on the
    resulting transformer stack (asserted; SI-12's suite
    green); the full existing suite is green (below);
    the demo path is unchanged; `si serve` still returns
    501 for every transformer bundle (existing suite); no
    CLI wiring added. Per-cycle `train_delta` wall times are recorded in the
    cycles' hyperparams; total decisive two-cycle wall time
    is on the order of seconds on this VM. All on this VM,
    CPU only.

Full suite on the author's own run: **353 passed,
2 skipped** in 342.71s — the SI-27 baseline (340 passed,
2 skipped) plus this slice's 13 new tests, zero
regressions. Focused run:
`tests/test_worst_input_composition.py`, 13 passed in
63.35s.

## What is still NOT claimed

- **No authored stack deeper than one.** Cycle 2
  REJECTed under worst-input training as it did under
  both anchorings unpenalised and under the mean
  composition penalty; the deepest landed authored stack
  on record remains depth 1 (`d0001`). No authored
  two-delta stack exists, and the transformer-squash
  precondition remains unmet.
- **Worst-input composition training is not shown to
  work — at the judged maximum.** On allowed inputs it
  is shown to work decisively (the tail priced directly
  moved, and the tail–bulk gap closed there). On the
  one figure the gate prices, the treatment (0.31216)
  improves on SI-25's control (0.33290) and SI-27's
  treatment (0.32509) and worsens on SI-23's (0.29634);
  at the binding probe its movements align *more* than
  either composition predecessor's. One decisive
  configuration was measured; no claim is made about
  worst-input objectives at other heads, other depths,
  other λ_w, other k, or any τ (none was run, and none
  is reported as anything but an open question for a
  future RFC), or for signal 1 as a decisive matter.
- **No λ_w / k conclusion beyond λ_w = 1.0, k = 4.**
  Both were the configuration's pre-registered
  definition. Whether a different single λ_w or k would
  behave differently is an open question this slice does
  not measure and does not answer; no sweep was run.
- **No third cycle, no continual learning, no
  mixed-configuration stack claim.** Exactly two
  authored cycles were attempted in the decisive run, on
  constructed signals, on a constructed model. The
  controls are separate runs, labelled controls.
- **No Glimmer anything.** No Glimmer forward path,
  probes, training, or authorship; §6's objective is a
  miniature mechanism and claims nothing at Glimmer
  scale; the Glimmer seed is untouched and its deltas
  still escalate with checks 3/4 "not evaluated".
- **No widening.** Two signals, two anchorings, three
  composition configurations, one target, one bundle
  kind; no additive or multi-target authorship; no
  policy default changed; no tolerance moved; both
  signals are byte-for-byte as SI-19/SI-23 defined them;
  the default anchoring and the default composition
  configuration are unchanged.
- **No serving, no CLI wiring.** Transformer bundles
  keep returning 501.
- **Diagnostics are not verdicts.** The diagnostic set
  above attributes the gate's REJECT after the fact; it
  is not a gate check, it influenced no verdict, and
  none of it was a prediction (§17 items 12/14).
- **Training numbers are training numbers.** Every loss
  and worst-input value above is labelled with its
  signal id, partition, head, target source, anchoring,
  and composition configuration; none of them is a gate
  or probe result (§17 item 1). Allowed-input figures
  are never presented as probe figures or vice versa
  (§17 item 18), and a worst-case figure always names
  its population (§17 item 20).
- **Recorded selection is not a choice the mind made**
  (§17 item 13). The driver configured each cycle's
  signal, anchoring, and composition configuration; the
  journal shows all three.

## Terminology

Machine learning infrastructure. This document contains
no "custom LLM software" claims; the miniature is a
constructed, deterministically seeded, not trained
machine-learning test instrument, its authored deltas are
trained on constructed signals, and its second signal's
worst-input-trained second-cycle delta was judged — and
rejected — on cumulative identity drift, with the
worst-input term measured doing exactly what it was
designed to do on the worst inputs it was allowed to see
and the gate's binding probe measured moving the other
way, further than under either predecessor.
