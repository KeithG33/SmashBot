#!/usr/bin/env python
"""Port the slippi-ai TF "medium-v2" checkpoint to the SmashBot PyTorch stack.

Three phases, run in the right interpreter:

  1. export  (venv-ref: TF + full slippi_ai)
     Dumps the 141 policy variables, the name_map/config, and a golden set of
     synthetic encoded state-action sequences with the TF policy's per-step
     logits (all 13 controller components) and final LSTM states.

       /home/kage/drive2/ShineBot/venv-ref/bin/python scripts/port_ref_model.py export

  2. convert  (.venv: torch, no TF)
     Builds our Policy with medium-v2's config, maps the TF variables onto it
     (sonnet Linear w is [in, out] -> transposed; sonnet LSTM gate order
     (i, f, g, o) matches torch's), and writes a checkpoint loadable by
     smashbot.eval.game.load_policy.

       .venv/bin/python scripts/port_ref_model.py convert

  3. verify  (.venv)
     Replays the golden inputs through the ported model (two T=8 chunks with
     recurrent-state carry per group) and checks numerical equivalence:

       a. embedding: our fp32 encoded vector vs TF's, < 1e-5;
       b. exactness: the ported model run in fp64 vs an INDEPENDENT fp64
          reference implementation (numpy, straight from the sonnet equations,
          using the raw TF weights) -- max logit + final-state diff < 1e-6.
          This proves the weight mapping and wiring are exact: any transpose,
          gate-order, eps, or activation mistake shows up at ~1e0, not 1e-9.
       c. fp32 end-to-end vs the TF fp32 run, < 1e-3. This diff is dominated
          by cross-framework fp32 kernel rounding (Eigen vs oneDNN reduce in
          different orders), amplified by the autoregressive head; it is
          irreducible without bit-identical kernels. See ref_port_report.md.

       .venv/bin/python scripts/port_ref_model.py verify

     verify also writes the self-contained golden artifact consumed by
     smashbot/tests/test_ref_port.py to smashbot/tests/assets/.
"""

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DEFAULT = (
    "/tmp/claude-1000/-home-kage-smashbot-workspace/"
    "622f61f0-32b7-4321-b892-7040871af8a8/scratchpad"
)
TF_CKPT = "/home/kage/drive2/ShineBot/models/medium-v2"
TORCH_CKPT = "/home/kage/drive2/ShineBot/models/medium-v2-torch.pt"
GOLDEN_ASSET = os.path.join(REPO, "smashbot", "tests", "assets", "ref_port_golden.npz")

# Golden shape: 8 groups x batch 4 x two consecutive T=8 chunks (with recurrent
# state carried between them) = 64 (T=8, B=4) sequences total.
N_GROUPS = 8
BATCH = 4
CHUNK_T = 8
TOTAL_T = 2 * CHUNK_T  # network steps; TOTAL_T + 1 frames incl. target overlap
SEED = 20260812

EMBED_TOL = 1e-5  # our fp32 embedding vs TF's fp32 embedding
EXACT_TOL = 1e-6  # ported model in fp64 vs independent fp64 reference
FP32_TOL = 1e-3   # fp32 end-to-end vs TF fp32 (cross-framework rounding floor)


# ---------------------------------------------------------------------------
# Phase 1: export (runs in venv-ref, TF).
# ---------------------------------------------------------------------------


def _random_encoded_struct(embed, shape, rng):
    """Random *encoded* leaves for an embedding struct (their or our domain)."""
    from slippi_ai.tf import embed as embed_lib

    def rand_leaf(e):
        if isinstance(e, embed_lib.MLPWrapper):
            return e._embed.map(rand_leaf)
        if isinstance(e, embed_lib.BoolEmbedding):
            return rng.random(shape) < 0.5
        if isinstance(e, embed_lib.OneHotEmbedding):
            # For EXTRA-policy embeddings, e.size = input_size + 1 and the
            # extra bucket is a legal encoded value; exercise it too.
            hi = e.size if e.one_hot_policy is embed_lib.OneHotPolicy.EXTRA else e.input_size
            return rng.integers(0, hi, size=shape).astype(e.dtype)
        if isinstance(e, embed_lib.FloatEmbedding):
            return (rng.random(shape) * 200.0 - 50.0).astype(np.float32)
        raise TypeError(f"unhandled embedding {type(e)}")

    return embed.map(rand_leaf)


def export(args):
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")

    import tree
    from slippi_ai import saving

    state = saving.load_state_from_disk(args.tf_ckpt)
    policy = saving.load_policy_from_state(state)

    # --- weights ---
    variables = policy.variables
    names = [v.name for v in variables]
    weights = {f"w_{i:03d}": v.numpy() for i, v in enumerate(variables)}
    np.savez(os.path.join(args.scratch, "medium_v2_weights.npz"), **weights)
    print(f"dumped {len(variables)} variables")

    # --- meta: config + name_map + step ---
    from slippi_ai.tf import saving as tf_saving

    config = tf_saving.upgrade_config(dict(state["config"]))
    name_map = None
    for container in (state, state.get("state", {})):
        if isinstance(container, dict) and "name_map" in container:
            name_map = container["name_map"]
    meta = {
        "var_names": names,
        "var_shapes": [list(v.shape) for v in variables],
        "config": config,
        "name_map": name_map,
        "step": int(state["state"].get("step", 0)) if "step" in state.get("state", {}) else None,
    }
    with open(os.path.join(args.scratch, "medium_v2_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print("wrote meta json; state keys:", list(state.keys()), "state['state'] keys:",
          list(state.get("state", {}).keys()))

    # --- golden sequences ---
    embed_sa = policy.network._embed_state_action
    head = policy.controller_head
    rng = np.random.default_rng(SEED)

    golden = {}
    debug = {}
    logit_paths = None
    for g in range(N_GROUPS):
        sa_np = _random_encoded_struct(embed_sa, (TOTAL_T + 1, BATCH), rng)
        reset_np = rng.random((TOTAL_T, BATCH)) < 0.06  # a few mid-sequence resets

        flat_in = tree.flatten(sa_np)
        for i, leaf in enumerate(flat_in):
            golden[f"g{g}_in{i:03d}"] = leaf
        golden[f"g{g}_reset"] = reset_np

        sa = tree.map_structure(tf.constant, sa_np)
        inputs = tree.map_structure(lambda t: t[:TOTAL_T], sa)
        reset = tf.constant(reset_np)

        state0 = policy.network.initial_state(BATCH)
        out1, state1 = policy.network.unroll(
            tree.map_structure(lambda t: t[:CHUNK_T], inputs),
            reset[:CHUNK_T], state0)
        out2, state2 = policy.network.unroll(
            tree.map_structure(lambda t: t[CHUNK_T:], inputs),
            reset[CHUNK_T:], state1)
        outputs = tf.concat([out1, out2], axis=0)  # [T, B, 768]

        prev_action = tree.map_structure(lambda t: t[:TOTAL_T], sa.action)
        target_action = tree.map_structure(lambda t: t[1:], sa.action)
        dist = head.distance(outputs, prev_action, target_action)

        flat_logits = tree.flatten(dist.logits)
        if logit_paths is None:
            logit_paths = [
                "/".join(map(str, p))
                for p, _ in tree.flatten_with_path(dist.logits)
            ]
        for i, t in enumerate(flat_logits):
            golden[f"g{g}_logit{i:02d}"] = t.numpy()

        flat_state = tree.flatten(state2)  # h0, c0, h1, c1, h2, c2
        assert len(flat_state) == 6, len(flat_state)
        for i, t in enumerate(flat_state):
            golden[f"g{g}_fs{i}"] = t.numpy()

        debug[f"g{g}_embedded"] = embed_sa(inputs).numpy()
        debug[f"g{g}_core_out"] = outputs.numpy()
        print(f"group {g} done")

    golden["logit_paths"] = np.array(json.dumps(logit_paths))
    golden["n_inputs"] = np.array(len(tree.flatten(sa_np)))
    np.savez_compressed(os.path.join(args.scratch, "ref_port_golden.npz"), **golden)
    np.savez_compressed(os.path.join(args.scratch, "ref_port_debug.npz"), **debug)
    print("export complete")


# ---------------------------------------------------------------------------
# Phase 2: convert (runs in .venv, torch).
# ---------------------------------------------------------------------------


def _build_torch_policy():
    from smashbot import configs, embed as embed_lib
    from smashbot.policy import build_policy

    head_config = configs.ControllerHeadConfig(
        residual_size=128, component_depth=2, axis_spacing=32, shoulder_spacing=10
    )
    policy = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(
            axis_spacing=head_config.axis_spacing,
            shoulder_spacing=head_config.shoulder_spacing,
        ),
        network_config=_network_config(),
        head_config=head_config,
        policy_config=configs.PolicyConfig(delay=21),
        num_names=128,
    )
    return policy, head_config


def _network_config():
    from smashbot import configs

    return configs.NetworkConfig(
        name="tx_like",
        hidden_size=768,
        num_layers=3,
        ffw_multiplier=2,
        recurrent_layer="lstm",
        # slippi-ai's LayerNorm has no epsilon; torch's default is 1e-5.
        ln_eps=0.0,
    )


def convert(args):
    import torch

    torch.set_num_threads(4)

    weights = np.load(os.path.join(args.scratch, "medium_v2_weights.npz"))
    with open(os.path.join(args.scratch, "medium_v2_meta.json")) as f:
        meta = json.load(f)
    names = meta["var_names"]
    assert len(names) == 141, len(names)

    policy, head_config = _build_torch_policy()
    policy.eval()

    cursor = [0]
    assigned = set()

    def take(name_frag, shape):
        i = cursor[0]
        cursor[0] += 1
        name, arr = names[i], weights[f"w_{i:03d}"]
        assert name_frag in name, f"var {i}: expected *{name_frag}*, got {name}"
        assert tuple(arr.shape) == tuple(shape), f"var {i} {name}: {arr.shape} != {shape}"
        return arr

    def set_param(param, arr):
        t = torch.from_numpy(np.ascontiguousarray(arr))
        assert param.shape == t.shape, (param.shape, t.shape)
        with torch.no_grad():
            param.copy_(t)
        assert id(param) not in assigned, "double assignment"
        assigned.add(id(param))

    def set_linear(linear, name_prefix):
        """sonnet Linear stores w as [in, out]; torch nn.Linear as [out, in]."""
        b = take(f"{name_prefix}/b:0", (linear.out_features,))
        w = take(f"{name_prefix}/w:0", (linear.in_features, linear.out_features))
        set_param(linear.bias, b)
        set_param(linear.weight, w.T)

    # --- AR head: 13 components x (decoder linear, encoder mlp linear_0..2) ---
    for block in policy.controller_head.res_blocks:
        size = block.embedder.size
        set_linear(block.decoder, "AutoRegressive/ResBlock/linear")
        assert block.decoder.in_features == size
        set_linear(block.encoder[0], "AutoRegressive/ResBlock/mlp/linear_0")
        set_linear(block.encoder[2], "AutoRegressive/ResBlock/mlp/linear_1")
        set_linear(block.encoder[4], "AutoRegressive/ResBlock/mlp/linear_2")
        assert block.encoder[4].out_features == size

    # --- AR head to_residual ---
    set_linear(policy.controller_head.to_residual, "AutoRegressive/linear")

    # --- item MLP (one module shared by all 15 slots) ---
    embed_game = policy.network.embed_game
    items_embed = dict(embed_game.embedding)["items"]
    item_mlp = items_embed.embedding[0][1]._mlp  # Sequential(Linear, ReLU, Linear, ReLU)
    for slot_name, slot_embed in items_embed.embedding:
        assert slot_embed._mlp is item_mlp, slot_name  # shared weights
    set_linear(item_mlp[0], "MLP_Item/mlp/linear_0")
    set_linear(item_mlp[2], "MLP_Item/mlp/linear_1")

    # --- tx_like core ---
    core = policy.network.core
    set_linear(core._layers[0]._module, "TransformerLike/encoder")
    num_layers = (len(core._layers) - 1) // 2
    assert num_layers == 3
    for l in range(num_layers):
        lstm = core._layers[1 + 2 * l]._net._core
        H = lstm.hidden_size
        # sonnet snt.LSTM: gates = x @ w_i + h @ w_h + b, split (i, f, g, o) --
        # identical gate order and equations to torch nn.LSTM.
        w_h = take("lstm/w_h:0", (H, 4 * H))
        w_i = take("lstm/w_i:0", (H, 4 * H))
        b = take("lstm/b:0", (4 * H,))
        set_param(lstm.weight_hh_l0, w_h.T)
        set_param(lstm.weight_ih_l0, w_i.T)
        set_param(lstm.bias_ih_l0, b)
        set_param(lstm.bias_hh_l0, np.zeros(4 * H, np.float32))

        res = core._layers[2 + 2 * l]._module  # ResBlock
        ln, lin1, lin2 = res.block[0], res.block[1], res.block[3]
        set_param(ln.bias, take("ResBlock/LayerNorm/bias:0", (H,)))
        set_param(ln.weight, take("ResBlock/LayerNorm/scale:0", (H,)))
        set_linear(lin1, "ResBlock/linear")
        set_linear(lin2, "ResBlock/linear")

    # --- value head (untrained here; policy config used a separate value net) ---
    set_linear(policy.value_head, "value_head")

    assert cursor[0] == 141, cursor[0]
    missing = [n for n, p in policy.named_parameters() if id(p) not in assigned]
    assert not missing, f"unassigned parameters: {missing}"

    # --- save in our checkpoint format ---
    import dataclasses

    from smashbot import configs, saving
    from smashbot.train_bc import TrainConfig

    config = TrainConfig(
        policy=configs.PolicyConfig(delay=21),
        network=_network_config(),
        head=head_config,
    )
    config.data.max_names = 128
    state = {
        "policy": policy.state_dict(),
        "name_map": meta["name_map"] or {},
        "step": meta.get("step") or 0,
        "ported_from": "slippi-ai medium-v2 (TF), via scripts/port_ref_model.py",
    }
    saving.save_checkpoint(args.torch_ckpt, config, state, best_eval_loss=float("inf"))
    print(f"wrote {args.torch_ckpt}")

    # prove it loads through the eval path
    from smashbot.eval.game import load_policy

    policy2, name_map, step = load_policy(args.torch_ckpt, "cpu")
    n_params = sum(p.numel() for p in policy2.parameters())
    print(f"reloaded via eval.game.load_policy: {n_params} params, "
          f"{len(name_map)} names, step={step}")


# ---------------------------------------------------------------------------
# Phase 3: verify (runs in .venv, torch). Also used by the pytest.
# ---------------------------------------------------------------------------


def _erf(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erf
        return erf(x)
    except ImportError:
        import torch
        return torch.special.erf(torch.from_numpy(x)).numpy()


def _ref64_forward(w, emb32, reset, prev_encs, target_encs):
    """Independent fp64 reference: sonnet equations + raw TF weights.

    w: index -> np.ndarray (TF checkpoint order, sonnet [in, out] layout).
    emb32: [T, B, 2613] fp32 embedded input (from the TF run).
    prev_encs/target_encs: per AR component, fp64 one-hot/bool encodings
      [T, B, size] in *embedding* (sampling) order.
    Returns (logits list in embedding order, final [h0, c0, h1, c1, h2, c2]).
    """
    T, B, _ = emb32.shape
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    gelu = lambda z: 0.5 * z * (1.0 + _erf(z / np.sqrt(2.0)))

    def layernorm(x, scale, bias):
        xc = x - x.mean(-1, keepdims=True)
        sd = np.sqrt(np.square(xc).mean(-1, keepdims=True))  # their LN: no eps
        return (xc / sd) * scale + bias

    x = emb32.astype(np.float64) @ w[111] + w[110]  # encoder
    h = [np.zeros((B, 768)) for _ in range(3)]
    c = [np.zeros((B, 768)) for _ in range(3)]
    outputs = np.empty((T, B, 768))
    for t in range(T):
        xt = x[t]
        mask = reset[t][:, None]
        for l in range(3):
            base = 112 + 9 * l
            w_h, w_i, b = w[base], w[base + 1], w[base + 2]
            ln_bias, ln_scale = w[base + 3], w[base + 4]
            l1_b, l1_w, l2_b, l2_w = (w[base + 5], w[base + 6],
                                      w[base + 7], w[base + 8])
            hl = np.where(mask, 0.0, h[l])
            cl = np.where(mask, 0.0, c[l])
            # sonnet snt.LSTM: gates split (i, f, g, o)
            gates = xt @ w_i + hl @ w_h + b
            i_, f_, g_, o_ = np.split(gates, 4, axis=-1)
            c[l] = sig(f_) * cl + sig(i_) * np.tanh(g_)
            h[l] = sig(o_) * np.tanh(c[l])
            xt = xt + h[l]  # ResidualWrapper
            y = layernorm(xt, ln_scale, ln_bias)  # ResBlock
            y = gelu(y @ l1_w + l1_b)
            xt = xt + (y @ l2_w + l2_b)
        outputs[t] = xt

    # Autoregressive head, teacher forcing.
    residual = outputs @ w[105] + w[104]  # to_residual
    relu = lambda z: np.maximum(z, 0.0)
    logits_out = []
    for ci, (pe, te) in enumerate(zip(prev_encs, target_encs)):
        base = 8 * ci
        inp = np.concatenate([residual, pe], axis=-1)
        y = relu(inp @ w[base + 3] + w[base + 2])
        y = relu(y @ w[base + 5] + w[base + 4])
        logits_out.append(y @ w[base + 7] + w[base + 6])
        residual = residual + (te @ w[base + 1] + w[base])
    final = [h[0], c[0], h[1], c[1], h[2], c[2]]
    return logits_out, final


def _component_names_and_encodings(policy, sa_np):
    """Flat AR components in embedding (sampling) order.

    Returns (names, prev_encs, target_encs) where encodings are fp64
    [T, B, size] arrays for frames [0:T] and [1:T+1] of sa_np.action.
    """
    from smashbot import embed as embed_lib

    embed_controller = policy.controller_head.embed_controller

    def walk(emb, prefix):
        if isinstance(emb, embed_lib.StructEmbedding):
            for k, e in emb.embedding:
                yield from walk(e, f"{prefix}/{k}")
        else:
            yield prefix, emb

    names, embedders = zip(*walk(embed_controller, "controller"))

    def encode(embedder, leaf):  # leaf: [T, B] numpy
        if isinstance(embedder, embed_lib.BoolEmbedding):
            return np.where(leaf, embedder.on, embedder.off)[..., None].astype(np.float64)
        assert isinstance(embedder, embed_lib.OneHotEmbedding)
        return np.eye(embedder.size, dtype=np.float64)[leaf.astype(np.int64)]

    leaves = list(embed_controller.flatten(sa_np.action))  # [T+1, B] each
    prev_encs = [encode(e, l[:TOTAL_T]) for e, l in zip(embedders, leaves)]
    target_encs = [encode(e, l[1:]) for e, l in zip(embedders, leaves)]
    return list(names), prev_encs, target_encs


def _to_batch_major(x):
    import torch

    return torch.from_numpy(np.ascontiguousarray(np.swapaxes(np.asarray(x), 0, 1)))


def _unflatten_inputs(policy, asset, g):
    import tree

    n_inputs = int(asset["n_inputs"])
    dummy = policy.network.embed_state_action.dummy((TOTAL_T + 1, BATCH))
    leaves = [asset[f"g{g}_in{i:03d}"] for i in range(n_inputs)]
    return tree.unflatten_as(dummy, leaves)  # numpy, time-major [T+1, B, ...]


def _chunked_unroll(core_unroll, initial_state, inputs, reset, slicer):
    """Two T=8 chunks with recurrent-state carry, as the TF export did."""
    import torch

    out1, state = core_unroll(slicer(inputs, 0, CHUNK_T), reset[:, :CHUNK_T],
                              initial_state)
    out2, state = core_unroll(slicer(inputs, CHUNK_T, TOTAL_T),
                              reset[:, CHUNK_T:], state)
    return torch.cat([out1, out2], dim=1), state


def _head_distance_fp64(head64, outputs64, prev_action, target_action):
    """heads.AutoRegressive.distance with explicit fp64 casts.

    The embed leaf modules hard-code .float() outputs (exact 0/1/-1 values),
    so cast each embedding to fp64 before it enters the fp64 residual stream.
    """
    import torch

    residual = head64.to_residual(outputs64)
    prev_flat = list(head64.embed_controller.flatten(prev_action))
    target_flat = list(head64.embed_controller.flatten(target_action))
    logits_out = []
    for block, prev, target in zip(head64.res_blocks, prev_flat, target_flat):
        pe = block.embedder(prev).double()
        logits_out.append(block.encoder(torch.cat([residual, pe], dim=-1)))
        te = block.embedder(target).double()
        residual = residual + block.decoder(te)
    return logits_out


def run_report(policy, asset) -> dict:
    """All equivalence metrics: {metric: max_abs_diff} over the golden set.

    Sections:
      embed/...        our fp32 embedding vs TF's fp32 embedding
      exact64/...      ported model in fp64 vs independent fp64 reference
                       (logits per component + final LSTM states, state carry)
      fp32/...         our full fp32 pipeline vs the TF fp32 run
    """
    import copy

    import torch
    import tree

    logit_paths = json.loads(str(asset["logit_paths"]))
    policy64 = copy.deepcopy(policy).double()
    diffs = {}

    def acc(key, got, ref):
        d = float(np.abs(np.asarray(got) - np.asarray(ref)).max())
        diffs[key] = max(diffs.get(key, 0.0), d)

    with torch.no_grad():
        for g in range(N_GROUPS):
            sa_np = _unflatten_inputs(policy, asset, g)
            sa = tree.map_structure(_to_batch_major, sa_np)  # [B, T+1, ...]
            reset = _to_batch_major(asset[f"g{g}_reset"])  # [B, T] bool
            inputs = tree.map_structure(lambda t: t[:, :TOTAL_T], sa)
            prev_action = tree.map_structure(lambda t: t[:, :TOTAL_T], sa.action)
            target_action = tree.map_structure(lambda t: t[:, 1:], sa.action)
            tf_emb = asset[f"g{g}_embedded"]  # [T, B, 2613] fp32

            # --- gate a: embedding ---
            our_emb = policy.network.embed_sa(inputs)
            acc("embed", our_emb.numpy().swapaxes(0, 1), tf_emb)

            # --- gate b: fp64 exactness vs independent reference ---
            emb64 = _to_batch_major(tf_emb).double()
            state64 = tree.map_structure(
                lambda t: t.double(),
                policy64.network.core.initial_state(BATCH, device="cpu"))
            out64, state64 = _chunked_unroll(
                policy64.network.core.unroll, state64, emb64, reset,
                lambda x, a, b: x[:, a:b])
            logits64 = _head_distance_fp64(
                policy64.controller_head, out64, prev_action, target_action)
            comp_names, _, _ = _component_names_and_encodings(policy, sa_np)
            for name, t in zip(comp_names, logits64):
                acc(f"exact64/logits/{name}", t.numpy().swapaxes(0, 1),
                    asset[f"g{g}_ref64_{name.replace('/', '_')}"])
            for i, t in enumerate(tree.flatten(state64)):
                acc(f"exact64/final_state/{'hc'[i % 2]}{i // 2}",
                    t.numpy().reshape(BATCH, -1), asset[f"g{g}_ref64_fs{i}"])

            # --- gate c: fp32 end-to-end vs the TF fp32 run ---
            state = policy.network.initial_state(BATCH, device="cpu")
            out32, state = _chunked_unroll(
                policy.network.unroll, state, inputs, reset,
                lambda x, a, b: tree.map_structure(lambda t: t[:, a:b], x))
            dist = policy.controller_head.distance(
                out32, prev_action, target_action)
            for i, t in enumerate(tree.flatten(dist.logits)):
                acc(f"fp32/logits/{logit_paths[i]}",
                    t.numpy().swapaxes(0, 1), asset[f"g{g}_logit{i:02d}"])
            for i, t in enumerate(tree.flatten(state)):
                acc(f"fp32/final_state/{'hc'[i % 2]}{i // 2}",
                    t.numpy().reshape(BATCH, -1), asset[f"g{g}_fs{i}"])
    return diffs


def check_report(diffs: dict) -> list[str]:
    """Returns failure messages (empty = all gates pass)."""
    gates = [("embed", EMBED_TOL), ("exact64/", EXACT_TOL), ("fp32/", FP32_TOL)]
    failures = []
    for prefix, tol in gates:
        worst = max(v for k, v in diffs.items() if k.startswith(prefix))
        if worst >= tol:
            failures.append(f"{prefix}* max diff {worst:.3e} >= {tol}")
    return failures


def _build_asset(policy, golden, debug, weights):
    """Golden npz + TF embedded inputs + independent fp64 reference outputs."""
    import tree

    w = {int(k[2:]): weights[k] for k in weights.files}
    asset = {k: golden[k] for k in golden.files}
    for g in range(N_GROUPS):
        asset[f"g{g}_embedded"] = debug[f"g{g}_embedded"]
        sa_np = _unflatten_inputs(policy, golden, g)
        names, prev_encs, target_encs = _component_names_and_encodings(
            policy, sa_np)
        logits64, final64 = _ref64_forward(
            w, debug[f"g{g}_embedded"], golden[f"g{g}_reset"],
            prev_encs, target_encs)
        for name, arr in zip(names, logits64):
            asset[f"g{g}_ref64_{name.replace('/', '_')}"] = arr
        for i, arr in enumerate(final64):
            asset[f"g{g}_ref64_fs{i}"] = arr
    return asset


def verify(args):
    import torch

    torch.set_num_threads(4)
    from smashbot.eval.game import load_policy

    policy, _, _ = load_policy(args.torch_ckpt, "cpu")
    golden = np.load(os.path.join(args.scratch, "ref_port_golden.npz"))
    debug = np.load(os.path.join(args.scratch, "ref_port_debug.npz"))
    weights = np.load(os.path.join(args.scratch, "medium_v2_weights.npz"))

    asset = _build_asset(policy, golden, debug, weights)
    os.makedirs(os.path.dirname(GOLDEN_ASSET), exist_ok=True)
    np.savez_compressed(GOLDEN_ASSET, **asset)
    print(f"wrote {GOLDEN_ASSET} "
          f"({os.path.getsize(GOLDEN_ASSET) / 1e6:.1f} MB)")

    diffs = run_report(policy, np.load(GOLDEN_ASSET))
    for k in sorted(diffs):
        print(f"  {k:50s} {diffs[k]:.3e}")
    for prefix in ("embed", "exact64/", "fp32/"):
        worst = max(v for k, v in diffs.items() if k.startswith(prefix))
        print(f"max {prefix:10s} {worst:.3e}")
    failures = check_report(diffs)
    if failures:
        raise SystemExit("FAIL: " + "; ".join(failures))
    print("PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["export", "convert", "verify"])
    parser.add_argument("--scratch", default=SCRATCH_DEFAULT)
    parser.add_argument("--tf-ckpt", default=TF_CKPT)
    parser.add_argument("--torch-ckpt", default=TORCH_CKPT)
    args = parser.parse_args()
    os.makedirs(args.scratch, exist_ok=True)
    {"export": export, "convert": convert, "verify": verify}[args.phase](args)


if __name__ == "__main__":
    main()
