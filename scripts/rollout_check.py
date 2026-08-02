"""M4 open-loop rollout check.

Replays a memorized game through the inference path (Policy.sample, one frame
at a time). States always come from the recording; the previous-action input
is either the human's recorded action (teacher-forced) or the model's own
sampled action (self-fed). Comparing the two isolates the self-feedback
channel — the exposure-bias failure mode — with the world held fixed.

For a well-memorized game both modes should agree with the recording at high
rates, and self-fed should not collapse relative to teacher-forced.

Usage:
  .venv/bin/python scripts/rollout_check.py --ckpt /home/kage/drive2/ShineBot/runs/overfit.pt
"""

import argparse

import numpy as np
import torch
import tree


def to_torch(x, device):
    x = np.asarray(x)
    if x.dtype == np.uint16:
        x = x.astype(np.int32)
    return torch.from_numpy(x).to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/kage/drive2/ShineBot/runs/overfit.pt")
    ap.add_argument("--root", default="/home/kage/drive2/ShineBot/data/full/Root")
    ap.add_argument("--game_index", type=int, default=0, help="index into ckpt's game list")
    ap.add_argument("--max_frames", type=int, default=4000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from slippi_ai import data as data_lib
    from slippi_ai.types import StateAction

    from shinebot import configs, embed as embed_lib
    from shinebot.policy import build_policy

    ckpt = torch.load(args.ckpt, weights_only=False)
    policy = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(),
        head_config=configs.ControllerHeadConfig(),
        policy_config=configs.PolicyConfig(delay=ckpt["delay"]),
        num_names=ckpt["num_names"],
    ).to(args.device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    delay = ckpt["delay"]

    md5 = ckpt["games"][args.game_index]
    game = data_lib.read_table(f"{args.root}/Parsed/{md5}", compressed=True)
    print(f"Rolling out game {md5[:8]} ({game.stage.shape[0]} frames), delay={delay}")

    # Encode once (numpy), like the data thread does.
    encoded_state = policy.network.encode_game(game)
    controller_embed = policy.controller_head.controller_embedding
    encoded_actions = controller_embed.from_state(game.p0.controller)

    T = min(game.stage.shape[0], args.max_frames)
    name = np.zeros((1,), dtype=np.int32)

    def run(self_fed: bool) -> dict:
        hidden = policy.initial_state(1, device=args.device)
        prev_action = tree.map_structure(
            lambda a: to_torch(a[delay : delay + 1], args.device), encoded_actions
        )
        matches: dict[str, list] = {"buttons": [], "main_x": [], "main_y": [],
                                    "c_x": [], "c_y": [], "shoulder": []}
        # input t: state[t], prev action slot t+delay, target slot t+delay+1
        for t in range(T - delay - 1):
            state_t = tree.map_structure(
                lambda a: to_torch(np.asarray(a)[t : t + 1], args.device), encoded_state
            )
            if not self_fed:
                prev_action = tree.map_structure(
                    lambda a: to_torch(a[t + delay : t + delay + 1], args.device),
                    encoded_actions,
                )
            sa = StateAction(state=state_t, action=prev_action,
                             name=to_torch(name, args.device))
            sampled, hidden = policy.sample(sa, hidden)

            target = tree.map_structure(
                lambda a: to_torch(a[t + delay + 1 : t + delay + 2], args.device),
                encoded_actions,
            )
            s, g = sampled.controller_state, target
            all_buttons = torch.stack(
                [getattr(s.buttons, f) == getattr(g.buttons, f) for f in g.buttons._fields]
            ).all()
            matches["buttons"].append(all_buttons.item())
            matches["main_x"].append((s.main_stick.x == g.main_stick.x).item())
            matches["main_y"].append((s.main_stick.y == g.main_stick.y).item())
            matches["c_x"].append((s.c_stick.x == g.c_stick.x).item())
            matches["c_y"].append((s.c_stick.y == g.c_stick.y).item())
            matches["shoulder"].append((s.shoulder == g.shoulder).item())

            if self_fed:
                prev_action = sampled.controller_state

        return {k: float(np.mean(v)) for k, v in matches.items()}

    with torch.no_grad():
        tf = run(self_fed=False)
        sf = run(self_fed=True)

    print(f"\nAgreement with recording over {T - delay - 1} frames (sampled, not argmax):")
    print(f"{'component':>10}  {'teacher-forced':>14}  {'self-fed':>9}")
    for k in tf:
        print(f"{k:>10}  {tf[k]:>13.1%}  {sf[k]:>8.1%}")


if __name__ == "__main__":
    main()
