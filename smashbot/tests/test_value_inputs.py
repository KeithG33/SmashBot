"""The value net takes the policy's inputs (embedding and tech mask) unless
its config says simple; configs saved before 2026-10-02 have no such key and
built simple value nets."""
from smashbot.value import value_network_config

NETWORK = {"name": "sgu", "num_heads": 1, "window": 4, "embed": "enhanced", "embed_hidden_size": 16,
           "tech_mask_window": 4}
VALUE = {"name": "match", "hidden_size": 32, "num_layers": 1, "layout": ""}


def test_value_net_takes_the_policys_inputs_unless_saved_without_them():
    matched = value_network_config(NETWORK, {**VALUE, "inputs": "policy"})
    assert (matched.embed, matched.embed_hidden_size, matched.tech_mask_window) == ("enhanced", 16, 4)
    for value in ({**VALUE, "inputs": "simple"}, VALUE):   # explicit, and a config saved before the key
        simple = value_network_config(NETWORK, value)
        assert (simple.embed, simple.tech_mask_window) == ("simple", 0)
