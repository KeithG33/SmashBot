"""Player-name conditioning codes, importable without Dolphin."""


def resolve_name_code(name_map: dict, name: str, verbose: bool = True) -> int:
    if name in name_map:
        return name_map[name]
    if name_map and verbose:
        print(f"'{name}' not in name_map {name_map}; using code 0")
    return 0
