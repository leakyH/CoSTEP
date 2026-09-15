import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "assets"


def load_dataset(dataset, root=None):
    root = Path(root) if root is not None else DEFAULT_DATA_ROOT
    folder = "us_state" if dataset == "us" else "japan"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing packaged data manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    entries = {key: row for key, row in manifest.items() if key.startswith(folder + "/")}
    if not entries:
        raise ValueError(f"Data manifest has no {folder} entries")
    data = {}
    for name, row in entries.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required data file: {path}; large data files are excluded from Git"
            )
        if row.get("format") in {"table", "schema"}:
            continue
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != row["shape"] or str(array.dtype) != row["dtype"]:
            raise ValueError(f"Data shape/dtype mismatch: {path}")
        data[path.stem] = array
    if dataset == "us":
        schema = json.loads((root / folder / "table_schema.json").read_text())
        table = pd.read_csv(root / folder / "observations.csv.gz",
                            dtype={"region_id": str}, float_precision="round_trip")
        ids, dates = schema["node_ids"], schema["dates"]
        if (len(ids) != len(set(ids)) or len(dates) != len(set(dates))
                or not np.array_equal(table["region_id"], np.tile(ids, len(dates)))
                or not np.array_equal(table["date"], np.repeat(dates, len(ids)))):
            raise ValueError("US table date/region order differs from frozen schema")
        for key, spec in schema["arrays"].items():
            columns = [f"{key}_{c}" for c in range(spec["shape"][-1])]
            data[key] = table[columns].to_numpy(dtype=spec["dtype"]).reshape(spec["shape"])
        data["node_ids"] = np.asarray(ids)
        data["node_names"] = np.asarray(schema["node_names"])
        data["dates"] = np.asarray(dates, dtype="datetime64[ns]")
    if dataset != "us":
        # Geographic IDs are a maintainer-added external population match, not
        # MepoGNN annotations; they never reorder the original training arrays.
        metadata = json.loads((root / folder / "identity.json").read_text())
        if len(metadata["node_ids"]) != data["y"].shape[1]:
            raise ValueError("JAPAN node identity count mismatch")
        data["node_ids"] = np.asarray(metadata["node_ids"])
        data["node_identity_provenance"] = metadata["node_identity_provenance"]
    required = {"y", "node", "SIR", "od", "commute"}
    if not required <= data.keys():
        raise ValueError(f"Missing dataset arrays: {required-data.keys()}")
    identity = {"dataset": dataset, "node_ids": data.get("node_ids", []).tolist(),
                "arrays": {key: list(data[key].shape) for key in sorted(required)}}
    return data, identity
