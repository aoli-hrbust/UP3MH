from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data


DatasetTuple = namedtuple(
    "DatasetTuple",
    [
        "I_db", "T_db", "L_db",
        "I_te", "T_te", "L_te",
    ],
)

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_RELATIVE_PATHS = {
    "flickr": Path("flickr25k/mir_clip_all.mat"),
    "nuswide": Path("nuswide/nus_clip_all.mat"),
    "coco": Path("coco/coco_clip_all.mat"),
}
REQUIRED_KEYS = (
    "I_db", "T_db", "L_db",
    "I_te", "T_te", "L_te",
)


def resolve_data_path(dataset, data_root=None, data_path=None):
    dataset = dataset.lower()
    if data_path:
        path = Path(data_path).expanduser()
    else:
        if dataset not in DEFAULT_RELATIVE_PATHS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Choose from "
                f"{sorted(DEFAULT_RELATIVE_PATHS)} or pass --data_path."
            )
        root = Path(data_root).expanduser() if data_root else DEFAULT_DATA_ROOT
        path = root / DEFAULT_RELATIVE_PATHS[dataset]
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Use --data_path with the full MAT path, or --data_root with the "
            "directory containing flickr25k/, nuswide/ and coco/."
        )
    return path.resolve()


def _read_mat(path):
    """Read both traditional MAT files and MATLAB v7.3/HDF5 MAT files."""
    try:
        import scipy.io as sio

        raw = sio.loadmat(path, variable_names=REQUIRED_KEYS)
        return {key: np.asarray(raw[key]) for key in REQUIRED_KEYS}
    except (NotImplementedError, ValueError, OSError):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required to read MATLAB v7.3 data files."
            ) from exc
        with h5py.File(path, "r") as handle:
            missing = [key for key in REQUIRED_KEYS if key not in handle]
            if missing:
                raise KeyError(f"Missing MAT variables: {missing}")
            return {key: np.asarray(handle[key]) for key in REQUIRED_KEYS}


def _orient_split(image, text, label, split_name):
    """Jointly choose the only orientation with a common sample dimension."""
    arrays = [np.asarray(image), np.asarray(text), np.asarray(label)]
    if any(array.ndim != 2 for array in arrays):
        shapes = [array.shape for array in arrays]
        raise ValueError(f"{split_name} arrays must all be 2-D, got {shapes}.")

    candidates = []
    for image_t in (False, True):
        for text_t in (False, True):
            for label_t in (False, True):
                oriented = [
                    arrays[0].T if image_t else arrays[0],
                    arrays[1].T if text_t else arrays[1],
                    arrays[2].T if label_t else arrays[2],
                ]
                if len({item.shape[0] for item in oriented}) == 1:
                    candidates.append(oriented)

    if not candidates:
        raise ValueError(
            f"Cannot align {split_name} image/text/label arrays: "
            f"{[array.shape for array in arrays]}"
        )

    # Prefer the orientation in which labels have the smallest feature width.
    image, text, label = min(
        candidates,
        key=lambda items: (
            items[2].shape[1],
            items[0].shape[0] < items[0].shape[1],
            items[1].shape[0] < items[1].shape[1],
        ),
    )
    return (
        np.asarray(image, dtype=np.float32, order="C"),
        np.asarray(text, dtype=np.float32, order="C"),
        np.asarray(label, dtype=np.float32, order="C"),
    )


def load_data(dataset, data_root=None, data_path=None):
    path = resolve_data_path(dataset, data_root=data_root, data_path=data_path)
    raw = _read_mat(path)

    I_db, T_db, L_db = _orient_split(
        raw["I_db"], raw["T_db"], raw["L_db"], "database"
    )
    I_te, T_te, L_te = _orient_split(
        raw["I_te"], raw["T_te"], raw["L_te"], "query"
    )

    if I_db.shape[1] != I_te.shape[1]:
        raise ValueError("Image feature dimensions differ across data splits.")
    if T_db.shape[1] != T_te.shape[1]:
        raise ValueError("Text feature dimensions differ across data splits.")
    if L_db.shape[1] != L_te.shape[1]:
        raise ValueError("Label dimensions differ across data splits.")

    result = DatasetTuple(
        I_db, T_db, L_db,
        I_te, T_te, L_te,
    )
    return result, path


class my_dataset(data.Dataset):
    def __init__(self, img_feature, txt_feature, label):
        self.img_feature = torch.from_numpy(np.asarray(img_feature, dtype=np.float32))
        self.txt_feature = torch.from_numpy(np.asarray(txt_feature, dtype=np.float32))
        self.label = torch.from_numpy(np.asarray(label, dtype=np.float32))

    def __getitem__(self, index):
        return (
            index,
            self.img_feature[index],
            self.txt_feature[index],
            self.label[index],
        )

    def __len__(self):
        return self.img_feature.shape[0]
