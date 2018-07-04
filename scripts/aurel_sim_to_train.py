#!/usr/bin/env python3

import argparse
import pathlib
import random
import csv
import PIL.Image


def load_sim_folder(sim_dir: pathlib.Path, tgt_dir: pathlib.Path, test_percentage=0.3) -> None:
    data_csv = sim_dir / "collected_data" / "processed_data.csv"
    images = sim_dir / "collected_data" / "images"
    test_dir = tgt_dir / "test"
    train_dir = tgt_dir / "train"

    if not data_csv.is_file():
        raise RuntimeError("Could not find the required file: {}".format(data_csv))

    with data_csv.open() as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        print("Extracting from {}".format(data_csv))
        for line in csv_reader:
            if line[0] == "angular.z":
                continue
            omega = line[0]
            img_src = images / line[1]

            if not img_src.is_file():
                raise RuntimeError("Could not find image at: {}".format(img_src))

            if random.random() < test_percentage:
                img_tgt = test_dir / img_src.name
                lbl_tgt = test_dir / "{}.txt".format(img_src.stem)
            else:
                img_tgt = train_dir / img_src.name
                lbl_tgt = train_dir / "{}.txt".format(img_src.stem)

            img = PIL.Image.open(img_src.as_posix())
            img.resize((160, 120)).save(img_tgt.as_posix())
            lbl_tgt.write_text(str(omega))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", "-s", required=True, help="directory containing sim dirs")
    parser.add_argument("--tgt_dir", "-t", required=True, help="where to store extracted data")

    args = parser.parse_args()

    src_dir = pathlib.Path(args.src_dir)
    tgt_dir = pathlib.Path(args.tgt_dir)

    if not src_dir.is_dir():
        raise ValueError("--src_dir must be a valid directory, but it not: ".format(src_dir))
    tgt_dir.mkdir(parents=True, exist_ok=True)
    (tgt_dir / "test").mkdir(exist_ok=True)
    (tgt_dir / "train").mkdir(exist_ok=True)

    for folder in src_dir.iterdir():
        if not folder.is_dir():
            continue
        load_sim_folder(folder, tgt_dir)


if __name__ == "__main__":
    main()
