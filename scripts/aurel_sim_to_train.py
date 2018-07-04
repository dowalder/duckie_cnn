#!/usr/bin/env python3

import pathlib
import csv
import PIL.Image


def load_sim_folder(sim_dir: pathlib.Path, tgt_dir: pathlib.Path) -> None:
    data_csv = sim_dir / "collected_data" / "processed_data.csv"
    images = sim_dir / "collected_data" / "images"

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

            img_tgt = tgt_dir / img_src.name
            lbl_tgt = tgt_dir / "{}.txt".format(img_src.stem)

            img = PIL.Image.open(img_src.as_posix())
            img.resize((160, 120)).save(img_tgt.as_posix())
            lbl_tgt.write_text(str(omega))


def main():
    img_dir = pathlib.Path("/home/dominik/dataspace/images/aurel_sim/sim_0")
    tgt_dir = pathlib.Path("/home/dominik/tmp/sim_0")
    load_sim_folder(img_dir, tgt_dir)


if __name__ == "__main__":
    main()
