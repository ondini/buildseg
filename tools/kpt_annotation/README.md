# Roof corner keypoint annotation (Mar 2024)

The pipeline that produced the corner-refiner training data:

1. `generate_filenames.py` (was `mserver:~/Dev/data/generate_filenames.py`): samples photos from the first 60%
   of `db_updated_05_03_24.db` rows (to stay out of the val split) and copies them to `kpts_imgs/`
   (191 images in total, listed in `kp_filepaths*.txt`).
2. `annotate_points.py` (was `ptAnnotation/test.py`): matplotlib click tool. It pre-loads existing points from
   `in_file`. Right click adds a roof corner, middle click adds a chimney point. Results are saved to `out_file` / `chim.json`.
   The successive passes were `pts.json` → `pts2.json` → `ptsR.json` → `ptsREA.json` → **`ptsRE.json`** (final, 185 images).
   Format: `{image_name: [[x, y], ...]}`. `pts.json` is at half resolution (`collect_keypoints` resizes by 2); all later files are in
   full-resolution pixels.
   `chim.json` (chimney points) was not found when archiving and is presumably lost.
3. `dataset/generators/generator_points_coco2.py`: `ptsRE.json` → `COCO_KPTS_2803` (224 px patches).
4. Training: `configs/config_kp.json` → `MaxVitUnetS`. The released `corner_ref.pth` =
   `out/train/SolAR_KPdet_MaxVitS/0328_090301/best_ckpt_ep53.pth` (init: `SolAR_KPdet/0323_091757/best_ckpt_ep21.pth`).

Archives: point JSONs in HF dataset `reframed-cz/roof-annotations-2024` (`keypoints/`), images in its `ground`
config, weights + run config/log in HF `reframed-cz/roof-models-2024` (`fvapp/`).
