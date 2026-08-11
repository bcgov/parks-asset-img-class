# PII screening summary

Local screen of 5310 images using YOLOv8n (general) — no face model. Confidence threshold ≥ 0.40. No images left the machine.

## Counts

- **Total images**: 5310
- **Flagged** (any face / person / vehicle): 481 (9.1%)
- Contain a face: 0 (0.0%)
- Contain a person body (no face needed): 459 (8.6%)
- Contain a vehicle: 24 (0.5%)

## Recommended action before cloud upload

- **Manually review** the 481 flagged images.
- For images with faces: blur / redact faces, OR exclude from upload.
- For images with vehicles: consider blurring license plates (YOLO finds the vehicle bounding box; a license-plate model can narrow it).
- Re-screen after redaction to confirm faces are gone.

## First 20 flagged images (by face count → person → vehicle)

| image_path                                                             |   n_persons |   n_faces |   n_vehicles | flagged_reason   |
|:-----------------------------------------------------------------------|------------:|----------:|-------------:|:-----------------|
| data/raw/citywide/images/359/82850/5060__AST_IM_20210711_103758.jpeg   |           8 |         0 |            0 | person×8         |
| data/raw/citywide/images/359/82850/17668__AST_EX_20210823_115204.jpeg  |           6 |         0 |            0 | person×6         |
| data/raw/citywide/images/253/106578/53319__AST_IM_20220712_110256.jpeg |           4 |         0 |            0 | person×4         |
| data/raw/citywide/images/253/46927/59886__IMG_9059.JPG                 |           4 |         0 |            0 | person×4         |
| data/raw/citywide/images/253/52242/1723__AST_EX_20210622_111115.jpeg   |           4 |         0 |            0 | person×4         |
| data/raw/citywide/images/253/91711/69144__IMG_0135.JPEG                |           4 |         0 |            0 | person×4         |
| data/raw/citywide/images/337/126159/83432__Picture1.png                |           4 |         0 |            0 | person×4         |
| data/raw/citywide/images/359/62340/2515__AST_EX_20210625_110834.jpeg   |           4 |         0 |            0 | person×4         |
| data/raw/citywide/images/253/109727/64858__AST_EX_20230323_111243.jpeg |           3 |         0 |            0 | person×3         |
| data/raw/citywide/images/253/114963/76557__20.jpg                      |           3 |         0 |            0 | person×3         |
| data/raw/citywide/images/253/114963/76558__19.jpg                      |           3 |         0 |            0 | person×3         |
| data/raw/citywide/images/253/117211/65243__AST_IM_20230616_131857.jpeg |           3 |         0 |            0 | person×3         |
| data/raw/citywide/images/253/54327/339__20210529_110126.jpg            |           3 |         0 |            0 | person×3         |
| data/raw/citywide/images/253/62089/53197__AST_EX_20220711_151917.jpeg  |           3 |         0 |            0 | person×3         |
| data/raw/citywide/images/253/88133/18905__AST_IM_20210826_132843.jpeg  |           3 |         0 |            0 | person×3         |
| data/raw/citywide/images/253/88866/21666__MI_20210912_113311.jpg       |           3 |         0 |            0 | person×3         |
| data/raw/citywide/images/337/114560/62844__AST_IM_20220901_122451.jpeg |           3 |         0 |            0 | person×3         |
| data/raw/citywide/images/337/48933/15279__AST_EX_20210811_144952.jpeg  |           3 |         0 |            0 | person×3         |
| data/raw/citywide/images/573/109183/56775__AST_IM_20220722_115812.jpeg |           3 |         0 |            0 | person×3         |
| data/raw/citywide/images/573/117011/65053__AST_EX_20230607_110148.jpeg |           3 |         0 |            0 | person×3         |

## Notes

- Detector models: YOLOv8n (general) — no face model.
- This screen is **recall-oriented**: borderline detections are flagged. Manual review is still needed.
- Re-run with `--conf 0.3` for an even more cautious screen.
- For a closer face-specific pass, download `yolov8n-face.pt` from https://github.com/akanametov/yolov8-face and pass `--face-model yolov8n-face.pt`.
- Source CSV: `results/predictions/pii_screen.csv` — full per-image rows.
- Regenerate with `python scripts/screen_images_for_pii.py`.
