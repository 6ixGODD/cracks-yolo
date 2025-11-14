@echo off
setlocal

poetry run python -m cracks_yolo dataset merge ^
    -i data/raw/BDS/annotations.json coco BDS ^
    -i data/raw/KN/annotations.json coco KN ^
    -i data/raw/LG/annotations.json coco LG ^
    -i data/raw/SXX/annotations.json coco SXX ^
    -i data/raw/SZY/annotations_train.json coco SZY ^
    -i data/raw/SZZY/annotations.json coco SZZY ^
    -o data/cracksdet_coco_701515 ^
    --output-format coco ^
    --train-ratio 0.7 ^
    --val-ratio 0.15 ^
    --test-ratio 0.15 ^
    --naming sequential_prefix


endlocal
