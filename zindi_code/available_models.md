# Models (previous evals)

| Experience | State | Appr | Family                                        | Model                                     | Parameters | COCO Eval | Zindi   |
| ---------- | ----- | ---- | --------------------------------------------- | ----------------------------------------- | ---------- | --------- | ------- |
| 9          | 1     | 0    | DETR (End-to-End Object Detection)            | facebook/detr-resnet-101                  | 60.7       | 43.5      |         |
| 1          | 1     | 0    |                                               | facebook/detr-resnet-50                   | 41.6       | 42.0      |         |
| 2          | 1     | 0.25 |                                               | facebook/detr-resnet-50-dc5               | 41.6       | 43.6      |         |
| 10         | 1     | 0    |                                               | facebook/detr-resnet-101-dc5              | 60.7       | 44.9      |         |
| 8          | -     | 0    | Deformable DETR                               | SenseTime/deformable-detr-single-scale    | 34.2       |           |         |
| 11         | 1     | 0.25 |                                               | SenseTime/deformable-detr                 | 40.2       |           |         |
| 4          | -     | 0    |                                               | SenseTime/deformable-detr-with-box-refine | 41         |           |         |
| 3          | 1     | 1    | Conditional DETR                              | microsoft/conditional-detr-resnet-50      | 43.5       | 45.1      | 0.303   |
| 5          | 1     | 0    | DETA : Detection Transformers with Assignment | jozhang97/deta-resnet-50-24-epochs        | 48.5       | 50.2      |         |
| 6          | 1     | 0.5  | Yolos                                         | hustvl/yolos-base                         |            |           | 0.3440  |
| 7          | 1     | 1    |                                               | hustvl/yolos-small                        | 30.7       |           | 0.35598 |
