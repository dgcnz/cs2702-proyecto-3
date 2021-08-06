# Backend

How to use RTree:

```python
from lib.rtree import RTree
import face_recognition

D = 128
rtree = RTree(D)

image_path = '...'
image = face_recognition.load_image_file(image_path)
features_vector = face_recognition.api.face_encodings(image)

answer_paths = []
indices = rtree.knn(features_vector, k)
for i in indices:
    answer_paths.append(rtree.get(i))
```


Run benchmarks


```python
pip install pytest-benchmark
python -m pytest tests/benchmarks.py
```
