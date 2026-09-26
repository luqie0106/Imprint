# Native burst sort scoring

This independent C++17 shared library computes only grayscale ROI sharpness
(Laplacian variance plus Sobel energy) and grayscale exposure threshold counts.
Python retains image decoding, grayscale conversion, face detection, model
inference, group ranking, and file operations. `BurstFilter` uses this library
only when `sort_backend="native"`; the default is the existing Python scoring
path and each native scoring error falls back to that Python implementation.

Build and validate locally:

```sh
cmake -S native-sort -B native-sort/build -DCMAKE_BUILD_TYPE=Release
cmake --build native-sort/build
ctest --test-dir native-sort/build --output-on-failure
python -m pytest -q native-sort/tests/test_native_sort.py
```

The exported C ABI is declared in `include/imprint_sort.h`. The Python bridge
searches `native-sort/build` during development and the PyInstaller `_MEIPASS`
root (including its `native-sort` subdirectory) in bundled runs.
