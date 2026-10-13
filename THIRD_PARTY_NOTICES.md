# Third-party notices

Imprint includes or dynamically uses the following third-party components.
These notices do not change the Apache-2.0 license of Imprint itself.

## Lensfun library

- Project: https://github.com/lensfun/lensfun
- Library license: GNU Lesser General Public License, version 3.0
- License text: `third_party/licenses/LENSFUN_LGPL_3.0.txt`
- Referenced GNU GPL version 3 text: `third_party/licenses/GNU_GPL_3.0.txt`
- The binary library is supplied by the `lensfunpy` wheel as a separate shared
  library. Corresponding Lensfun 0.3.4 source is available from
  https://github.com/lensfun/lensfun/releases/tag/v0.3.4.
- Imprint does not modify the Lensfun library.
- Release packages keep Lensfun as a separate shared library inside the
  `dist-python/imprint_api` sidecar directory. Recipients may replace or relink
  it with a compatible modified build and may reverse engineer Imprint only as
  needed to debug such LGPL modifications, as permitted by LGPL-3.0 section 4.
- Imprint imposes no additional restriction on exercising the rights granted by
  the Lensfun license. The notices and complete license texts are included in
  both this repository and packaged sidecar resources.

## LibRaw metadata / rendering and TicoRAW reconstruction reference

- Source: https://github.com/pdcgomes/redlamp-libraw/tree/4abfcd2ab82f2211b1f7e7c3cd83974c87e0d812
- Fixed archive SHA-256: `2ca1d9673531fccabcc7c8bce589444163868bcde625aa1f09eb442df1bcce2d`.
- This component is used under LGPL-2.1, with license and attribution in
  `third_party/licenses/LIBRAW_LGPL_2.1.txt` and `LIBRAW_COPYRIGHT.txt`.
- `native-raw` builds a separate replaceable shared library containing LibRaw's
  metadata and color rendering with Imprint's C++/Metal TicoRAW operators.
  Its adapted reconstruction sources retain the upstream attribution and are
  provided under LGPL-2.1. The rest of Imprint retains its existing license.
- Corresponding source and build instructions are in `native-raw`, together
  with the fixed downloadable LibRaw source above. Recipients can rebuild and
  replace the shared library, including for debugging their modifications.
- Release packages include this notice and the full license texts.

## Lensfun database

- Project: https://github.com/lensfun/lensfun
- Database license: Creative Commons Attribution-ShareAlike 3.0 Unported
- License text: `third_party/licenses/LENSFUN_DATABASE_CC_BY_SA_3.0.txt`
- Database snapshot: commit `2f1be1deab50da7869fc5d23f481939803d845bf`
  (2026-09-21)
- The XML files in `third_party/lensfun-db` were generated from that upstream
  snapshot with Lensfun's official `tools/update_database/generate_db.py`
  converter. The source database format v2 was converted to format v1 for the
  Lensfun 0.3.x library bundled by lensfunpy. This converted database remains
  available under CC BY-SA 3.0 and is loaded as separate data files.

## lensfunpy

- Project: https://github.com/letmaik/lensfunpy
- Version: 1.18.0
- License: MIT
- License text: `third_party/licenses/LENSFUNPY_MIT.txt`

## imagecodecs

- Project: https://github.com/cgohlke/imagecodecs
- Version: 2025.x
- License: BSD 3-Clause; bundled codec libraries retain their respective licenses.
- Used to encode lossless JPEG and JPEG XL image data in DNG exports.
- Package and bundled library license files are collected with the Python sidecar.
