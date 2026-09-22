# Third-party notices

Imprint includes or dynamically uses the following third-party components for
lens correction. These notices do not change the Apache-2.0 license of Imprint
itself.

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
