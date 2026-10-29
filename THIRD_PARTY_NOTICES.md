# Third-party notices

This repository is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)). It contains the
following files from other projects, under their own licences.

| File | Source | Licence | Copyright |
|---|---|---|---|
| `utility/generate_pattern.py` | OpenCV, [`apps/pattern-tools/generate_pattern.py`](https://github.com/opencv/opencv/blob/4.x/apps/pattern-tools/generate_pattern.py), 4.x branch | Apache License 2.0 | the OpenCV authors (see below) |
| `utility/svgfig.py` | OpenCV, [`apps/pattern-tools/svgfig.py`](https://github.com/opencv/opencv/blob/4.x/apps/pattern-tools/svgfig.py), 4.x branch (earlier revision) | BSD 3-Clause (full text in the file header) | 2022, Jim Pivarski |

`generate_pattern.py` is modified: an added attribution header and the `-b`/`--background`
option (a white background behind the pattern). `svgfig.py` is
unmodified apart from an added attribution line above its licence header.

## OpenCV copyright notice

From OpenCV's [`COPYRIGHT`](https://github.com/opencv/opencv/blob/4.x/COPYRIGHT) file:

```
Copyright (C) 2000-2022, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2023, OpenCV Foundation, all rights reserved.
Copyright (C) 2008-2016, Itseez Inc., all rights reserved.
Copyright (C) 2019-2023, Xperience AI, all rights reserved.
Copyright (C) 2019-2022, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
Copyright (C) 2022-2023, Southern University of Science And Technology, all rights reserved.
Copyright (C) 2023-2025, OpenCV AI, all rights reserved.

Third party copyrights are property of their respective owners.
```
