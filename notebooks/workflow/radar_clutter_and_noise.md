---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: conda-env-sphinx-wradlib-py
  display_name: Python [conda env:sphinx-wradlib]
  language: python
---

```{image} ../../images/logos/wradlib_logo.svg.png
:width: 250px
:alt: wradlib Logo
```

# Radar Clutter and Noise

```{code-cell} ipython3
import numpy as np
import datetime as dt
import wradlib as wrl
import matplotlib.pyplot as plt
import xarray as xr
import xradar as xd
import fsspec
import icechunk
import hvplot
import hvplot.xarray

from dask.diagnostics import ProgressBar
```

## Claim Data

```{code-cell} ipython3
OSN_ENDPOINT = "https://umn1.osn.mghpcc.org"
BUCKET = "nexrad-arco"
```

```{code-cell} ipython3
fs = fsspec.filesystem(
    "s3", anon=True, client_kwargs={"endpoint_url": OSN_ENDPOINT},
)

for site, prefix in [("FGora", "fgora_vol"), ("Jastrebac", "jastrebac_vol")]:
    files = sorted(fs.glob(f"{BUCKET}/{prefix}/**/*.vol"))
    print(f"{site} raw files: {len(files)}")
    for f in files[:4]:
        print(f"  {f.split('/')[-1]}")
    print()
```

```{code-cell} ipython3
#prefix = "Fgora"  # single-pol, 12 sweeps × 360 az × 250 range, 2014 + 2017 + 2026
prefix = "jastrebac_250m"  # dual-pol, 12 × 360 × 1000, 2014 only
#prefix = "jastrebac_500m"  # dual-pol, 12 × 360 × 500,  2017 + 2026

storage = icechunk.s3_storage(
    bucket=BUCKET,
    prefix=prefix,
    endpoint_url=OSN_ENDPOINT,
    region="us-east-1",
    anonymous=True,
    force_path_style=True,
)
repo = icechunk.Repository.open(storage)
dtree = xr.open_datatree(
    repo.readonly_session("main").store,
    engine="zarr",
    consolidated=False,
    chunks={},
)
dtree
```

## Get lowest elevation sweep

```{code-cell} ipython3
swp = (
    dtree["JSTB_250_Dp_leto/sweep_0"]
    .to_dataset()
    .assign_coords(dtree["JSTB_250_Dp_leto"].coords)
    .assign_coords(sweep_mode="azimuth_surveillance")
    .wrl.georef.georeference(crs=wrl.georef.get_earth_projection())
)
swp.x.attrs = xd.model.get_longitude_attrs()
swp.y.attrs = xd.model.get_latitude_attrs()
swp.z.attrs = xd.model.get_altitude_attrs()
```

## Clutter Filter

Clutter filtering is a very complex issue and several algorithms have been invented to detect and mitigate the different types of clutter.

Anyway, we will use a simple Heuristic clutter detection based on distribution properties (“histo cut”). For this we will use all available data from the given day and sweep. See [EURADCLIM: The European climatological high-resolution gauge-adjusted radar precipitation dataset](https://essd.copernicus.org/preprints/essd-2022-334/) for further information.


## Convert data and calculate sum

First we convert the complete day of sweep data into a zarr archive. The we calculate the rainsum.

```{code-cell} ipython3
%%time
with ProgressBar():
    RSUM = swp.DBTH.sum("vcp_time")
swp = swp.assign(RSUM=RSUM)
display(swp)
```

```{code-cell} ipython3
swp.RSUM.wrl.vis.plot(vmin=0)
```

## Histo-Cut

Calculate Clutter and Beam Blockage fields and create simple plot.

```{code-cell} ipython3
import wradlib as wrl

clmap = swp.RSUM.wrl.classify.histo_cut(upper_frequency=0.05, lower_frequency=0.01)
clmap.wrl.vis.plot()
```

```{code-cell} ipython3
%%time
with rp.ProgressBar():
    swp = swp.pipe(
        rp.estimate_cmap,
        swp.RSUM,
        upper_frequency=0.05,
        lower_frequency=0.05,
        name="CMAP2",
    )
```

```{code-cell} ipython3
swp.CMAP2.wrl.vis.plot()
```

## Exercise 1: Remove clutter

Use clutter map to remove clutter from whole dataset. Plot 2-by-2 images

```{code-cell} ipython3

```

## Exercise 2: Remove Beam Blockage

Use beam blockage map to remove clutter from whole dataset. Plot 2-by-2 images.

```{code-cell} ipython3

```

## Exercise 3: Remove clutter and beam blockage

Use clutter and beamblockage map to remove clutter and beamblockage from whole dataset. Plot 2-by-2 images.

```{code-cell} ipython3

```

## Noise calculation and correction

This calculates signal-to-noise ratio SNRH and corrects RHOHV for noise in far ranges.

```{code-cell} ipython3
noise = (-40, -10, 1)
hist = rp.noise_hist(swp.DBZH, swp.RHOHV, noise=noise)
nl = rp.noise_level(hist)
```

```{code-cell} ipython3
rp.plot_noise_hist(hist)
```

```{code-cell} ipython3
swp = swp.pipe(rp.noise_correction, "DBZH", "RHOHV", nl)
display(swp)
```

## Save sweep to NetCDF4 file

```{code-cell} ipython3
print(rp.data_path())
```

```{code-cell} ipython3
stage = "01_pre"
fname = rp.data_path() / "lecture" / f"{stage}_{start_time:%Y%m%dT%H%M}.zarr"
print(fname)

swp.to_zarr(fname, mode="w", zarr_version=2)
```

## Exercise 4: Plot DBTH and SNRH

Check if SNRH can be used to mask DBTH.

```{code-cell} ipython3

```

## Solution 1: Remove clutter

```{code-cell} ipython3
swp_cl = swp.where(swp.CMAP2 != 1)
rp.plot_2by2(swp_cl)
```

## Solution 2: Remove Beam Blockage

```{code-cell} ipython3
swp_bb = swp.where(swp.CMAP2 != 2)
rp.plot_2by2(swp_bb)
```

## Solution 3: Remove clutter and beam blockage

```{code-cell} ipython3
swp_map = swp.where(swp.CMAP2 == 0)
rp.plot_2by2(swp_map)
```

## Solution 4: Plot DBTH and SNRH

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
rp.plot_moment(swp.DBTH, ax=ax1)
rp.plot_moment(swp.SNRH, ax=ax2, vmin=0, vmax=65)
fig.tight_layout()
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
rp.plot_moment(swp.DBTH.where(swp.SNRH > 0), ax=ax1)
rp.plot_moment(swp.SNRH.where(swp.SNRH > 0), ax=ax2, vmin=0, vmax=65)
fig.tight_layout()
```

## Footer

Clutter and Noise
