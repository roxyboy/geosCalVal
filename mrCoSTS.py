import numpy as np
import xarray as xr

import glob

import os.path as op

import warnings
warnings.filterwarnings("ignore")

from pydmd.mrcosts import mrCOSTS

grav = 9.807
meth = "linear"
ddir = "/storage/tartar/takaya/swot/"

for ipass in range(7,29):
    ds = xr.open_zarr(op.join(ddir,"SWOT_no_IT/CalVal_pass%03d_v2.0.1.zarr"
                            % ipass))
    ds.coords["num_lines"] = ("num_lines",range(len(ds.num_lines)))
    ds.coords["num_pixels"] = ("num_pixels",range(len(ds.num_pixels)))

    maskn = xr.DataArray(ds.latitude.where(ds.latitude>5.).to_masked_array().mask,
                     dims=ds.latitude.dims,
                     coords=ds.latitude.coords
                    )
    masks = xr.DataArray(ds.latitude.where(ds.latitude<-5.).to_masked_array().mask,
                     dims=ds.latitude.dims,
                     coords=ds.latitude.coords
                    )
    dsns = ds.where(~maskn,drop=True)
    dsns = xr.concat([dsns, ds.where(~masks,drop=True)
                 ], "num_lines"
                ).sortby("num_lines")

    inp = 3
    maskns = xr.DataArray(dsns.ssha_detided.to_masked_array().mask.mean(axis=0),
                     dims=dsns.ssha_detided.isel(cycle_num=0).dims,
                     coords=dsns.ssha_detided.isel(cycle_num=0).coords
                    )
    mm = xr.DataArray(maskns.where(maskns<.5
                              ).to_masked_array().mask,
                   dims=dsns.ssha_detided.isel(cycle_num=0).dims,
                   coords=dsns.ssha_detided.isel(cycle_num=0).coords
                  ).isel(num_pixels=slice(inp,-inp))

    fscale = np.ceil(14/np.sqrt(12)/2)

    h_interp = xr.open_zarr(op.join(ddir,
                        'CalVal/Pass%03d/ssha_rawgrid_%02dpoints.zarr' 
                        % (ipass,fscale))
                       ).ssha.interp(cycle_num=ds.cycle_num, 
                                     method="slinear",
                                     kwargs={"fill_value": "extrapolate"}
                                    ) * 1e-2   # centimeters to meters
    h_dropped = h_interp.where(mm.isel(num_pixels=slice(1,-1),
                                   num_lines=slice(1,-1)
                                  ).drop("cycle_num")!=1, 
                           drop=True)

    zchunk = 10000
    h_stacked = (h_dropped
             - h_dropped.mean(['num_lines','num_pixels'],skipna=True)
            ).stack(z=('num_lines','num_pixels')
                          ).chunk({'z':zchunk}).fillna(0.)

    nh = 24.*3600.
    window_lengths = np.ceil( np.array([9, 10, 30, 60])
                         * 86400/nh ).astype(int) # 24 hourly
    step_sizes = np.array([1, 1, 1, 1])

    svd_ranks = [4, 4, 10, 12]
    suppress_growth = True
    transform_method = "absolute"
    global_svd_array = [False] * len(window_lengths)

    dmd = mrCOSTS(
    svd_rank_array=svd_ranks,
    window_length_array=window_lengths,
    step_size_array=step_sizes,
    global_svd_array=global_svd_array,
    cluster_sweep=True,
    transform_method=transform_method,
    )

    dmd.fit(h_stacked.values.T, 
        np.atleast_2d(np.arange(len(h_stacked.cycle_num)
                               ) * nh)
       )

    dmd.to_netcdf(op.join(ddir, 
              "CalVal/mrCOSTS/Pass%03d_%02d_raw_%02dpoints"
              % (ipass,len(window_lengths),fscale)
                     )
             )

