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

for ipass in [4,19]:
    if ipass == 21:
        reg = "PS"
    else:
        reg = "Kx"
    ds = xr.open_zarr(op.join(ddir,"tssh_swot1day/Pass%03d/" % ipass)
                      + reg + "/ssh-detided_rawgrid.zarr"
                     )
    # ds = xr.open_dataset(op.join(ddir,
    #                "HYCOM-tidal-correction/hycom_ssh_pass%02d_swot1day_v2.0.1.nc"
    #                               % ipass)
    #                      ).isel(num_pass=0)
    # ds.coords["latitude"] = (("num_lines","num_pixels"), dsH,latitude.data)
    # ds.coords["num_pixels"] = ("num_pixels",range(len(ds.num_pixels)))

    # maskn = xr.DataArray(ds.latitude.where(ds.latitude>5.).to_masked_array().mask,
    #                  dims=ds.latitude.dims,
    #                  coords=ds.latitude.coords
    #                 )
    # masks = xr.DataArray(ds.latitude.where(ds.latitude<-5.).to_masked_array().mask,
    #                  dims=ds.latitude.dims,
    #                  coords=ds.latitude.coords
    #                 )
    # dsns = ds.where(~maskn,drop=True)
    # dsns = xr.concat([dsns, ds.where(~masks,drop=True)
    #              ], "num_lines"
    #             ).sortby("num_lines")

    inp = 3
    # maskns = xr.DataArray(dsns.hycom_total.to_masked_array().mask.mean(axis=0),
    #                  dims=dsns.hycom_total.isel(num_cycle=0).dims,
    #                  coords=dsns.hycom_total.isel(num_cycle=0).coords
    #                 )
    # mm = xr.DataArray(maskns.where(maskns<.5
    #                           ).to_masked_array().mask,
    #                dims=dsns.hycom_total.isel(num_cycle=0).dims,
    #                coords=dsns.hycom_total.isel(num_cycle=0).coords
    #               ).isel(num_pixels=slice(inp,-inp))


    # h_interp = xr.open_zarr(op.join(ddir,
    #                     'tssh_swot1day/Pass%03d/ssh-detided_rawgrid.zarr' 
    #                     % (ipass))
    #                    ).ssha.interp(cycle_num=ds.num_cycle.data, 
    #                                  method="slinear",
    #                                  kwargs={"fill_value": "extrapolate"}
    #                                 ) * 1e-2   # centimeters to meters
    # h_dropped = h_interp.where(mm.drop("num_cycle")!=1, 
    #                            drop=True
    #                           ).isel(num_pixels=slice(1,-1),
    #                                  num_lines=slice(1,-1)
    #                                 )
    #h_dropped = xr.concat([ds.ssha.isel(num_pixels=slice(inp,33)),
    #                       ds.ssha.isel(num_pixels=slice(-33,-inp))
    #                      ], "num_pixels"
    #                     ).sortby("num_pixels") * 1e-2
    h_dropped = ds.ssha.dropna("num_lines",how="all") * 1e-2    

    zchunk = 10000
    h_stacked = (h_dropped
             - h_dropped.mean(['num_lines','num_pixels'],skipna=True)
            ).stack(z=('num_lines','num_pixels')
                          ).chunk({'z':zchunk}).fillna(0.)

    nh = 24.*3600.
    if reg == "PS":
        window_lengths = np.ceil( np.array([12, 13, 30, 60])
                         * 86400./nh ).astype(int) # 24 hourly
    else:
        window_lengths = np.ceil( np.array([12, 13, 30, 60])
                         * 86400./nh ).astype(int)
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

    dmd.to_netcdf(ddir 
              + "tssh_swot1day/mrCOSTS/"
              + reg 
              + op.join("/Pass%03d_%02d_detided"
              % (ipass,len(window_lengths))
                     )
             )

