import numpy as np
import xarray as xr

import gsw

from scipy.interpolate import griddata

import os.path as op

import warnings
warnings.filterwarnings("ignore")


grav = 9.807
meth = "linear"
ddir = "/storage/tartar/takaya/swot/"

for ipass in range(1,29):
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

    low = xr.open_zarr(op.join(ddir,"CalVal/geos/Pass%03d/SSHa.zarr" 
                               % ipass)
                      ).ssha
    
    longi = low.longitude
    if (longi.max(skipna=True) - longi.min(skipna=True)) > 180.:
        long_adjusted = xr.where(longi > 180, longi - 360, longi)
    else:
        long_adjusted = longi

    lati = low.latitude

    dnum = 2000
    iy = 2
    for cc in low.cycle_num:
    
        for sy in np.arange(iy,len(lati.num_lines),dnum):

            ey = sy+dnum+iy
            if ey > len(lati.num_lines):
                ey = None
            
            LA = lati.isel(num_lines=slice(sy-iy,ey)).max(skipna=True).values
            la = lati.isel(num_lines=slice(sy-iy,ey)).min(skipna=True).values
            dla = np.abs(lati.where((lati<0.) & (mm.isel(num_pixels=slice(1,-1), 
                                                         num_lines=slice(1,-1)
                                                        )!=1)
                                   ).diff('num_lines')
                        ).min(skipna=True).values
            LO = long_adjusted.isel(num_lines=slice(sy-iy,ey)
                               ).max(skipna=True).values
            lo = long_adjusted.isel(num_lines=slice(sy-iy,ey)
                               ).min(skipna=True).values
            dlo = np.abs(longi.where(mm.isel(num_pixels=slice(1,-1), 
                                             num_lines=slice(1,-1)
                                            )!=1
                                    ).diff('num_pixels')
                        ).min(skipna=True).values
	    
            yy, xx = np.mgrid[la:LA:dla,
                              lo:LO:dlo
                             ]

            lon = griddata((lati.values.flatten(),
                        long_adjusted.values.flatten()
                       ), 
                       long_adjusted.values.flatten(), 
                       (yy, xx), method=meth
                      )
            lat = griddata((lati.values.flatten(),
                        long_adjusted.values.flatten()
                       ), 
                       lati.values.flatten(), 
                       (yy, xx), method=meth
                      )
        
            ssh = griddata((lati.values.flatten(),
                        long_adjusted.values.flatten()
                       ), 
                       low.sel(cycle_num=cc).values.flatten(), 
                       (yy, xx), method=meth
                      )
    
            low_gridded = xr.DataArray(ssh, dims=['YC','XC'], 
                                   coords={'YC':np.mean(yy, axis=1),
                                           'XC':np.mean(xx, axis=0)}
                                  )
    	    # print(low_gridded.XC, low_gridded.YC)
            xx, yy = np.meshgrid(low_gridded.XC, low_gridded.YC)
            print(xx.shape, yy.shape)
            dx = gsw.distance(xx, yy, p=0, axis=-1)
            dy = gsw.distance(xx, yy, p=0, axis=0)
        
            cori = gsw.f(low_gridded.YC)
            coriY = gsw.f(low_gridded.YC.rolling(YC=2).mean().dropna("YC"))
        
            u = -(grav * low_gridded.diff("YC")
              / coriY
              / dy)
            v = (grav * low_gridded.diff("XC")
             / cori 
             / dx)
        
            u_x = u.diff("XC").data / (.5*(dx[1:] + dx[:-1]))
            v_y = v.diff("YC").data / (.5*(dy[:,1:] + dy[:,:-1]))
            u_x = .25*(u_x[:-1,:-1] + u_x[1:,:-1] + u_x[1:,1:] + u_x[:-1,1:])
            v_y = .25*(v_y[:-1,:-1] + v_y[1:,:-1] + v_y[1:,1:] + v_y[:-1,1:])
        
            u_y = (u.diff("YC").data / (.5*(dy[1:] + dy[:-1])))[:,1:-1]
            v_x = (v.diff("XC").data / (.5*(dx[:,1:] + dx[:,:-1])))[1:-1]
            
            _vort = v_x - u_y
        
            sn = u_x - v_y
            ss = v_x + u_y
            _strain = np.sqrt(sn**2 + ss**2)
        
            YG = (.5*(low_gridded.YC[1:].data + low_gridded.YC[:-1].data))
            XG = (.5*(low_gridded.XC[1:].data + low_gridded.XC[:-1].data))
        
            vor_g = xr.DataArray(_vort, dims=['YF','XF'], 
                             coords={'YF':low_gridded.YC[1:-1].data, 
                                     'XF':low_gridded.XC[1:-1].data}
                            )
            str_g = xr.DataArray(_strain, dims=['YF','XF'], 
                             coords={'YF':low_gridded.YC[1:-1].data, 
                                     'XF':low_gridded.XC[1:-1].data}
                            )
            u_g = xr.DataArray(u.data, dims=['YG','XC'], 
                           coords={'YG':YG,
                                   'XC':low_gridded.XC.data}
                          )
            v_g = xr.DataArray(v.data, dims=['YC','XG'], 
                           coords={'YC':low_gridded.YC.data,
                                   'XG':XG}
                          )
         
            xf, yf = np.meshgrid(low_gridded.XC[1:-1].data, 
                             low_gridded.YC[1:-1].data)
            xg, yc = np.meshgrid(XG, low_gridded.YC.data)
            xc, yg = np.meshgrid(low_gridded.XC.data, YG)
    
            maskh = low.sel(cycle_num=cc).isel(num_pixels=slice(None,None),
                                           num_lines=slice(sy-iy,ey)
                                          ).to_masked_array().mask
        
            _vorg = griddata((yf.flatten(), xf.flatten()), 
                         vor_g.values.flatten(), 
                         (lati.isel(num_lines=slice(sy-iy,ey)).values,
                          long_adjusted.isel(num_lines=slice(sy-iy,ey)).values), 
                         method=meth
                        )
        
            _strg = griddata((yf.flatten(), xf.flatten()), 
                         str_g.values.flatten(), 
                         (lati.isel(num_lines=slice(sy-iy,ey)).values,
                          long_adjusted.isel(num_lines=slice(sy-iy,ey)).values),  
                         method=meth
                        )
        
            _vgo = griddata((yc.flatten(), xg.flatten()), 
                        v_g.values.flatten(), 
                        (lati.isel(num_lines=slice(sy-iy,ey)).values,
                         long_adjusted.isel(num_lines=slice(sy-iy,ey)).values), 
                        method=meth
                       )
        
            _ugo = griddata((yg.flatten(), xc.flatten()), 
                        u_g.values.flatten(), 
                        (lati.isel(num_lines=slice(sy-iy,ey)).values,
                         long_adjusted.isel(num_lines=slice(sy-iy,ey)).values), 
                        method=meth
                       )
    
            array = np.ma.masked_array(_vorg)
            # get only the valid values
            y1 = lati.isel(num_lines=slice(sy-iy,ey)).values[~array.mask][0]
            x1 = long_adjusted.isel(num_lines=slice(sy-iy,ey)).values[~array.mask][0]
            newarr = array[~array.mask][0]
            _vort = xr.DataArray(griddata((y1.flatten(),x1.flatten()),
                                      newarr.flatten(), 
                                      (lati.isel(num_pixels=slice(None,None),
                                                 num_lines=slice(sy-iy,ey)
                                                ).values,
                                       long_adjusted.isel(num_pixels=slice(None,None),
                                                          num_lines=slice(sy-iy,ey)
                                                         ).values), 
                                      method=meth,
                                               # fill_value=0.
                                     ), 
                             dims=low.sel(cycle_num=cc).isel(num_pixels=slice(None,None),
                                                             num_lines=slice(sy-iy,ey)
                                                ).dims,
                             coords=low.sel(cycle_num=cc).isel(num_pixels=slice(None,None),
                                                               num_lines=slice(sy-iy,ey)
                                                ).coords
                            ).where(~maskh)
            array = np.ma.masked_array(_strg)
            # get only the valid values
            y1 = lati.isel(num_lines=slice(sy-iy,ey)).values[~array.mask][0]
            x1 = long_adjusted.isel(num_lines=slice(sy-iy,ey)).values[~array.mask][0]
            newarr = array[~array.mask][0]
            _strain = xr.DataArray(griddata((y1.flatten(),x1.flatten()),
                                        newarr.flatten(), 
                                        (lati.isel(num_pixels=slice(None,None),
                                                 num_lines=slice(sy-iy,ey)
                                                ).values,
                                       long_adjusted.isel(num_pixels=slice(None,None),
                                                          num_lines=slice(sy-iy,ey)
                                                         ).values), 
                                        method=meth,
                                                   # fill_value=0.
                                       ), 
                               dims=low.sel(cycle_num=cc).isel(num_pixels=slice(None,None),
                                                                 num_lines=slice(sy-iy,ey)
                                                    ).dims,
                               coords=low.sel(cycle_num=cc).isel(num_pixels=slice(None,None),
                                                                 num_lines=slice(sy-iy,ey)
                                                    ).coords
                              ).where(~maskh)
            array = np.ma.masked_array(_ugo)
            # get only the valid values
            y1 = lati.isel(num_lines=slice(sy-iy,ey)).values[~array.mask][0]
            x1 = long_adjusted.isel(num_lines=slice(sy-iy,ey)).values[~array.mask][0]
            newarr = array[~array.mask][0]
            _ug = xr.DataArray(griddata((y1.flatten(),x1.flatten()),
                                    newarr.flatten(), 
                                    (lati.isel(num_pixels=slice(None,None),
                                                 num_lines=slice(sy-iy,ey)
                                                ).values,
                                     long_adjusted.isel(num_pixels=slice(None,None),
                                                          num_lines=slice(sy-iy,ey)
                                                         ).values), 
                                    method=meth,
                                               # fill_value=0.
                                   ), 
                           dims=low.sel(cycle_num=cc).isel(num_pixels=slice(None,None),
                                                             num_lines=slice(sy-iy,ey)
                                                ).dims,
                           coords=low.sel(cycle_num=cc).isel(num_pixels=slice(None,None),
                                                             num_lines=slice(sy-iy,ey)
                                                ).coords
                          ).where(~maskh)
            array = np.ma.masked_array(_vgo)
            # get only the valid values
            y1 = lati.isel(num_lines=slice(sy-iy,ey)).values[~array.mask][0]
            x1 = long_adjusted.isel(num_lines=slice(sy-iy,ey)).values[~array.mask][0]
            newarr = array[~array.mask][0]
            _vg = xr.DataArray(griddata((y1.flatten(),x1.flatten()),
                                    newarr.flatten(), 
                                    (lati.isel(num_pixels=slice(None,None),
                                                 num_lines=slice(sy-iy,ey)
                                                ).values,
                                     long_adjusted.isel(num_pixels=slice(None,None),
                                                          num_lines=slice(sy-iy,ey)
                                                         ).values), 
                                    method=meth,
                                               # fill_value=0.
                                   ), 
                           dims=low.sel(cycle_num=cc).isel(num_pixels=slice(None,None),
                                                             num_lines=slice(sy-iy,ey)
                                                ).dims,
                           coords=low.sel(cycle_num=cc).isel(num_pixels=slice(None,None),
                                                             num_lines=slice(sy-iy,ey)
                                                ).coords
                          ).where(~maskh)
    
            _ug[dict(num_pixels=slice(None,5))] = np.nan
            _ug[dict(num_pixels=slice(-4,None))] = np.nan
            _vg[dict(num_pixels=slice(None,5))] = np.nan
            _vg[dict(num_pixels=slice(-4,None))] = np.nan
            _vort[dict(num_pixels=slice(None,5))] = np.nan
            _vort[dict(num_pixels=slice(-4,None))] = np.nan
            _strain[dict(num_pixels=slice(None,5))] = np.nan
            _strain[dict(num_pixels=slice(-4,None))] = np.nan
            _ug[dict(num_pixels=slice(23,39))] = np.nan
            _vg[dict(num_pixels=slice(23,39))] = np.nan
            _vort[dict(num_pixels=slice(23,39))] = np.nan
            _strain[dict(num_pixels=slice(23,39))] = np.nan

            if sy == iy:
                vort = _vort.isel(num_lines=slice(None,-iy))
                strain = _strain.isel(num_lines=slice(None,-iy))
                ug = _ug.isel(num_lines=slice(None,-iy))
                vg = _vg.isel(num_lines=slice(None,-iy))
            else:
                if ey is not None:
                    vort = xr.concat([vort, _vort.isel(num_lines=slice(iy,-iy))
                                 ], "num_lines")
                    strain = xr.concat([strain, _strain.isel(num_lines=slice(iy,-iy))
                                   ], "num_lines")
                    ug = xr.concat([ug, _ug.isel(num_lines=slice(iy,-iy))
                               ], "num_lines")
                    vg = xr.concat([vg, _vg.isel(num_lines=slice(iy,-iy))
                               ], "num_lines")
                else:
                    vort = xr.concat([vort, _vort.isel(num_lines=slice(iy,None))
                                 ], "num_lines")
                    strain = xr.concat([strain, _strain.isel(num_lines=slice(iy,None))
                                   ], "num_lines")
                    ug = xr.concat([ug, _ug.isel(num_lines=slice(iy,None))
                               ], "num_lines")
                    vg = xr.concat([vg, _vg.isel(num_lines=slice(iy,None))
                               ], "num_lines")
    
            del _ug, _vg, _ugo, _vgo
            del _vort, _strain, _vorg, _strg
            del ssh, low_gridded

        if cc == low.cycle_num[0]:
            Vort = vort
            Strain = strain
            Ug = ug
            Vg = vg
        # ssha = low.sel(cycle_num=ii)
        else:
            Vort = xr.concat([Vort, vort], "cycle_num")
            Strain = xr.concat([Strain, strain], "cycle_num")
            Ug = xr.concat([Ug, ug], "cycle_num")
            Vg = xr.concat([Vg, vg], "cycle_num")
        # ssha = xr.concat([ssha, low.sel(cycle_num=ii)
        #                  ], "cycle_num")

    dsave = Vort.to_dataset(name="vort")
    dsave["strain"] = Strain
    dsave["vg"] = Vg
    dsave["ug"] = Ug
# dsave["ssha"] = ssha

    dsave.coords['cycle_num'] = low.cycle_num

    dsave.chunk({"cycle_num":1, "num_pixels":-1, "num_lines":2000}
           ).to_zarr(op.join(ddir,
            'CalVal/geos/Pass%03d/SSVs.zarr' 
                             % (ipass)
                            ), mode='w')
    dsave.close()
