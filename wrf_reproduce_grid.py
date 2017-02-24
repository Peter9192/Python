#!usr/bin/env python
import xarray as xr
import numpy as np
import pyproj

def get_wrf_proj(wrfout):
    ''' Given a wrfout file as xarray dataset object,
        Return a pyproj object consistent with the wrf projection '''

    # Check which projection is used
    if wrfout.MAP_PROJ ==1:
        # Obtain projection parameters from WRFout
        clon=wrfout.CEN_LON
        clat=wrfout.CEN_LAT # equal to MOAD_CEN_LAT if 1 domain is used
        lat1=wrfout.TRUELAT1
        lat2=wrfout.TRUELAT2

        # Cartopy not installed on cartesius, use pyproj instead (ugly!)
        proj_string = ("+proj=lcc "
                       "+lat_1=%f "
                       "+lat_2=%f "
                       "+lat_0=%f "
                       "+lon_0=%f "
                       "+a=6370000"
                       )%(lat1,lat2,clat,clon)
        wrf_proj = pyproj.Proj(proj_string)
    else:
        raise ValueError("This WRF output is not on a Lambert grid")

    return wrf_proj

def get_wrf_grid_params(wrfout):
    ''' Given a wrfout file as xarray dataset object,
        return the relevant grid parameters as a dict '''

    wrf_grid_params = {
        "dx":wrfout.DX,
        "dy":wrfout.DY,
        "nx":wrfout.attrs["WEST-EAST_PATCH_END_UNSTAG"],
        "ny":wrfout.attrs["SOUTH-NORTH_PATCH_END_UNSTAG"]
    }
    return wrf_grid_params

def reconstruct_grid(grid_params):
    ''' Given dx,dy,nx and ny as dict,
        return x,y as 1d and X,Y as 2d grid coordinates '''

    def make_grid_coordinate(nx,dx):
        ''' Given a number of grid points and grid spacing,
        return a 1D-array centered around 0 '''
        if nx%2 == 1:
            ix = np.arange(-(nx//2),nx//2+1,1)
        else:
            ix = np.arange(-(nx-1)/2.,(nx+1)//2.,1)
        return ix*dx

    # Mass point coordinates
    x = make_grid_coordinate(grid_params["nx"],grid_params["dx"])
    y = make_grid_coordinate(grid_params["ny"],grid_params["dy"])

    # Staggered coordinates
    x_stag = make_grid_coordinate(grid_params["nx"]+1,grid_params["dx"])
    y_stag = make_grid_coordinate(grid_params["nx"]+1,grid_params["dx"])

    # 2d meshgrids
    mass_grid = np.meshgrid(x,y)
    u_grid = np.meshgrid(x_stag,y)
    v_grid = np.meshgrid(x,y_stag)

    coords = {"x":x,"y":y,"x_stag":x_stag,"y_stag":y_stag},
    grids = {"mass_grid":mass_grid,"u_grid":u_grid,"v_grid":v_grid}
    return coords,grids

def load_wrf_grids(wrfout):
    ''' Given a wrfout file as xarray dataset object,
        return a dictionary with the mass and staggered coordinates '''

    x = wrfout.XLONG.isel(Time=0).values
    y = wrfout.XLAT.isel(Time=0).values
    mass_grid = (x,y)

    x_u_stag = wrfout.XLONG_U.isel(Time=0).values
    y_u_stag = wrfout.XLAT_U.isel(Time=0).values
    u_grid = (x_u_stag,y_u_stag)

    x_v_stag = wrfout.XLONG_V.isel(Time=0).values
    y_v_stag = wrfout.XLAT_V.isel(Time=0).values
    v_grid = (x_v_stag,y_v_stag)

    return {"mass_grid":mass_grid,"u_grid":u_grid,"v_grid":v_grid}

def check_grid_consistency(grid1,grid2,proj1,proj2):
    ''' Given two 2D (mesh)grids and their respective projections,
        perform a number of checks to verify their consistency '''

    def print_diffs(xt,yt,x2,y2):
        print 'Total accumulated longitude mismatch:',(xt-x2).sum()
        print 'Total accumulated latitude mismatch:',(yt-y2).sum()
        print 'Average longitudinal grid mismatch:',(xt-x2).mean()
        print 'Average latitudinal mismatch:',(yt-y2).mean()
        print 'Maximum longitudinal grid mismatch:',(xt-x2).max(),np.unravel_index((xt-x2).argmax(),xt.shape)
        print 'Maximum latitudinal mismatch:',(yt-y2).max(),np.unravel_index((yt-y2).argmax(),yt.shape),'\n'
        return

    for grid in ['mass_grid','u_grid','v_grid']:
        print 'Performing transformation on the %s'%grid

        x1,y1 = grid1['mass_grid']
        x2,y2 = grid2['mass_grid']

        # Transform proj1 to proj 2
        print 'Transform proj1 to proj2 (probably LCC to lat/lon)'
        xt,yt = pyproj.transform(proj1,proj2,x1,y1)
        print_diffs(xt,yt,x2,y2)

        # Transform proj2 to proj 1
        print 'Transform proj2 to proj1 (probably lat/lon to LCC)'
        xt,yt = pyproj.transform(proj2,proj1,x2,y2)
        print_diffs(xt,yt,x1,y1)

    return

def reproduce_wrf_grid(wrfout):
    ''' Given a wrfout file as xarray dataset object,
        - get projection information from wrfout (only works for Lambert),
        - reproduce a regular grid in LCC coordinates,
        - also read the irregular, 2D WRF lat/lon variables,
        - check whether these grids are consistent
        - return the 1d coordinate arrays and 2d meshgrids as dicts '''

    # Get a (pyproj) projection object using the projection parameters from wrf
    proj_wrf_lcc = get_wrf_proj(wrfout)

    # Also initialize a 'normal' lat/lon projection object for transformations
    proj_wgs84 = pyproj.Proj("+init=EPSG:4326") # Regular lat/lon, equivalent to:
    # Use spherical lat/lon for WRF datums: http://www.pkrc.net/wrf-lambert.html
    proj_sll = pyproj.Proj("+proj=latlong +a=6370 +b=6370 +towgs84=0,0,0 +no_defs")

    # Get the grid parameters from the wrfout files
    grid_params_wrf = get_wrf_grid_params(wrfout)

    # Reconstruct the WRF grid, in LCC coordinates
    coords_peter, grids_peter = reconstruct_grid(grid_params_wrf)

    # Load the WRF grid from the wrfout files (irregular latlon arrays !?!)
    grids_wrfout = load_wrf_grids(wrfout)

    # Check that the reconstructed LCC grid is consistent with the WRF Lat/Lons
    check_grid_consistency(grids_peter,grids_wrfout,proj_wrf_lcc,proj_wgs84)
    return coords_peter, grids_peter

if __name__=="__main__":
    # Sample wrf output
    wrfpath = '/scratch-shared/peter919/case00_ACM2/wrfout_d01_2013-03-20_00:00:00'
    wrfout = xr.open_dataset(wrfpath)
    coords,grids = reproduce_wrf_grid(wrfout)
