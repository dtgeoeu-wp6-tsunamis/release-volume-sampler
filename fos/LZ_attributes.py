#This code computes the probability distributions of landslide size, movement 
#and direction

#by Laide Ojomo, 2023, UT Austin

# Import necessary libraries
import rasterio as rio
import geopandas as gpd
import pandas as pd
import time
import numpy as np
import scipy.integrate as spi
from rasterio import features
from shapely.geometry import shape, mapping
from rasterio.features import shapes
from geopandas import GeoDataFrame
import rioxarray as rxr

# calculate the area of polygons
def get_area(ls_poly):
    Area = []
  # Iterate over rows and print the area of a Polygon
    for index, row in ls_poly.iterrows():
      # Get the area of the polygon
        poly_area = row['geometry'].area
        Area.append(poly_area)
    return Area

# compute the CDF for slope aspect
def get_cdf(asp): # without considering weights
    aspect, counts = np.unique(asp, return_counts=True)
    cusum = np.cumsum(counts)
    cdf = cusum / cusum[-1]
    return aspect, cdf

# compute the probabilities for slope aspect
def prob_aspect(aspect):
    bins = np.arange(0,365,5) #bins at 5 deg intervals
    heights,bins = np.histogram(aspect,bins=bins)
    heights = heights/sum(heights)
    
    return heights

def get_prob(a, b):
        # area_poly = area of polygon in sq.m
    f = lambda x : (1/(0.00128*0.88726))*((0.00128/(x+0.000132))**2.4)*(np.exp(-0.00128/(x+0.000132))) #full PDF

    # x and y values for the trapezoid rule function
    x1 = np.log10(a); x2 = np.log10(b); N = 200
    x = np.logspace(x1,x2,N+1); y = f(x)
    
    prob = spi.trapezoid(y,x)
    
    return prob

area_bins = [0.00001, 0.0001, 0.0025, 0.01, 0.04, 10] # in sq km
len_bins = [3, 10, 50, 100, 200, 500]

def get_distributions(area):
    global prob1,prob2,prob3,prob4,prob5
    area_poly = area/1e6
    a = area_bins[0]; b = area_bins[1]; c = area_bins[2]; d = area_bins[3]; e = area_bins[4]; f = area_bins[5]
    total_prob = get_prob(a, f) # total prob should be 1
    if area_poly >= a and area_poly > b:
        prob1 = get_prob(a, b)
    elif area_poly >= a and area_poly < b:
        prob1 = get_prob(a, area_poly)
        
    if area_poly >= b and area_poly > c:
        prob2 = get_prob(b, c)
    elif area_poly >= a and area_poly < c:
        prob2 = get_prob(b, area_poly)

    if area_poly > c and area_poly > d:
        prob3 = get_prob(c, d)
    elif area_poly >= a and area_poly < d:
        prob3 = get_prob(c, area_poly)

    if area_poly > d and area_poly > e:
        prob4 = get_prob(d, e)
    elif area_poly >= a and area_poly < e:
        prob4 = get_prob(d, area_poly)

    if area_poly > e and area_poly > f:
        prob5 = get_prob(e, f)
    elif area_poly >= a and area_poly < f:
        prob5 = get_prob(e, area_poly)

    prob = np.array((prob1, prob2, prob3, prob4, prob5))
    prob[prob<0] = 0
    final_prob = prob/sum(prob)
    
    return area_poly, prob, final_prob

def get_lz(raster):
    with rio.open(raster) as src:
        data = src.read(1)
        is_valid = (data ==2).astype(np.uint8)
        # Use a generator instead of a list
        shape_gen = ((shape(s), v) for s, v in shapes(is_valid, transform=src.transform))
        # build a dict from unpacked shapes
        gdf = GeoDataFrame(dict(zip(["geometry", "class"], zip(*shape_gen))), crs=src.crs)
        #gdf = gdf[~(gdf.is_empty)]

    # Delete areas <= 100 m^2
    Area = get_area(gdf)
    gdf['area_sqm'] = Area
    gdf = gdf[(gdf['area_sqm'] > 100)]
    gdf = gdf[(gdf['class'] >= 1.0)]
    gdf.drop(gdf[gdf['area_sqm'] <= 100].index, inplace=True)
    gdf.drop(gdf[gdf['class'] < 1.0].index, inplace=True)
    return gdf

def get_lzp(gdf, slu):
    ls_poly = gpd.overlay(gdf, slu, how='intersection')
    Area = get_area(ls_poly)
    ls_poly['area'] = Area
    ls_poly.drop(ls_poly[ls_poly['area'] <= 100].index, inplace=True)
    ls_poly.drop(['area_sqm'], axis=1, inplace=True)
    ls_poly.set_crs(epsg=32611, inplace=True)
    ls_poly.index = np.arange(1, len(ls_poly)+1)
    ls_poly.reset_index(inplace=True)
    ls_poly.rename(columns = {'index':'landslide_zone_ID'}, inplace = True)
    return ls_poly

def get_points(raster, disp, asp, ls_poly):

    rds = rxr.open_rasterio(raster)
    rds.name = "disp"
    df = rds.squeeze().to_dataframe().reset_index()
    points = gpd.GeoDataFrame(df, crs=rds.rio.crs, geometry=gpd.points_from_xy(df.x, df.y))
    del rds, df
    points = points[points['disp'] == 2.0]
    # join points to LZPs for gridded deliverable
    ls_poly.set_crs(epsg=32611, inplace=True)
    ls_pts = gpd.sjoin(points, ls_poly, how='inner', predicate='intersects')
    ls_pts.drop(['disp','index_right'], axis=1, inplace=True)
    ls_pts_copy = ls_pts.copy()
    ls_pts_copy = ls_pts_copy.to_crs(4326)
    ls_pts_copy['longitude'] = ls_pts_copy.geometry.x  
    ls_pts_copy['latitude'] = ls_pts_copy.geometry.y
    ls_pts_copy = ls_pts_copy[['landslide_zone_ID', 'latitude', 'longitude']]
#     add raster attributes
    coord_list = [(x,y) for x,y in zip(ls_pts['geometry'].x , ls_pts['geometry'].y)]
    ls_pts['disp_prob'] = [x for x in disp.sample(coord_list)]
    ls_pts['aspect'] = [x for x in asp.sample(coord_list)]

    ls_pts = ls_pts[['landslide_zone_ID', 'geometry', 'area', 'disp_prob', 'aspect']]
    del coord_list, points
    return ls_pts, ls_pts_copy

def lzp(ls_poly, ls_pts):
    ls_combined = gpd.sjoin(ls_poly, ls_pts, how='inner', predicate='contains')
    ls_combined.drop(['index_right','landslide_zone_ID_right','area_right'], axis=1, inplace=True)
    ls_combined.rename(columns = {'landslide_zone_ID_left':'landslide_zone_ID'}, inplace = True)
    ls_combined.rename(columns = {'area_left':'area'}, inplace = True)
    ls_agg_disp = (ls_combined.groupby(['landslide_zone_ID'])['disp_prob'].apply(list)).reset_index()
    ls_agg_asp = (ls_combined.groupby(['landslide_zone_ID'])['aspect'].apply(list)).reset_index()
    
    prob_dist = []
    for direction in ls_agg_asp['aspect']:
        aspect, cdf = get_cdf(direction)
        probabilities = prob_aspect(aspect) # probabilities for aspect
        prob_dist.append(probabilities.round(3))
    ls_agg_asp['asp_prob'] = prob_dist
    
    mean_prob = []
    for index, row in ls_agg_disp.iterrows():
        mean_arr = np.mean(row['disp_prob'], axis=0)
        mean_prob.append(mean_arr.round(3))
    ls_agg_disp['disp_prob2'] = mean_prob
    
    ls_poly2 = pd.merge(ls_poly, ls_agg_disp, on='landslide_zone_ID', how='outer')
    ls_poly2 = pd.merge(ls_poly2, ls_agg_asp, on='landslide_zone_ID', how='outer')
    return ls_poly2

def lzp_attr(ls_poly2):
    
    prob_dist = []
    for area_poly in ls_poly2['area']:
        area, prob, final_prob = get_distributions(area_poly)
        prob_dist.append(final_prob.round(3))
    ls_poly2['size_prob_dist'] = prob_dist
    ls_poly2.drop(['geometry','area','disp_prob','aspect'], axis=1, inplace=True)
    ls_poly2.rename(columns = {'size_prob_dist':'PMF of pipeline exposure length', 'asp_prob':'PMF of direction', 
                              'disp_prob2':'PMF of movement'}, inplace = True)
    ls_poly2 = ls_poly2[['landslide_zone_ID', 'PMF of pipeline exposure length', 'PMF of direction', 'PMF of movement']]
    return ls_poly2

# Open files and store metadata
raster = "loc1_dispgt5.tif"
disp = rio.open('loc1_binned_disp.tif')
asp = rio.open('loc1_aspect.tif')

shapefile = "loc1_su.shp"

gdf = get_lz(raster)

slu = gpd.read_file(shapefile)
#slu = slu[~(slu.is_empty)]

del shapefile

ls_poly = get_lzp(gdf, slu)

ls_pts, ls_pts_copy = get_points(raster, disp, asp, ls_poly)
ls_pts_copy.to_csv('loc1_lz_grids.csv', index=False)
del disp, asp, ls_pts_copy


ls_poly2 = lzp(ls_poly, ls_pts)
del ls_poly, ls_pts

ls_poly2 = lzp_attr(ls_poly2)

ls_poly2.to_csv('loc1_lz.csv', index=False)