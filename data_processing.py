# data_processing.py

import numpy as np
from numpy.typing import NDArray
from datetime import datetime, timedelta
from pathlib import Path
from shapely.geometry import Polygon, LineString, Point
import requests
import logging
from tempfile import tempdir
import gzip
import shutil
from coordinates import satellite_xyz
from numpy.typing import NDArray

if not tempdir:
    tempdir = "./"

def load_nav_file(epoch: datetime) -> Path:        
    yday = str(epoch.timetuple().tm_yday).zfill(3)
    file_name = f"BRDC00IGS_R_{epoch.year}{yday}0000_01D_MN.rnx"
    url = f"https://simurg.space/files2/{epoch.year}/{yday}/nav/{file_name}.gz"
    gziped_file = Path(tempdir) / (file_name + ".gz")
    local_file = Path(tempdir) / file_name
    with open(gziped_file, "wb") as f:
        print(f"Downloading {gziped_file}")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Could not load file {response}")
        f.write(response.content)
    print(f"Unpack file {gziped_file} to {local_file}")
    with gzip.open(gziped_file, 'rb') as f_in:
        with open(local_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return local_file

station_cache = {}

def get_site_data_by_id(site_id: str):

    if site_id in station_cache:
        return station_cache[site_id]

    try:
        simurg_data = requests.get(f"https://api.simurg.space/sites/{site_id.lower()}").json()

        station_data = {
            'id': simurg_data['code'],
            'lat': simurg_data['location']['lat'],
            'lon': simurg_data['location']['lon'],
            'xyz': simurg_data['xyz']
        }
    
        station_cache[site_id] = station_data
        return station_data

    except requests.exceptions.RequestException as e:
        station_cache[site_id] = None
        return None


GNSS_SATS = []
GNSS_SATS.extend(['G' + str(i).zfill(2) for i in range(1, 33)])
GNSS_SATS.extend(['R' + str(i).zfill(2) for i in range(1, 25)])
GNSS_SATS.extend(['E' + str(i).zfill(2) for i in range(1, 37)])
GNSS_SATS.extend(['C' + str(i).zfill(2) for i in range(1, 41)])

TIME_STEP_SECONDS = 30
HEIGHT_OF_THIN_IONOSPHERE = 300000 # meters
RE = 6378000.0

def get_sat_xyz(
    nav_file: Path, 
    start: datetime, 
    end: datetime,
    sats: list = GNSS_SATS, 
    timestep: int = TIME_STEP_SECONDS
) -> tuple[dict[str, NDArray], list[datetime]]:
    xyz = {}
    times = []
    _timestep = timedelta(seconds=timestep)
    current = start + timedelta(0)
    while current < end:
        times.append(current)
        current = current + _timestep
        
    for sat in sats:
        sat_xyz = []
        keep_satellite = True
        for epoch in times:
            try:
                epoch_xyz = satellite_xyz(str(nav_file), sat[0], int(sat[1:]), epoch)
                sat_xyz.append(epoch_xyz)
            except Exception as e:
                print(f"Check nav file: {sat} and {epoch}. Error occured {e}. Skip satellite")    
                keep_satellite = False
                break # probably satellite not in nav-file skip all epochs
        if keep_satellite:
            xyz[sat] = np.array(sat_xyz)
    return xyz, times

def xyz_to_el_az(
    xyz_site: tuple[float], 
    xyz_sat: NDArray[float], 
    earth_radius=RE
) -> tuple[NDArray[float]]:
    """Computes elevation and azimuth to satellite.

    Parameters
    ----------
    xyz_site : tuple (x, y, z); cartesian coordinates of the observer
    xyz_sat : NDArray of (N, 3) shape; cartesian coordinates of the satellite
    Returns 
    -------
    elevation and azimuth as 2 numpy arrays
    """
    def cartesian_to_latlon(x, y, z, earth_radius=earth_radius):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        lon = np.arctan2(y, x)
        lat = np.arcsin(z / r)
        return lat, lon, r - earth_radius


    (x_0, y_0, z_0) = xyz_site
    (x_s, y_s, z_s) = xyz_sat[:, 0], xyz_sat[:, 1], xyz_sat[:, 2]

    (b_0, l_0, h_0) = cartesian_to_latlon(*xyz_site)
    (b_s, l_s, h_s) = cartesian_to_latlon(x_s, y_s, z_s)

    r_k = np.sqrt(x_s ** 2 + y_s ** 2 + z_s ** 2)

    sigma = np.arctan2(
        np.sqrt(1 - (np.sin(b_0) * np.sin(b_s) + np.cos(b_0) * np.cos(b_s) * np.cos(l_s - l_0)) ** 2), 
        (np.sin(b_0) * np.sin(b_s) + np.cos(b_0) * np.cos(b_s) * np.cos(l_s - l_0))
    )

    x_t = -(x_s - x_0) * np.sin(l_0) + (y_s - y_0) * np.cos(l_0)
    y_t = (
        -1 * (x_s - x_0) * np.cos(l_0) * np.sin(b_0) -
        (y_s - y_0) * np.sin(l_0) * np.sin(b_0) + 
        (z_s - z_0) * np.cos(b_0)
    )

    el = np.arctan2(
        (np.cos(sigma) - earth_radius / r_k),  
        np.sin(sigma)
    )
    az = np.arctan2(x_t, y_t)

    az = np.where(az < 0, az + 2*np.pi, az)
    return np.concatenate([[el], [az]]).T # to keep same notation as for xyz i.e. azimuth = elaz[:, 1]

def calculate_sips(
    site_lat: NDArray | float,           
    site_lon: NDArray | float,           
    elevation: NDArray,          
    azimuth: NDArray,            
    ionospheric_height: float = HEIGHT_OF_THIN_IONOSPHERE, 
    earth_radius: float = RE  
) -> NDArray:
    """
    Calculates subionospheric point and delatas from site
    Parameters:
        s_lat, slon - site latitude and longitude in radians
        hm - ionposheric maximum height (meters)
        az, el - azimuth and elevation of the site-sattelite line of sight in
            radians
        R - Earth radius (meters)
    """
    psi = (
        (np.pi / 2 - elevation) - 
        np.arcsin(np.cos(elevation) * earth_radius / (earth_radius + ionospheric_height))
    )
    lat = np.arcsin(
        np.sin(site_lat) * np.cos(psi) + 
        np.cos(site_lat) * np.sin(psi) * np.cos(azimuth)
    )
    lon = site_lon + np.arcsin(np.sin(psi) * np.sin(azimuth) / np.cos(site_lat))

    # Normalize longitude to [-pi, pi]
    lon = np.where(lon > np.pi, lon - 2 * np.pi, lon)
    lon = np.where(lon < -np.pi, lon + 2 * np.pi, lon)

    return np.concatenate([[np.degrees(lat)], [np.degrees(lon)]]).T

def get_elaz_for_site(site_xyz: tuple, sats_xyz: dict) -> dict:
    elaz = {}
    for sat, sat_xyz_data in sats_xyz.items():
        elaz[sat] = xyz_to_el_az(site_xyz, sat_xyz_data)
    return elaz

def get_sips_for_site(site_latlon: tuple, sats_elaz: dict) -> dict:
    sip_latlon = {}
    site_lat, site_lon = site_latlon
    for sat, elaz in sats_elaz.items():
        visible_mask = elaz[:, 0] > 0
        if np.any(visible_mask):
            sips = calculate_sips(
                np.radians(site_lat), 
                np.radians(site_lon), 
                elaz[visible_mask, 0], 
                elaz[visible_mask, 1]
            )
            sip_latlon[sat] = sips
    return sip_latlon

def get_main_map_data(study_date: datetime):

    anomaly_polygon_coords = [(0, 40), (120, 400), (120, 60), (0, 60)]
    anomaly_polygon_shapely = Polygon(anomaly_polygon_coords)
    effective_radius_km = 2000

    site_id="irkj"
    site = get_site_data_by_id(site_id)
    nav_file = load_nav_file(study_date)
    end_time = study_date + timedelta(days=1, seconds=-30)

    all_sats_xyz, times = get_sat_xyz(nav_file, study_date, end_time)

    final_trajectories = []
    # anomaly_center = anomaly_polygon_shapely.centroid

    site_point = Point(int(site['lon']), int(site['lat']))
    
    if  not(anomaly_polygon_shapely.contains(site_point) or site_point.distance(anomaly_polygon_shapely) * 111 < effective_radius_km):
        return 0
            
    print("Станция подошла") 
    sats_elaz = get_elaz_for_site(site['xyz'], all_sats_xyz)
    site_sips = get_sips_for_site((site['lat'], site['lon']), sats_elaz)

    for sat_id, sips_coords in site_sips.items():
        if len(sips_coords) < 2:
            continue 
                
        trajectory_line = LineString(sips_coords[:, ::-1])
        #TODO сделать функцию проверки пересечения на сфере   
        #TODO сохранянять данные о времени, и сохранять данные о точках пересечения траекторий sip-ов и полигона
        if trajectory_line.intersects(anomaly_polygon_shapely):
            print(f"    -> Найдено пересечение: станция {site['id']}, спутник {sat_id}")
            unique_id = f"{site['id']}-{sat_id}"
            trajectory_data = {
            'id': unique_id, 
            'points': sips_coords.tolist(),
            'station_id': site['id']
            }
            final_trajectories.append(trajectory_data)
    return final_trajectories 
   


def get_series_data_for_trajectory(trajectory_id: str):

    # Генерируем случайные данные для примера
    import random
    data_points = 30
    mock_data = {
        'time': [i for i in range(data_points)],
        'value': [random.uniform(0.5, 2.0) for _ in range(data_points)]
    }
    return mock_data