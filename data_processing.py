import math
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
import s2geometry as s2
from dateutil import tz

import h5py
import os
import sys
from dataclasses import dataclass
from enum import Enum

if not tempdir:
    tempdir = "./"


def load_nav_file(epoch: datetime) -> Path:        
    yday = str(epoch.timetuple().tm_yday).zfill(3)
    file_name = f"BRDC00IGS_R_{epoch.year}{yday}0000_01D_MN.rnx"
    url = f"https://simurg.space/files2/{epoch.year}/{yday}/nav/{file_name}.gz"
    gziped_file = Path(tempdir) / (file_name + ".gz")
    local_file = Path(tempdir) / file_name
    
    if local_file.exists():
        print(f"Using cached nav file {local_file}")
        return local_file
        
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
        station_data = {'id': simurg_data['code'], 'lat': simurg_data['location']['lat'], 'lon': simurg_data['location']['lon'], 'xyz': simurg_data['xyz']}
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

nav_file_cache = {} 
xyz_cache = {}
trajectory_details_cache = {}

def get_sat_xyz(nav_file: Path, start: datetime, end: datetime, sats: list = GNSS_SATS, timestep: int = TIME_STEP_SECONDS) -> tuple[dict[str, NDArray], list[datetime]]:
    xyz, times = {}, []
    _timestep = timedelta(seconds=timestep)
    current = start
    while current < end:
        times.append(current)
        current += _timestep
    for sat in sats:
        try:
            sat_xyz = [satellite_xyz(str(nav_file), sat[0], int(sat[1:]), epoch) for epoch in times]
            xyz[sat] = np.array(sat_xyz)
        except Exception as e:
            pass
    return xyz, times

def xyz_to_el_az(xyz_site, xyz_sat, earth_radius=RE):
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
    sigma = np.arctan2(np.sqrt(1 - (np.sin(b_0) * np.sin(b_s) + np.cos(b_0) * np.cos(b_s) * np.cos(l_s - l_0)) ** 2), (np.sin(b_0) * np.sin(b_s) + np.cos(b_0) * np.cos(b_s) * np.cos(l_s - l_0)))
    x_t = -(x_s - x_0) * np.sin(l_0) + (y_s - y_0) * np.cos(l_0)
    y_t = (-1 * (x_s - x_0) * np.cos(l_0) * np.sin(b_0) - (y_s - y_0) * np.sin(l_0) * np.sin(b_0) + (z_s - z_0) * np.cos(b_0))
    el = np.arctan2((np.cos(sigma) - earth_radius / r_k), np.sin(sigma))
    az = np.arctan2(x_t, y_t)
    az = np.where(az < 0, az + 2*np.pi, az)
    return np.concatenate([[el], [az]]).T 

def split_trajectory_by_gaps(points_with_time: list[dict]) -> list[list[dict]]:
    if not points_with_time: return []
    gap_threshold = timedelta(seconds=TIME_STEP_SECONDS * 120)
    all_segments, current_segment = [], [points_with_time[0]]
    for i in range(1, len(points_with_time)):
        time_difference = points_with_time[i]['time'] - points_with_time[i-1]['time']
        if time_difference > gap_threshold:
            all_segments.append(current_segment)
            current_segment = [points_with_time[i]]
        else:
            current_segment.append(points_with_time[i])
    all_segments.append(current_segment)
    return all_segments

def calculate_sips(site_lat, site_lon, elevation, azimuth, ionospheric_height=HEIGHT_OF_THIN_IONOSPHERE, earth_radius=RE):
    psi = ((np.pi / 2 - elevation) - np.arcsin(np.cos(elevation) * earth_radius / (earth_radius + ionospheric_height)))
    lat = np.arcsin(np.sin(site_lat) * np.cos(psi) + np.cos(site_lat) * np.sin(psi) * np.cos(azimuth))
    lon = site_lon + np.arcsin(np.sin(psi) * np.sin(azimuth) / np.cos(site_lat))
    lon = np.where(lon > np.pi, lon - 2 * np.pi, lon)
    lon = np.where(lon < -np.pi, lon + 2 * np.pi, lon)
    return np.concatenate([[np.degrees(lat)], [np.degrees(lon)]]).T

def get_elaz_for_site(site_xyz, sats_xyz):
    elaz = {}
    for sat, sat_xyz_data in sats_xyz.items():
        elaz[sat] = xyz_to_el_az(site_xyz, sat_xyz_data)
    return elaz

def generate_equatorial_poly():
    lat_min, lat_max, lon_min, lon_max, num_segments = -5, 5, -180, 180, 100
    path1 = list(zip(np.linspace(lon_max, lon_min, num_segments), [lat_min] * num_segments))
    path2 = [(lon_max, lat_max)]
    path3 = list(zip(np.linspace(lon_min, lon_max, num_segments), [lat_max] * num_segments))
    path4 = [(lon_min, lat_min)]
    return path4 + path3 + path2 + path1

def is_segment_valid(
    segment_points: list[dict],
    polygon_s2: s2.S2Polygon
) -> tuple[bool, list[dict]]:
    """
    Проверяет, пересекает ли сегмент границу аномалии >= 2 раз.
    Учитывает случай, когда сегмент обрывается внутри зоны.
    Возвращает (is_valid, список точек пересечения с КООРДИНАТАМИ и ВРЕМЕНЕМ)
    """
    if len(segment_points) < 2:
        return False, []

    intersections = []
    crossings_count = 0 
    
    try:
        prev_point_data = segment_points[0]
        prev_point_s2 = s2.S2LatLng.FromDegrees(prev_point_data['lat'], prev_point_data['lon']).ToPoint()
        prev_status = polygon_s2.Contains(prev_point_s2)

        for i in range(1, len(segment_points)):
            curr_point_data = segment_points[i]
            curr_point_s2 = s2.S2LatLng.FromDegrees(curr_point_data['lat'], curr_point_data['lon']).ToPoint()
            curr_status = polygon_s2.Contains(curr_point_s2)

            if curr_status != prev_status:
                crossings_count += 1
                
                # Интерполяция времени и координат пересечения
                prev_time = prev_point_data['time']
                curr_time = curr_point_data['time']
                intersection_time = prev_time + (curr_time - prev_time) / 2
                
                intersection_lat = (prev_point_data['lat'] + curr_point_data['lat']) / 2
                intersection_lon = (prev_point_data['lon'] + curr_point_data['lon']) / 2

                intersections.append({
                    'lat': intersection_lat,
                    'lon': intersection_lon,
                    'time': intersection_time
                })

            prev_point_data = curr_point_data
            prev_status = curr_status

        # Если количество пересечений нечетное, значит, сегмент закончился внутри зоны.
        # Считаем это событием "выхода" для полноты картины.
        if crossings_count > 1 and crossings_count % 2 != 0:
            crossings_count += 1
            last_point_data = segment_points[-1]
            # Добавляем последнюю точку сегмента как финальное пересечение
            intersections.append({
                'lat': last_point_data['lat'],
                'lon': last_point_data['lon'],
                'time': last_point_data['time']
            })

        # Сегмент валиден, если есть хотя бы одна пара "вход-выход"
        is_valid = crossings_count >= 2
        
        if is_valid:
            intersections.sort(key=lambda p: p['time'])

        return is_valid, intersections

    except Exception as e:
        print(f"Ошибка при проверке сегмента: {e}")
        return False, []
@dataclass
class ColorLimits():
    min: float; max: float; units: str

@dataclass(frozen=True)
class DataProduct():
    long_name: str; hdf_name: str; color_limits: ColorLimits

class DataProducts(Enum):
    roti = DataProduct("ROTI", "roti", ColorLimits(-0, 0.5, 'TECU/min'))
    dtec_2_10 = DataProduct("2-10 minute TEC variations", "dtec_2_10", ColorLimits(-0.4, 0.4, 'TECU'))
    timestamp = DataProduct("Timestamp", "timestamp", None)
    time = DataProduct("Time", None, None)

@dataclass
class GnssSat:
    name: str; system: str
    def __hash__(self): return hash(self.name)
    def __eq__(self, other):
        if not isinstance(other, GnssSat): return NotImplemented
        return self.name == other.name

h5_file_cache = {}
series_data_cache = {}

def load_h5_data(study_date: datetime, override: bool = False) -> Path:
    """Загружает HDF5 файл с данными временных рядов."""
    date_str = study_date.strftime("%Y-%m-%d")
    filename = Path(tempdir) / f"{date_str}.h5"
    
    if filename in h5_file_cache and not override:
        print(f"Using cached h5 file {filename}")
        return h5_file_cache[filename]
    
    if filename.exists() and not override:
        print(f"File {filename} exists. Use override=True to download it again.")
        h5_file_cache[filename] = filename
        return filename

    url = f"https://simurg.space/gen_file?data=obs&date={date_str}"
    
    with open(filename, "wb") as f:
        print(f"Downloading {filename} from {url}")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            h5_file_cache[filename] = None
            raise ValueError(f"Could not load h5 file {response.status_code}")
        f.write(response.content)
        
    h5_file_cache[filename] = filename
    return filename

def retrieve_series_data(local_file: Path) -> dict:
    """Читает все данные из HDF5 файла в память. Возвращает словарь."""
    date_str = local_file.stem
    if date_str in series_data_cache:
        print(f"Using cached series data for {date_str}")
        return series_data_cache[date_str]

    print(f"Reading HDF5 file {local_file} into memory cache...")
    all_data = {}
    try:
        with h5py.File(local_file, 'r') as f:
            for site_name in f.keys():
                all_data[site_name] = {}
                for sat_name in f[site_name].keys():
                    all_data[site_name][sat_name] = {}
                    sat_group = f[site_name][sat_name]
                    
                    timestamps = sat_group[DataProducts.timestamp.value.hdf_name][:]
                    times = [datetime.fromtimestamp(t, tz=tz.gettz("UTC")) for t in timestamps]
                    all_data[site_name][sat_name][DataProducts.time] = np.array(times)
                    
                    for dp in [DataProducts.roti, DataProducts.dtec_2_10]:
                        if dp.value.hdf_name in sat_group:
                            all_data[site_name][sat_name][dp] = sat_group[dp.value.hdf_name][:]
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        series_data_cache[date_str] = None
        return None
        
    series_data_cache[date_str] = all_data
    return all_data

def is_site_near_polygon(
    site_latlon: tuple[float, float],
    polygon_center_latlon: tuple[float, float],
    max_distance_km: float = 3000.0  
) -> bool:
    """
    Проверяет, находится ли станция в пределах max_distance_km от центра полигона.
    Использует расчет по большому кругу (great-circle distance).
    """
    site_lat, site_lon = site_latlon
    center_lat, center_lon = polygon_center_latlon

    # Преобразование градусов в радианы
    site_lat_rad = math.radians(site_lat)
    site_lon_rad = math.radians(site_lon)
    center_lat_rad = math.radians(center_lat)
    center_lon_rad = math.radians(center_lon)

    # Преобразование lat/lon в 3D декартовы единичные векторы
    site_vec = np.array([
        math.cos(site_lat_rad) * math.cos(site_lon_rad),
        math.cos(site_lat_rad) * math.sin(site_lon_rad),
        math.sin(site_lat_rad)
    ])
    center_vec = np.array([
        math.cos(center_lat_rad) * math.cos(center_lon_rad),
        math.cos(center_lat_rad) * math.sin(center_lon_rad),
        math.sin(center_lat_rad)
    ])

    # Вычисление угла между векторами через скалярное произведение
    dot_product = np.dot(site_vec, center_vec)
    # Ограничение значения для избежания ошибок с плавающей точкой
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = math.acos(dot_product)

    # Расчет расстояния
    earth_radius_km = RE / 1000.0
    distance_km = angle_rad * earth_radius_km

    return distance_km <= max_distance_km


def get_main_map_data(study_date: datetime):
    """
    Главная функция, которая сначала фильтрует СТАНЦИИ по близости к аномалии,
    а затем итерируется по ним.
    """
    print(f"Начинаю обработку для ВСЕХ станций за {study_date.date()}")
    
    SEGMENT_LIMIT_FOR_DEBUG = 10 
    DISTANCE_LIMIT_KM = 3000.0 

    all_valid_segments_for_all_sats = []
    
    try:
        h5_file = load_h5_data(study_date)
        if not h5_file:
            print("Крит. ошибка: не удалось загрузить H5 файл.")
            return [], study_date
        all_series_data = retrieve_series_data(h5_file)
        if not all_series_data:
            print("Крит. ошибка: не удалось прочитать данные из H5 файла.")
            return [], study_date
    except Exception as e:
        print(f"Крит. ошибка при работе с H5 файлом: {e}")
        return [], study_date

    nav_file = load_nav_file(study_date)
    end_time = study_date + timedelta(days=1, seconds=-30)

    anomaly_polygon_s2 = s2.S2Polygon(s2.S2Loop([
        p.ToPoint() for p in [s2.S2LatLng.FromDegrees(lat, lon) for lon, lat in generate_equatorial_poly()]
    ]))
    poly_center_latlon = (0, 0)

    available_stations = list(all_series_data.keys())
    print(f"Всего найдено станций в H5 файле: {len(available_stations)}.")
    print(f"Начинаю фильтрацию станций по расстоянию до экваториальной аномалии (< {DISTANCE_LIMIT_KM} км)...")

    for i, station_id in enumerate(available_stations):
        # Проверка лимита перед обработкой новой станции
        if SEGMENT_LIMIT_FOR_DEBUG is not None and len(all_valid_segments_for_all_sats) >= SEGMENT_LIMIT_FOR_DEBUG:
            print(f"Достигнут лимит в {SEGMENT_LIMIT_FOR_DEBUG} сегментов. Завершаю поиск.")
            break

        site_data = get_site_data_by_id(station_id)
        if not site_data:
            # Пропускаем, если нет данных о станции
            continue

        if not is_site_near_polygon(
            site_latlon=(site_data['lat'], site_data['lon']),
            polygon_center_latlon=poly_center_latlon,
            max_distance_km=DISTANCE_LIMIT_KM
        ):
            continue
        
        print(f"\n--- Обработка станции {i+1}/{len(available_stations)}: {station_id.upper()} (прошла фильтр) ---")
        
        available_sats_for_station = list(all_series_data.get(station_id, {}).keys())
        if not available_sats_for_station:
            print(f"    -> Пропуск: нет спутников для станции {station_id}")
            continue
        
        print(f"    Расчет XYZ для {len(available_sats_for_station)} спутников...")
        all_sats_xyz, times = get_sat_xyz(nav_file, study_date, end_time, sats=available_sats_for_station)
        if not all_sats_xyz:
            print(f"    -> Пропуск: не удалось рассчитать XYZ для спутников станции {station_id}")
            continue

        sats_elaz = get_elaz_for_site(site_data['xyz'], all_sats_xyz)
        
        for sat_id, elaz_data in sats_elaz.items():
            if SEGMENT_LIMIT_FOR_DEBUG is not None and len(all_valid_segments_for_all_sats) >= SEGMENT_LIMIT_FOR_DEBUG:
                break

            visible_mask = elaz_data[:, 0] > 0
            if not np.any(visible_mask): continue
            
            sips_coords = calculate_sips(np.radians(site_data['lat']), np.radians(site_data['lon']), elaz_data[visible_mask, 0], elaz_data[visible_mask, 1])
            points_with_datetime = [{'lat': sip[0], 'lon': sip[1], 'time': time} for sip, time in zip(sips_coords, np.array(times)[visible_mask])]
            
            time_based_segments = split_trajectory_by_gaps(points_with_datetime)

            part_number = 1
            for segment in time_based_segments:
                is_valid, intersections = is_segment_valid(segment, anomaly_polygon_s2)
                if is_valid:
                    print(f"        ✅ Сегмент {part_number} для {site_data['id']}-{sat_id} прошел проверку.")
                    segment_for_json = [{**p, 'time': p['time'].isoformat()} for p in segment]
                    intersections_for_json = [{**p, 'time': p['time'].isoformat()} for p in intersections]
                    all_valid_segments_for_all_sats.append({
                        'id': f"{site_data['id']}-{sat_id} ({part_number})",
                        'station_id': site_data['id'],
                        'satellite_id': sat_id,
                        'points': segment_for_json, 
                        'intersections': intersections_for_json,
                    })
                    part_number += 1
                
                if SEGMENT_LIMIT_FOR_DEBUG is not None and len(all_valid_segments_for_all_sats) >= SEGMENT_LIMIT_FOR_DEBUG:
                    break
            
            if SEGMENT_LIMIT_FOR_DEBUG is not None and len(all_valid_segments_for_all_sats) >= SEGMENT_LIMIT_FOR_DEBUG:
                break
        
        if SEGMENT_LIMIT_FOR_DEBUG is not None and len(all_valid_segments_for_all_sats) >= SEGMENT_LIMIT_FOR_DEBUG:
            break

    print(f"\nОбработка завершена. Всего найдено валидных сегментов: {len(all_valid_segments_for_all_sats)}")
    return all_valid_segments_for_all_sats, study_date



def get_trajectory_details(
    study_date: datetime,
    station_id: str,
    satellite_id: str
):
    """
    Получает детальные данные (точки со временем) для одной конкретной траектории.
    """
    print(f"Запрос деталей для траектории: {station_id}-{satellite_id}")

    def get_sips_with_time_for_site(
        site_latlon: tuple[float],
        sats_elaz: dict[str, NDArray],
        times: list[datetime]  
    ) -> dict:
        """
        Рассчитывает SIP'ы и сохраняет связь с временными метками.
        Возвращает словарь, где для каждого спутника есть 'sips' и 'times'.
        """
        sips_with_time = {}
        site_lat, site_lon = site_latlon
        
        times_arr = np.array(times)

        for sat, elaz in sats_elaz.items():
            visible_mask = elaz[:, 0] > 0
            
            if np.any(visible_mask):
                sips = calculate_sips(
                    np.radians(site_lat), 
                    np.radians(site_lon), 
                    elaz[visible_mask, 0], 
                    elaz[visible_mask, 1]
                )
                
                visible_times = times_arr[visible_mask]
                
                sips_with_time[sat] = {
                    'sips': sips,
                    'times': visible_times.tolist() 
                }
                
        return sips_with_time
    
    cache_key = f"{station_id}-{satellite_id}-{study_date.date()}"
    if cache_key in trajectory_details_cache:
        print(f"--> [КЭШ] Возвращаю готовые детали для {cache_key}")
        return trajectory_details_cache[cache_key]

    print(f"==> [РАСЧЕТ] Запрос деталей для {station_id}-{satellite_id}...")

    date_str = str(study_date.date())
    
    if date_str in nav_file_cache:
        nav_file = nav_file_cache[date_str]
        print(f"--> [КЭШ] Использую nav-файл: {nav_file}")
    else:
        print("==> [РАСЧЕТ] Загружаю новый nav-файл...")
        nav_file = load_nav_file(study_date)
        if nav_file:
            nav_file_cache[date_str] = nav_file
    
    if not nav_file: return None

    if date_str in xyz_cache:
        all_sats_xyz, times = xyz_cache[date_str]
        print(f"--> [КЭШ] Использую XYZ-координаты для {len(all_sats_xyz)} спутников.")
    else:
        print("==> [РАСЧЕТ] Вычисляю XYZ-координаты для всех спутников...")
        end_time = study_date + timedelta(days=1, seconds=-30)
        all_sats_xyz, times = get_sat_xyz(nav_file, study_date, end_time)
        xyz_cache[date_str] = (all_sats_xyz, times)
    
    if satellite_id not in all_sats_xyz:
        print(f"Ошибка: нет данных XYZ для спутника {satellite_id}")
        return None
        
    site_data = get_site_data_by_id(station_id)
    if not site_data: return None

    single_sat_xyz = {satellite_id: all_sats_xyz[satellite_id]}
    sats_elaz = get_elaz_for_site(site_data['xyz'], single_sat_xyz)

    site_sips_data = get_sips_with_time_for_site(
        (site_data['lat'], site_data['lon']),
        sats_elaz,
        times
    )

    if satellite_id in site_sips_data:
        sips_info = site_sips_data[satellite_id]
        
        points_with_time = []
        for sip_coord, sip_time in zip(sips_info['sips'], sips_info['times']):
            points_with_time.append({
                'lat': sip_coord[0],
                'lon': sip_coord[1],
                'time': sip_time.isoformat()
            })
        
        final_result = {
            'id': f"{station_id}-{satellite_id}",
            'points': points_with_time
        }
        
        trajectory_details_cache[cache_key] = final_result
        return final_result
    else:
        trajectory_details_cache[cache_key] = None 
        return None


def get_series_data_for_trajectory(
    study_date: datetime, 
    station_id: str, 
    satellite_id: str,
    product: DataProducts = DataProducts.roti
):
    """
    Получает ВСЕ данные временного ряда для указанной пары станция-спутник за сутки.
    Фильтрация по времени сегмента больше не производится.
    """
    try:
        h5_file = load_h5_data(study_date)
        if not h5_file:
            print("Не удалось загрузить H5 файл.")
            return None
            
        all_series_data = retrieve_series_data(h5_file)
        if not all_series_data:
            print("Не удалось прочитать данные из H5 файла.")
            return None

        station_data = all_series_data.get(station_id)
        if not station_data:
            print(f"Станция {station_id} не найдена в H5 файле.")
            return None
            
        sat_data = station_data.get(satellite_id)
        if not sat_data:
            print(f"Спутник {satellite_id} не найден для станции {station_id} в H5 файле.")
            return None

        if product not in sat_data:
            print(f"Продукт {product.name} не найден для {station_id}-{satellite_id}.")
            return None

        times = sat_data[DataProducts.time]
        values = sat_data[product]
        
        if len(times) == 0:
            print(f"Нет данных временного ряда для {station_id}-{satellite_id}.")
            return None

        return {
            'time': [t.isoformat() for t in times], # Возвращаем ВСЕ временные метки
            'value': values.tolist(),              # Возвращаем ВСЕ значения
            'product_name': product.value.long_name,
            'product_units': product.value.color_limits.units
        }

    except Exception as e:
        print(f"Произошла ошибка при получении данных для графика: {e}")
        return None
