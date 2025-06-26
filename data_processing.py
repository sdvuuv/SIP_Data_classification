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
import asyncio
import aiohttp
import json

import h5py
import os
import sys
from dataclasses import dataclass
from enum import Enum

if not tempdir:
    tempdir = "./"


CACHE_DIR = Path("geometry_cache")
CACHE_DIR.mkdir(exist_ok=True) # Создаем папку для кэша, если ее нет

def cache_segment_data(segment_id: str, segment_data: dict):
    """Сохраняет данные сегмента в отдельный JSON-файл."""
    # Используем безопасное имя файла
    safe_filename = segment_id.replace('/', '_').replace('\\', '_') + ".json"
    file_path = CACHE_DIR / safe_filename
    try:
        with open(file_path, 'w') as f:
            json.dump(segment_data, f)
    except Exception as e:
        print(f"Ошибка при сохранении кэша для {segment_id}: {e}")

def get_segment_from_cache(segment_id: str) -> dict | None:
    """Читает данные сегмента из JSON-файла."""
    safe_filename = segment_id.replace('/', '_').replace('\\', '_') + ".json"
    file_path = CACHE_DIR / safe_filename
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка при чтении кэша для {segment_id}: {e}")
            return None
    else:
        # Это важный случай: если файла нет (после перезапуска), мы должны сообщить об этом.
        return None

def clear_geometry_cache():
    """Очищает папку с файлами кэша."""
    print("Очистка файлового кэша геометрии...")
    for item in CACHE_DIR.iterdir():
        item.unlink() # Удаляем файл


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
    gap_threshold = timedelta(seconds=TIME_STEP_SECONDS * 30)
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

async def fetch_site_data_async(session, site_id: str):
    """Асинхронно запрашивает данные для ОДНОЙ станции."""
    if site_id in station_cache:
        return station_cache[site_id]
        
    url = f"https://api.simurg.space/sites/{site_id.lower()}"
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            simurg_data = await response.json()
            station_data = {
                'id': simurg_data['code'],
                'lat': simurg_data['location']['lat'],
                'lon': simurg_data['location']['lon'],
                'xyz': simurg_data['xyz']
            }
            station_cache[site_id] = station_data
            return station_data
    except Exception as e:
        # print(f"Ошибка при запросе {site_id}: {e}")
        station_cache[site_id] = None
        return None

async def get_all_site_data_concurrently(site_ids: list[str]):
    """
    Запускает запросы для всех нужных ID станций параллельно.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_site_data_async(session, site_id) for site_id in site_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Отфильтровываем None и возможные исключения
        return [res for res in results if res is not None and not isinstance(res, Exception)]

def get_elaz_for_site(site_xyz, sats_xyz):
    elaz = {}
    for sat, sat_xyz_data in sats_xyz.items():
        elaz[sat] = xyz_to_el_az(site_xyz, sat_xyz_data)
    return elaz

def generate_equatorial_poly():
    """
    Генерирует координаты для полигона "экваториальный пояс"
    с правильным порядком обхода вершин (против часовой стрелки).
    """
    lat_min, lat_max = -15, 15
    lon_min, lon_max = -180, 180
    num_segments = 50 # Достаточное количество для гладкости

    # 1. Движемся по нижней границе слева направо
    bottom_edge = list(zip(np.linspace(lon_min, lon_max, num_segments), [lat_min] * num_segments))

    # 2. Движемся по верхней границе справа налево
    top_edge = list(zip(np.linspace(lon_max, lon_min, num_segments), [lat_max] * num_segments))

    # 3. Соединяем их в один контур против часовой стрелки.
    # [точки нижней границы] + [точки верхней границы в обратном порядке]
    # Первая и последняя точки будут совпадать, что хорошо для S2Loop.
    polygon_coords = bottom_edge + top_edge
    
    return polygon_coords

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
    anomaly_polygon_s2: s2.S2Polygon,
    max_distance_km: float
) -> bool:
    """
    Проверяет, находится ли станция внутри полигона аномалии или на заданном
    расстоянии от его БЛИЖАЙШЕЙ ГРАНИЦЫ, используя s2geometry.
    """
    # Создаем объект S2LatLng для станции
    site_s2_latlng = s2.S2LatLng.FromDegrees(site_latlon[1], site_latlon[0])

    # 1. Проверяем, не находится ли станция уже внутри полигона.
    # Для этого S2LatLng нужно преобразовать в S2Point.
    if anomaly_polygon_s2.Contains(site_s2_latlng.ToPoint()):
        return True

    # 2. Если нет, вычисляем расстояние до полигона.
    # Метод Project() находит ближайшую точку на полигоне к нашей станции.
    projected_point_s2 = anomaly_polygon_s2.Project(site_s2_latlng.ToPoint())
    
    # Создаем S2LatLng из спроецированной точки
    projected_latlng = s2.S2LatLng(projected_point_s2)

    # Метод GetDistance() между двумя S2LatLng возвращает расстояние в виде угла S1Angle.
    s1_angle_distance = site_s2_latlng.GetDistance(projected_latlng)

    # Конвертируем угол в километры
    earth_radius_km = RE / 1000.0
    distance_in_km = s1_angle_distance.radians() * earth_radius_km
    print(site_latlon,' ', distance_in_km)
    return distance_in_km <= max_distance_km

def get_filtered_stations(study_date: datetime):
    """
    БЫСТРАЯ функция для инициализации сессии.
    1. Получает список всех станций из H5 файла.
    2. Асинхронно загружает их координаты.
    3. Фильтрует станции по близости к аномалии.
    Возвращает отфильтрованный список словарей с данными о станциях.
    """
    print(f"Инициализация сессии для {study_date.date()}. Фильтрация станций...")
    DISTANCE_LIMIT_KM = 100.0 

    try:
        h5_file_path = load_h5_data(study_date)
        if not h5_file_path:
            raise FileNotFoundError("Не удалось загрузить H5 файл для получения списка станций.")
        
        with h5py.File(h5_file_path, 'r') as f:
            all_stations_in_h5 = list(f.keys())
        
        print(f"Найдено станций в H5: {len(all_stations_in_h5)}. Загружаю координаты...")
        all_stations_data = asyncio.run(get_all_site_data_concurrently(all_stations_in_h5))
        
        anomaly_poly_coords = generate_equatorial_poly()
        anomaly_s2_loop = s2.S2Loop([
            s2.S2LatLng.FromDegrees(lat, lon).ToPoint() for lat, lon in anomaly_poly_coords
        ])
        anomaly_polygon_s2 = s2.S2Polygon(anomaly_s2_loop)

        relevant_stations = []
        for site_data in all_stations_data:
            if is_site_near_polygon(
                site_latlon=(site_data['lat'], site_data['lon']),
                anomaly_polygon_s2=anomaly_polygon_s2,
                max_distance_km=DISTANCE_LIMIT_KM
            ):
                relevant_stations.append(site_data)
        
        print(f"После фильтрации осталось {len(relevant_stations)} станций.")
        return relevant_stations

    except Exception as e:
        print(f"Критическая ошибка при фильтрации станций: {e}")
        return []


# ==============================================================================
# НОВАЯ ФУНКЦИЯ №2: find_next_valid_segment
# ==============================================================================
def find_next_valid_segment(study_date: datetime, station_list: list, current_station_idx: int, current_sat_idx: int):
    """
    "Ленивая" поисковая функция. Ищет следующий валидный сегмент,
    начиная с указанной позиции в списке станций и спутников.
    Возвращает (метаданные_сегмента, новый_индекс_станции, новый_индекс_спутника) или (None, None, None).
    """
    print(f"\nПоиск следующего сегмента, начиная со станции #{current_station_idx}, спутника #{current_sat_idx}...")
    
    try:
        nav_file = load_nav_file(study_date)
        if not nav_file: raise FileNotFoundError("Не удалось загрузить NAV файл для поиска сегмента.")
        h5_file_path = load_h5_data(study_date)
        if not h5_file_path: raise FileNotFoundError("Не удалось загрузить H5 файл для поиска сегмента.")
        
        end_time = study_date + timedelta(days=1, seconds=-30)
        anomaly_poly_coords = generate_equatorial_poly()
        anomaly_s2_loop = s2.S2Loop([
            s2.S2LatLng.FromDegrees(lat, lon).ToPoint() for lon, lat in anomaly_poly_coords
        ])
        anomaly_polygon_s2 = s2.S2Polygon(anomaly_s2_loop)
    except Exception as e:
        print(f"Ошибка при подготовке к поиску сегмента: {e}")
        return None, None, None

    # Итерируемся по станциям, начиная с текущей
    for i in range(current_station_idx, len(station_list)):
        site_data = station_list[i]
        
        with h5py.File(h5_file_path, 'r') as f:
            available_sats = list(f.get(site_data['id'], {}).keys())
        
        # Если мы на новой станции, начинаем с первого спутника. Иначе - продолжаем.
        start_sat_idx = current_sat_idx if i == current_station_idx else 0
        
        print(f"--- Проверка станции {site_data['id'].upper()} (спутники с #{start_sat_idx}) ---")

        # Итерируемся по спутникам этой станции
        for j in range(start_sat_idx, len(available_sats)):
            sat_id = available_sats[j]
            
            # --- ЗАПУСКАЕМ ТЯЖЕЛЫЕ ВЫЧИСЛЕНИЯ ТОЛЬКО ДЛЯ ОДНОЙ ПАРЫ ---
            all_sats_xyz, times = get_sat_xyz(nav_file, study_date, end_time, sats=[sat_id])
            if not all_sats_xyz: continue
            
            sats_elaz = get_elaz_for_site(site_data['xyz'], all_sats_xyz)
            if sat_id not in sats_elaz: continue
            
            elaz_data = sats_elaz[sat_id]
            visible_mask = elaz_data[:, 0] > 0
            if not np.any(visible_mask): continue
            
            sips_coords = calculate_sips(np.radians(site_data['lat']), np.radians(site_data['lon']), elaz_data[visible_mask, 0], elaz_data[visible_mask, 1])
            points_with_datetime = [{'lat': sip[0], 'lon': sip[1], 'time': time} for sip, time in zip(sips_coords, np.array(times)[visible_mask])]
            
            time_based_segments = split_trajectory_by_gaps(points_with_datetime)

            part_number = 1
            for segment in time_based_segments:
                is_valid, intersections = is_segment_valid(segment, anomaly_polygon_s2)
                if is_valid:
                    # НАШЛИ!
                    print(f"    ✅ Найден валидный сегмент: {site_data['id']}-{sat_id} (часть {part_number})")
                    segment_id = f"{site_data['id']}-{sat_id}-{part_number}"
                    
                    full_segment_data = {
                        'id': segment_id,
                        'points': [{**p, 'time': p['time'].isoformat()} for p in segment],
                        'intersections': [{**p, 'time': p['time'].isoformat()} for p in intersections]
                    }
                    cache_segment_data(segment_id, full_segment_data)
                    
                    event_metadata = {
                        'id': segment_id,
                        'station_id': site_data['id'],
                        'satellite_id': sat_id,
                        'entry_time': intersections[0]['time'].isoformat(),
                        'exit_time': intersections[-1]['time'].isoformat(),
                        'has_effect': False 
                    }
                    
                    # Возвращаем метаданные и НОВЫЕ индексы для СЛЕДУЮЩЕГО поиска
                    return event_metadata, i, j + 1
                
                part_number += 1

    # Если мы прошли все циклы и ничего не нашли, значит, разметка закончена.
    print("Поиск завершен. Больше валидных сегментов не найдено.")
    return None, None, None

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
    ОПТИМИЗИРОВАННАЯ ВЕРСЯ.
    Читает из HDF5 файла данные ТОЛЬКО для одной запрошенной траектории.
    """
    print(f"Ленивый запрос данных для графика: {station_id}-{satellite_id}, продукт: {product.name}")
    try:
        # Получаем путь к файлу (он будет взят из кэша, если уже скачан)
        h5_file = load_h5_data(study_date)
        if not h5_file:
            raise FileNotFoundError("HDF5 файл для этой даты не найден.")
            
        # Открываем файл для чтения
        with h5py.File(h5_file, 'r') as f:
            # Проверяем наличие нужных "папок" (групп) в файле
            if station_id not in f:
                raise KeyError(f"Станция {station_id} не найдена в H5 файле.")
            if satellite_id not in f[station_id]:
                raise KeyError(f"Спутник {satellite_id} не найден для станции {station_id}.")
            
            sat_group = f[station_id][satellite_id]
            
            if product.value.hdf_name not in sat_group:
                raise KeyError(f"Продукт {product.name} не найден для траектории.")

            # --- ВОТ ОНА, МАГИЯ! ---
            # Читаем ТОЛЬКО нужные нам датасеты
            timestamps = sat_group[DataProducts.timestamp.value.hdf_name][:]
            values = sat_group[product.value.hdf_name][:]

        # Обработка и возврат данных
        times = [datetime.fromtimestamp(t, tz=tz.gettz("UTC")) for t in timestamps]
        
        return {
            'time': [t.isoformat() for t in times],
            'value': values.tolist(),
            'product_name': product.value.long_name,
            'product_units': product.value.color_limits.units
        }

    except (KeyError, FileNotFoundError) as e:
        print(f"  -> Ошибка: {e}")
        return None
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при чтении H5 файла: {e}")
        return None
