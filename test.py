import data_processing # Импортируем твой модуль с логикой
from datetime import datetime

def run_tests():

    # --- Тест 1: Получение данных для одной станции ---
    print("\n[Тест 1/3] Проверка get_site_data_by_id...")
    test_station_id = 'irkj'
    try:
        station_data = data_processing.get_site_data_by_id(test_station_id)
    except Exception as e:
        print(f"Ошибка при получении данных станции: {e}")
        return

    print("\n[Тест 2/3] Проверка get_main_map_data...")
    test_date = datetime(2025, 5, 1) 
    
    try:

        result = data_processing.get_main_map_data(
            study_date=test_date
        )
        print(len(result))
        print(result[0])
    except Exception as e:
        print(f"Ошибка во время выполнения get_main_map_data: {e}")
        return



# Запускаем тесты
if __name__ == '__main__':
    run_tests()