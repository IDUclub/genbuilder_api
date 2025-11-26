from enum import IntEnum

class ServiceType(IntEnum):
    SCHOOL = 22
    KINDERGARTEN = 21
    POLYCLINIC = 28


SERVICE_TITLE_RU = {
    ServiceType.SCHOOL: "Школа",
    ServiceType.KINDERGARTEN: "Детский сад",
    ServiceType.POLYCLINIC: "Поликлиника",
}


service_projects = {
    ServiceType.SCHOOL: [
        {
            "service": "Школа",
            "type_id": "school_550",
            "capacity": 550,
            "floors_count": 3,
            "plot_length": [120.0, 160.0],
            "plot_width": [90.0, 120.0],
            "osm_url": "https://www.openstreetmap.org/way/42273865",
        },
        {
            "service": "Школа",
            "type_id": "school_825",
            "capacity": 825,
            "floors_count": 4,
            "plot_length": [140.0, 180.0],
            "plot_width": [90.0, 140.0],
            "osm_url": "https://www.openstreetmap.org/way/139855509",
        },
        {
            "service": "Школа",
            "type_id": "school_1100",
            "capacity": 1100,
            "floors_count": 4,
            "plot_length": [150.0, 220.0],
            "plot_width": [120.0, 160.0],
            "osm_url": "https://www.openstreetmap.org/way/1268542756",
        },
    ],

    ServiceType.KINDERGARTEN: [
        {
            "service": "Детский сад",
            "type_id": "kindergarten_140",
            "capacity": 140,
            "floors_count": 2,
            "plot_length": [70.0, 100.0],
            "plot_width": [60.0, 90.0],
            "osm_url": "https://www.openstreetmap.org/way/28202955",
        },
        {
            "service": "Детский сад",
            "type_id": "kindergarten_220",
            "capacity": 220,
            "floors_count": 3,
            "plot_length": [80.0, 120.0],
            "plot_width": [70.0, 100.0],
            "osm_url": "https://www.openstreetmap.org/way/1340484069",
        },
        {
            "service": "Детский сад",
            "type_id": "kindergarten_280",
            "capacity": 280,
            "floors_count": 3,
            "plot_length": [110.0, 150.0],
            "plot_width": [90.0, 130.0],
            "osm_url": "https://www.openstreetmap.org/way/23130101",
        },
    ],

    ServiceType.POLYCLINIC: [
        {
            "service": "Поликлиника",
            "type_id": "polyclinic_spb_49",
            "capacity": 750, 
            "floors_count": 5,
            "plot_length": [70.0, 120.0],
            "plot_width": [50.0, 90.0],
            "osm_url": "https://www.openstreetmap.org/way/23048458",
        },
        {
            "service": "Поликлиника",
            "type_id": "polyclinic_volgodonsk_1",
            "capacity": 600,
            "floors_count": 3,
            "plot_length": [70.0, 110.0],
            "plot_width": [50.0, 80.0],
            "osm_url": "https://www.openstreetmap.org/way/104048492",
        },
        {
            "service": "Поликлиника",
            "type_id": "polyclinic_kazan_20",
            "capacity": 400,
            "floors_count": 5,
            "plot_length": [90.0, 140.0],
            "plot_width": [60.0, 100.0],
            "osm_url": "https://www.openstreetmap.org/relation/18102614",
        },
    ],
}