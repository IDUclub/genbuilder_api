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
