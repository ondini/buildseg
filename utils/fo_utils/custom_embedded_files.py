import sys
from fiftyone.core.odm import DynamicEmbeddedDocument, EmbeddedDocument
import fiftyone.core.fields as fof
import fiftyone.core.utils as fou


class BoatBusDataEmbeddedDocument(DynamicEmbeddedDocument):
    """Base class for storing boatbus data about generic samples."""

    date = fof.DateTimeField()
    time = fof.DateTimeField()
    rateOfTurn = fof.FloatField()
    heading = fof.FloatField()
    roll = fof.FloatField()
    yaw = fof.FloatField()
    pitch = fof.FloatField()
    courseOverGround = fof.FloatField()
    speedOverGround = fof.FloatField()
    speedOverWater = fof.FloatField()
    windAngle = fof.FloatField()
    waterTemperature = fof.FloatField()
    windSpeed = fof.FloatField()
    longitude = fof.FloatField()
    latitude = fof.FloatField()


class ImuDataEmbeddedDocument(DynamicEmbeddedDocument):
    """Base class for storing boatbus data about generic samples."""

    roll = fof.FloatField()
    pitch = fof.FloatField()
    yaw = fof.FloatField()
    aPerpendicular = fof.FloatField()
    aLeteral = fof.FloatField()
    aLongitundinal = fof.FloatField()
    

class SensorsEmbeddedDocument(DynamicEmbeddedDocument):
    """Base class for storing sensors.json data about generic samples."""

    version = fof.StringField()
    UUID = fof.StringField()

    @classmethod
    def build_from(cls, sensors_version, sensors_UUID):
        return cls._build_from_dict(sensors_version, sensors_UUID)

    @classmethod
    def _build_from_dict(cls, sensors_version, sensors_UUID):
        version = sensors_version
        UUID = sensors_UUID

        return cls(
            version = version,
            UUID = UUID
        )



class SentryBoatBusDataEmbeddedDocument(DynamicEmbeddedDocument):
    """Base class for storing sentry boatbus data about generic samples."""

    counter=fof.FloatField()
    roll=fof.FloatField()
    pitch=fof.FloatField()
    yaw=fof.FloatField()
    pan=fof.FloatField()
    tilt=fof.FloatField()
    lati=fof.FloatField()
    longi=fof.FloatField()
    timestp=fof.FloatField()
    heading=fof.FloatField()
    rateofturn=fof.FloatField()
    cog=fof.FloatField()
    sog2=fof.FloatField()
    sow=fof.FloatField()
    sog=fof.FloatField()
    temp=fof.FloatField()
    humidity=fof.FloatField()
    pressure=fof.FloatField()
    wind_speed=fof.FloatField()
    wind_angle=fof.FloatField()
    bb_roll=fof.FloatField()
    bb_pitch=fof.FloatField()
    bb_yaw=fof.FloatField()
    proc_ptu_mode=fof.FloatField()
    proc_colav_limit=fof.FloatField()
    proc_colav_speed=fof.FloatField()
    proc_surveil_speed=fof.FloatField()


class SentrySensorsDataEmbeddedDocument(DynamicEmbeddedDocument):
    """Base class for storing sentry sensors data about generic samples."""

    hardware_version=fof.StringField()
    setup_date=fof.StringField()
    serial_number=fof.StringField()
    jetson_serial_number=fof.IntField()
    license=fof.StringField()
    json_version=fof.IntField()
    calibrated_by=fof.StringField()
    camera_calibration_version=fof.IntField()
    imu_calibration_version=fof.IntField()

    camera_name = fof.StringField()


