import urllib.parse
import urllib.request
import json
import math

### ----- Google Maps API ----- ###

class GoogleAPI:
    def __init__(self, key_path='/home/kafkaon1/buildseg/api-keys.json'):
        with open(key_path, "r") as read_file:
            keys = json.load(read_file)
        self.api_key = keys['google']

    def get_satellite_image(self, coords, zoom=20, save_path='./tmp_sat.jpg', size=[600, 600]):
        url_sat = f'https://maps.googleapis.com/maps/api/staticmap?center={coords[0]},{coords[1]}&zoom={zoom}&size={size[0]}x{size[1]}&maptype=satellite&key={self.api_key}'
        print(url_sat)
        urllib.request.urlretrieve(url_sat, save_path)

    def get_streetview_image(self, coords, size=[600, 600], fov=100, pitch=0, heading=0, save_path='./tmp_str.jpg'):
        url_str = f'https://maps.googleapis.com/maps/api/streetview?size={size[0]}x{size[1]}&location={coords[0]},{coords[1]}&fov={fov}&pitch={pitch}&heading={heading}&key={self.api_key}'
        urllib.request.urlretrieve(url_str, save_path)

    def geoencode_address(self, address):
        address_encoded = urllib.parse.quote(address)
        url_geo = f'https://maps.googleapis.com/maps/api/geocode/json?address={address_encoded}&key={self.api_key}'
        response = urllib.request.urlopen(url_geo).read()
        data = json.loads(response.decode('utf-8'))
        return data['results'][0]['geometry']['location']['lat'], data['results'][0]['geometry']['location']['lng']

    def get_streetview_metadata(self, address):
        # TODO BY : https://developers.google.com/maps/documentation/streetview/metadata
        #https://maps.googleapis.com/maps/api/streetview/metadata?parameters
        pass 

    def validate_address(self, address):
        # TODO BY : https://developers.google.com/maps/documentation/address-validation/requests-validate-address
        pass

### ----- Map Utility functions ----- ###

def distance_from_pixels_to_meters(lat, lng, pixel_x, pixel_y, zoom):
    #     double scale = Math.Pow(2, m_intZoom);
# NW_Latitude = CenterLat + (PicSizeY / 2) / scale;
# NW_Longitude = CenterLon - (PicSizeX / 2) / scale;
# NE_Longitude = CenterLon + (PicSizeX / 2) / scale;
# WS_Latitude = CenterLat - (PicSizeY / 2) / scale;

# double earthR = 6371;

# double deltaLat = 0;

# double deltaLon = (m_dblLocationNWLongitude - m_dblLocationNELongitude) * Math.PI / 180;

# double a = Math.Sin(deltaLat / 2) * Math.Sin(deltaLat / 2) +
#             Math.Cos(m_dblLocationNWLatitude * Math.PI / 180) * Math.Cos(m_dblLocationNWLatitude * Math.PI / 180) *
#             Math.Sin(deltaLon / 2) * Math.Sin(deltaLon / 2);
# double b = 2 * Math.Atan2(Math.Sqrt(a), Math.Sqrt(1 - a));

# MapWidthDistance = earthR * b;
#  ///////////////////////////
# deltaLat = (m_dblLocationNWLatitude - m_dblLocationWSLatitude) * Math.PI / 180;

# deltaLon = 0;

# a = Math.Sin(deltaLat / 2) * Math.Sin(deltaLat / 2) +
#      Math.Cos(m_dblLocationNWLatitude * Math.PI / 180) *
#      Math.Cos(m_dblLocationNWLatitude * Math.PI / 180) *
#      Math.Sin(deltaLon / 2) * Math.Sin(deltaLon / 2);

# b = 2 * Math.Atan2(Math.Sqrt(a), Math.Sqrt(1 - a));

# MapHeightDistance = earthR * b;

# double earthC = 6371000 * 2 * Math.PI;
# double factor = Math.Pow(2, 8 + m_intZoomLevel);
# double MeterPerPixel = (Math.Cos(CenterLat * Math.PI / 180) * earthC / factor)/2;
# double MapWidthDistance = OrginalImageWidth * MeterPerPixel;
# double ActualMeterPerPixel = MapWidthDistance / imgWidthAfterResize;
    pass

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6372800  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


