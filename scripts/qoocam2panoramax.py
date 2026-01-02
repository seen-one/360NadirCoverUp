#! /usr/bin/env python3

import sys, subprocess, os, re, time, argparse, copy, shutil
from datetime import datetime
from math import radians, cos, sin, asin, sqrt, degrees, atan2
from struct import unpack

try:
    import piexif
    from bs4 import BeautifulSoup
    from tqdm import tqdm
    import cv2
    from turbojpeg import TurboJPEG, TJSAMP_420, TJFLAG_PROGRESSIVE

except:
    print("Missing python modules: piexif, pyturbojpeg, lxml, opencv-python, tqdm and bs4 are required", file=sys.stderr)
    print("  pip install piexif lxml bs4 opencv-python tqdm pyturbojpeg", file=sys.stderr)
    exit(1)


def bytes2str(b):
    return b.decode().strip(chr(0))

def kvar_load(kvar_file):
    kvar = {}
    with open(kvar_file, 'rb') as kv:
        dummy, sign, sections = unpack('<4s4sI',kv.read(12))
        if bytes2str(sign) != 'kvar':
            # no header in file ? Let's restart !
            kv.seek(0)
            sections, = unpack('<I',kv.read(4))
        for section in range(sections):
            title, data_type, data_count = unpack('<32s8sI',kv.read(44))
            title = bytes2str(title)
            if title == 'TOTAL_FRAME':
                frames, = unpack('<I',kv.read(4))
                kvar[title] = frames
            elif title == 'TOTAL_TIME_MS':
                time_ms, = unpack('<I',kv.read(4))
                kvar[title] = time_ms/1000
            elif title == 'LENS':
                lens, = unpack('%ss' % data_count ,kv.read(data_count))
                kvar[title] = lens
            elif title == 'IMU':
                kvar[title] = []
                for idx in range(int(data_count/20)):
                    pts, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z = unpack('<Qhhhhhh', kv.read(20))
                    kvar[title].append({'pts': pts, 'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
                                        'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z
                    })
            elif title == 'GPS':
                kvar[title] = []
                kv.read(4)
                for idx in range(int(data_count/28)):
                    pts, lat, lon, alt = unpack('<Lddd', kv.read(28))
                    kvar[title].append({'pts': pts/1000, 'lat': round(lat,7), 'lon': round(lon,7), 'alt': alt })
            elif title == 'EXP':
                kv.read(data_count)
            elif title in ('ISP', 'ISP0', 'ISP1'):
                kv.read(data_count)
            elif title == 'FRAME_ISP':
                kv.read(data_count)
            elif title == 'INFO':
                info, = unpack('%ss' % data_count ,kv.read(data_count))
                kvar[title] = bytes2str(info)
            elif title == 'PTS_UNIT':
                pts_unit, = unpack('<I',kv.read(4))
                kvar[title] = pts_unit
                pts_unit = 1000000 if pts_unit == 1 else 1000
            elif title == 'PTS':
                kvar[title] = []
                for idx in range(int(data_count)):
                    pts, = unpack('<Q', kv.read(8))
                    if idx == 0:
                        pts_start = pts
                    kvar[title].append(pts/pts_unit)
            elif title == 'GPSX':
                kv.read(data_count)

            else:
                print('!!!! UNKOWN SECTION in kvar file : ', title, data_type, data_count, file=sys.stderr)
                kv.read(data_count)
    return kvar

def format_offset_string(hours):
    """Convert a timezone offset in hours to ±HH:MM format."""
    sign = '+' if hours >= 0 else '-'
    h = int(abs(hours))
    m = int((abs(hours) - h) * 60)
    return f"{sign}{h:02d}:{m:02d}"

def set_timestamp(filename, epoch, lat, lon, alt, direction, speed, tz_offset_hours=0):
    new_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))
    zeroth_ifd = {
        piexif.ImageIFD.Make: args.make,
        piexif.ImageIFD.Model: args.model
    }
    exif = {
        piexif.ExifIFD.DateTimeOriginal: new_date,
        piexif.ExifIFD.SubSecTimeOriginal: "{:.3f}".format(epoch)[-3:],
        piexif.ExifIFD.FocalLength: [17, 10],
        piexif.ExifIFD.FNumber: [16, 10]
    }
    if tz_offset_hours != 0:
        offset_str = format_offset_string(tz_offset_hours)
        exif[piexif.ExifIFD.OffsetTime] = offset_str
        exif[piexif.ExifIFD.OffsetTimeOriginal] = offset_str
        exif[piexif.ExifIFD.OffsetTimeDigitized] = offset_str
    gps = {piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
            piexif.GPSIFD.GPSMapDatum: 'WGS-84',
            piexif.GPSIFD.GPSLatitudeRef: 'N' if lat>=0 else 'S',
            piexif.GPSIFD.GPSLatitude: ((int(abs(lat) * 10000000), 10000000),(0,1),(0,1)),
            piexif.GPSIFD.GPSLongitudeRef: 'E' if lon>=0 else 'W',
            piexif.GPSIFD.GPSLongitude: ((int(abs(lon) * 10000000), 10000000),(0,1),(0,1)),
            piexif.GPSIFD.GPSImgDirection: (int(direction * 10), 10),
            piexif.GPSIFD.GPSImgDirectionRef: 'T',
            piexif.GPSIFD.GPSTrack: (int(direction * 10), 10),
            piexif.GPSIFD.GPSTrackRef: 'T',
            piexif.GPSIFD.GPSSpeed: (int(speed * 10), 10),
            piexif.GPSIFD.GPSSpeedRef: 'K'
    }
    if alt:
        gps[piexif.GPSIFD.GPSAltitude] = (int(abs(alt) * 10), 10)
    
    exif_bytes = piexif.dump({"0th":zeroth_ifd, "Exif": exif, "GPS": gps})
    try:
        piexif.insert(exif_bytes, filename)
    except:
        print("EXIF tag write error on", filename, file=sys.stderr)
        pass

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6372800 # Radius of earth in meters.
    return c * r 

def bearing(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
 
    dlon = lon2_rad - lon1_rad
 
    y = sin(dlon) * cos(lat2_rad)
    x = cos(lat1_rad) * sin(lat2_rad) - \
        sin(lat1_rad) * cos(lat2_rad) * cos(dlon)
 
    bearing_rad = atan2(y, x)
    bearing_deg = degrees(bearing_rad)
 
    # Normalize to [0, 360) degrees
    bearing_deg = (bearing_deg + 360) % 360
 
    return bearing_deg


def save_jpeg(path, pic):
    with open(path,'wb') as jpg:
        jpg.write(jpeg.encode(pic, quality=95,jpeg_subsample=TJSAMP_420))


def get_horizon(kvar):
    check_time = min(2000,len(kvar['IMU']))
    acc_x,acc_y,acc_z = (0.0,0.0,0.0)
    for idx in range(check_time):
        acc_x += kvar['IMU'][idx]['acc_x']
        acc_y += kvar['IMU'][idx]['acc_y']
        acc_z += kvar['IMU'][idx]['acc_z']
    acc_x = acc_x / check_time
    acc_y = acc_y / check_time
    acc_z = acc_z / check_time
    acc = sqrt(acc_x*acc_x + acc_y*acc_y + acc_z*acc_z)
    angle_x = round(degrees(asin(acc_x/acc)),1)
    angle_y = round(degrees(asin(acc_y/acc)),1)
    angle_z = round(degrees(asin(acc_z/acc)),1)
    return (angle_x,angle_y,angle_z)


def gps_interpolate(prev_val, next_val, prev_epoch, next_epoch, cur_epoch):
    return prev_val + (next_val-prev_val) * (cur_epoch-prev_epoch)/(next_epoch-prev_epoch)


parser = argparse.ArgumentParser(
    prog='qoocam2panoramax',
    description='Post-processing for Qoocam JPG extracted from timelapse video',
    epilog='written by cquest, shared under MIT License')
parser.add_argument('--gpx', help='GPX file containing timestamps and locations')
parser.add_argument('--kvar', help='.kvar file containing video GPS and IMU data')
parser.add_argument('--fps', type=int, default=2, help='Frame per second in the MP4 timelapse video')
parser.add_argument('--input', help='Directory containing pictures to process')
parser.add_argument('--output', help='Directory to store processed pictures (default = same as input)')
parser.add_argument('--offset', type=int, default=2, help='Timestamp offset in seconds')
parser.add_argument('--heading', type=int, default=0, help='Camera heading compared to GPS track')
parser.add_argument('--distance', type=int, default=5, help='Minimum distance between pictures')
parser.add_argument('--make', default="Kandao", help='Camera "Make" EXIF tag')
parser.add_argument('--model', default="Qoocam 3 Ultra", help='Camera "Model" EXIF tag')
parser.add_argument('--nadir', help='Image to add at nadir on final picture')
parser.add_argument('--limit', type=int, help='Maximum number of pictures to extract from timelapse video (default = no limit)')
parser.add_argument('--timezone', type=float, default=0, help='Timezone offset in hours to apply to GPX timestamps (e.g. 11 for AEDT)')

#parser.add_argument('--api-url', type=str, help='Set API to query, default from ~/.config/geovisio/config.toml')
#parser.add_argument('--token', type=str, help='Set TOKEN to use for API auth, default from ~/.config/geovisio/config.toml')
global args
args = parser.parse_args()

global jpeg

if args.input is None:
    print("missing --input", file=sys.stderr)
    exit(1)
if args.output:
    args.output = os.path.normpath(args.output)

FPS = int(args.fps)

if 'MP4' not in args.input.upper():
    DIR = args.input
    video = None
else:
    video = cv2.VideoCapture(args.input)
    if args.output:
        try:
            os.mkdir(args.output)
        except:
            pass       
    DIR = None

OFFSET = int(args.offset)
if OFFSET != 0:
    print('%ss offset applied on GPX/GPS timestamps' % OFFSET, file=sys.stderr)

if args.nadir:
    nadir = cv2.imread(args.nadir)

kvar = None
if args.gpx:
    # read GPX file
    with open(args.gpx) as gpxfile:
        gpx = BeautifulSoup(gpxfile,"xml")
        trkpts = gpx.find_all('trkpt')
        START = trkpts[0].time.string
        EPOCH = datetime.strptime(START, "%Y-%m-%dT%H:%M:%S%z").timestamp() + OFFSET
        # Note: Timezone offset is applied only when writing EXIF tags, not here
        # This keeps GPS interpolation consistent with the GPX trackpoint timestamps
        
elif args.kvar:
    # get the start time from the .kvar filename
    try:
        START = re.sub(r'^.*(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2}).*$',r'\1-\2-\3T\4:\5:\6',args.kvar)
        EPOCH = datetime.strptime(START, "%Y-%m-%dT%H:%M:%S").timestamp()
    except:
        print('Then .kvar file does not contain date-time info, please restore its original name', file=sys.stderr)
        exit(1)

    kvar = kvar_load(args.kvar)
    # PTS are relative to the camera start time, compute absolute timestamps
    for idx in range(len(kvar['GPS'])):
        timestamp = EPOCH + kvar['GPS'][idx]['pts'] - kvar['PTS'][0] - OFFSET
        kvar['GPS'][idx]['epoch'] = timestamp
        kvar['GPS'][idx]['time'] = time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        kvar['GPS'][idx]['time'] += "."+"{:.3f}".format(timestamp)[-3:]
    kvar['EPOCH'] = []
    for idx in range(len(kvar['PTS'])):
        kvar['EPOCH'].append(EPOCH + kvar['PTS'][idx] - kvar['PTS'][0])

    # check horizon
    angle_x,angle_y,angle_z = get_horizon(kvar)
    print("Horizon check %s : %s° / %s° / %s°" % ('OK' if abs(angle_x)<3 and abs(angle_z)<3 else 'PROBLEM', angle_x,angle_y,angle_z), file=sys.stderr)
else:
    print('missing --gpx or --kvar', file=sys.stderr)
    exit(1)

prev_pt = -1
prev_epoch = 0
next_epoch = 0

tags = []

if DIR:
    print('timelapse with %s fps'% FPS, file=sys.stderr)
    for JPG in sorted(os.listdir(DIR)):
        if JPG[0:2] == '._':
            os.remove(DIR+'/'+JPG)
        if JPG[0] == '.' or '.jpg' not in JPG:
            continue
        
        IDX = re.sub('^.*Output_','', JPG)
        IDX = int(re.sub('.jpg','',IDX))
        if kvar:
            TS = kvar['EPOCH'][IDX-1]
            while next_epoch < TS and prev_pt+2 < len(kvar['GPS']):
                prev_pt += 1
                prev_epoch = kvar['GPS'][prev_pt]['epoch']
                if (prev_pt+1<len(kvar['GPS'])):
                    next_epoch = kvar['GPS'][prev_pt+1]['epoch']

            lat = gps_interpolate(kvar['GPS'][prev_pt]['lat'],kvar['GPS'][prev_pt+1]['lat'],prev_epoch,next_epoch,TS)
            lon = gps_interpolate(kvar['GPS'][prev_pt]['lon'],kvar['GPS'][prev_pt+1]['lon'],prev_epoch,next_epoch,TS)
            alt = gps_interpolate(kvar['GPS'][prev_pt]['alt'],kvar['GPS'][prev_pt+1]['alt'],prev_epoch,next_epoch,TS)
        else:
            TS = EPOCH + (float(IDX)-1) / FPS
            while next_epoch < TS and prev_pt+2 < len(trkpts):
                prev_pt += 1
                prev_epoch = datetime.strptime(trkpts[prev_pt].time.string, "%Y-%m-%dT%H:%M:%S%z").timestamp()
                if (prev_pt+1<len(trkpts)):
                    next_epoch = datetime.strptime(trkpts[prev_pt+1].time.string, "%Y-%m-%dT%H:%M:%S%z").timestamp()

            lat = gps_interpolate(float(trkpts[prev_pt]['lat']),float(trkpts[prev_pt+1]['lat']),prev_epoch,next_epoch,TS)
            lon = gps_interpolate(float(trkpts[prev_pt]['lon']),float(trkpts[prev_pt+1]['lon']),prev_epoch,next_epoch,TS)
            alt = None

        keep = True
        if len(tags) > 0:
            for dedup in reversed(range(len(tags))):
                prev_frame = tags[dedup]
                if haversine(lat, lon, prev_frame['lat'], prev_frame['lon']) < args.distance:
                    keep = False
                    break
        
        if keep:
            tags.append({'file': DIR+'/'+JPG, 'time': TS, 'lat': lat, 'lon': lon})

    print(len(tags),'files analyzed, applying EXIF tags', file=sys.stderr)
elif video:
    jpeg = TurboJPEG(lib_path=r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll")
    last_frame = 0
    print(len(kvar['EPOCH']),"frames in timelapse video", file=sys.stderr)
    frames = len(kvar['EPOCH'])
    if args.limit and frames > args.limit:
        frames = args.limit
    gps = kvar['GPS']
    for frame in tqdm(range(0,frames), desc='Step 1/2 : Processing frames '):
        isread, pic = video.read()
        TS = kvar['EPOCH'][frame]
        while next_epoch < TS and prev_pt+2 < len(kvar['GPS']):
            prev_pt += 1
            prev_epoch = kvar['GPS'][prev_pt]['epoch']
            if (prev_pt+1<len(kvar['GPS'])):
                next_epoch = kvar['GPS'][prev_pt+1]['epoch']

        lat = gps_interpolate(gps[prev_pt]['lat'], gps[prev_pt+1]['lat'], prev_epoch, next_epoch, TS)
        lon = gps_interpolate(gps[prev_pt]['lon'], gps[prev_pt+1]['lon'], prev_epoch, next_epoch, TS)
        alt = gps_interpolate(gps[prev_pt]['alt'], gps[prev_pt+1]['alt'], prev_epoch, next_epoch, TS)

        JPG = "frame{:06d}.jpg".format(frame)

        if len(tags) > 0:
            for dedup in reversed(range(len(tags))):
                prev_frame = tags[dedup]
                if haversine(lat, lon, prev_frame['lat'], prev_frame['lon']) < args.distance:
                    JPG = ''
                    break

        # do not keep the first video frame due to exposure problem
        if frame == 0 or JPG == '':
            continue

        if args.output:
            JPG = os.path.join(args.output, JPG.replace('frame',os.path.basename(args.output)+'_'))
            if args.heading != 0:
                # roll image by swapping image columns
                height = len(pic)
                width = height*2
                cut_left = int(width * ((args.heading + 360) % 360) / 360)
                cut_right = width-cut_left
                pic1 = cv2.transpose(pic) # row/col -> col/row
                pic2 = cv2.transpose(pic)
                for col in range(cut_right):
                    pic2[col] = copy.deepcopy(pic1[col+cut_left])
                for col in range(cut_left):
                    pic2[cut_right+col] = copy.deepcopy(pic1[col])
                pic = cv2.transpose(pic2)
            if args.nadir:
                nadir_start = len(pic)-len(nadir)
                for row in range(len(nadir)):
                    pic[nadir_start + row] = nadir[row]
            #cv2.imwrite(JPG, pic, [cv2.IMWRITE_JPEG_QUALITY, 90])
            save_jpeg(JPG, pic)
        tags.append({'file': JPG, 'time': TS, 'lat': lat, 'lon': lon, 'alt': alt})
    print(len(tags),'pictures extracted, applying EXIF tags', file=sys.stderr)

# compute direction, speed + add tags in files    
for idx in tqdm(range(0,len(tags)), desc='Step 2/2 : Applying EXIF tags'):
    if idx == 0:
        direction = bearing(tags[0]['lat'], tags[0]['lon'], tags[1]['lat'], tags[1]['lon'])
        speed = haversine(tags[0]['lat'], tags[0]['lon'], tags[1]['lat'], tags[1]['lon']) / (tags[1]['time']-tags[0]['time'])
    elif idx == len(tags)-1:
        direction = bearing(tags[idx-1]['lat'], tags[idx-1]['lon'], tags[idx]['lat'], tags[idx]['lon'])
        speed = haversine(tags[idx-1]['lat'], tags[idx-1]['lon'], tags[idx]['lat'], tags[idx]['lon']) / (tags[idx]['time']-tags[idx-1]['time'])
    else:
        dir_prev = bearing(tags[idx-1]['lat'], tags[idx-1]['lon'],
                        tags[idx]['lat'], tags[idx]['lon'])
        dir_next = bearing(tags[idx]['lat'], tags[idx]['lon'],
                        tags[idx+1]['lat'], tags[idx+1]['lon'])
        direction = (dir_prev+dir_next)/2
        if abs(direction-dir_prev) > 120:
            direction = (direction + 180) % 360

        dist = haversine(tags[idx-1]['lat'], tags[idx-1]['lon'],
                        tags[idx]['lat'], tags[idx]['lon'])
        dist = dist + haversine(tags[idx]['lat'], tags[idx]['lon'],
                        tags[idx+1]['lat'], tags[idx+1]['lon'])
        speed = dist / (tags[idx+1]['time']-tags[idx-1]['time']) * 3.6
        # print(dist, tags[idx+1]['time']-tags[idx-1]['time'], speed)

    target_file = tags[idx]['file']
    if args.output and args.output != os.path.dirname(target_file):
        out_file = os.path.join(args.output, os.path.basename(target_file))
        shutil.copy2(target_file, out_file)
        target_file = out_file

    set_timestamp(target_file,tags[idx]['time'],
            tags[idx]['lat'],tags[idx]['lon'],tags[idx]['alt'] if 'alt' in tags[idx] else None,direction, speed, args.timezone)
    os.utime(target_file, (tags[idx]['time'], tags[idx]['time']))
    
if args.output or args.input:
    target_dir = args.output if args.output else args.input
    run = subprocess.run('exiftool -P -overwrite_original -ProjectionType=equirectangular "%s"/*.jpg -UsePanoramaViewer=true' % target_dir, shell=True, capture_output=True)
    if run.returncode != 0:
        print('ERROR:', run.stderr.decode(), file=sys.stderr)
    else:
        print('Equirectangular projection tags added', file=sys.stderr)

print('Done', file=sys.stderr)
