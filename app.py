import math
import cv2
import matplotlib
from matplotlib import pyplot as plt
from ultralytics import YOLO

matplotlib.use('TkAgg')

VIDEO_PATH = '/home/ftheir/PycharmProjects/DroneCrossroadTracking/footage/crossroad6.mp4'
MODEL = 'best.pt'
TRACKER = 'botsort.yaml'

coordinates = {}  # {[id тр. ср.]:{[время]:[координаты]}}
car_lengths = {}  # {[время]: средняя длина тр. средств}


# car_nums = {}  # {[время]: кол-во тр.ср. на перекрёстке}


def calculate_distance(m0: list, m1: list, t: float) -> float:  # расстояние между двумя точками, м
    s = math.sqrt((m0[0] - m1[0]) ** 2 + (m0[1] - m1[1]) ** 2)
    return pixels_to_meters(s, car_lengths[t])


def ms_to_kmh(v: float) -> float:
    return 3.6 * v


def calculate_velocities(times: list) -> (list, list):  # скорость, км/ч
    time_segments = split_time(times)
    velocities = {}
    for vehicle_id in coordinates.keys():
        vehicle_times = []
        for t in coordinates[vehicle_id].keys():
            if t in time_segments:
                vehicle_times.append(t)
        for i in range(1, len(vehicle_times)):
            t = vehicle_times[i]
            t0 = vehicle_times[i - 1]
            delta_t = t - t0
            if delta_t < 1.15:
                v = calculate_distance(coordinates[vehicle_id][t],
                                       coordinates[vehicle_id][t0], t) / delta_t
                v = ms_to_kmh(v)
                if t in velocities:
                    velocities[t].append(v)
                else:
                    velocities[t] = [v]
    time_segments = list(velocities.keys())
    for i in range(len(velocities)):
        try:
            velocities[time_segments[i]].sort()
            velocities[time_segments[i]] = velocities[time_segments[i]][2:len(velocities[time_segments[i]]) - 3]
        except IndexError:
            pass
        finally:
            velocities[time_segments[i]] = sum(velocities[time_segments[i]]) / len(velocities[time_segments[i]])
    return list(velocities.keys())[2:], list(velocities.values())[2:]  # time, vel


def split_time(times: list) -> list:  # разбивает на отрезки времени по секунде
    closest_dict = {}
    for t in times:
        integer_part = int(t)
        if integer_part not in closest_dict:
            closest_dict[integer_part] = t
        else:
            if abs(t - integer_part) < abs(closest_dict[integer_part] - integer_part):
                closest_dict[integer_part] = t
    return list(closest_dict.values())


def median(l: list) -> float:  # медиана
    l.sort()  # плотность =  n_mash / l_polotno, n_mash/v*delta_t
    n = len(l)
    if n % 2 == 1 or n == 0:
        return l[n // 2]
    else:
        return (l[n // 2 - 1] + l[n // 2]) / 2


def sort_close_cars(distances: list, length: float, meters: int = 20) -> list:  # отсеивает дальние машины
    res = [0]
    for i in distances:
        if pixels_to_meters(i, length) < meters:
            res.append(i)
    return res


def get_data(boxes: list, time: float) -> (float, float):  # -> (ср. длина машин, ср. расстояние до сл. машины)
    global coordinates
    lengths = []
    distances = []
    for i in range(len(boxes)):
        boxes[i] = [[(boxes[i][2] + boxes[i][0]) / 2, (boxes[i][3] + boxes[i][1]) / 2],  # [x0, y0] center
                    [boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]]]  # [width, height]
    for i in range(len(boxes)):
        single_distances = []
        x, y = boxes[i][0]
        width = boxes[i][1][0]
        height = boxes[i][1][1]
        if 0.75 < width / height < 1.3:
            lengths.append(math.sqrt(width ** 2 + height ** 2))
        else:
            lengths.append(max(width, height))
        for j in range(1, len(boxes) - 1):
            x1, y1 = boxes[j][0]
            single_distances.append(math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2))
        distances.append(median(sort_close_cars(single_distances, lengths[-1])))
    median_length = median(lengths)
    car_lengths[time] = median_length
    return median_length, median(distances)


def pixels_to_meters(s: float, avg_car_length: float) -> float:  # переводит пиксели в метры
    return 4.5 * s / avg_car_length


def flow(totals: list, times: list) -> (list, list):  # интенсивность, s**(-1)
    f = []
    time_segments = split_time(times)
    data = dict(zip(times, totals))
    totals = []
    for t in time_segments:
        totals.append(data[t])
    for i in range(1, len(time_segments)):
        delta_t = time_segments[i] - time_segments[i - 1]
        growth = totals[i] - totals[i - 1]
        f.append(growth / delta_t)
    return time_segments[1:], f


def main():
    delta_time = []
    times = []
    density_stat = []
    totals = []
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time = 0
    model = YOLO(MODEL)

    while True:
        time += 1 / fps
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=True, tracker=TRACKER)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        car_len, distance = get_data(boxes.tolist(), time)
        delta_time.append(time)
        density_stat.append(1000 / (pixels_to_meters(distance, car_len) + 4.5))
        times.append(time)
        print(f'{time}s')
        totals.append(max(ids))

        for box, vehicle_id, vehicle_class in zip(boxes, ids, classes):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                          [(255, 0, 0), (0, 0, 255)][int(vehicle_class)], 1)
            cv2.putText(frame,
                        f'{model.model.names[int(vehicle_class)]} {vehicle_id}',
                        (box[0], box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, [(255, 0, 0), (0, 0, 255)][int(vehicle_class)], 1,
                        )
            if vehicle_id not in coordinates:
                coordinates[vehicle_id] = {time: [(box[2] + box[0]) / 2, (box[3] + box[1]) / 2]}
            else:
                coordinates[vehicle_id][time] = [(box[2] + box[0]) / 2, (box[3] + box[1]) / 2]
        cv2.imshow('Drone Crossroad Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(30) == 27:
            break

    vel_times, vels = calculate_velocities(times)
    plt.plot(vel_times, vels)
    plt.xlabel('время, с')
    plt.ylabel('средняя скорость, км/ч')
    plt.title('График средней скорости потока')
    plt.show()

    flow_times, flows = flow(totals, times)
    plt.plot(flow_times, flows)
    plt.xlabel('время, с')
    plt.ylabel('интенсивность потока, n/с времени')
    plt.title('График интенсивности потока от времени')
    plt.show()

    plt.plot(times, density_stat)
    plt.xlabel('время, с')
    plt.ylabel('плотность потока, n/км дорожного полотна')
    plt.title('График плотности потока от времени')
    plt.show()

    plt.plot(times, totals)
    plt.xlabel('время, с')
    plt.ylabel('кол-во тр.ср. всего')
    plt.title('График зависимости кол-ва тр.ср. потока от времени')
    plt.show()


if __name__ == '__main__':
    main()
