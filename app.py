import math
import cv2
import matplotlib
from matplotlib import pyplot as plt
from ultralytics import YOLO

matplotlib.use('TkAgg')

VIDEO_PATH = '/home/ftheir/PycharmProjects/DroneCrossroadTracking/footage/crossroad6.mp4'
MODEL = 'best.pt'
TRACKER = 'botsort.yaml'


def median(l):
    l.sort()
    n = len(l)
    if n % 2 == 1 or n == 0:
        return l[n // 2]
    else:
        return (l[n // 2 - 1] + l[n // 2]) / 2


def sort_close_cars(distances, length):
    res = [0]
    for i in distances:
        if pixels_to_meters(i, length) < 12:
            res.append(i)
    return res


def get_data(boxes):
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
    return median(lengths), median(distances)


def pixels_to_meters(s, avg_car_length):
    return 4.5 * s / avg_car_length


def density(distance, length):
    return pixels_to_meters(distance, length)


# def flow(number_of_vehicles, time):
#     return number_of_vehicles / time


def main():
    delta_time = []
    times = []
    density_stat = []
    flow_stat = []
    # total = 0
    # delta_vehicle = []
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
        car_len, distance = get_data(boxes.tolist())
        delta_time.append(time)
        density_stat.append(density(distance, car_len))
        times.append(time)
        print(time)
        # if not len(delta_time):
        #     delta_time.append(time)
        # elif len(delta_time) == 1:
        #     delta_time = [abs(time - delta_time[0])] * 2
        # else:
        #     delta_time.append(abs(time - delta_time[-1]))
        # if not len(delta_vehicle):
        #     delta_vehicle.append(max(ids))
        # elif len(delta_vehicle) == 1:
        #     delta_vehicle = [abs(max(ids) - delta_vehicle[0])] * 2
        # else:
        #     delta_vehicle.append(abs(max(ids) - delta_vehicle[-1]))
        # flow_stat.append(flow(delta_vehicle[-1], delta_time[-1]))
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
        cv2.imshow('Drone Crossroad Detection', frame)
        # total = max(ids)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(30) == 27:
            break

    plt.plot(times, density_stat)
    plt.xlabel('время, с')
    plt.ylabel('плотность потока, среднее расстояние до следующей машины, м')
    plt.title('График зависимости плотности потока от времени')
    plt.show()

    plt.plot(times, totals)
    plt.xlabel('время, с')
    plt.ylabel('кол-во тр.ср.')
    plt.title('График зависимости кол-ва тр.ср. потока от времени')
    plt.show()
    print(flow_stat)


if __name__ == '__main__':
    main()
