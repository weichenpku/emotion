#! /usr/bin/python3
# -*-coding:utf-8
import os
import numpy
import cv2


WORK_DIR = "/home/soar/avis/"
fps = 30
TYPES = ["ne", "ne", "jo", "jo", "sa", "sa", "fe", "fe"]
idx = {}
durations = [[[420, 450], [645, 675], [948, 978], [1239, 1269], [1540, 1570], [1868, 1898], [-1, -1], [2320, 2350]],
            [[245, 275], [488, 518], [842, 872], [1145, 1175], [1444, 1474], [1746, 1776], [1795, 1825], [2390, 2420]],
            [[267, 297], [520, 550], [886, 916], [1180, 1210], [1468, 1498], [1792, 1822], [2045, 2075], [2452, 2482]],
            [[267, 297], [531, 561], [903, 933], [1187, 1217], [1510, 1540], [1843, 1873], [2118, 2148], [2549, 2579]],
            [[270, 300], [544, 574], [806, 836], [1085, 1115], [1377, 1407], [1670, 1700], [1926, 1956], [2319, 2349]],
            [[325, 355], [568, 598], [861, 891], [1138, 1168], [1510, 1540], [1840, 1870], [2191, 2221], [2625, 2655]],
            [[252, 282], [509, 539], [769, 799], [1124, 1154], [1427, 1457], [1768, 1798], [2090, 2120], [2527, 2557]],
            [[284, 314], [531, 561], [797, 827], [1062, 1092], [1341, 1371], [1644, 1674], [1881, 1911], [2276, 2306]],
            [[227, 257], [520, 550], [817, 847], [1074, 1104], [1374, 1404], [1642, 1672], [2083, 2113], [2470, 2500]],
            [[262, 292], [490, 520], [760, 790], [1014, 1044], [1308, 1338], [1621, 1651], [1878, 1908], [2284, 2304]],
            [[358, 388], [594, 624], [850, 880], [1101, 1131], [1365, 1395], [1659, 1689], [1890, 1920], [2269, 2299]],
            [[316, 346], [567, 597], [851, 881], [1117, 1147], [1407, 1437], [1712, 1742], [1971, 2001], [2376, 2406]],
            [[292, 322], [589, 619], [851, 881], [1187, 1217], [1487, 1517], [1800, 1830], [2068, 2098], [2480, 2510]],
            [[242, 272], [538, 568], [845, 875], [1134, 1164], [1520, 1550], [1843, 1873], [2093, 2123], [2498, 2528]]]
#fps: 30 resolution: 640*480

def initIdx():
    global index
    idx["y04"] = 0
    idx["y05"] = 1
    idx["y06"] = 2
    idx["y07"] = 3
    idx["y08"] = 4
    idx["y09"] = 5
    idx["y10"] = 6
    idx["y12"] = 7
    idx["y14"] = 8
    idx["y16"] = 9
    idx["y17"] = 10
    idx["y18"] = 11
    idx["y20"] = 12
    idx["y21"] = 13


def gen_name(prefix, emotion, cnt):
    return os.path.join(WORK_DIR, emotion, prefix + "_" + str(cnt) + ".jpg")


if __name__ == "__main__":
    items = os.listdir(WORK_DIR)
    initIdx()
    for item in items:
        parts = item.split(".")
        if len(parts) < 2 or parts[1] != "avi":
            continue
        row = durations[idx[parts[0][0:3]]]
        video = cv2.VideoCapture(item)
        frame_num = video.get(7)
        print(item, " ", frame_num)
        cnt = 0
        curEmotion = 0
#        print(row[curEmotion][0])
        while cnt < int(frame_num):
            ret, frame = video.read()
            cnt += 1
            if cnt < row[curEmotion][0] * 30:
                continue
            elif cnt >= row[curEmotion][0] * 30 and cnt <= row[curEmotion][1] * 30:
                print("Write ", curEmotion, " ", TYPES[curEmotion], " ", cnt)
                cv2.imwrite(gen_name(parts[0], TYPES[curEmotion], cnt), frame)
            else:
                curEmotion += 1
                if curEmotion > 7:
                    break
                if row[curEmotion][0] < 0:
                    curEmotion += 1
                    if curEmotion > 7:
                        break
        video.release()
        