import cv2

from ultralytics import YOLO
from collections import defaultdict

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon

model_path = "/home/bcml1/WBC-Counting/ultralytics-Yolov8/model/best_recon.pt"
input_path = "/home/bcml1/WBC-Counting/data/videos/Reconstructed/video6_right.mp4"
output_path = "/home/bcml1/WBC-Counting/ultralytics-Yolov8/video6Right_Counting_recon_1022.mp4"

#모델 경로
model = YOLO(model_path)
#영상 경로
#cap = cv2.VideoCapture("/home/bcml1/WBC_project/EVREAL/outputs/t10ms/no_18/dvs_event_210914_18/ET-Net_100Hz.mp4")
cap = cv2.VideoCapture(input_path)
assert cap.isOpened(), "Error reading video file"
#영상 fps및 frame 크기 가져오기
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Counting region height
counting_region_height = 20

# Offset of Counting region location
#  100: bottom
#    0: mid
# -100: top
offset = 0

# 영상 저장
video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

class ObjectCounter:

    def __init__(
        self,
        names,
        reg_pts=None,
        count_reg_color=(255, 0, 255),
        count_txt_color=(0, 0, 0),
        count_bg_color=(255, 255, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        track_color=None,
        region_thickness=5,
        line_dist_thresh=15,
        cls_txtdisplay_gap=50,
    ):

        # Region & Line Information
        self.line_dist_thresh = line_dist_thresh
        self.counting_region = None
        self.region_color = count_reg_color
        self.region_thickness = region_thickness

        # Image and annotation Information
        self.im0 = None
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts

        self.names = names  # Classes names
        self.annotator = None  # Annotator

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.count_txt_thickness = 0
        self.count_txt_color = count_txt_color
        self.count_bg_color = count_bg_color
        self.cls_txtdisplay_gap = cls_txtdisplay_gap
        self.fontsize = 0.6

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        self.track_color = track_color

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""

        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            if clss[0] is not None:
                self.reg_center = (boxes[0][1] + boxes[0][3]) / 2
                self.reg_pts = [(boxes[0][0], self.reg_center - counting_region_height/2 + offset), (boxes[0][0], self.reg_center + counting_region_height/2 + offset), (boxes[0][2], self.reg_center + counting_region_height/2 + offset), (boxes[0][2], self.reg_center - counting_region_height/2 + offset)]
                self.counting_region = Polygon(self.reg_pts)
                self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

                # Store class info
                if self.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.names[cls]] = {"IN": 0, "OUT": 0}

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color or colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Count objects in any polygon
                if len(self.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))

                    if prev_position is not None and is_inside and track_id not in self.count_ids:
                        self.count_ids.append(track_id)

                        if (box[1] - prev_position[1]) * (self.counting_region.centroid.y - prev_position[1]) < 0:
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]["IN"] += 1
                        else:
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]["OUT"] += 1

                # Count objects using line
                elif len(self.reg_pts) == 2:
                    if prev_position is not None and track_id not in self.count_ids:
                        distance = Point(track_line[-1]).distance(self.counting_region)
                        if distance < self.line_dist_thresh and track_id not in self.count_ids:
                            self.count_ids.append(track_id)

                            if (box[1] - prev_position[1]) * (self.counting_region.centroid.y - prev_position[1]) < 0:
                                self.in_counts += 1
                                self.class_wise_count[self.names[cls]]["IN"] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.names[cls]]["OUT"] += 1

        labels_dict = {}

        for key, value in self.class_wise_count.items():
            if value["IN"] != 0 or value["OUT"] != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    continue
                elif not self.view_in_counts:
                    labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                elif not self.view_out_counts:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                else:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

        if labels_dict:
            self.annotator.display_analytics(self.im0, labels_dict, self.count_txt_color, self.count_bg_color, 2)

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks)  # draw region even if no objects

        return self.im0

# Init Object Counter
counter = ObjectCounter(
    view_in_counts=True,
    view_out_counts=True,

    count_reg_color = (255,0,255),      #region point 색깔
    region_thickness=2,                 #region point 두께
    line_thickness=1,                   #?? 두께

    count_txt_color = (0,0,0),          #결과 박스 글자 색깔
    count_bg_color = (255,255,255),     #결과 박스 배경 색깔

    names=model.names,
    draw_tracks=True,                   #WBC 추적 선
    track_thickness=2,                  #WBC 추적 선 두께

    line_dist_thresh=15,
    cls_txtdisplay_gap=50,

)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)

    # im0 <- 최종 이미지 결과물
    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()