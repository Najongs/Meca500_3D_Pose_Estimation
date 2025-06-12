import os
import cv2
import json
import numpy as np
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as R, Slerp
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

save_path = "/home/zed_box/Documents/ZED_Cap_make_dataset/ArUco_cap"

view_name = "leftcam"  
view_map = {
    "leftcam": sl.VIEW.LEFT,
    "rightcam": sl.VIEW.RIGHT
}
view = view_map[view_name]

camera_serial = 49045152
camera_position_map = {
    41182735: "front",
    49429257: "right",
    44377151: "left",
    49045152: "top"
}
camera_position = camera_position_map.get(camera_serial, "unknown")

# 1. ZED 카메라 초기화
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1200
init_params.camera_fps = 30
init_params.coordinate_units = sl.UNIT.METER
init_params.set_from_serial_number(camera_serial)

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"ZED 초기화 실패: {status}")
    exit()

runtime_parameters = sl.RuntimeParameters()
image_zed = sl.Mat()

# 2. 아루코 딕셔너리 및 파라미터 설정
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters

# right camera
focal_right_x = calibration_params.right_cam.fx
focal_right_y = calibration_params.right_cam.fy
center_right_x = calibration_params.right_cam.cx
center_right_y = calibration_params.right_cam.cy
dist_coeffs = calibration_params.right_cam.disto

camera_right_matrix = np.array([[focal_right_x, 0, center_right_x],
                          [0, focal_right_y, center_right_y],
                          [0, 0, 1]])
dist_right_coeffs = np.array([dist_coeffs[0], dist_coeffs[1], dist_coeffs[2], dist_coeffs[3], dist_coeffs[4],dist_coeffs[5], dist_coeffs[6], dist_coeffs[7]])

# left camera
focal_left_x = calibration_params.left_cam.fx
focal_left_y = calibration_params.left_cam.fy
center_left_x = calibration_params.left_cam.cx
center_left_y = calibration_params.left_cam.cy
dist_coeffs = calibration_params.left_cam.disto

camera_left_matrix = np.array([[focal_left_x, 0, center_left_x],
                                [0, focal_left_y, center_left_y],
                                [0, 0, 1]])
dist_left_coeffs = np.array([dist_coeffs[0], dist_coeffs[1], dist_coeffs[2], dist_coeffs[3], dist_coeffs[4],dist_coeffs[5], dist_coeffs[6], dist_coeffs[7]])

if view_name == "leftcam":
    camera_matrix = camera_left_matrix
    dist_coeffs = dist_left_coeffs
else:
    camera_matrix = camera_right_matrix
    dist_coeffs = dist_right_coeffs

print("실행 중... ESC로 종료")
marker_size = 0.05
marker_3d_edges = np.array([
    [0, 0, 0],                       # top-left
    [marker_size, 0, 0],            # top-right
    [marker_size, marker_size, 0],  # bottom-right
    [0, marker_size, 0]             # bottom-left
], dtype='float32')

# 색상 정의
blue_BGR = (255, 0, 0)

# EMA 상태 저장용 변수 (마커별)
ema_tvecs = {}
ema_rvecs = {}
alpha = 0.1  # EMA 계수

def update_ema(prev, current, alpha):
    return alpha * current + (1 - alpha) * prev
def rvec_to_quat(rvec):
    return R.from_rotvec(rvec.flatten()).as_quat()  # [x, y, z, w]
def quat_to_rvec(quat):
    return R.from_quat(quat).as_rotvec().reshape((3, 1))
def update_quat_ema(prev_quat, current_quat, alpha):
    # Quaternion SLERP (선형 근사)
    return (1 - alpha) * prev_quat + alpha * current_quat
def slerp_ema_stable(prev_quat, curr_quat, alpha):
    # 부호 정렬 (dot product < 0이면 반대 방향)
    if np.dot(prev_quat, curr_quat) < 0.0:
        curr_quat = -curr_quat
    # SLERP 수행
    key_rots = R.from_quat([prev_quat, curr_quat])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    interp_rot = slerp([alpha])[0]
    return interp_rot.as_quat()

while True:
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_image(image_zed, view)
        frame = image_zed.get_data()

        # 이미지 왜곡 보정
        frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
        # 마커 검출
        corners, ids, rejected = detector.detectMarkers(frame_undistorted)
        
        # 마커가 검출되면 표시 및 포즈 추정
        if corners:
            gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
            
            # 코너 보정 (subpixel refinement)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            for i in range(len(corners)):
                cv2.cornerSubPix(gray, corners[i], winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)
    
            for i, corner in enumerate(corners):
                marker_id = int(ids[i][0])  # 마커 ID
                corner = np.array(corner).reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corner

                # 코너 포인트 좌표 변환
                topRightPoint = (int(topRight[0]), int(topRight[1]))
                topLeftPoint = (int(topLeft[0]), int(topLeft[1]))
                bottomRightPoint = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeftPoint = (int(bottomLeft[0]), int(bottomLeft[1]))

                # 코너 포인트 표시
                cv2.circle(frame_undistorted, topLeftPoint, 4, blue_BGR, -1)
                cv2.circle(frame_undistorted, topRightPoint, 4, blue_BGR, -1)
                cv2.circle(frame_undistorted, bottomRightPoint, 4, blue_BGR, -1)
                cv2.circle(frame_undistorted, bottomLeftPoint, 4, blue_BGR, -1)

                # PnP로 포즈 추정
                ret, rvec, tvec = cv2.solvePnP(marker_3d_edges, corner, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                rvec, tvec = cv2.solvePnPRefineLM(marker_3d_edges, corner, camera_matrix, dist_coeffs, rvec, tvec)

                if ret:
                    curr_quat = R.from_rotvec(rvec.flatten()).as_quat()

                    # EMA 적용
                    if marker_id not in ema_tvecs:
                        ema_tvecs[marker_id] = tvec
                        ema_rvecs[marker_id] = curr_quat
                    else:
                        ema_tvecs[marker_id] = update_ema(ema_tvecs[marker_id], tvec, alpha)
                        ema_rvecs[marker_id] = slerp_ema_stable(ema_rvecs[marker_id], curr_quat, alpha)

                    # EMA 결과로 좌표 및 각도 추출
                    tvec_filtered = ema_tvecs[marker_id]
                    rvec_filtered = R.from_quat(ema_rvecs[marker_id]).as_rotvec().reshape((3, 1))

                    # 위치 및 회전 정보 계산
                    x = round(tvec_filtered[0][0], 6)
                    y = round(tvec_filtered[1][0], 6)
                    z = round(tvec_filtered[2][0], 6)
                    rx = round(np.rad2deg(rvec_filtered[0][0]), 4)
                    ry = round(np.rad2deg(rvec_filtered[1][0]), 4)
                    rz = round(np.rad2deg(rvec_filtered[2][0]), 4)

                    # 정보 표시
                    pos_text = f"Pos: ({x}, {y}, {z})m"
                    rot_text = f"Rot: ({rx}, {ry}, {rz})deg"

                    # 마커 ID 표시
                    id_text = f"ID: {marker_id}"
                    cv2.putText(frame_undistorted,
                                id_text,
                                (int(topLeft[0]-10), int(topLeft[1]-20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                    cv2.putText(frame_undistorted, 
                                pos_text,
                                (int(topLeft[0]-10), int(topLeft[1]+10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)

                    cv2.putText(frame_undistorted,
                                rot_text,
                                (int(topLeft[0]-10), int(topLeft[1]+40)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)

                    # 좌표축 표시 (EMA 적용된 포즈로)
                    cv2.drawFrameAxes(frame_undistorted, camera_matrix, dist_coeffs,
                                    rvec_filtered, tvec_filtered, marker_size/2)

        
        # 프레임 표시
        cv2.imshow('ArUco Marker Detection', frame_undistorted)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            base_filename = f"{camera_position}_{camera_serial}_{view_name}_{timestamp}"

            output_image_path = os.path.join(save_path, f"{base_filename}.png")
            cv2.imwrite(output_image_path, frame_undistorted)

            # 마커 정보 저장
            marker_data = {}
            for marker_id in ema_tvecs:
                tvec = ema_tvecs[marker_id].flatten().tolist()
                quat = ema_rvecs[marker_id].tolist()
                
                # 해당 마커의 픽셀 좌표를 찾기 위해 반복
                for i, id_array in enumerate(ids):
                    if int(id_array[0]) == marker_id:
                        marker_corners = corners[i].reshape((4, 2)).tolist()
                        break
                else:
                    marker_corners = []

                marker_data[marker_id] = {
                    "position_m": {
                        "x": round(tvec[0], 6),
                        "y": round(tvec[1], 6),
                        "z": round(tvec[2], 6)
                    },
                    "rotation_quat": {
                        "x": round(quat[0], 4),
                        "y": round(quat[1], 4),
                        "z": round(quat[2], 4),
                        "w": round(quat[3], 4)
                    },
                    "corners_pixel": marker_corners
                }

            # JSON 저장
            output_json_path = os.path.join(save_path, f"{base_filename}.json")
            with open(output_json_path, "w") as json_file:
                json.dump(marker_data, json_file, indent=4)

            print(f"[INFO] 저장 완료: {output_image_path}, {output_json_path}")

            break

zed.close()
cv2.destroyAllWindows()
