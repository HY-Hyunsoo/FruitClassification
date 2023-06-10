import cv2
import datetime

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# YOLOv8 모델 불러오기
model = YOLO("best.pt")

# Tracker 불러오기
# max_age: 얼마나 많은 프레임들이 트랙킹에서 잊혀질 수 있는지 결정하는 파라미터
tracker = DeepSort(max_age=50)

# 0.8 이상의 confidence를 가지고 있다면 해당 물체인 것으로 판단
CONFIDENCE_THRESHOLD = 0.8

# 경계 박스의 색 설정, 초록색
GREEN = (0, 255, 0)

# 트랙ID 텍스트의 색 설정, 흰색
WHITE = (255, 255, 255)

# 비디오 캡처 객체 초기화
video_cap = cv2.VideoCapture(0)

# 메인 반복문
while True:
	# FPS(초당 프레임 수)를 계산하기 위해 현재 시간(시작 시간) 구하기
	start = datetime.datetime.now()

	# 비디오 캡처 객체로부터 프레임 읽어오기
	ret, frame = video_cap.read()

	# 만약 더 이상 읽어올 프레임이 없다면 종료
	if not ret:
		break

	# 프레임 상에서 YOLO 모델 실행
	detections = model(frame)[0]
	
	# 경계 박스와 confidence들의 리스트 초기화
	results = []

	# 발견한 물체들에 대한 반복문
	for data in detections.boxes.data.tolist():
		# 물체 인식과 관련된 confidence 값 추출
		confidence = data[4]

		# confidence 값이 낮은 물체는 필터링
		if float(confidence) < CONFIDENCE_THRESHOLD:
			continue

		# 프레임에 경계 박스 그리기
		xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

		# 클래스ID 얻어오기
		class_id = int(data[5])

		# 인식된 물체의 이름 구하기
		name = detections.names[class_id]

		# 경계 박스, confidence, 클래스ID를 results 리스트에 추가
		results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
		cv2.putText(frame, str(name), (xmin + 10, ymin - 8), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

	# 발견한 물체들의 트랙킹에 대한 반복문
	tracks = tracker.update_tracks(results, frame=frame)
	for track in tracks:
		# 만약 컨펌되지 않은 트랙킹이라면 무시
		if not track.is_confirmed():
			continue

		# 트랙 ID와 경계 박스 얻어오기
		track_id = track.track_id
		ltrb = track.to_ltrb()

		xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

		# 경계 박스와 트랙 ID 그리기
		cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
		cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
		cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

	# FPS를 계산하기 위해 현재 시간(끝 시간) 구하기
	end = datetime.datetime.now()

	# 1 프레임을 처리하는데 걸린 시간 표시
	total = (end - start).total_seconds()
	print(f"1 프레임을 처리하는 데 소요된 시간: {total * 1000:.0f}ms")

	# FPS 계산 및 프레임 상에 그리기
	fps = f"FPS: {1 / total:.2f}"
	cv2.putText(frame, fps, (50, 50),
		cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

	# 프레임을 스크린에 표시
	cv2.imshow("Frame", frame)

	# q 키를 입력할 경우 종료
	if cv2.waitKey(1) == ord("q"):
		break

video_cap.release()
cv2.destroyAllWindows()
