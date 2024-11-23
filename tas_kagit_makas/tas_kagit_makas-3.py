from inference import get_model
import supervision as sv
import cv2
import api_anahtar

model = get_model(model_id="rock-paper-scissors-sxsw/14", api_key=api_anahtar.api_key)

cap = cv2.VideoCapture("tas_kagit_makas-2.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
    results = model.infer(frame)[0]

    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results)

    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    # display the image
    cv2.imshow("Webcam", annotated_image)

    k = cv2.waitKey(1)

    if k % 256 == 27:
        print("Esc tuşuna basıldı.. Kapatılıyor..")
        break
