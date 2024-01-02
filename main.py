from main_detect import detect,load_model,show_results
import torch
import cv2

if __name__ == '__main__':
    img_size = 640
    weights = "./weights/yolov5n-face.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)

    frame = cv2.imread("./images/zidane.jpg")
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    dict_list = detect(model, frame, device, img_size)
    result_img = show_results(frame, dict_list)
    cv2.imshow("Detection", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()