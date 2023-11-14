import streamlit as st
from streamlit_webrtc import webrtc_streamer
from pyzbar.pyzbar import decode
import cv2
import numpy as np

st.title("App Streamlit WebRTC Barcode Scanner")

# Configurazione WebRTC
webrtc_ctx = webrtc_streamer.stream()

@st.cache
def detect_barcodes(frame):
    decoded_objects = decode(frame)
    for obj in decoded_objects:
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            cv2.polylines(frame, [hull], True, (0, 255, 0), 2)
        else:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)

        barcode_data = obj.data.decode("utf-8")
        cv2.putText(frame, barcode_data, (obj.rect[0], obj.rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

if webrtc_ctx.video_transformer is None:
    webrtc_ctx.video_transformer = VideoTransformer()

class VideoTransformer():
    def __init__(self):
        self.threshold = 0.5

    def transform(self, frame):
        frame_with_barcodes = detect_barcodes(frame.to_ndarray(format="bgr24"))
        return frame_with_barcodes

if __name__ == "__main__":
    import streamlit as st

    # Esegui l'app Streamlit
    st.set_page_config(
        page_title="App Streamlit WebRTC Barcode Scanner",
        page_icon="📊",
        layout="centered",
    )
