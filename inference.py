import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
import torchvision
import json
import numpy as np
from collections import defaultdict

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Inference():

    def process_video(self,model, video_path, output_json,plot_path,anomalies_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        anomaly_scores = []
        timestamps = []

        for i in range(frame_count):
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (320, 240))
            frame = torchvision.transforms.ToTensor()(frame)
            frame = (frame - 0.5) / 0.5
            frame = frame.unsqueeze(0)

            with torch.no_grad():
                reconstructed_frame = model(frame)

            mse_loss = nn.MSELoss(reduction='none')
            error = mse_loss(frame, reconstructed_frame).sum().item()

            anomaly_scores.append(error)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)  # Convert to seconds

            # Update the output JSON
            self.update_json_with_timestamp(output_json, timestamps[-1], error)
            self.plot_anomaly_scores(timestamps, anomaly_scores,plot_path)
            self.detect_anomalies(output_json, 5, 3,4,2,anomalies_path)


        cap.release()

        return timestamps, anomaly_scores

    def update_json_with_timestamp(self,output_json, timestamp, anomaly_score):
        try:
            with open(output_json, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
            
        data[str(timestamp)] = anomaly_score
        
        with open(output_json, 'w') as f:
            json.dump(data, f)

    def detect_anomalies(self,json_path, window_size, min_duration, max_duration, top_k, output_anomalies_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        timestamps = list(map(float, data.keys()))
        scores = list(data.values())

        heightened_periods = defaultdict(float)
        max_heightened_score = 0.0
        start_time = 0

        for i in range(len(scores)):
            if i - window_size >= 0:
                avg_score = np.mean(scores[i-window_size+1:i+1])
                heightened_periods[(start_time, timestamps[i])] = avg_score
                max_heightened_score = max(max_heightened_score, avg_score)

            if scores[i] > max_heightened_score:
                start_time = timestamps[i]
                max_heightened_score = scores[i]

        sorted_periods = sorted(heightened_periods.items(), key=lambda x: x[1], reverse=True)
        filtered_periods = [(start, end, score) for (start, end), score in sorted_periods 
                            if end - start >= min_duration and end - start <= max_duration]
        
        
        with open(output_anomalies_path, 'w') as f:
            json.dump(filtered_periods[:top_k], f)


    def plot_anomaly_scores(self,timestamps, anomaly_scores, save_path):
        plt.plot(timestamps, anomaly_scores, label='Anomaly Score', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Detection')
        plt.savefig(save_path)  

    

model = Autoencoder()
model.load_state_dict(torch.load('models/anomaly_detector.pth', map_location='cpu'))  # Load on CPU
model.eval()

video_path = 'data/anomaly/theftPick.avi'
output_json = 'results/output-demo.json'
plot_path = 'results/plot-demo.png'
anomalies_path = 'results/anomalies-demo.json'

inf = Inference()
inf.process_video(model, video_path, output_json,plot_path,anomalies_path)
