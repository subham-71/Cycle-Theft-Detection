import json
import numpy as np
from collections import defaultdict

def update_heightened_periods(json_path, window_size, min_duration, max_duration, top_k):
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
    

    return filtered_periods[:top_k]

json_path = 'results/output2.json'
window_size = 3  
min_duration = 3  
max_duration = 4  
top_k = 10
sorted_heightened_periods = update_heightened_periods(json_path, window_size, min_duration, max_duration, top_k)

print(f"Top {top_k} Heightened Periods:")
for i, (start, end, score) in enumerate(sorted_heightened_periods, 1):
    print(f"{i}. Start time: {start} s, End time: {end} s, Score: {score}")
