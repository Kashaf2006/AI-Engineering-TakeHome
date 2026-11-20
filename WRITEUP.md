# Write-up Questions

## 1. Your timings
My first working version took about 1 day and one night on my short test video. My final version took 50 minutes. The single change that saved the most time was that my computer dont have that graphics to play or check 150 housr video so i have to test short video for that and for adjusting and changing its extension is the hardest part. This made the testing much faster.
## 2. Your pipeline
i write a code in google colab that firsts change my .mp video to .parquet file .The script first reads the .parquet file using pandas. It loops through each row, extracts the raw video bytes from the mp4 column, and saves it temporarily so OpenCV can read it. Then, it loops through each frame, passes it to the YOLO11L model, and counts the objects. It groups the counts by the second they appear in. Finally, it calculates the median people count and checks if an object is present in over 50% of the frames before writing the result to a JSON line.

## 3. Where the time goes
Most of the time is spent on YOLO model inference and testing in google colab . First i have to download the video from youtube to computer and adjust it extension i have to use cookies extension and other stuff . I measured this by checking the progress bar and seeing that decoding the frames with OpenCV is fast, but the GPU takes about 9-10ms per frame. Since there are thousands of frames, this adds up quickly

## 4. Checking correctness
 I checked correctness by running the script on a short test video (about 3 min) and printing the output. I verified that the seconds were not skipped and that the "objects" list only contained items found in at least half the frames, as the assignment required. I also checked that the people count was the median, not the average.

## 5. Working with AI
I used AI to help write the code and testing . One suggestion it gave was to rename the video file to inputvideo.mp4 directly in the code. However, I found out it was wrong because my computer was hiding the file extension, causing errors. I rejected it and instead wrote a Python snippet to rename the file safely inside the code after checking its real name

## 6. Next steps
right now it only check one video but we can do it for multiple videos in the future. 
