import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press:")
print("n - Normal Speed")
print("s - Slow Motion")
print("f - Fast Motion")
print("q - Quit")

delay = 30  # Normal speed

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Display the video
    cv2.imshow("Video Processing", frame)

    # Wait according to selected speed
    key = cv2.waitKey(delay) & 0xFF

    if key == ord('s'):
        delay = 100      # Slow motion
    elif key == ord('n'):
        delay = 30       # Normal speed
    elif key == ord('f'):
        delay = 1        # Fast motion
    elif key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
