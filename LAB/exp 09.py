import cv2

# Read the image
image = cv2.imread("sample.png")

# Check if the image is loaded
if image is None:
    print("Error: Image not found!")
else:
    # Resize image to a bigger size (2x)
    bigger = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Resize image to a smaller size (0.5x)
    smaller = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # Display the images
    cv2.imshow("Original Image", image)
    cv2.imshow("Bigger Image", bigger)
    cv2.imshow("Smaller Image", smaller)

    # Wait until a key is pressed
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()
